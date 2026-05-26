import argparse
import inspect
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / ".deps"))


def init_dist(local_rank: int, num_ranks: int, master_addr: str, master_port: int):
    torch.cuda.set_device(local_rank)
    params = {
        "backend": "nccl",
        "init_method": f"tcp://{master_addr}:{master_port}",
        "world_size": num_ranks,
        "rank": local_rank,
    }
    if "device_id" in inspect.signature(dist.init_process_group).parameters:
        params["device_id"] = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(**params)


def wait_works(works):
    for work in works or []:
        if work.work is not None:
            work.work.wait()


def make_group(local_rank: int, group_ranks, group_name: str):
    from sglang.srt.distributed.parallel_state import GroupCoordinator

    return GroupCoordinator(
        group_ranks=group_ranks,
        local_rank=local_rank,
        torch_distributed_backend="nccl",
        use_pynccl=False,
        use_pymscclpp=False,
        use_custom_allreduce=False,
        use_torch_symm_mem_all_reduce=False,
        use_hpu_communicator=False,
        use_xpu_communicator=False,
        use_npu_communicator=False,
        use_message_queue_broadcaster=False,
        group_name=group_name,
    )


def worker(local_rank: int, args: argparse.Namespace):
    sys.path.insert(0, os.path.join(args.sglang_root, "python"))
    init_dist(local_rank, args.num_processes, args.master_addr, args.master_port)

    rank = dist.get_rank()
    assert args.num_processes % args.tp_size == 0
    pp_size = args.num_processes // args.tp_size
    assert pp_size == 2, "This smoke test currently expects pp_size=2."

    pp_group_ranks = [
        [tp_rank + pp_rank * args.tp_size for pp_rank in range(pp_size)]
        for tp_rank in range(args.tp_size)
    ]
    attn_group_ranks = [
        [pp_rank * args.tp_size + tp_rank for tp_rank in range(args.tp_size)]
        for pp_rank in range(pp_size)
    ]
    pp_group = make_group(local_rank, pp_group_ranks, "pp")
    attn_group = (
        make_group(local_rank, attn_group_ranks, "attention_tp")
        if args.tp_size > 1
        else None
    )

    pp_rank = rank // args.tp_size
    src_pp_rank = 0
    dst_pp_rank = 1

    if pp_rank == src_pp_rank:
        for i in range(args.iters):
            hidden_states = torch.full(
                (args.num_tokens, args.hidden),
                float(i + 1),
                dtype=torch.bfloat16,
                device="cuda",
            )
            residual = torch.full_like(hidden_states, float(i + 2))
            fallback_tensor = (
                torch.arange(17, dtype=torch.int64, device="cuda") + i * 100
            )
            tensor_dict = {
                "hidden_states": hidden_states,
                "residual": residual,
                "fallback_tensor": fallback_tensor,
                "metadata_value": f"kept-on-cpu-path-{i}",
            }
            works = pp_group.send_tensor_dict(
                tensor_dict=tensor_dict,
                dst=dst_pp_rank,
                all_gather_group=attn_group,
                async_send=True,
            )
            wait_works(works)
        torch.cuda.synchronize()
        assert pp_group._deepep_pp_sent_tensors == 2 * args.iters, (
            "expected hidden_states and residual to use DeepEP PP, got "
            f"{pp_group._deepep_pp_sent_tensors}"
        )
    elif pp_rank == dst_pp_rank:
        for i in range(args.iters):
            recv_dict = pp_group.recv_tensor_dict(
                src=src_pp_rank, all_gather_group=attn_group
            )
            torch.cuda.synchronize()
            expected_hidden_states = torch.full(
                (args.num_tokens, args.hidden),
                float(i + 1),
                dtype=torch.bfloat16,
                device="cuda",
            )
            expected_residual = torch.full_like(
                expected_hidden_states, float(i + 2)
            )
            expected_fallback_tensor = (
                torch.arange(17, dtype=torch.int64, device="cuda") + i * 100
            )
            assert torch.equal(recv_dict["hidden_states"], expected_hidden_states)
            assert torch.equal(recv_dict["residual"], expected_residual)
            assert torch.equal(recv_dict["fallback_tensor"], expected_fallback_tensor)
            assert recv_dict["metadata_value"] == f"kept-on-cpu-path-{i}"
        assert pp_group._deepep_pp_recv_tensors == 2 * args.iters, (
            "expected hidden_states and residual to use DeepEP PP, got "
            f"{pp_group._deepep_pp_recv_tensors}"
        )

    dist.barrier()
    pp_group.destroy()
    if attn_group is not None:
        attn_group.destroy()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(
        description="Correctness smoke for SGLang tensor_dict DeepEP PP payloads"
    )
    parser.add_argument("--num-processes", type=int, default=2)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--num-tokens", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--sglang-root", type=str, default="/root/sglang")
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=8491)
    args = parser.parse_args()

    max_tensor_bytes = args.num_tokens * args.hidden * 2 // args.tp_size
    os.environ.setdefault("SGLANG_DEEPEP_PP", "1")
    os.environ.setdefault("SGLANG_DEEPEP_PP_MIN_BYTES", str(max_tensor_bytes))
    os.environ.setdefault("SGLANG_DEEPEP_PP_MAX_TENSOR_BYTES", str(max_tensor_bytes))
    os.environ.setdefault("SGLANG_DEEPEP_PP_NUM_INFLIGHT", "4")

    torch.multiprocessing.spawn(
        worker, args=(args,), nprocs=args.num_processes
    )


if __name__ == "__main__":
    main()

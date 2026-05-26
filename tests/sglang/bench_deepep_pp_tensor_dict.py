import argparse
import inspect
import os
import sys
import time
from pathlib import Path
from typing import Iterable

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


def make_pp_group(local_rank: int, world_size: int):
    from sglang.srt.distributed.parallel_state import GroupCoordinator

    return GroupCoordinator(
        group_ranks=[list(range(world_size))],
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
        group_name="pp",
    )


def summarize(name: str, elapsed_s: float, payload_bytes: int, rank: int, world_size: int):
    values = [None for _ in range(world_size)]
    dist.all_gather_object(values, elapsed_s)
    if rank != 0:
        return

    active_values = [values[0], values[1]]
    max_s = max(active_values)
    min_s = min(active_values)
    avg_s = sum(active_values) / len(active_values)
    print(
        f"{name:18s} avg={avg_s * 1e6:9.3f} us "
        f"min={min_s * 1e6:9.3f} us max={max_s * 1e6:9.3f} us "
        f"payload={payload_bytes / 1024 / 1024:7.2f} MiB "
        f"bw={payload_bytes / max_s / 1e9:8.3f} GB/s",
        flush=True,
    )


def bench_one(
    pp_group,
    num_tokens: int,
    hidden: int,
    warmup: int,
    iters: int,
    rank: int,
    world_size: int,
):
    src_rank = 0
    dst_rank = 1
    hidden_states = torch.full(
        (num_tokens, hidden), 1.0, dtype=torch.bfloat16, device="cuda"
    )
    residual = torch.full_like(hidden_states, 2.0)
    payload_bytes = hidden_states.nbytes + residual.nbytes

    def send_once():
        works = pp_group.send_tensor_dict(
            {
                "hidden_states": hidden_states,
                "residual": residual,
                "metadata_value": "bench",
            },
            dst=dst_rank,
            async_send=True,
        )
        wait_works(works)

    def recv_once():
        recv_dict = pp_group.recv_tensor_dict(src=src_rank)
        if recv_dict["metadata_value"] != "bench":
            raise AssertionError("metadata mismatch")
        return recv_dict

    dist.barrier()
    for _ in range(warmup):
        if rank == src_rank:
            send_once()
        elif rank == dst_rank:
            recv_once()
        torch.cuda.synchronize()

    dist.barrier()
    begin = time.perf_counter()
    for _ in range(iters):
        if rank == src_rank:
            send_once()
        elif rank == dst_rank:
            recv_once()
        torch.cuda.synchronize()
    dist.barrier()
    elapsed_s = (time.perf_counter() - begin) / iters
    summarize(f"tokens={num_tokens}", elapsed_s, payload_bytes, rank, world_size)
    dist.barrier()


def parse_tokens(spec: str) -> Iterable[int]:
    return [int(x) for x in spec.split(",") if x]


def worker(local_rank: int, args: argparse.Namespace):
    sys.path.insert(0, os.path.join(args.sglang_root, "python"))
    init_dist(local_rank, args.num_processes, args.master_addr, args.master_port)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2, "This benchmark only measures PP2."
    pp_group = make_pp_group(local_rank, world_size)

    if rank == 0:
        mode = "deepep" if os.getenv("SGLANG_DEEPEP_PP") == "1" else "sglang"
        print(
            f"Config: mode={mode}, pp=2, hidden={args.hidden}, "
            f"warmup={args.warmup}, iters={args.iters}, "
            f"deepep_sent={getattr(pp_group, '_deepep_pp_sent_tensors', 'n/a')}",
            flush=True,
        )

    for num_tokens in parse_tokens(args.tokens):
        bench_one(
            pp_group,
            num_tokens,
            args.hidden,
            args.warmup,
            args.iters,
            rank,
            world_size,
        )

    if rank == 0:
        print(
            f"DeepEP counters: sent={pp_group._deepep_pp_sent_tensors}, "
            f"recv={pp_group._deepep_pp_recv_tensors}",
            flush=True,
        )

    dist.barrier()
    pp_group.destroy()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(
        description="PP2 SGLang tensor_dict benchmark with optional DeepEP PP payloads"
    )
    parser.add_argument("--num-processes", type=int, default=2)
    parser.add_argument("--tokens", type=str, default="128,1024,4096")
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--sglang-root", type=str, default="/root/sglang")
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=8501)
    args = parser.parse_args()

    max_tokens = max(parse_tokens(args.tokens))
    max_tensor_bytes = max_tokens * args.hidden * 2
    os.environ.setdefault("SGLANG_DEEPEP_PP_MAX_TENSOR_BYTES", str(max_tensor_bytes))
    os.environ.setdefault("SGLANG_DEEPEP_PP_MIN_BYTES", "1")
    os.environ.setdefault("SGLANG_DEEPEP_PP_NUM_INFLIGHT", "4")

    torch.multiprocessing.spawn(worker, args=(args,), nprocs=args.num_processes)


if __name__ == "__main__":
    main()

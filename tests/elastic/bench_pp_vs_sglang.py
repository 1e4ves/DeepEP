import argparse
import inspect
import math
import os
import sys
import time
from typing import Callable, Dict, List

import torch
import torch.distributed as dist


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


def wait_p2p(works):
    for work in works or []:
        if work.work is not None:
            work.work.wait()


def assert_received(label: str, rank: int, prev_rank: int,
                    tensors: List[torch.Tensor]):
    for i, tensor in enumerate(tensors):
        expected = torch.full_like(tensor, float(prev_rank * 10 + i))
        if not torch.equal(tensor, expected):
            raise AssertionError(f"{label}: rank {rank} tensor {i} received wrong data")


def summarize(name: str, elapsed_s: float, payload_bytes: int, rank: int,
              world_size: int):
    values = [None for _ in range(world_size)]
    dist.all_gather_object(values, elapsed_s)
    if rank != 0:
        return

    avg_s = sum(values) / len(values)
    min_s = min(values)
    max_s = max(values)
    gbps = payload_bytes / avg_s / 1e9
    print(
        f"{name:22s} avg={avg_s * 1e6:9.3f} us "
        f"min={min_s * 1e6:9.3f} us max={max_s * 1e6:9.3f} us "
        f"payload={payload_bytes / 1024 / 1024:7.2f} MiB/rank "
        f"bw={gbps:8.3f} GB/s/rank",
        flush=True,
    )


def bench(
    name: str,
    fn: Callable[[], List[torch.Tensor]],
    payload_bytes: int,
    warmup: int,
    iters: int,
    rank: int,
    world_size: int,
):
    dist.barrier()
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()

    dist.barrier()
    begin = time.perf_counter()
    for _ in range(iters):
        fn()
        torch.cuda.synchronize()
    dist.barrier()
    elapsed_s = (time.perf_counter() - begin) / iters
    summarize(name, elapsed_s, payload_bytes, rank, world_size)
    dist.barrier()


def worker(local_rank: int, args: argparse.Namespace):
    sys.path.insert(0, os.path.join(args.sglang_root, "python"))

    import deep_ep
    from sglang.srt.distributed.parallel_state import GroupCoordinator

    init_dist(local_rank, args.num_processes, args.master_addr, args.master_port)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    ranks = list(range(world_size))
    next_rank = (rank + 1) % world_size
    prev_rank = (rank - 1) % world_size
    pair_src = args.src_rank
    pair_dst = args.dst_rank
    if args.pattern == "pair":
        if pair_src == pair_dst:
            raise ValueError("src-rank and dst-rank must be different")
        if pair_dst not in ((pair_src + 1) % world_size,
                            (pair_src - 1) % world_size):
            raise ValueError("DeepEP PP only supports adjacent ranks in the PP ring")

    # SGLang's current PP proxy path uses GroupCoordinator.send_tensor_dict.
    sglang_pp_group = GroupCoordinator(
        group_ranks=[ranks],
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
        group_name="pp_compare",
    )

    deep_ep_group = dist.new_group(ranks, backend="nccl")

    shape = (args.num_tokens, args.hidden)
    dtype = getattr(torch, args.dtype)
    send_tensors = [
        torch.full(shape, float(rank * 10 + i), dtype=dtype, device="cuda")
        for i in range(args.num_tensors)
    ]
    deepep_recv_tensors = [torch.empty_like(t) for t in send_tensors]
    torch_recv_tensors = [torch.empty_like(t) for t in send_tensors]
    tensor_dict: Dict[str, torch.Tensor] = {
        f"t{i}": tensor for i, tensor in enumerate(send_tensors)
    }
    payload_bytes = sum(t.numel() * t.element_size() for t in send_tensors)

    num_max_tensor_bytes = math.prod(shape) * send_tensors[0].element_size()
    num_max_inflight = max(args.num_max_inflight_tensors, args.num_tensors + 1)
    deepep_buffer = deep_ep.ElasticBuffer(
        deep_ep_group,
        explicitly_destroy=True,
        allow_hybrid_mode=False,
        num_bytes=deep_ep.ElasticBuffer.get_pp_buffer_size_hint(
            num_max_tensor_bytes, num_max_inflight
        ),
    )
    deepep_buffer.pp_set_config(num_max_tensor_bytes, num_max_inflight)

    def deepep_once():
        if args.pattern == "ring":
            for tensor in send_tensors:
                deepep_buffer.pp_send(tensor, next_rank)
            for tensor in deepep_recv_tensors:
                deepep_buffer.pp_recv(tensor, prev_rank)
            return deepep_recv_tensors

        if rank == pair_src:
            for tensor in send_tensors:
                deepep_buffer.pp_send(tensor, pair_dst)
        elif rank == pair_dst:
            for tensor in deepep_recv_tensors:
                deepep_buffer.pp_recv(tensor, pair_src)
            return deepep_recv_tensors
        return []

    def sglang_tensor_dict_once():
        if args.pattern == "ring":
            works = sglang_pp_group.send_tensor_dict(
                tensor_dict, dst=next_rank, async_send=True
            )
            recv_dict = sglang_pp_group.recv_tensor_dict(src=prev_rank)
            wait_p2p(works)
            return [recv_dict[f"t{i}"] for i in range(args.num_tensors)]

        if rank == pair_src:
            works = sglang_pp_group.send_tensor_dict(
                tensor_dict, dst=pair_dst, async_send=True
            )
            wait_p2p(works)
        elif rank == pair_dst:
            recv_dict = sglang_pp_group.recv_tensor_dict(src=pair_src)
            return [recv_dict[f"t{i}"] for i in range(args.num_tensors)]
        return []

    def torch_p2p_once():
        works = []
        if args.pattern == "ring":
            for tensor in torch_recv_tensors:
                works.append(
                    dist.irecv(
                        tensor, src=prev_rank, group=sglang_pp_group.device_group)
                )
            for tensor in send_tensors:
                works.append(
                    dist.isend(
                        tensor, dst=next_rank, group=sglang_pp_group.device_group)
                )
        elif rank == pair_dst:
            for tensor in torch_recv_tensors:
                works.append(
                    dist.irecv(tensor, src=pair_src, group=sglang_pp_group.device_group)
                )
        elif rank == pair_src:
            for tensor in send_tensors:
                works.append(
                    dist.isend(tensor, dst=pair_dst, group=sglang_pp_group.device_group)
                )
        for work in works:
            work.wait()
        return torch_recv_tensors if args.pattern == "ring" or rank == pair_dst else []

    if rank == 0:
        print(
            f"Config: ranks={world_size}, shape={shape}, dtype={args.dtype}, "
            f"num_tensors={args.num_tensors}, pattern={args.pattern}, "
            f"pair={pair_src}->{pair_dst}, warmup={args.warmup}, iters={args.iters}",
            flush=True,
        )

    modes = set(args.modes.split(","))
    funcs = {
        "deepep": deepep_once,
        "sglang_tensor_dict": sglang_tensor_dict_once,
        "torch_p2p": torch_p2p_once,
    }

    for name, fn in funcs.items():
        if name not in modes:
            continue
        received = fn()
        torch.cuda.synchronize()
        if received:
            expected_src = prev_rank if args.pattern == "ring" else pair_src
            assert_received(name, rank, expected_src, received)
        bench(name, fn, payload_bytes, args.warmup, args.iters, rank, world_size)

    deepep_buffer.destroy()
    sglang_pp_group.destroy()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(
        description="Compare DeepEP PP with SGLang PP communication")
    parser.add_argument("--num-processes", type=int, default=2)
    parser.add_argument("--num-tokens", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--num-tensors", type=int, default=1)
    parser.add_argument("--dtype",
                        type=str,
                        default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--num-max-inflight-tensors", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--pattern",
                        type=str,
                        default="pair",
                        choices=["pair", "ring"])
    parser.add_argument("--src-rank", type=int, default=0)
    parser.add_argument("--dst-rank", type=int, default=1)
    parser.add_argument(
        "--modes",
        type=str,
        default="deepep,sglang_tensor_dict,torch_p2p",
        help="Comma-separated: deepep,sglang_tensor_dict,torch_p2p",
    )
    parser.add_argument("--sglang-root", type=str, default="/root/sglang")
    parser.add_argument("--master-addr",
                        type=str,
                        default=os.getenv("MASTER_ADDR", "127.0.0.1"))
    parser.add_argument("--master-port",
                        type=int,
                        default=int(os.getenv("MASTER_PORT", "8371")))
    args = parser.parse_args()

    torch.multiprocessing.spawn(worker, args=(args,), nprocs=args.num_processes)


if __name__ == "__main__":
    main()

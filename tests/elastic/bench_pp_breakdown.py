import argparse
import inspect
import math
import os
import time
from typing import Callable, Iterable

import torch
import torch.distributed as dist

import deep_ep


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


def summarize(name: str, elapsed_s: float, active_ranks: Iterable[int],
              payload_bytes: int, rank: int, world_size: int):
    values = [None for _ in range(world_size)]
    dist.all_gather_object(values, elapsed_s)
    if rank != 0:
        return None

    active = list(active_ranks)
    active_values = [values[i] for i in active]
    max_s = max(active_values)
    min_s = min(active_values)
    bw = payload_bytes / max_s / 1e9
    print(
        f"{name:24s} max={max_s * 1e6:9.3f} us "
        f"min={min_s * 1e6:9.3f} us "
        f"payload={payload_bytes / 1024 / 1024:7.2f} MiB "
        f"bw={bw:8.3f} GB/s",
        flush=True,
    )
    return max_s


def bench(name: str, fn: Callable[[], None], active_ranks: Iterable[int],
          payload_bytes: int, warmup: int, iters: int, rank: int,
          world_size: int):
    dist.barrier()
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()

    dist.barrier()
    begin = time.perf_counter()
    for _ in range(iters):
        fn()
        torch.cuda.synchronize()
    elapsed_s = (time.perf_counter() - begin) / iters
    dist.barrier()
    return summarize(name, elapsed_s, active_ranks, payload_bytes, rank,
                     world_size)


def bench_deepep_send_then_recv(send_fn: Callable[[], None],
                                recv_fn: Callable[[], None], src_rank: int,
                                dst_rank: int, payload_bytes: int, warmup: int,
                                iters: int, rank: int, world_size: int):
    dist.barrier()
    for _ in range(warmup):
        if rank == src_rank:
            send_fn()
            torch.cuda.synchronize()
        dist.barrier()
        if rank == dst_rank:
            recv_fn()
            torch.cuda.synchronize()
        dist.barrier()

    send_elapsed = 0.0
    recv_elapsed = 0.0
    dist.barrier()
    for _ in range(iters):
        if rank == src_rank:
            begin = time.perf_counter()
            send_fn()
            torch.cuda.synchronize()
            send_elapsed += time.perf_counter() - begin
        dist.barrier()

        if rank == dst_rank:
            begin = time.perf_counter()
            recv_fn()
            torch.cuda.synchronize()
            recv_elapsed += time.perf_counter() - begin
        dist.barrier()

    send_avg = send_elapsed / iters if rank == src_rank else 0.0
    recv_avg = recv_elapsed / iters if rank == dst_rank else 0.0
    send_stage = summarize("deepep_send_only", send_avg, (src_rank, ),
                           payload_bytes, rank, world_size)
    recv_stage = summarize("deepep_recv_after_send", recv_avg, (dst_rank, ),
                           payload_bytes, rank, world_size)
    return send_stage, recv_stage


def worker(local_rank: int, args: argparse.Namespace):
    init_dist(local_rank, args.num_processes, args.master_addr, args.master_port)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    src_rank = args.src_rank
    dst_rank = args.dst_rank
    if src_rank == dst_rank:
        raise ValueError("src-rank and dst-rank must be different")
    if dst_rank not in ((src_rank + 1) % world_size,
                        (src_rank - 1) % world_size):
        raise ValueError("DeepEP PP only supports adjacent ranks in the PP ring")

    shape = (args.num_tokens, args.hidden)
    dtype = getattr(torch, args.dtype)
    payload_bytes = math.prod(shape) * torch.empty((),
                                                   dtype=dtype).element_size()

    src = torch.full(shape, float(rank), dtype=dtype, device="cuda")
    local_stage = torch.empty_like(src)
    remote_stage = torch.empty_like(src)
    dst = torch.empty_like(src)

    group = dist.new_group(list(range(world_size)), backend="nccl")
    num_inflight = max(args.num_max_inflight_tensors, 2)
    buffer = deep_ep.ElasticBuffer(
        group,
        explicitly_destroy=True,
        allow_hybrid_mode=False,
        num_bytes=deep_ep.ElasticBuffer.get_pp_buffer_size_hint(
            payload_bytes, num_inflight),
    )
    buffer.pp_set_config(payload_bytes, num_inflight)

    def full_deepep_once():
        if rank == src_rank:
            buffer.pp_send(src, dst_rank)
        elif rank == dst_rank:
            buffer.pp_recv(dst, src_rank)

    def local_copy_once():
        if rank == src_rank:
            local_stage.copy_(src)

    def nccl_p2p_push_once():
        works = []
        if rank == dst_rank:
            works.append(dist.irecv(remote_stage, src=src_rank, group=group))
        elif rank == src_rank:
            works.append(dist.isend(local_stage, dst=dst_rank, group=group))
        for work in works:
            work.wait()

    def remote_copy_once():
        if rank == dst_rank:
            dst.copy_(remote_stage)

    # Fill staging buffers and sanity check the link once.
    if rank == src_rank:
        local_stage.copy_(src)
    dist.barrier()
    nccl_p2p_push_once()
    torch.cuda.synchronize()
    if rank == dst_rank and not torch.equal(
            remote_stage, torch.full_like(remote_stage, float(src_rank))):
        raise AssertionError("NCCL P2P staging transfer sanity check failed")
    dist.barrier()

    if rank == 0:
        print(
            f"Config: ranks={world_size}, pair={src_rank}->{dst_rank}, "
            f"shape={shape}, dtype={args.dtype}, payload={payload_bytes / 1024 / 1024:.2f} MiB, "
            f"warmup={args.warmup}, iters={args.iters}",
            flush=True,
        )

    active = (src_rank, dst_rank)
    full = bench("deepep_full", full_deepep_once, active, payload_bytes,
                 args.warmup, args.iters, rank, world_size)
    send_stage, recv_stage = bench_deepep_send_then_recv(
        lambda: buffer.pp_send(src, dst_rank),
        lambda: buffer.pp_recv(dst, src_rank),
        src_rank,
        dst_rank,
        payload_bytes,
        args.warmup,
        args.iters,
        rank,
        world_size,
    )
    local = bench("local_d2d_copy", local_copy_once, (src_rank, ), payload_bytes,
                  args.warmup, args.iters, rank, world_size)
    push = bench("nccl_p2p_push", nccl_p2p_push_once, active, payload_bytes,
                 args.warmup, args.iters, rank, world_size)
    remote = bench("remote_d2d_copy", remote_copy_once, (dst_rank, ),
                   payload_bytes, args.warmup, args.iters, rank, world_size)

    if rank == 0 and all(x is not None for x in (full, local, push, remote)):
        staged_sum = local + push + remote
        residual = full - staged_sum
        print(
            f"{'staged_sum':24s} max={staged_sum * 1e6:9.3f} us "
            f"(local + nccl_p2p + remote)",
            flush=True,
        )
        print(
            f"{'deepep_minus_staged_sum':24s} {residual * 1e6:9.3f} us",
            flush=True,
        )
        if send_stage is not None and recv_stage is not None:
            sequential_sum = send_stage + recv_stage
            print(
                f"{'deepep_send_then_recv_sum':24s} {sequential_sum * 1e6:9.3f} us",
                flush=True,
            )

    buffer.destroy()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(
        description="Break down PP staging transfer costs")
    parser.add_argument("--num-processes", type=int, default=2)
    parser.add_argument("--num-tokens", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--dtype",
                        type=str,
                        default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--num-max-inflight-tensors", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--src-rank", type=int, default=0)
    parser.add_argument("--dst-rank", type=int, default=1)
    parser.add_argument("--master-addr",
                        type=str,
                        default=os.getenv("MASTER_ADDR", "127.0.0.1"))
    parser.add_argument("--master-port",
                        type=int,
                        default=int(os.getenv("MASTER_PORT", "8381")))
    args = parser.parse_args()

    torch.multiprocessing.spawn(worker, args=(args,), nprocs=args.num_processes)


if __name__ == "__main__":
    main()

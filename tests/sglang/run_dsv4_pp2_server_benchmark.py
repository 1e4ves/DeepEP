import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def wait_health(base_url: str, timeout_s: int, proc: subprocess.Popen):
    deadline = time.time() + timeout_s
    last_error = ""
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5) as resp:
                if resp.status == 200:
                    return
        except Exception as exc:
            last_error = str(exc)
        time.sleep(5)
    raise TimeoutError(f"server did not become healthy: {last_error}")


def terminate_tree(proc: subprocess.Popen):
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + 60
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(1)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_case(args: argparse.Namespace, mode: str):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    deepep_nccl_root = Path("/root/DeepEP/.deps/nvidia/nccl")
    deepep_nccl_so = deepep_nccl_root / "lib" / "libnccl.so.2"
    torch_cuda_deps_override = Path("/root/DeepEP/.torch_cuda_deps_override")
    nccl_override_so = torch_cuda_deps_override / "nvidia" / "nccl" / "lib" / "libnccl.so.2"
    if deepep_nccl_so.exists():
        nccl_override_so.parent.mkdir(parents=True, exist_ok=True)
        if nccl_override_so.is_symlink() or nccl_override_so.exists():
            nccl_override_so.unlink()
        nccl_override_so.symlink_to(deepep_nccl_so)

    base_url = f"http://127.0.0.1:{args.port}"
    case_prefix = f"dsv4_pp{args.pp_size}_tp{args.tp_size}"
    server_log = out_dir / f"{case_prefix}_{mode}_server.log"
    bench_jsonl = out_dir / f"{case_prefix}_{mode}_bench.jsonl"
    bench_log = out_dir / f"{case_prefix}_{mode}_bench.log"

    env = os.environ.copy()
    pythonpath = [
        str(torch_cuda_deps_override),
        "/root/DeepEP/.deps",
        "/root/DeepEP",
        "/root/sglang/python",
        env.get("PYTHONPATH", ""),
    ]
    env["PYTHONPATH"] = ":".join(x for x in pythonpath if x)
    env.setdefault("CUDA_VISIBLE_DEVICES", args.cuda_visible_devices)
    env.setdefault("XDG_CACHE_HOME", "/root/DeepEP/.cache")
    env.setdefault("TRITON_CACHE_DIR", "/root/DeepEP/.triton_cache")
    env.setdefault("TORCH_EXTENSIONS_DIR", "/root/DeepEP/.torch_extensions")
    env.setdefault("SGLANG_JIT_DEEPGEMM_FAST_WARMUP", "1")
    env.setdefault("SGLANG_DSV4_FP4_EXPERTS", "0")
    env.setdefault("EP_NIC_NAME", "mlx5_bond_0")
    if deepep_nccl_root.exists():
        env["EP_NCCL_ROOT_DIR"] = str(deepep_nccl_root)
    if args.force_nccl_symmetric_env:
        env["NCCL_CUMEM_ENABLE"] = "1"
        env["NCCL_NVLS_ENABLE"] = "1"

    env.pop("SGLANG_DEEPEP_PP", None)
    if mode == "deepep":
        env["SGLANG_DEEPEP_PP"] = "1"
        env["SGLANG_DEEPEP_PP_MIN_BYTES"] = str(args.deepep_min_bytes)
        env["SGLANG_DEEPEP_PP_MAX_TENSOR_BYTES"] = str(args.deepep_max_tensor_bytes)
        env["SGLANG_DEEPEP_PP_NUM_INFLIGHT"] = str(args.deepep_num_inflight)
        env["SGLANG_DEEPEP_PP_EAGER_INIT"] = "1"
        env["SGLANG_DEEPEP_PP_LOG_COUNTERS"] = "1"
        env["EP_REUSE_NCCL_COMM"] = "0"
        env["NCCL_CUMEM_ENABLE"] = "1"
        env["NCCL_NVLS_ENABLE"] = "1"
        env.setdefault("EP_JIT_CACHE_DIR", "/root/DeepEP/.deep_ep_jit_cache_sglang_pp_server")

    server_cmd = [
        args.python,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model_path,
        "--tokenizer-path",
        args.model_path,
        "--host",
        "0.0.0.0",
        "--port",
        str(args.port),
        "--trust-remote-code",
        "--tp-size",
        str(args.tp_size),
        "--pp-size",
        str(args.pp_size),
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--max-running-requests",
        str(args.max_running_requests),
        "--chunked-prefill-size",
        str(args.chunked_prefill_size),
        "--max-prefill-tokens",
        str(args.max_prefill_tokens),
        "--cuda-graph-max-bs",
        str(args.cuda_graph_max_bs),
        "--model-loader-extra-config",
        json.dumps({"enable_multithread_load": "true", "num_threads": args.load_threads}),
    ]
    if args.disable_cuda_graph:
        server_cmd.append("--disable-cuda-graph")

    print(f"[{mode}] launching server, log={server_log}", flush=True)
    with server_log.open("w") as log_file:
        proc = subprocess.Popen(
            server_cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            cwd="/root/sglang",
            preexec_fn=os.setsid,
        )

    try:
        wait_health(base_url, args.launch_timeout, proc)
        print(f"[{mode}] server healthy, running benchmark", flush=True)

        bench_cmd = [
            args.python,
            "/root/sglang/python/sglang/bench_serving.py",
            "--backend",
            "sglang",
            "--base-url",
            base_url,
            "--model",
            args.model_path,
            "--dataset-name",
            "random-ids",
            "--random-input-len",
            str(args.input_len),
            "--random-output-len",
            str(args.output_len),
            "--num-prompts",
            str(args.num_prompts),
            "--max-concurrency",
            str(args.max_concurrency),
            "--warmup-requests",
            str(args.warmup_requests),
            "--output-file",
            str(bench_jsonl),
            "--tokenize-prompt",
            "--disable-tqdm",
            "--flush-cache",
        ]
        with bench_log.open("w") as log_file:
            subprocess.run(
                bench_cmd,
                check=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                cwd="/root/sglang",
            )
        print(f"[{mode}] benchmark done, result={bench_jsonl}", flush=True)
    finally:
        print(f"[{mode}] stopping server", flush=True)
        terminate_tree(proc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "deepep", "both"], default="both")
    parser.add_argument("--model-path", default="/root/models/DeepSeek-V4-Flash-FP8")
    parser.add_argument("--python", default="/root/sglang/.venv-cu128/bin/python")
    parser.add_argument("--out-dir", default="/root/DeepEP/results/sglang_pp2")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--cuda-visible-devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--pp-size", type=int, default=2)
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--num-prompts", type=int, default=128)
    parser.add_argument("--max-concurrency", type=int, default=64)
    parser.add_argument("--warmup-requests", type=int, default=8)
    parser.add_argument("--launch-timeout", type=int, default=3600)
    parser.add_argument("--mem-fraction-static", type=float, default=0.78)
    parser.add_argument("--max-running-requests", type=int, default=128)
    parser.add_argument("--chunked-prefill-size", type=int, default=8192)
    parser.add_argument("--max-prefill-tokens", type=int, default=16384)
    parser.add_argument("--cuda-graph-max-bs", type=int, default=128)
    parser.add_argument("--load-threads", type=int, default=64)
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--force-nccl-symmetric-env", action="store_true")
    parser.add_argument("--deepep-min-bytes", type=int, default=1)
    parser.add_argument("--deepep-max-tensor-bytes", type=int, default=67108864)
    parser.add_argument("--deepep-num-inflight", type=int, default=4)
    args = parser.parse_args()

    modes = ["baseline", "deepep"] if args.mode == "both" else [args.mode]
    for mode in modes:
        run_case(args, mode)


if __name__ == "__main__":
    main()

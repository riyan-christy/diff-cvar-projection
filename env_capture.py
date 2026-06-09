"""
env_capture.py
==============
Record the exact environment a run was produced in -> results/env.json.
This is what makes a benchmark reproducible: hardware, driver/CUDA, OS, Python,
and the full pip freeze. Safe to run anywhere (missing pieces are recorded as
null rather than crashing).
"""
import json, os, platform, subprocess, sys, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)


def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def main():
    info = {
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version,
        "cpu_count": os.cpu_count(),
        "uname": sh("uname -a"),
        "nvidia_smi": sh("nvidia-smi"),
        "nvcc": sh("nvcc --version"),
        "git_commit": sh("git rev-parse HEAD"),
        "git_dirty": bool(sh("git status --porcelain")),
    }

    # torch / GPU details (optional)
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["torch_cuda_available"] = torch.cuda.is_available()
        info["torch_cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(0)
            info["gpu_total_mem_gb"] = round(props.total_memory / 1e9, 2)
    except Exception as e:
        info["torch_version"] = None
        info["torch_note"] = f"torch not importable: {e}"

    # full dependency snapshot
    info["pip_freeze"] = sh(f"{sys.executable} -m pip freeze")

    path = os.path.join(RESULTS, "env.json")
    with open(path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"wrote {path}")
    print(f"  platform : {info['platform']}")
    print(f"  python   : {platform.python_version()}")
    print(f"  torch    : {info.get('torch_version')}  cuda={info.get('torch_cuda_available')}")
    print(f"  gpu      : {info.get('gpu_name')}")


if __name__ == "__main__":
    main()


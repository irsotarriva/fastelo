"""Make onnxruntime-gpu work on whatever CUDA the box's PyTorch was built for.

The newest onnxruntime-gpu targets CUDA 13, while most images ship PyTorch built for
CUDA 12 with the CUDA/cuDNN libraries as pip packages. `ensure_onnxruntime()` installs a matching
build if needed, puts torch's NVIDIA libraries on LD_LIBRARY_PATH, and proves in a fresh process
that a session on the given network really runs on the GPU. Returns the environment for the
labelling subprocesses.
"""
import glob
import os
import subprocess
import sys

# onnxruntime-gpu builds to try per CUDA major version of the installed torch (first that works wins)
CANDIDATES = {"12": ["1.23.2", "1.22.0", "1.21.1"], "13": ["", "1.24.1"]}


def torch_env():
    import torch
    libs = sorted(glob.glob(os.path.join(os.path.dirname(torch.__file__), "..", "nvidia", "*", "lib")))
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ":".join([os.path.abspath(p) for p in libs] + [env.get("LD_LIBRARY_PATH", "")])
    return env, (torch.version.cuda or "").split(".")[0]


def _works(onnx, env):
    code = ("import onnxruntime as ort; s = ort.InferenceSession(%r, providers=['CUDAExecutionProvider']); "
            "print('PROVIDERS', s.get_providers())" % onnx)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    return "CUDAExecutionProvider" in r.stdout, (r.stdout + r.stderr).strip()[-400:]


def ensure_onnxruntime(onnx, log=print):
    env, cuda_major = torch_env()
    ok, msg = _works(onnx, env)
    if ok:
        log(f"onnxruntime already uses the GPU (torch CUDA {cuda_major})")
        return env
    for version in CANDIDATES.get(cuda_major, [""]):
        spec = f"onnxruntime-gpu=={version}" if version else "onnxruntime-gpu"
        subprocess.run(f"{sys.executable} -m pip -q uninstall -y onnxruntime onnxruntime-gpu; "
                       f"{sys.executable} -m pip -q install {spec}", shell=True, check=False)
        ok, msg = _works(onnx, env)
        log(f"{spec}: {'GPU OK' if ok else 'no GPU: ' + msg}")
        if ok:
            return env
    raise RuntimeError(f"no onnxruntime-gpu build uses the GPU with torch CUDA {cuda_major}; last error: {msg}")

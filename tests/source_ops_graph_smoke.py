"""Run the optional native Graph test hook under an externally held board lease.

Build with RPU_BUILD_SOURCE_OPS_GRAPH_TEST=ON. Run in separate processes with
RPU_SOURCE_OPS unset, gelu, layernorm, and gelu,layernorm, and set the matching
RPU_KERNEL_LIB_PATH before Python starts. Use the Python/Torch used to build it.
"""
import argparse
import ctypes
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--rmsnorm", action="store_true", help="Run the eight-core RMSNorm lifecycle hook")
    args = parser.parse_args()
    import torch  # Load the extension's Torch dependencies before dlopen.

    native = ctypes.CDLL(str(args.native.resolve(strict=True)), mode=ctypes.RTLD_GLOBAL)
    hook = native.rpu_test_source_rmsnorm_graph if args.rmsnorm else native.rpu_test_source_ops_graph
    hook.argtypes = []
    hook.restype = ctypes.c_int
    return hook()


if __name__ == "__main__":
    raise SystemExit(main())

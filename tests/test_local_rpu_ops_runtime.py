"""Repository runtime selection, without importing Torch or using an RPU."""

import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
ASSET = "rhinoOpLib_rhinoforge_v1.0.0.ref"
HEADERS = (
    "rhino_launch_def.h", "rhino_launch_features.h", "rhino_launch_program.h",
    "rhino_launch_kernel.h", "rhino_launch_buffer.h", "rhino_launch_queue.h", "version.h",
)
RUNTIME_KEYS = (
    "RPU_OPS_ROOT", "RPU_KERNEL_LIB_PATH", "RHINO_LAUNCH_LIB_DIR", "RPU_SOURCE_OPS",
    "PYTHONPATH", "TORCH_PROFILE_DIR", "HW_PERF_DIR", "FLOW_NOISE",
)


def make_runtime(root):
    runtime = root / "runtime"
    launch = runtime / "launch"
    (launch / "lib").mkdir(parents=True)
    (launch / "include").mkdir()
    (runtime / ASSET).touch()
    (runtime / f"{ASSET}.kernels").touch()
    (runtime / "RELEASE.txt").write_text("launch_version=1.0.0\n")
    (launch / "lib/librhino_launch.so").touch()
    for header in HEADERS:
        (launch / "include" / header).touch()
    return runtime


@pytest.fixture
def wrapper_env(tmp_path):
    framework = tmp_path / "RhinoForge"
    framework.mkdir()
    wrapper = framework / "run_wall_qwen35_openloop.sh"
    shutil.copyfile(ROOT / wrapper.name, wrapper)
    runtime = make_runtime(tmp_path / "rpu_ops")
    env_sh = tmp_path / "env.sh"
    env_sh.write_text(f"export CONDA_PREFIX={shlex.quote(str(tmp_path))}\n")
    fake_python = tmp_path / "python"
    fake_python.write_text(
        '#!/bin/bash\nif [[ "$1" == -c ]]; then printf "/tmp/torch/lib\\n"; else\n'
        + "\n".join(f'printf "TEST_{k}=%s\\n" "${{{k}-<unset>}}"'
                    for k in (*RUNTIME_KEYS[:5], "LD_LIBRARY_PATH"))
        + "\nfi\n"
    )
    fake_python.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k not in RUNTIME_KEYS}
    env.update(ENV_SH=str(env_sh), PYTHON_BIN=str(fake_python),
               DATASET_DIR=str(tmp_path), CHECKPOINT_PATH=str(tmp_path),
               WALL_QWEN35_OPT="1", LD_LIBRARY_PATH="/old/launch/lib")
    return wrapper, runtime, env_sh, env


def run_wrapper(wrapper, env, *, sudo=False):
    return subprocess.run(
        ["bash", str(wrapper), "--check", "--sudo" if sudo else "--no-sudo"],
        env=env, capture_output=True, text=True,
    )


def test_wrapper_defaults_to_sibling_repository(wrapper_env):
    wrapper, runtime, _, env = wrapper_env
    result = run_wrapper(wrapper, env)
    assert result.returncode == 0, result.stderr
    assert f"TEST_RPU_KERNEL_LIB_PATH={runtime / ASSET}" in result.stdout
    assert f"TEST_RHINO_LAUNCH_LIB_DIR={runtime / 'launch/lib'}" in result.stdout
    assert f"TEST_RPU_OPS_ROOT={runtime.parent}" in result.stdout
    assert f"TEST_LD_LIBRARY_PATH={runtime / 'launch/lib'}:" in result.stdout
    assert "TEST_RPU_SOURCE_OPS=<unset>" in result.stdout
    assert "operator reference:" in result.stdout and "Rhino Launch:" in result.stdout


@pytest.mark.parametrize("from_caller", [False, True])
def test_runtime_paths_resolve_after_env_with_caller_priority(wrapper_env, tmp_path, from_caller):
    wrapper, _, env_sh, env = wrapper_env
    env_runtime = make_runtime(tmp_path / "environment ops")
    caller_runtime = make_runtime(tmp_path / "caller ops")
    settings = {
        "RPU_OPS_ROOT": str(env_runtime.parent),
        "RPU_KERNEL_LIB_PATH": str(env_runtime / ASSET),
        "RHINO_LAUNCH_LIB_DIR": str(env_runtime / "launch/lib"),
    }
    env_sh.write_text(env_sh.read_text() + "".join(
        f"export {key}={shlex.quote(value)}\n" for key, value in settings.items()
    ))
    selected = caller_runtime if from_caller else env_runtime
    if from_caller:
        env.update(RPU_OPS_ROOT=str(selected.parent),
                   RPU_KERNEL_LIB_PATH=str(selected / ASSET),
                   RHINO_LAUNCH_LIB_DIR=str(selected / "launch/lib"))
    result = run_wrapper(wrapper, env)
    assert result.returncode == 0, result.stderr
    assert f"TEST_RPU_OPS_ROOT={selected.parent}" in result.stdout
    assert f"TEST_RPU_KERNEL_LIB_PATH={selected / ASSET}" in result.stdout
    assert f"TEST_RHINO_LAUNCH_LIB_DIR={selected / 'launch/lib'}" in result.stdout


def test_root_override_alone_selects_both_runtime_assets(wrapper_env, tmp_path):
    wrapper, _, env_sh, env = wrapper_env
    runtime = make_runtime(tmp_path / "custom ops")
    env_sh.write_text(env_sh.read_text() +
                      f"export RPU_OPS_ROOT={shlex.quote(str(runtime.parent))}\n")
    result = run_wrapper(wrapper, env)
    assert result.returncode == 0, result.stderr
    assert f"TEST_RPU_KERNEL_LIB_PATH={runtime / ASSET}" in result.stdout
    assert f"TEST_RHINO_LAUNCH_LIB_DIR={runtime / 'launch/lib'}" in result.stdout


@pytest.mark.parametrize("missing,message", [
    (ASSET, "operator reference is missing"),
    (f"{ASSET}.kernels", "operator reference manifest is missing"),
    ("launch/lib/librhino_launch.so", "Rhino Launch shared library is missing"),
])
def test_missing_runtime_fails_before_python_without_legacy_fallback(wrapper_env, missing, message):
    wrapper, runtime, _, env = wrapper_env
    (runtime / missing).unlink()
    result = run_wrapper(wrapper, env)
    assert result.returncode == 2
    assert message in result.stderr
    assert "TEST_" not in result.stdout


@pytest.mark.parametrize("selectors", [False, True])
def test_sudo_explicitly_forwards_source_selection_and_pythonpath(wrapper_env, tmp_path, selectors):
    wrapper, runtime, env_sh, env = wrapper_env
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # Emulate sudo's environment reset; never invoke real privilege escalation.
    sudo = bin_dir / "sudo"
    sudo.write_text('#!/bin/bash\nexec /usr/bin/env -i "$@"\n')
    sudo.chmod(0o755)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    if selectors:
        env_sh.write_text(env_sh.read_text() +
                          "export RPU_SOURCE_OPS=rmsnorm\nexport PYTHONPATH='/local/framework code'\n")
    result = run_wrapper(wrapper, env, sudo=True)
    assert result.returncode == 0, result.stderr
    assert f"TEST_RPU_KERNEL_LIB_PATH={runtime / ASSET}" in result.stdout
    assert f"TEST_LD_LIBRARY_PATH={runtime / 'launch/lib'}:" in result.stdout
    assert f"TEST_RPU_SOURCE_OPS={'rmsnorm' if selectors else '<unset>'}" in result.stdout
    assert f"TEST_PYTHONPATH={'/local/framework code' if selectors else '<unset>'}" in result.stdout


def configure_launch(tmp_path, launch=None, *, extra_args=()):
    if shutil.which("cmake") is None:
        pytest.skip("CMake is unavailable")
    source = tmp_path / "cmake_source"
    source.mkdir(exist_ok=True)
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.18)\nproject(LaunchContract NONE)\n"
        f'include("{ROOT / "cmake/RhinoLaunch.cmake"}")\n'
        "get_target_property(location rhino_launch::rhino_launch IMPORTED_LOCATION)\n"
        'message(STATUS "SELECTED_LAUNCH=${location}")\n'
    )
    args = ["cmake", "-S", str(source), "-B", str(tmp_path / "build")]
    if launch is not None:
        args.append(f"-DRHINO_LAUNCH_DIR={launch}")
    return subprocess.run([*args, *extra_args], capture_output=True, text=True)


def test_cmake_imports_exact_repository_library(tmp_path):
    runtime = make_runtime(tmp_path / "rpu_ops")
    result = configure_launch(tmp_path, runtime / "launch")
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"SELECTED_LAUNCH={runtime / 'launch/lib/librhino_launch.so'}" in result.stdout


@pytest.mark.parametrize("change", ["release", "version", "duplicate", "header", "library"])
def test_explicit_cmake_runtime_never_uses_another_install(tmp_path, change):
    runtime = make_runtime(tmp_path / "rpu_ops")
    other = make_runtime(tmp_path / "other_ops")
    if change == "release":
        (runtime / "RELEASE.txt").unlink()
    elif change == "version":
        (runtime / "RELEASE.txt").write_text("launch_version=1.1.0\n")
    elif change == "duplicate":
        (runtime / "RELEASE.txt").write_text("launch_version=1.0.0\nlaunch_version=1.0.0\n")
    elif change == "header":
        (runtime / "launch/include/rhino_launch_queue.h").unlink()
    else:
        (runtime / "launch/lib/librhino_launch.so").unlink()
    result = configure_launch(tmp_path, runtime / "launch", extra_args=(
        f"-DCMAKE_PREFIX_PATH={other / 'launch'}",))
    assert result.returncode != 0


def test_cmake_reconfigure_does_not_reuse_cached_library(tmp_path):
    first = make_runtime(tmp_path / "first")
    second = make_runtime(tmp_path / "second")
    assert configure_launch(tmp_path, first / "launch").returncode == 0
    result = configure_launch(tmp_path, second / "launch")
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"SELECTED_LAUNCH={second / 'launch/lib/librhino_launch.so'}" in result.stdout


def test_no_explicit_cmake_root_preserves_exact_package_lookup(tmp_path):
    package = tmp_path / "installed/lib/cmake/rhino_launch"
    package.mkdir(parents=True)
    (package / "rhino_launchConfig.cmake").write_text(
        'if(NOT rhino_launch_FIND_VERSION STREQUAL "1.0.0" OR NOT rhino_launch_FIND_VERSION_EXACT)\n'
        '  message(FATAL_ERROR "Missing exact Launch 1.0.0 requirement")\nendif()\n'
        'add_library(rhino_launch::rhino_launch SHARED IMPORTED)\n'
        'set_target_properties(rhino_launch::rhino_launch PROPERTIES IMPORTED_LOCATION "/installed/launch.so")\n'
    )
    (package / "rhino_launchConfigVersion.cmake").write_text(
        'set(PACKAGE_VERSION "1.0.0")\nset(PACKAGE_VERSION_COMPATIBLE TRUE)\n'
        'set(PACKAGE_VERSION_EXACT TRUE)\n'
    )
    result = configure_launch(tmp_path, extra_args=(f"-Drhino_launch_DIR={package}",))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SELECTED_LAUNCH=/installed/launch.so" in result.stdout

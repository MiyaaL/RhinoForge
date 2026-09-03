"""Board-free contract checks for the diagnostic Wall GEMM tile sweep."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CPP = (ROOT / "src/fused/rpu_qwen3_5_model.cpp").read_text(encoding="utf-8")
LINEAR_CPP = (ROOT / "src/ops/rpu_linear.cpp").read_text(encoding="utf-8")
DECLS = (ROOT / "src/core/rpu_kernel_decls.h").read_text(encoding="utf-8")


def _section(source: str, start: str, end: str) -> str:
    begin = source.index(start)
    return source[begin : source.index(end, begin)]


def test_wall_tile_spec_is_cold_and_shape_keyed():
    parser = _section(
        CPP,
        "int wall_action_gemm_tile_override(",
        "// PARTIAL_MROPE",
    )
    assert 'std::getenv("RPU_QWEN35_WALL_GEMM_TILES")' in parser
    assert "if (spec == nullptr || *spec == '\\0') return 0;" in parser
    assert '"%lldx%lldx%lld=%d%n"' in parser
    assert "fields == 4 && consumed > 0" in parser
    assert "if (em == m && en == n && ek == k) return et;" in parser
    # Unknown shapes retain generated auto-tiling rather than inheriting a
    # neighboring experiment; malformed entries fail before graph BUILD.
    assert "return 0;" in parser
    assert "expects MxNxK=n entries" in parser


def test_wall_launch_maps_global_to_local_shape_and_non_wall_is_unchanged():
    launch = _section(
        CPP,
        "void Qwen3_5Model::launch_linear(",
        "void Qwen3_5Model::load_adaptive_mod_row(",
    )
    assert "if (wall_action_mode_)" in launch
    assert "partition == 0 ? n : n / num_cores" in launch
    assert "partition == 0 ? k / num_cores : k" in launch
    assert "if (env_tile != 0) tile_override_n = env_tile;" in launch
    # The override is passed only from this Wall-gated member; every other
    # caller keeps the declaration default of zero.
    assert "/*force_acc32=*/linear_acc32_, tile_override_n" in launch
    header = (ROOT / "src/fused/rpu_qwen3_5_model.h").read_text(encoding="utf-8")
    assert "int tile_override_n = 0" in header


def test_only_release_admitted_fp16_acc32_tiles_are_accepted():
    dispatch = _section(
        LINEAR_CPP,
        "auto tile = rpu_pl_tiling::select_tile_acc32(",
        "// Weight in DDR",
    )
    for tile in (128, 112, 96, 80, 64, 48, 32):
        assert f"tile_override_n == {tile}" in dispatch
    assert "FP16 ACC32 tile override must be one of" in dispatch
    assert "tile = rpu_pl_tiling::TilePick{" in dispatch
    assert "mtile_fp16_acc32(tile_override_n)" in dispatch
    # The ABI has a zero default on both direct and quant-dispatch entrypoints.
    assert "int tile_override_n = 0" in DECLS
    assert "bool force_acc32 = false" in DECLS


def test_wall_exact_shapes_are_documented_in_hook_comments():
    assert "32x256x1024=128" in CPP
    assert "RPU_QWEN35_WALL_GEMM_TILES" in CPP
    assert "tile_override_n" in LINEAR_CPP

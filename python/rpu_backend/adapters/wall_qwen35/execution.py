"""One cold execution switch for the controlled Wall profile.

Both arms use the same FP16 Action math. These presets own the process-wide
Graph/SDK budgets; start a fresh process to change the selected arm.
"""

import os

WALL_QWEN35_OPT_ABI = 1


def resolve_wall_qwen35_opt() -> bool:
    value = os.environ.get("WALL_QWEN35_OPT", "1").lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise ValueError("WALL_QWEN35_OPT must be boolean (1 or 0)")


def configure_wall_qwen35_execution(opt: bool) -> None:
    if type(opt) is not bool:
        raise ValueError("WALL_QWEN35_OPT must resolve to bool")
    os.environ.update({
        "WALL_QWEN35_OPT": "1" if opt else "0",
        "RPU_GRAPH_MAX_SEGMENT_ENTRIES": "32768" if opt else "8192",
        "RPU_GRAPH_MAX_SEGMENT_COMMAND_MB": "8" if opt else "4",
        "RPU_GRAPH_MAX_SEGMENT_INSTRUCTION_MB": "64" if opt else "32",
        "LKN_MAX_BATCH_ENTRIES": "65536",
        "LKN_KD_BUF_MB": "16" if opt else "8",
        "LKN_INSTR_BUF_MB": "128" if opt else "64",
    })

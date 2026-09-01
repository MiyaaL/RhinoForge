#!/usr/bin/env python3
"""Run one controlled-evaluation Wall Qwen3.5 action request."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


CAMERAS = ("face_view", "left_wrist_view", "right_wrist_view")


def _json_vector(value: str, *, name: str, binary: bool = False) -> list[float]:
    source = Path(value).expanduser()
    text = source.read_text() if source.is_file() else value
    try:
        result = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be a JSON array or a JSON file") from exc
    if not isinstance(result, list) or len(result) != 26:
        raise ValueError(f"{name} must contain exactly 26 values")
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(float(item))
        for item in result
    ):
        raise ValueError(f"{name} must contain only finite numbers")
    values = [float(item) for item in result]
    if binary and any(item not in (0.0, 1.0) for item in values):
        raise ValueError(f"{name} must contain only 0 or 1")
    return values


def _images(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--image must use CAMERA=/path/to/image")
        name, raw_path = value.split("=", 1)
        if name not in CAMERAS or name in result:
            raise ValueError(f"--image camera must be unique and one of {CAMERAS}")
        path = Path(raw_path).expanduser()
        if not path.is_file():
            raise ValueError(f"image file not found: {path}")
        result[name] = path
    if tuple(name for name in CAMERAS if name in result) != CAMERAS:
        raise ValueError(f"three --image arguments are required: {CAMERAS}")
    return {name: result[name] for name in CAMERAS}


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dataset-key", choices=("x2_normal",), default="x2_normal")
    parser.add_argument("--robot-id", default="10070")
    parser.add_argument("--image", action="append", default=[], metavar="CAMERA=PATH")
    parser.add_argument("--instruction")
    parser.add_argument("--state-json")
    parser.add_argument("--state-mask-json")
    parser.add_argument("--dof-mask-json")
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-numeric-blocked-vision", action="store_true")
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args(argv)

    # Every repository example supports a dependency-free, argument-free
    # ``--check-config`` probe.  If request arguments are supplied, validate
    # the complete request exactly as a live run would.
    request_values = (
        args.checkpoint,
        args.instruction,
        args.state_json,
        *args.image,
    )
    if args.check_config and not any(value is not None for value in request_values):
        return args
    missing = [
        flag
        for flag, value in (
            ("--checkpoint", args.checkpoint),
            ("--instruction", args.instruction),
            ("--state-json", args.state_json),
        )
        if value is None
    ]
    if missing:
        parser.error("the following arguments are required: " + ", ".join(missing))

    args.checkpoint = args.checkpoint.expanduser()
    if not args.checkpoint.is_dir():
        parser.error(f"checkpoint directory not found: {args.checkpoint}")
    try:
        args.images = _images(args.image)
        args.state = _json_vector(args.state_json, name="--state-json")
        args.state_mask = (
            [1.0] * 20 + [0.0] * 6
            if args.state_mask_json is None
            else _json_vector(
                args.state_mask_json, name="--state-mask-json", binary=True
            )
        )
        args.dof_mask = (
            [1.0] * 20 + [0.0] * 6
            if args.dof_mask_json is None
            else _json_vector(
                args.dof_mask_json, name="--dof-mask-json", binary=True
            )
        )
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.check_config:
        if args.checkpoint is None:
            print("configuration OK: wall_qwen35 exact-profile CLI")
            return 0
        print(
            f"configuration OK: checkpoint={args.checkpoint} "
            f"dataset={args.dataset_key} robot_id={args.robot_id} "
            f"cameras={','.join(CAMERAS)}"
        )
        return 0
    if not args.allow_numeric_blocked_vision:
        raise SystemExit(
            "Wall Qwen3.5 uses the experimental/numeric-blocked Qwen3.5 vision "
            "path. Pass --allow-numeric-blocked-vision for controlled evaluation."
        )

    import numpy as np
    from PIL import Image

    from rpu_backend.api import WallQwen35Policy

    images = {
        name: Image.open(path).convert("RGB") for name, path in args.images.items()
    }
    policy = WallQwen35Policy.from_checkpoint(
        args.checkpoint,
        dataset_key=args.dataset_key,
        robot_id=args.robot_id,
        allow_numeric_blocked_vision=True,
    ).to("rpu")
    try:
        output = policy.predict_action_chunk(
            images=images,
            instruction=args.instruction,
            proprioception=args.state,
            agent_pos_mask=args.state_mask,
            dof_mask=args.dof_mask,
            noise_seed=args.noise_seed,
        )
    finally:
        policy.close()
    actions = output.actions.numpy()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, actions)
    print(
        f"actions={actions.shape} dtype={actions.dtype} "
        f"min={actions.min():.6g} max={actions.max():.6g}"
    )
    print("first_action=" + json.dumps(actions[0, 0].tolist()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

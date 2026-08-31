#!/usr/bin/env python3
"""Print the canonical SHA-256 identity of a campaign reference tree."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from campaign_common import ValidationError, content_tree_sha256


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_root", type=Path)
    args = parser.parse_args(argv)
    try:
        digest = content_tree_sha256(args.reference_root)
    except (OSError, ValidationError) as error:
        print(
            json.dumps(
                {"schema_version": 1, "ok": False, "error": str(error)},
                sort_keys=True,
            )
        )
        return 2
    print(digest)
    return 0


if __name__ == "__main__":
    sys.exit(main())

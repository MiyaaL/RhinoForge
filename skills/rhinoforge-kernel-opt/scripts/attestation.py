#!/usr/bin/env python3
"""Verify externally issued DSSE/Ed25519 RhinoForge release proofs."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from campaign_common import ValidationError, canonical_json, read_limited_bytes, strict_json_loads


PAYLOAD_TYPE = "application/vnd.rhinoforge.kernel-release-proof.v1+json"
MAX_ENVELOPE_BYTES = 8 * 1024 * 1024


def _decode_base64(value: Any, label: str, length: int | None = None) -> bytes:
    if not isinstance(value, str):
        raise ValidationError(f"{label} must be base64 text")
    try:
        decoded = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValidationError(f"{label} is not canonical base64") from error
    if base64.b64encode(decoded).decode("ascii") != value:
        raise ValidationError(f"{label} is not canonical base64")
    if length is not None and len(decoded) != length:
        raise ValidationError(f"{label} has the wrong byte length")
    return decoded


def _pae(payload_type: bytes, payload: bytes) -> bytes:
    """DSSE pre-authentication encoding, version 1."""

    return (
        b"DSSEv1 "
        + str(len(payload_type)).encode("ascii")
        + b" "
        + payload_type
        + b" "
        + str(len(payload)).encode("ascii")
        + b" "
        + payload
    )


def key_id(public_key: bytes) -> str:
    if len(public_key) != 32:
        raise ValidationError("Ed25519 public key must be 32 bytes")
    return "ed25519:sha256:" + hashlib.sha256(public_key).hexdigest()


def verify_release_proof(
    envelope_path: Path,
    *,
    public_key_base64: str,
    verifier_executable: str,
    verifier_sha256: str,
    expected_signer_id: str,
    expected_challenge: str,
) -> tuple[dict[str, Any], str]:
    """Verify one canonical DSSE envelope and return its strict JSON statement."""

    if envelope_path.is_symlink() or not envelope_path.is_file():
        raise ValidationError("release proof must be a regular non-symlink file")
    raw_envelope = read_limited_bytes(
        envelope_path, MAX_ENVELOPE_BYTES, "release proof"
    )
    try:
        envelope = strict_json_loads(raw_envelope.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValidationError("release proof envelope is invalid JSON") from error
    if not isinstance(envelope, dict) or set(envelope) != {
        "payloadType",
        "payload",
        "signatures",
    }:
        raise ValidationError("release proof envelope schema is not exact")
    if envelope["payloadType"] != PAYLOAD_TYPE:
        raise ValidationError("release proof payload type is not admitted")
    signatures = envelope["signatures"]
    if not isinstance(signatures, list) or len(signatures) != 1:
        raise ValidationError("release proof must contain exactly one signature")
    signature_entry = signatures[0]
    if not isinstance(signature_entry, dict) or set(signature_entry) != {"keyid", "sig"}:
        raise ValidationError("release proof signature schema is not exact")

    public_key = _decode_base64(
        public_key_base64, "attestation public key", length=32
    )
    expected_key_id = key_id(public_key)
    if signature_entry["keyid"] != expected_key_id:
        raise ValidationError("release proof key ID does not match the contract")
    payload = _decode_base64(envelope["payload"], "release proof payload")
    signature = _decode_base64(signature_entry["sig"], "release proof signature", 64)
    verifier_path = Path(verifier_executable)
    verifier_descriptor = -1
    try:
        verifier_descriptor = os.open(
            verifier_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        )
        verifier_stat = os.fstat(verifier_descriptor)
        if (
            not stat.S_ISREG(verifier_stat.st_mode)
            or stat.S_IMODE(verifier_stat.st_mode) & 0o022
            or not stat.S_IMODE(verifier_stat.st_mode) & 0o111
            or verifier_stat.st_size > 64 * 1024 * 1024
        ):
            raise ValidationError("Ed25519 verifier file mode is unsafe")
        chunks: list[bytes] = []
        remaining = verifier_stat.st_size
        while remaining:
            block = os.read(verifier_descriptor, min(remaining, 1024 * 1024))
            if not block:
                raise ValidationError("Ed25519 verifier changed while hashing")
            chunks.append(block)
            remaining -= len(block)
        verifier_bytes = b"".join(chunks)
    except OSError as error:
        raise ValidationError("Ed25519 verifier is unavailable") from error
    except ValidationError:
        if verifier_descriptor >= 0:
            os.close(verifier_descriptor)
            verifier_descriptor = -1
        raise
    if (
        hashlib.sha256(verifier_bytes).hexdigest() != verifier_sha256
    ):
        os.close(verifier_descriptor)
        verifier_descriptor = -1
        raise ValidationError("Ed25519 verifier identity does not match the contract")
    public_key_der = bytes.fromhex("302a300506032b6570032100") + public_key
    try:
        with tempfile.TemporaryDirectory(prefix="rhinoforge-proof-verify-") as temporary:
            root = Path(temporary)
            key_path = root / "public-key.der"
            data_path = root / "dsse-pae.bin"
            signature_path = root / "signature.bin"
            key_path.write_bytes(public_key_der)
            data_path.write_bytes(_pae(PAYLOAD_TYPE.encode("utf-8"), payload))
            signature_path.write_bytes(signature)
            completed = subprocess.run(
                [
                    f"/proc/self/fd/{verifier_descriptor}",
                    "pkeyutl",
                    "-verify",
                    "-rawin",
                    "-pubin",
                    "-keyform",
                    "DER",
                    "-inkey",
                    str(key_path),
                    "-in",
                    str(data_path),
                    "-sigfile",
                    str(signature_path),
                ],
                check=False,
                capture_output=True,
                pass_fds=(verifier_descriptor,),
                timeout=10,
            )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValidationError("Ed25519 verifier execution failed") from error
    finally:
        if verifier_descriptor >= 0:
            os.close(verifier_descriptor)
    if completed.returncode != 0:
        raise ValidationError("release proof signature is invalid")

    try:
        statement = strict_json_loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValidationError("release proof payload is invalid JSON") from error
    if not isinstance(statement, dict):
        raise ValidationError("release proof payload must be an object")
    canonical_payload = canonical_json(statement).encode("utf-8")
    if payload != canonical_payload:
        raise ValidationError("release proof payload is not canonical JSON")
    if statement.get("signer_id") != expected_signer_id:
        raise ValidationError("release proof signer ID does not match the contract")
    if statement.get("challenge") != expected_challenge:
        raise ValidationError("release proof challenge is stale or mismatched")
    return statement, hashlib.sha256(raw_envelope).hexdigest()

#!/usr/bin/env python3
"""Verify immutable Core image security metadata without changing the image."""

from __future__ import annotations

import json
import subprocess
import sys


def main() -> int:
    image = sys.argv[1]
    if not image:
        raise SystemExit("Core image ID was not resolved")
    output = subprocess.check_output(["docker", "image", "inspect", image], text=True)
    config = json.loads(output)[0]["Config"]
    failures: list[str] = []
    if config.get("User") in {None, "", "0", "root"}:
        failures.append("image must declare a non-root user")
    if not config.get("Healthcheck", {}).get("Test"):
        failures.append("image must declare a healthcheck")
    if config.get("StopSignal") != "SIGTERM":
        failures.append("image must declare SIGTERM as its stop signal")
    if config.get("Cmd") != ["core-rag-api"]:
        failures.append("image command must be the locked Core console entrypoint")
    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1
    print(f"Core image security metadata verified for {image[:20]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

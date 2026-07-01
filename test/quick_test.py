#!/usr/bin/env python3
"""快速流程测试：Sigmoid + Gaussian Ring 合成模型 + Real pipeline。

Usage:
    python quick_test.py              # 运行全部 3 个测试
    python quick_test.py --skip-real  # 跳过 real pipeline
"""

import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent

TESTS = [
    ("Sigmoid", ["python", "run_test_plan.py", "--model", "Sigmoid"]),
    ("Gaussian Ring", ["python", "run_test_plan.py", "--model", "Gaussian Ring"]),
    ("Real Pipeline", ["python", "run_test_plan_real.py"]),
]


def main() -> int:
    skip_real = "--skip-real" in sys.argv
    total = 0
    failed = []

    for name, cmd in TESTS:
        if skip_real and name == "Real Pipeline":
            print(f"\n⏭  Skipping: {name}\n")
            continue

        total += 1
        print(f"\n{'=' * 60}")
        print(f"[{total}/{len(TESTS)}] Running: {name}")
        print(f"  {' '.join(cmd)}")
        print(f"{'=' * 60}\n")

        t0 = time.time()
        result = subprocess.run(cmd, cwd=_HERE)
        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"\n  FAIL: {name} (exit {result.returncode}, {elapsed:.0f}s)")
            failed.append(name)
        else:
            print(f"\n  PASS: {name} ({elapsed:.0f}s)")

    print(f"\n{'=' * 60}")
    print(f"Summary: {total - len(failed)}/{total} passed")
    if failed:
        print(f"Failed: {', '.join(failed)}")
        return 1
    print("All tests passed ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())

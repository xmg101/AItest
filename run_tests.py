#!/usr/bin/env python3
"""
CLI entry point for the Android Unity automated test suite.

Usage:
  python run_tests.py                        # run all tests in tests/
  python run_tests.py tests/example_test.py  # run specific file
  python run_tests.py -k test_launch         # run by keyword
  python run_tests.py --list                 # list collected tests
  python run_tests.py --device <serial>      # specify device serial
  python run_tests.py --info                 # print connected device info
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Android Unity Automated Test Runner")
    parser.add_argument("targets", nargs="*", default=["tests/"], help="Test files or dirs")
    parser.add_argument("-k", "--keyword", help="Filter tests by keyword")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--list", action="store_true", help="List collected tests without running")
    parser.add_argument("--device", help="ADB device serial (overrides DEVICE_SERIAL env var)")
    parser.add_argument("--info", action="store_true", help="Print device info and exit")
    parser.add_argument("--timeout", type=float, help="Override default timeout (seconds)")
    parser.add_argument("--threshold", type=float, help="Override image match threshold (0-1)")
    return parser.parse_args()


def print_device_info(serial: str = ""):
    sys.path.insert(0, str(Path(__file__).parent))
    from core.device import ADBDevice
    dev = ADBDevice(serial)
    if not dev.connect():
        print("No device connected. Run: adb devices")
        sys.exit(1)
    info = dev.get_device_info()
    print("\n=== Connected Device ===")
    for k, v in info.items():
        print(f"  {k:<12} {v}")
    print()


def main():
    args = parse_args()

    # Apply env overrides before importing config
    if args.device:
        os.environ["DEVICE_SERIAL"] = args.device
    if args.timeout:
        os.environ["DEFAULT_TIMEOUT"] = str(args.timeout)
    if args.threshold:
        os.environ["DEFAULT_THRESHOLD"] = str(args.threshold)

    if args.info:
        print_device_info(args.device or "")
        return

    # Build pytest command
    pytest_args = list(args.targets)

    if args.list:
        pytest_args += ["--collect-only", "-q"]
    else:
        pytest_args += ["-v"] if args.verbose else []

    if args.keyword:
        pytest_args += ["-k", args.keyword]

    # Add conftest path so pytest finds base_test fixtures
    pytest_args += ["--import-mode=importlib"]

    print(f"Running: pytest {' '.join(pytest_args)}\n")
    sys.path.insert(0, str(Path(__file__).parent))

    import pytest as _pytest
    exit_code = _pytest.main(pytest_args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

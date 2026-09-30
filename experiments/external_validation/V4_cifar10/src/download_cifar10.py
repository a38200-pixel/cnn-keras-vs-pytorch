"""Explicit, user-invoked CIFAR-10 downloader. No other V4 script downloads."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="Required explicit network-download opt-in")
    parser.add_argument("--data-root", type=Path, default=Path("data/cifar10"))
    args = parser.parse_args()
    if not args.download:
        parser.error("--download is required; implicit download is disabled")
    from torchvision.datasets import CIFAR10

    CIFAR10(root=str(args.data_root), train=True, download=True)
    CIFAR10(root=str(args.data_root), train=False, download=True)
    print(f"CIFAR-10 download complete: {args.data_root}")


if __name__ == "__main__":
    main()

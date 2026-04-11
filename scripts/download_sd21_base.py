#!/usr/bin/env python3
"""
Download the SD 2.1 base diffusers weights through the HF mirror.

This downloads only the files required by diffusers `from_pretrained()`,
and skips the large root-level single-file checkpoints.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO_ID = "sd2-community/stable-diffusion-2-1-base"
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
DEFAULT_HF_HOME = PROJECT_ROOT / ".hf-home"
DIFFUSERS_ALLOW_PATTERNS = [
    "feature_extractor/*",
    "scheduler/*",
    "text_encoder/*",
    "tokenizer/*",
    "unet/*",
    "vae/*",
    "model_index.json",
    "README.md",
    ".gitattributes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download SD 2.1 base diffusers weights")
    parser.add_argument("--repo_id", type=str, default=DEFAULT_REPO_ID, help="HF repo id")
    parser.add_argument("--hf_endpoint", type=str, default=DEFAULT_HF_ENDPOINT, help="HF endpoint")
    parser.add_argument("--hf_home", type=str, default=str(DEFAULT_HF_HOME), help="HF cache dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.environ.setdefault("HF_ENDPOINT", args.hf_endpoint)
    os.environ.setdefault("HF_HOME", args.hf_home)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    snapshot_path = snapshot_download(
        repo_id=args.repo_id,
        allow_patterns=DIFFUSERS_ALLOW_PATTERNS,
    )
    print(snapshot_path)


if __name__ == "__main__":
    main()

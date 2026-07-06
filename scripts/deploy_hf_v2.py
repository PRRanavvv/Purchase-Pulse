from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, upload_folder


ROOT = Path(__file__).resolve().parents[1]
SPACE_FOLDER = ROOT / "spaces" / "purchase-pulse-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy Purchase Pulse V2 to a Hugging Face Space.")
    parser.add_argument("--owner", required=True, help="Hugging Face username or organization.")
    parser.add_argument("--space", default="Purchase-Pulse-v2", help="Space name.")
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the Space as private. Defaults to public.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise SystemExit(
            "Set HF_TOKEN first. Example: PowerShell `$env:HF_TOKEN='hf_...'`, then rerun this script."
        )

    if not SPACE_FOLDER.exists():
        raise SystemExit(f"Missing Space folder: {SPACE_FOLDER}")

    repo_id = f"{args.owner}/{args.space}"
    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="gradio",
        private=args.private,
        exist_ok=True,
    )
    upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=str(SPACE_FOLDER),
        token=token,
        commit_message="Deploy Purchase Pulse V2",
        ignore_patterns=[
            "**/__pycache__/**",
            "**/*.pyc",
            "**/.DS_Store",
        ],
    )
    print(f"Deployed: https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()

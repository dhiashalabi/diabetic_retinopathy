#!/usr/bin/env python
"""Download the Diabetic Retinopathy dataset from Kaggle.

Usage:
    # With env vars (recommended for servers):
    KAGGLE_USERNAME=xxx KAGGLE_KEY=yyy python scripts/download_data.py

    # With ~/.kaggle/kaggle.json already configured:
    python scripts/download_data.py

    # Custom output directory:
    python scripts/download_data.py --data-dir /data/dr_dataset

    # Using uv:
    uv run scripts/download_data.py
"""

import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_manager import ensure_dataset


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Download DR dataset from Kaggle")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Target directory (default: <project_root>/dataset)",
    )
    args = parser.parse_args()

    data_dir = ensure_dataset(data_dir=args.data_dir)
    print(f"\nDataset ready at: {data_dir}")


if __name__ == "__main__":
    main()

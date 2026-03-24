from __future__ import annotations

import argparse
import sys
from pathlib import Path

from update_dashboard_data import (
    DEFAULT_DATA_DIR,
    DEFAULT_ENV_PATH,
    ValidationError,
    run_query_group,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run billing DB queries plus KYC query exports.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_PATH,
        help="Path to .env credentials file.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory where CSV exports will be written.",
    )
    parser.add_argument(
        "--allow-billing-fallback",
        action="store_true",
        help="If connect_licenses extraction fails, keep the existing CSV when available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_query_group(
        "kyc_and_billing",
        env_file=args.env_file,
        data_dir=args.data_dir,
        allow_billing_fallback=args.allow_billing_fallback,
    )


if __name__ == "__main__":
    try:
        main()
    except (ValidationError, FileNotFoundError, ValueError) as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)

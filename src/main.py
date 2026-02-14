"""
Main entry point for the ML Ops Titanic Survival Prediction project.

This script provides a command-line interface to run different components
of the ML pipeline.
"""

import argparse
import sys


def run_preprocessing():
    """Run the data preprocessing pipeline."""
    from src.preprocessing import preprocess_titanic

    print("Running preprocessing pipeline...\n")
    result = preprocess_titanic()
    if result is None:
        sys.exit(1)


def run_training():
    """Run the model training pipeline."""
    from src.train import main as train_main

    print("Running training pipeline...\n")
    train_main()


def run_download():
    """Run the data download script."""
    import sys
    from pathlib import Path

    # Add scripts to path
    scripts_path = Path(__file__).parent.parent / "scripts"
    sys.path.insert(0, str(scripts_path))
    from download_data import main as download_main

    print("Running data download...\n")
    download_main()


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="ML Ops Titanic Survival Prediction - Main Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main preprocess    # Run data preprocessing
  python -m src.main train         # Train the model
  python -m src.main download     # Optional: Download dataset via Kaggle API
  python -m src.main all           # Run full pipeline (preprocess -> train)
                                   # Note: Assumes data is already in data/raw/
        """,
    )

    parser.add_argument(
        "command",
        choices=["preprocess", "train", "download", "all"],
        help="Command to execute (download is optional, requires Kaggle API)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("ML Ops Final Project - Titanic Survival Prediction")
    print("=" * 60)
    print()

    if args.command == "download":
        run_download()
    elif args.command == "preprocess":
        run_preprocessing()
    elif args.command == "train":
        run_training()
    elif args.command == "all":
        print("Running full pipeline: preprocess -> train\n")
        print("Note: Assuming data files are already in data/raw/")
        print("If you need to download, run: python -m src.main download\n")
        run_preprocessing()
        print("\n" + "-" * 60 + "\n")
        run_training()

    print("\n" + "=" * 60)
    print("Pipeline execution completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

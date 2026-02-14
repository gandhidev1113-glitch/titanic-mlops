# Project Structure Documentation

## Overview

This document describes the professional organization of the ML Ops Final Project repository.

## Directory Structure

### `/data/`
Contains all data files organized by stage:
- `raw/`: Original datasets downloaded from Kaggle
- `derived/`: Preprocessed and cleaned data
- `output/`: Generated outputs (predictions, reports, feature importance)

### `/src/`
Core machine learning code package:
- `main.py`: Main CLI entry point
- `preprocessing.py`: Data cleaning and feature engineering
- `train.py`: Model training with MLflow integration
- `utils.py`: Shared utility functions

### `/scripts/`
Utility and pipeline scripts:
- `download_data.py`: Kaggle API data download script
- `setup_checkpoint2.bat/sh`: Setup scripts for different platforms

### `/tests/`
Comprehensive test suite:
- `test_preprocessing.py`: Tests for preprocessing pipeline
- `test_train.py`: Tests for training functions
- `test_utils.py`: Tests for utility functions

### `/models/`
Saved model artifacts (created during training):
- Trained models in `.pkl` format
- Model metadata

### `/docs/`
Documentation files:
- `CHECKPOINT2_VERIFICATION.md`: Verification checklist
- `CONVENTIONAL_COMMITS.md`: Commit message guidelines
- `PROFESSOR_FEEDBACK.md`: Feedback and improvements
- `PROJECT_STRUCTURE.md`: This file

### `/configs/`
Configuration files (for future use):
- Model configurations
- Pipeline settings

## Root Files

- `pyproject.toml`: Project dependencies and configuration
- `uv.lock`: Locked dependency versions
- `.gitignore`: Git ignore rules
- `.pre-commit-config.yaml`: Pre-commit hooks configuration
- `README.md`: Main project documentation

Note: `main.py` is located in `src/` directory, not root.

## Best Practices

1. **Separation of Concerns**:
   - Core ML logic in `src/`
   - Utility scripts in `scripts/`
   - Tests mirror `src/` structure

2. **Data Organization**:
   - Raw data never modified
   - Derived data is reproducible
   - Outputs are generated artifacts

3. **Documentation**:
   - README for overview
   - Docs folder for detailed guides
   - Inline code documentation

4. **Version Control**:
   - `.gitkeep` files maintain directory structure
   - `.gitignore` excludes generated files
   - Pre-commit hooks ensure code quality

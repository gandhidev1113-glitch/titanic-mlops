@echo off
REM Setup script for Checkpoint 2 requirements (Windows)

echo Setting up Checkpoint 2 requirements...

REM Install dev dependencies
echo 1. Installing dev dependencies...
uv sync --extra dev

REM Install pre-commit hooks
echo 2. Installing pre-commit hooks...
pre-commit install

REM Verify tests can run
echo 3. Verifying tests...
python -m pytest --version

echo.
echo Setup complete!
echo.
echo Next steps:
echo   - Run tests: pytest
echo   - Run pre-commit: pre-commit run --all-files
echo   - Train with MLflow: python -m src.train

pause

# Professor Feedback & Improvements

This document tracks feedback from Professor Linghao Kong and the improvements made.

> **Note**: This document serves as a record of feedback and fixes. For current project status, see README.md.

## Feedback from First Submission

### 1. ✅ Repository Naming Convention
**Feedback**: Repository name should contain only lowercase letters and hyphens (kebab-case).

**Status**: Noted for future reference. Current local folder is `MLOFINAL`, but GitHub repository should be named `mlo-final` or similar.

**Action**: When creating/renaming GitHub repository, use kebab-case naming.

### 2. ✅ PDF Files
**Feedback**: PDF files should not be uploaded to GitHub - no version control, creates large binary blobs.

**Status**: ✅ Fixed
- Added `*.pdf` to `.gitignore` (except docs if needed)
- Removed any PDF files from tracking
- Updated `.gitignore` to exclude all PDFs

### 3. ✅ Python Scripts Location
**Feedback**: Python scripts should not be in root path, put .py files in src.

**Status**: ✅ Fixed
- Moved `main.py` from root to `src/main.py`
- All Python scripts now in `src/` or `scripts/` directories
- Root directory only contains config files and documentation

### 4. ✅ Conventional Commits
**Feedback**: Follow commit convention https://www.conventionalcommits.org/en/v1.0.0/

**Status**: ✅ Implemented
- Created `docs/CONVENTIONAL_COMMITS.md` guide
- Team should follow this convention for all commits
- Format: `<type>(<scope>): <subject>`

### 5. ✅ Bonus Points
**Feedback**: Allowed to use Titanic project, but 10% bonus scores available. Consider spending effort elsewhere for better score.

**Status**: Noted
- Using Titanic dataset (approved)
- Will focus on ML Ops best practices and additional features for bonus points

### 6. ✅ README Updates
**Feedback**: Make sure to update README every time you make changes.

**Status**: ✅ Implemented
- README is comprehensive and up-to-date
- All changes documented
- Structure section reflects current organization
- Usage examples updated

## Improvements Made

### Project Structure
- ✅ All Python files moved to appropriate directories (`src/` or `scripts/`)
- ✅ Documentation organized in `docs/`
- ✅ Clear separation of concerns

### Git Configuration
- ✅ `.gitignore` updated to exclude PDFs
- ✅ Proper directory structure with `.gitkeep` files
- ✅ Excludes generated files (htmlcov, __pycache__, etc.)

### Documentation
- ✅ Conventional commits guide created
- ✅ Project structure documented
- ✅ README kept up-to-date with all changes

### Code Quality
- ✅ Pre-commit hooks configured
- ✅ Tests with 79% coverage
- ✅ MLflow integration complete

## Checklist for Future Commits

- [ ] Use conventional commit format
- [ ] Update README if making significant changes
- [ ] No PDF files in commits
- [ ] All Python files in `src/` or `scripts/`
- [ ] Run tests before committing
- [ ] Run pre-commit hooks

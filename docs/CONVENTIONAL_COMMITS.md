# Conventional Commits Guide

This project follows the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification for commit messages.

## Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

## Types

- **feat**: A new feature
- **fix**: A bug fix
- **docs**: Documentation only changes
- **style**: Code style changes (formatting, missing semi colons, etc.)
- **refactor**: Code refactoring without bug fixes or new features
- **test**: Adding or updating tests
- **chore**: Maintenance tasks, dependency updates, etc.
- **perf**: Performance improvements
- **ci**: CI/CD changes

## Scope (Optional)

The scope should be the name of the package affected:
- `preprocessing`: Changes to preprocessing pipeline
- `train`: Changes to training code
- `utils`: Changes to utility functions
- `tests`: Changes to test suite
- `docs`: Documentation changes
- `config`: Configuration changes

## Examples

### Feature
```
feat(train): add MLflow experiment tracking

- Log model parameters and hyperparameters
- Track metrics (accuracy, precision, recall, F1)
- Save model artifacts
```

### Bug Fix
```
fix(preprocessing): handle missing values in Age column

Previously, missing Age values caused errors during feature engineering.
Now using median imputation as fallback.
```

### Documentation
```
docs(readme): update project structure section

Added new directory organization and updated usage examples.
```

### Test
```
test(train): add unit tests for evaluate_model function

- Test metrics calculation
- Test return value structure
- Achieve 79% coverage
```

### Refactoring
```
refactor(src): reorganize project structure

- Move scripts to scripts/ directory
- Move main.py to src/
- Organize documentation in docs/
```

### Chore
```
chore(deps): update pytest to 9.0.2

Update testing dependencies for better compatibility.
```

## Best Practices

1. **Use imperative mood**: "add feature" not "added feature" or "adds feature"
2. **Keep subject line under 50 characters** when possible
3. **Capitalize the subject line**
4. **No period at the end of subject line**
5. **Use body to explain what and why**, not how
6. **Reference issues/PRs in footer**: `Closes #123` or `Fixes #456`

## Full Example

```
feat(train): integrate MLflow for experiment tracking

Add comprehensive MLflow integration to track:
- Model hyperparameters (n_estimators, max_depth, etc.)
- Training and validation metrics
- Model artifacts and feature importance

This enables reproducible experiments and easy comparison
of different model configurations.

Closes #42
```

## For Pull Requests

When creating PRs, use the same convention in the PR title:
- `feat(train): add MLflow integration`
- `fix(preprocessing): handle edge cases in feature engineering`
- `docs(readme): update setup instructions`

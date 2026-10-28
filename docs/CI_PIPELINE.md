# Continuous Integration Pipeline Documentation

## 1. Overview

This document describes the GitHub Actions CI pipeline for the mlops-iris-classifier project.

The workflow is defined in `.github/workflows/ci.yml` and automates Python compilation, linting, testing, and model-quality verification.

## 2. Workflow Triggers

The workflow runs on:
- Push events to the `main` and `develop` branches.
- Pull requests targeting the `main` and `develop` branches.

## 3. CI Jobs

### Job 1: build-and-lint

Purpose: Check Python syntax and code quality.

Steps:
1. Check out the repository.
2. Set up Python 3.11.
3. Install project dependencies, Flake8, pytest, and pytest-cov.
4. Compile Python files in `src/` and `tests/`.
5. Run Flake8 on `src/` with a maximum line length of 100 characters.

### Job 2: test-suite

Purpose: Run automated tests and produce test reports.

This job runs after `build-and-lint` succeeds.

Steps:
1. Check out the repository and set up Python 3.11.
2. Install dependencies.
3. Create the raw and processed data directories.
4. Collect Iris data.
5. Preprocess the data.
6. Engineer features.
7. Run the complete pytest suite with coverage reporting.
8. Upload the test and coverage reports as workflow artifacts.

Reports:
- `test-results.xml`
- `coverage.xml`

### Job 3: model-quality-gate

Purpose: Verify that the model satisfies the project's quality tests.

This job runs after `test-suite` succeeds.

Steps:
1. Check out the repository and set up Python 3.11.
2. Install project dependencies and pytest.
3. Create the data directories.
4. Collect and preprocess Iris data.
5. Engineer features.
6. Run `tests/test_model_quality.py`.

## 4. Job Dependency Order

The jobs run in this order:

`build-and-lint` -> `test-suite` -> `model-quality-gate`

If an upstream job fails, its dependent job does not run.

## 5. Local Verification Results

The following checks were performed locally:
- Python compilation completed successfully.
- Flake8 completed without lint errors.
- All 18 tests passed.
- `test-results.xml` and `coverage.xml` were generated.

## 6. GitHub Actions Verification

After committing and pushing the workflow, open the repository's Actions tab on GitHub to inspect workflow runs, job results, and uploaded artifacts.

A successful local run does not guarantee a successful GitHub Actions run. The remote workflow must also be verified.

## 7. Summary

This CI pipeline automates compilation, linting, testing, report generation, and model-quality checks to help detect code problems before changes are merged.

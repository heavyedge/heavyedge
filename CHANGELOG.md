# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - Unreleased

### Added

- `__len__` and `__getitem__` to `RawProfileBase`.
- `__len__` and `__getitem__` to `ProfileData`.

## [1.0.1] - 2025-06-05

### Fixed

- `api.preprocess()` returns profile length instead of contact point index.

## [1.0.0] - 2025-06-04

### Added

- Raw data preprocessing and outlier filtering.
- Profile averaging.
- Landmark detection.
- File I/O and command line API.

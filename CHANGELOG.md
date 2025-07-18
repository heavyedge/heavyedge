# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.3] - 2025-07-13

### Fixed

- `landmarks_type3()` when trough does not exist.

## [1.1.2] - 2025-07-17

### Fixed

- Numpy array is allowed for data file indexing.

## [1.1.1] - 2025-06-16

### Fixed

- Any integer-like object is allowed for data file indexing.

## [1.1.0] - 2025-06-15

### Added

- `RawProfileBase` now supports `len()` and indexing.
- `ProfileData` now supports `len()` and indexing.

### Changed

- Functions in `heavyedge.wasserstein` are changed.

## [1.0.1] - 2025-06-05

### Fixed

- `api.preprocess()` returns profile length instead of contact point index.

## [1.0.0] - 2025-06-04

### Added

- Raw data preprocessing and outlier filtering.
- Profile averaging.
- Landmark detection.
- File I/O and command line API.

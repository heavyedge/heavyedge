# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.7.0] - 2025-10-26

### Added

- `api.fill()` function is added.
- `heavyedge fill` command is added.

## [1.6.2] - 2025-10-25

### Fixed

- `prep()` now does not fill the processed array if `fill_value` argument is `None`.

## [1.6.1] - 2025-10-25

### Added

- `heavyedge mean` command now accepts `--fill-value` argument.

### Fixed

- `mean_wasserstein()` fills input profiles with zero values after their contact points.

### Deprecated

- Passing profile data which are not filled with zero values after their contact points
to `mean_wasserstein()` is deprecated.

## [1.6.0] - 2025-10-25

### Added

- `heavyedge prep` command now accepts `--fill-value` argument.
- `heavyedge prep` command now accepts `--z-thres` argument for outlier detection.
- `heavyedge merge` command now accepts `--batch-size` argument.
- `heavyedge mean` command now accepts `--batch-size` argument.
- `api.mean_euclidean()` and `api.mean_wasserstein()` functions are added.
- `heavyedge filter` command is introduced.
- `api.edge` module is introduced.
- `api.scale_area()` function is introduced.
- `api.scale_plateau()` function is introduced.

### Fixed

- `preprocess()` now ensures that the detected contact point has the lowest height.

### Changed

- `wasserstein.quantile()` function now takes multiple PDFs and their lengths.
- `wasserstein.wmean()` function now takes a single x, multiple PDFs and their lengths, and pre-defined grid.
- `wasserstein.wmean()` function now returns function interpolated over input `x`, and the length of its support.
- `api.mean_wasserstein()` function now returns function interpolated over input `x`, and the length of its support.
- `scale` command now no loger fills data outside the contact point with `nan`.
- `trim` command and `pad` command now no longer fills data outside the contact point with `nan`.
Instead, they result data whose length of the second axis different from the original data. 

### Deprecated

- `RawProfileBase.count_profiles()` method is deprecated. Implement `__len__()` method and use `len()` function instead.
- `RawProfileBase.profiles()` method is deprecated. Implement `__getitem__()` and directly index the object instead.
- `RawProfileBase.all_profiles()` method is deprecated. Implement `__getitem__()` and directly index the object instead.
- `RawProfileBase.profile_names()` method is deprecated. Implement `__getitem__()` and directly index the object instead.
- `wasserstein.wdist()` function is deprecated. Use HeavyEdge-Distance package instead.
- `outlier` command is deprecated. Filter in `prep` command instead.
- `api.preprocess()` function is deprecated. Use `profile.preprocess()` function instead.
- `api.fill_after()` function is deprecated. Use `profile.fill_after()` function instead.
- `api.outlierr()` function is deprecated. Use outlier detection in preprocessing step instead.
- `api.mean()` function is deprecated. Use functions in `api.mean` module instead.

## [1.5.0] - 2025-10-21

### Added

- `fill_after()` function is added.

### Deprecated

- `landmarks_type2()`, `landmarks_type3()`, `plateau_type2()`, `plateau_type3()` functions are deprecated. Use HeavyEdge-Landmarks package instead.
- `ProfileData.profile_names()` and `ProfileData.all_profiles()` methods are deprecated. Directly index the object instead.

## [1.4.1] - 2025-09-16

### Fixed

- Add scaling type to `heavyedge scale` command.

## [1.4.0] - 2025-09-15

### Added

- `heavyedge --list-plugins` command.

- `heavyedge scale` command.
- `heavyedge trim` command.
- `heavyedge pad` command.

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

# Changelog

<!--next-version-placeholder-->

## Unreleased
- JIT refactor and performance improvements: moved hot loops into Numba JIT helpers and vectorized key routines.
- `src/zepyros/common.py`: vectorized `rotate_patch`, vectorized `log10_factorial` (gammaln), JIT helpers for `isolate_surfaces` and `contact_points`.
- `src/zepyros/surface.py`: `create_plane` rewritten to use `scipy.stats.binned_statistic_2d`, gap-filling delegated to JIT helpers.
- `src/zepyros/zernike.py`: vectorized conversions and optimizations in `Zernike2D` and `Zernike3D`.
- Added `numba` dependency in `pyproject.toml`.

## v0.1.4 (01/08/2023)
- Update documentation
- Update `get_zernike`

## v0.1.3 (24/06/2023)

- First release of `zepyros`!
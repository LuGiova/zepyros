# zepyros

ZErnike Polynomials analYsis of pROtein Shapes. A tool for image characterizarion

> **Performance Optimization Update**
> This version includes a major refactoring of the main numerical and geometric hot paths (within `common.py`, `surface.py`, and `zernike.py`) to reduce Python-level looping. By introducing Numba JIT compilation and vectorized operations, runtime execution is reduced by up to 99% on repeated benchmark runs (after JIT warm-up). The public API remains completely unchanged, and the numerical outputs are strictly aligned with the original implementation.

## Installation
`zepyros` is not yet released on `PyPi`.
You can install it from the GitHub repo as follows

```bash
$ pip install git+https://github.com/matmi8/zepyros.git
```

## Usage

```python
import zepyros as zp
```

## License and Credits

`zepyros` was created by Mattia Miotto. It is licensed under the terms of the Apache License 2.0 license.

**Contributors:**
* **Giovanni Marzioni**: Performance optimizations, JIT compilation (via Numba), and vectorization of core numerical modules.

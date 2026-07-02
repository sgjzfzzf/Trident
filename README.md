# Trident

Trident is an experimental bridge that compiles Python functions containing
Triton/Torch-style compute into MLIR/LLVM, then executes through tvm_ffi.

The repository contains:

- A Python package (`trident`) with `jit` and `compile` entry points.
- A C++/MLIR core (`trident-core`) built with CMake.
- End-to-end examples and tests for ATen op lowering/execution.

## Current Status

- Build system is wired through `scikit-build-core` + CMake.
- LLVM/MLIR is fetched and built as an external project.
- Python extension artifacts are installed under `trident/_C` in wheel/editable builds.

## Repository Layout

- `python/trident`: Python APIs (`jit`, `compile`) and runtime backend.
- `trident-core`: MLIR/C++ implementation, dialects, passes, runtime glue.
- `examples`: Functional demos (`add.py`, `mm.py`, `softmax.py`, `attention.py`).
- `test`: Python end-to-end tests for lowered ATen ops.

## Prerequisites

- Linux
- Python 3.10+
- CMake 3.29+
- Ninja
- Clang/Clang++
- CUDA Toolkit (for GPU execution)
- `uv` (recommended Python package/build workflow)

Python build dependencies are declared in `pyproject.toml`, including:

- `torch>=2.10.0`
- `triton==3.6.0`
- `apache-tvm-ffi`
- `scikit-build-core`, `nanobind`, `cmake`, `ninja`

## Build And Install

### Editable Debug Build (recommended during development)

```bash
CC=clang CXX=clang++ uv pip install -ve . \
	--no-build-isolation \
	-Cbuild-dir=build \
	-Ccmake.define.CMAKE_EXPORT_COMPILE_COMMANDS=ON
```

### Build Wheel

```bash
CC=clang CXX=clang++ uv build
```

## Run Examples

After install, run from repository root:

```bash
python examples/*.py
```

All examples compare direct Torch/Triton outputs with Trident-jitted outputs.

## Run Tests

```bash
python -m unittest discover -s test -p "test_*.py"
```

Tests in `test/` validate end-to-end lowering and execution for selected ATen
ops (for example `empty` and `empty_like`).

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Notes

- Initial builds can take a long time because LLVM/MLIR and torch-mlir are
	built/fetched as dependencies.
- GPU/CUDA availability is expected by the provided examples/tests.

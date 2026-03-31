# LibTriton

## Instruction

### Debug Building
```bash
CC=clang CXX=clang++ uv pip install -ve . --no-build-isolation -Cbuild-dir=build -Ccmake.define.CMAKE_EXPORT_COMPILE_COMMANDS=ON
```

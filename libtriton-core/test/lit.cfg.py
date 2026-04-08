import os

import lit.formats

from lit.llvm import llvm_config

config.name = "LIBTRITON_CORE"
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.libtriton_core_obj_root, "test")

config.substitutions.append(("%PATH%", config.environment["PATH"]))
# Some LLVM packages may leave SHLIBEXT empty; default to .so on Linux.
shlibext = config.llvm_shlib_ext if config.llvm_shlib_ext else ".so"
config.substitutions.append(("%shlibext", shlibext))

# E2E test tools
config.substitutions.append(
    (
        "%e2e-dlpack-runner-asan",
        os.path.join(config.libtriton_core_obj_root, "bin", "e2e-dlpack-runner-asan"),
    )
)

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])
llvm_config.with_environment("PATH", config.libtriton_core_tools_dir, append_path=True)
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

config.excludes = [
    "Inputs",
    "CMakeLists.txt",
    "README.txt",
    "lit.cfg.py",
    "lit.site.cfg.py",
]

tool_dirs = [config.libtriton_core_tools_dir, config.llvm_tools_dir]
tools = ["libtriton-core-opt", "FileCheck"]

llvm_config.add_tool_substitutions(tools, tool_dirs)

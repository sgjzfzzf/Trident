import pathlib
import sys
import unittest


def main() -> int:
    tests_dir = pathlib.Path(__file__).resolve().parent
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))

    loader = unittest.defaultTestLoader
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromName("test_torch_compile_execution_engine"))
    suite.addTests(loader.loadTestsFromName("test_tvm_ffi_add"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())

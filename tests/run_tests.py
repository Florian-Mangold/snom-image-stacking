"""Run all tests:   python tests/run_tests.py   (or simply: pytest)"""
import os, sys, glob, importlib, traceback

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
failed = 0
for path in sorted(glob.glob(os.path.join(here, "test_*.py"))):
    module = importlib.import_module(os.path.basename(path)[:-3])
    for name in sorted(n for n in dir(module) if n.startswith("test_")):
        try:
            getattr(module, name)()
            print(f"PASS  {module.__name__}.{name}")
        except Exception:
            failed += 1
            print(f"FAIL  {module.__name__}.{name}")
            traceback.print_exc()
print("\nall tests passed" if failed == 0 else f"\n{failed} test(s) failed")
sys.exit(failed > 0)

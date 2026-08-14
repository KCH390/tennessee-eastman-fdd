"""
Tiny, dependency-free stand-in for the pieces of pytest these test files
use (approx, raises, importorskip/skip), plus a runner that lets each test
file execute standalone with `python tests/test_whatever.py` -- no pytest
install required.

Why this exists: the three test files in this directory (test_nrtl.py,
test_physics_cstr.py, test_column.py) were originally written against
`import pytest`, with no `if __name__ == "__main__"` block. That combination
silently does nothing if you just run `python tests/test_column.py` --
it imports the file (which only defines functions) and exits with zero
output, and if pytest isn't installed, an actual `pytest` invocation fails
outright. This module fixes both problems: no external dependency, and an
explicit runner.

If you DO have pytest installed and prefer it: these files still work
fine under `pytest tests/` too, since `approx`/`raises` below behave the
same as pytest's own (plain equality/context-manager objects -- pytest's
assert-rewriting doesn't care that they're not literally pytest.approx).

Usage in a test file:
    from _test_support import approx, raises, run_tests
    ...
    if __name__ == "__main__":
        raise SystemExit(run_tests(globals()))
"""

import sys
import traceback


class approx:
    """Stand-in for pytest.approx -- supports `x == approx(val, rel=..., abs=...)`."""

    def __init__(self, expected, rel=None, abs=None):
        self.expected = expected
        self.rel = rel
        self.abs = abs

    def __eq__(self, actual):
        if self.abs is not None:
            return abs_diff(actual, self.expected) <= self.abs
        rel = self.rel if self.rel is not None else 1e-6
        if self.expected == 0:
            return abs_diff(actual, self.expected) <= rel
        return abs_diff(actual, self.expected) <= rel * abs_val(self.expected)

    def __req__(self, actual):
        return self.__eq__(actual)

    def __repr__(self):
        return f"approx({self.expected!r}, rel={self.rel!r}, abs={self.abs!r})"


def abs_diff(a, b):
    return (a - b) if (a - b) >= 0 else (b - a)


def abs_val(a):
    return a if a >= 0 else -a


class raises:
    """Stand-in for pytest.raises -- `with raises(SomeError): ...`."""

    def __init__(self, expected_exception):
        self.expected_exception = expected_exception

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            raise AssertionError(f"expected {self.expected_exception} to be raised, nothing was")
        return issubclass(exc_type, self.expected_exception)


class _Skip(Exception):
    """Raised by skip()/importorskip() to signal a test should be skipped, not failed."""


def skip(reason: str = ""):
    raise _Skip(reason)


def importorskip(module_name: str, reason: str = ""):
    try:
        import importlib
        return importlib.import_module(module_name)
    except ImportError:
        raise _Skip(reason or f"{module_name} not installed")


def run_tests(module_globals: dict) -> int:
    """
    Run every zero-argument callable in module_globals whose name starts
    with 'test_'. Prints PASS/FAIL/SKIP per test and a summary. Returns a
    process exit code (0 if nothing failed, 1 otherwise) so
    `raise SystemExit(run_tests(globals()))` does the right thing in CI
    or a plain terminal.
    """
    tests = {
        name: fn
        for name, fn in sorted(module_globals.items())
        if name.startswith("test_") and callable(fn)
    }

    if not tests:
        print("No test_* functions found in this module.")
        return 1

    passed, failed, skipped = 0, 0, 0
    for name, fn in tests.items():
        try:
            fn()
            print(f"PASS  {name}")
            passed += 1
        except _Skip as e:
            print(f"SKIP  {name}" + (f"  ({e})" if str(e) else ""))
            skipped += 1
        except AssertionError as e:
            print(f"FAIL  {name}")
            if str(e):
                print(f"      {e}")
            failed += 1
        except Exception:
            print(f"ERROR {name}")
            traceback.print_exc(limit=3, file=sys.stdout)
            failed += 1

    print(f"\n{passed} passed, {failed} failed, {skipped} skipped, {len(tests)} total")
    return 1 if failed else 0

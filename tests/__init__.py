from contextlib import contextmanager
from os.path import abspath, dirname, join

import numpy


@contextmanager
def disable_numpy(module):
    assert module.numpy, "numpy is not installed"

    old_numpy = module.numpy
    module.numpy = None

    try:
        yield
    finally:
        module.numpy = old_numpy


def resource(*x):
    return abspath(join(dirname(__file__), *x))


class PillowTestCase:
    def assertAlmostEqualLuts(self, left, right, diff=None):
        assert tuple(left.size) == tuple(right.size)
        assert left.channels == right.channels
        left = numpy.array(left.table, dtype=numpy.float64)
        right = numpy.array(right.table, dtype=numpy.float64)

        if diff is None:
            diff = 20
        diff = 1.0 / (1 << diff)

        with numpy.errstate(divide='ignore', invalid='ignore'):
            diffs = numpy.abs(numpy.nan_to_num(left / right - 1.0))
        if diffs.max() <= diff:
            return

        idx = diffs.argmax()
        msg = f'{left[idx]} != {right[idx]} within {diff} diff'
        raise AssertionError(msg)

    def assertEqualLuts(self, left, right):
        assert tuple(left.size) == tuple(right.size)
        assert left.channels == right.channels

        left = numpy.array(left.table, dtype=numpy.float64)
        right = numpy.array(right.table, dtype=numpy.float64)
        if numpy.array_equal(left, right):
            return

        lines = ['Tables are not equal. Different elements:']
        for idx in (left - right).nonzero()[0][:20]:
            lines.append(f"  {idx}: {left[idx]}, {right[idx]}")
        raise AssertionError("\n".join(lines))

    def assertNotEqualLutTables(self, left, right):
        assert tuple(left.size) == tuple(right.size)
        assert left.channels == right.channels

        left = numpy.array(left.table, dtype=numpy.float64)
        right = numpy.array(right.table, dtype=numpy.float64)
        if numpy.array_equal(left, right):
            raise AssertionError('Tables are equal')

import unittest
from os.path import join, abspath, dirname
from contextlib import contextmanager

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
    return abspath(join(abspath(dirname(__file__)), *x))


class PillowTestCase(unittest.TestCase):
    def assertImageEqual(self, a, b, msg=None):
        self.assertEqual(
            a.mode, b.mode,
            msg or "got mode %r, expected %r" % (a.mode, b.mode))
        self.assertEqual(
            a.size, b.size,
            msg or "got size %r, expected %r" % (a.size, b.size))
        if a.tobytes() != b.tobytes():
            self.fail(msg or "got different content")

    def assertAlmostEqualLuts(self, left, right, diff=None, msg=None):
        self.assertEqual(tuple(left.size), tuple(right.size))
        self.assertEqual(left.channels, right.channels)
        left = numpy.array(left.table, dtype=numpy.float64)
        right = numpy.array(right.table, dtype=numpy.float64)

        if diff is None:
            diff = 20
        diff = 1.0 / (1 << diff)

        diffs = numpy.abs(numpy.nan_to_num(left / right - 1.0))
        if diffs.max() <= diff:
            return

        if not msg:
            idx = diffs.argmax()
            msg = '{} != {} within {} diff'.format(left[idx], right[idx], diff)
        raise self.failureException(msg)

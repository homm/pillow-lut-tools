from __future__ import division, unicode_literals, absolute_import

import warnings

from pillow_lut import operations, generators
from pillow_lut import (ImageFilter, Image, identity_table, transform_lut,
                        sample_lut_linear, sample_lut_cubic)

from . import PillowTestCase, disable_numpy


class TestSampleLutLinear(PillowTestCase):
    def test_identity_2(self):
        identity = identity_table(2)
        data = [-1.1, -0.3, 0, 0.1, 0.5, 1, 1.1]
        for b in data:
            for g in data:
                for r in data:
                    point = sample_lut_linear(identity, (r, g, b))
                    for left, right in zip(point, (r, g, b)):
                        self.assertAlmostEqual(left, right)

    def test_identity_17(self):
        identity = identity_table(17)
        data = [-1.1, -0.3, 0, 0.1, 0.5, 1, 1.1]
        for b in data:
            for g in data:
                for r in data:
                    point = sample_lut_linear(identity, (r, g, b))
                    for left, right in zip(point, (r, g, b)):
                        self.assertAlmostEqual(left, right)

    def test_identity_sizes(self):
        identity = identity_table((5, 6, 7))
        data = [-1.1, -0.3, 0, 0.1, 0.5, 1, 1.1]
        for b in data:
            for g in data:
                for r in data:
                    point = sample_lut_linear(identity, (r, g, b))
                    for left, right in zip(point, (r, g, b)):
                        self.assertAlmostEqual(left, right)

    def test_interpolation(self):
        lut = ImageFilter.Color3DLUT.generate(3, lambda r, g, b:
            (r, g*g, b*b + r))
        for point, res in [
            (( 0, 0, 0), ( 0, 0,  0)),
            ((.3, 0, 0), (.3, 0, .3)),
            ((.6, 0, 0), (.6, 0, .6)),
            (( 1, 0, 0), ( 1, 0,  1)),
            ((0,  0, 0), (0,  0, 0)),
            ((0, .3, 0), (0,.15, 0)),
            ((0, .6, 0), (0, .4, 0)),
            ((0,  1, 0), (0,  1, 0)),
            ((0, 0,  0), (0, 0,  0)),
            ((0, 0, .3), (0, 0,.15)),
            ((0, 0, .6), (0, 0, .4)),
            ((0, 0,  1), (0, 0,  1)),
            (( 0,  0,  0), ( 0,  0,  0)),
            ((.3, .3, .3), (.3,.15,.45)),
            ((.6, .6, .6), (.6, .4,  1)),
            (( 1,  1,  1), ( 1,  1,  2)),
        ]:
            for l, r in zip(sample_lut_linear(lut, point), res):
                self.assertAlmostEqual(l, r)

class TestSampleLutCubic(PillowTestCase):
    def test_identity_2(self):
        identity = identity_table(2)
        with self.assertRaisesRegexp(ValueError, "requires a table of size 4"):
            sample_lut_cubic(identity, (0, 0, 0))

    def test_identity_4(self):
        identity = identity_table(4)
        data = [-1.1, -0.3, 0, 0.1, 0.5, 1, 1.1]
        for b in data:
            for g in data:
                for r in data:
                    point = sample_lut_cubic(identity, (r, g, b))
                    for left, right in zip(point, (r, g, b)):
                        self.assertAlmostEqual(left, right)

    def test_identity_17(self):
        identity = identity_table(17)
        data = [-1.1, -0.3, 0, 0.1, 0.5, 1, 1.1]
        for b in data:
            for g in data:
                for r in data:
                    point = sample_lut_cubic(identity, (r, g, b))
                    for left, right in zip(point, (r, g, b)):
                        self.assertAlmostEqual(left, right)

    def test_identity_sizes(self):
        identity = identity_table((5, 6, 7))
        data = [-1.1, -0.3, 0, 0.1, 0.5, 1, 1.1]
        for b in data:
            for g in data:
                for r in data:
                    point = sample_lut_cubic(identity, (r, g, b))
                    for left, right in zip(point, (r, g, b)):
                        self.assertAlmostEqual(left, right)

    def test_interpolation(self):
        lut = ImageFilter.Color3DLUT.generate(5, lambda r, g, b:
            (r, g*g, b*b + r))
        for point, res in [
            (( 0, 0, 0), ( 0, 0,  0)),
            ((.3, 0, 0), (.3, 0, .3)),
            ((.6, 0, 0), (.6, 0, .6)),
            (( 1, 0, 0), ( 1, 0,  1)),
            ((0,  0, 0), (0,  0, 0)),
            ((0, .3, 0), (0,.09, 0)),
            ((0, .6, 0), (0,.36, 0)),
            ((0,  1, 0), (0,  1, 0)),
            ((0, 0,  0), (0, 0,  0)),
            ((0, 0, .3), (0, 0,.09)),
            ((0, 0, .6), (0, 0,.36)),
            ((0, 0,  1), (0, 0,  1)),
            (( 0,  0,  0), ( 0,  0,  0)),
            ((.3, .3, .3), (.3,.09,.39)),
            ((.6, .6, .6), (.6,.36,.96)),
            (( 1,  1,  1), ( 1,  1,  2)),
        ]:
            for l, r in zip(sample_lut_cubic(lut, point), res):
                self.assertAlmostEqual(l, r)


class TestTransformLut(PillowTestCase):
    lut_in = ImageFilter.Color3DLUT.generate(7,
            lambda r, g, b: (r**1.2, g**1.2, b**1.2))
    lut_out = ImageFilter.Color3DLUT.generate(7,
        lambda r, g, b: (r**(1/1.2), g**(1/1.2), b**(1/1.2)))

    def test_wrong_args(self):
        lut_4c = ImageFilter.Color3DLUT.generate(5, channels=4,
            callback=lambda r, g, b: (r*r, g*g, b*b, 1.0))

        with self.assertRaisesRegexp(ValueError, "only 3-channel cubes"):
            result = transform_lut(lut_4c, identity_table(3))

        with self.assertRaisesRegexp(ValueError, "only 3-channel cubes"):
            result = transform_lut(lut_4c, identity_table(3), target_size=5)

        with self.assertRaisesRegexp(ValueError, "interpolations"):
            result = transform_lut(identity_table(4), identity_table(4),
                interp=Image.NEAREST)

    def test_correct_args(self):
        lut_4c = ImageFilter.Color3DLUT.generate(5, channels=4,
            callback=lambda r, g, b: (r*r, g*g, b*b, 1.0))

        result = transform_lut(identity_table((3, 4, 5), target_mode='RGB'),
                               identity_table((6, 7, 8), target_mode='HSV'))
        self.assertEqual(tuple(result.size), (3, 4, 5))
        self.assertEqual(result.mode, 'HSV')
        self.assertEqual(result.channels, 3)

        result = transform_lut(identity_table(3), lut_4c)
        self.assertEqual(tuple(result.size), (3, 3, 3))
        self.assertEqual(result.mode, None)
        self.assertEqual(result.channels, 4)

        with disable_numpy(operations):
            result = transform_lut(identity_table(3), lut_4c)
        self.assertEqual(tuple(result.size), (3, 3, 3))
        self.assertEqual(result.mode, None)
        self.assertEqual(result.channels, 4)

        result = transform_lut(identity_table(4, target_mode='RGB'),
                               identity_table(5), target_size=(6, 7, 8))
        self.assertEqual(tuple(result.size), (6, 7, 8))
        self.assertEqual(result.mode, 'RGB')
        self.assertEqual(result.channels, 3)

        with disable_numpy(operations):
            result = transform_lut(identity_table(4, target_mode='RGB'),
                                   identity_table(5), target_size=(6, 7, 8))
        self.assertEqual(tuple(result.size), (6, 7, 8))
        self.assertEqual(result.mode, 'RGB')
        self.assertEqual(result.channels, 3)

    def test_identity_linear(self):
        identity9 = identity_table(9)
        lut7 = ImageFilter.Color3DLUT.generate(7,
            lambda r, g, b: (r**1.4, g**1.4, b**1.4))
        lut9 = ImageFilter.Color3DLUT.generate(9,
            lambda r, g, b: (r**1.4, g**1.4, b**1.4))

        res_numpy = transform_lut(lut7, identity9)
        self.assertAlmostEqualLuts(res_numpy, lut7)

        with disable_numpy(operations):
            res_native = transform_lut(lut7, identity9)
        self.assertAlmostEqualLuts(res_native, res_numpy)

        res_numpy = transform_lut(identity9, lut7)
        self.assertAlmostEqualLuts(res_numpy, lut9, 3)

        with disable_numpy(operations):
            res_native = transform_lut(identity9, lut7)
        self.assertAlmostEqualLuts(res_native, res_numpy)

    def test_identity_cubic(self):
        identity9 = identity_table(9)
        lut7 = ImageFilter.Color3DLUT.generate(7,
            lambda r, g, b: (r**1.4, g**1.4, b**1.4))
        lut9 = ImageFilter.Color3DLUT.generate(9,
            lambda r, g, b: (r**1.4, g**1.4, b**1.4))

        result = transform_lut(lut7, identity9, interp=Image.CUBIC)
        self.assertAlmostEqualLuts(result, lut7)

        result = transform_lut(identity9, lut7, interp=Image.CUBIC)
        self.assertAlmostEqualLuts(result, lut9, 4)

    def test_correctness_linear(self):
        identity = identity_table(7)

        res_numpy = transform_lut(self.lut_in, self.lut_out)
        self.assertAlmostEqualLuts(res_numpy, identity, 4)

        with disable_numpy(operations):
            res_native = transform_lut(self.lut_in, self.lut_out)
        self.assertAlmostEqualLuts(res_native, res_numpy)

        res_numpy = transform_lut(self.lut_out, self.lut_in)
        self.assertAlmostEqualLuts(res_numpy, identity, 6)

        with disable_numpy(operations):
            res_native = transform_lut(self.lut_out, self.lut_in)
        self.assertAlmostEqualLuts(res_native, res_numpy)

    def test_correctness_cubic(self):
        identity = identity_table(7)

        result = transform_lut(self.lut_in, self.lut_out, interp=Image.CUBIC)
        self.assertAlmostEqualLuts(result, identity, 4)

        result = transform_lut(self.lut_out, self.lut_in, interp=Image.CUBIC)
        self.assertAlmostEqualLuts(result, identity, 7)

    def test_target_size_correctness_linear(self):
        identity = identity_table(9)

        res_numpy = transform_lut(self.lut_out, self.lut_in, target_size=9)
        self.assertAlmostEqualLuts(res_numpy, identity, 4)

        with disable_numpy(operations):
            res_native = transform_lut(self.lut_out, self.lut_in, target_size=9)
        self.assertAlmostEqualLuts(res_native, res_numpy)

    def test_target_size_correctness_cubic(self):
        identity = identity_table(9)

        result = transform_lut(self.lut_out, self.lut_in,
                               target_size=9, interp=Image.CUBIC)
        self.assertAlmostEqualLuts(result, identity, 4)

    def test_fallback_to_linear(self):
        lut3 = ImageFilter.Color3DLUT.generate((5, 5, 3),
            lambda r, g, b: (r**1.5, g**1.5, b**1.5))
        lut4 = ImageFilter.Color3DLUT.generate((5, 5, 4),
            lambda r, g, b: (r**1.5, g**1.5, b**1.5))

        with warnings.catch_warnings(record=True) as w:
            cubic = transform_lut(identity_table((5, 5, 3)), lut4,
                                  interp=Image.CUBIC)
            self.assertEqual(len(w), 0)
        linear = transform_lut(identity_table((5, 5, 3)), lut4)
        self.assertNotEqualLutTables(cubic, linear)

        with warnings.catch_warnings(record=True) as w:
            cubic = transform_lut(identity_table((5, 5, 4)), lut3,
                                  interp=Image.CUBIC)
            self.assertEqual(len(w), 1)
            self.assertIn('Cubic interpolation', "{}".format(w[0].message))
        linear = transform_lut(identity_table((5, 5, 4)), lut3)
        self.assertEqualLuts(cubic, linear)

        with warnings.catch_warnings(record=True) as w:
            cubic = transform_lut(identity_table((5, 5, 3)), lut4,
                                  target_size=(5, 5, 4), interp=Image.CUBIC)
            self.assertEqual(len(w), 1)
            self.assertIn('Cubic interpolation', "{}".format(w[0].message))
        linear = transform_lut(identity_table((5, 5, 3)), lut4,
                               target_size=(5, 5, 4))
        self.assertEqualLuts(cubic, linear)

    def test_application(self):
        im = Image.new('RGB', (10, 10))

        lut_numpy = transform_lut(identity_table(5), identity_table(5))
        self.assertEqual(lut_numpy.table.__class__.__name__, 'ndarray')
        im.filter(lut_numpy)

        with disable_numpy(operations):
            lut_native = transform_lut(identity_table(5), identity_table(5))
        self.assertEqual(lut_native.table.__class__.__name__, 'list')
        im.filter(lut_native)

        with disable_numpy(generators):
            args = identity_table(5), identity_table(5)
        self.assertEqual(args[0].table.__class__.__name__, 'list')
        lut_numpy = transform_lut(*args)
        self.assertEqual(lut_numpy.table.__class__.__name__, 'ndarray')
        im.filter(lut_numpy)

        args = identity_table(5), identity_table(5)
        self.assertEqual(args[0].table.__class__.__name__, 'ndarray')
        with disable_numpy(operations):
            lut_native = transform_lut(*args)
        self.assertEqual(lut_native.table.__class__.__name__, 'list')
        im.filter(lut_native)

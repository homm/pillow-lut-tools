from __future__ import division, unicode_literals, absolute_import

import warnings

from pillow_lut import operations
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
    def test_wrong_args(self):
        with self.assertRaisesRegexp(ValueError, "only 3-channel cubes"):
            lut = ImageFilter.Color3DLUT.generate(5, channels=4,
                callback=lambda r, g, b: (r*r, g*g, b*b, 1.0))
            result = transform_lut(lut, identity_table(3))

        with self.assertRaisesRegexp(ValueError, "interpolations"):
            result = transform_lut(identity_table(4), identity_table(4),
                interp=Image.NEAREST)

    def test_correct_args(self):
        result = transform_lut(identity_table((3, 4, 5), target_mode='RGB'),
                               identity_table((6, 7, 8), target_mode='HSV'))
        self.assertEqual(tuple(result.size), (3, 4, 5))
        self.assertEqual(result.mode, 'HSV')
        self.assertEqual(result.channels, 3)

        lut = ImageFilter.Color3DLUT.generate(5, channels=4,
            callback=lambda r, g, b: (r*r, g*g, b*b, 1.0))
        result = transform_lut(identity_table(3), lut)
        self.assertEqual(tuple(result.size), (3, 3, 3))
        self.assertEqual(result.mode, None)
        self.assertEqual(result.channels, 4)

        result = transform_lut(identity_table(4, target_mode='RGB'),
                               identity_table(5), target_size=(6, 7, 8))
        self.assertEqual(tuple(result.size), (6, 7, 8))
        self.assertEqual(result.mode, 'RGB')
        self.assertEqual(result.channels, 3)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            transform_lut(identity_table(5), identity_table(10))
            self.assertEqual(w, [])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            transform_lut(identity_table(10), identity_table(5))
            self.assertEqual(len(w), 1)
            self.assertIn('fairly slow', "{}".format(w[0].message))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            transform_lut(identity_table(6), identity_table(5),
                          interp=Image.CUBIC)
            self.assertEqual(len(w), 1)
            self.assertIn('fairly slow', "{}".format(w[0].message))

    def test_identity(self):
        identity9 = identity_table(9)
        lut7 = ImageFilter.Color3DLUT.generate(7,
            lambda r, g, b: (r*r, g*g, b*b))
        lut9 = ImageFilter.Color3DLUT.generate(9,
            lambda r, g, b: (r*r, g*g, b*b))

        result = transform_lut(lut7, identity9)
        self.assertEqual(result.size, lut7.size)
        for left, right in zip(result.table, lut7.table):
            self.assertAlmostEqual(left, right)

        result = transform_lut(identity9, lut7)
        self.assertEqual(result.size, identity9.size)
        for left, right in zip(result.table, lut9.table):
            self.assertAlmostEqual(left, right, delta=0.007)

    def test_identity_cubic(self):
        identity9 = identity_table(9)
        lut7 = ImageFilter.Color3DLUT.generate(7,
            lambda r, g, b: (r*r, g*g, b*b))
        lut9 = ImageFilter.Color3DLUT.generate(9,
            lambda r, g, b: (r*r, g*g, b*b))

        result = transform_lut(lut7, identity9, interp=Image.CUBIC)
        self.assertEqual(result.size, lut7.size)
        for left, right in zip(result.table, lut7.table):
            self.assertAlmostEqual(left, right)

        result = transform_lut(identity9, lut7, interp=Image.CUBIC)
        self.assertEqual(result.size, identity9.size)
        for left, right in zip(result.table, lut9.table):
            self.assertAlmostEqual(left, right, delta=0.002)

    def test_correctness(self):
        lut_in = ImageFilter.Color3DLUT.generate(7,
            lambda r, g, b: (r**1.2, g**1.2, b**1.2))
        lut_out = ImageFilter.Color3DLUT.generate(7,
            lambda r, g, b: (r**(1/1.2), g**(1/1.2), b**(1/1.2)))
        identity = identity_table(7)

        result = transform_lut(lut_in, lut_out)
        self.assertEqual(result.size, lut_in.size)
        for left, right in zip(result.table, identity.table):
            self.assertAlmostEqual(left, right, delta=0.01)

        result = transform_lut(lut_out, lut_in)
        self.assertEqual(result.size, lut_out.size)
        for left, right in zip(result.table, identity.table):
            self.assertAlmostEqual(left, right, delta=0.003)

    def test_correctness_cubic(self):
        lut_in = ImageFilter.Color3DLUT.generate(7,
            lambda r, g, b: (r**1.2, g**1.2, b**1.2))
        lut_out = ImageFilter.Color3DLUT.generate(7,
            lambda r, g, b: (r**(1/1.2), g**(1/1.2), b**(1/1.2)))
        identity = identity_table(7)

        result = transform_lut(lut_in, lut_out, interp=Image.CUBIC)
        self.assertEqual(result.size, lut_in.size)
        for left, right in zip(result.table, identity.table):
            self.assertAlmostEqual(left, right, delta=0.007)

        result = transform_lut(lut_out, lut_in, interp=Image.CUBIC)
        self.assertEqual(result.size, lut_out.size)
        for left, right in zip(result.table, identity.table):
            self.assertAlmostEqual(left, right, delta=0.001)

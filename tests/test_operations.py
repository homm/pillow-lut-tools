from __future__ import division, unicode_literals, absolute_import

import warnings

from pillow_lut import operations
from pillow_lut import (ImageFilter, Image, identity_table, point_lut_linear,
                        transform_lut)

from . import PillowTestCase, disable_numpy


class TestPointLutLinear(PillowTestCase):
    def test_identity_2(self):
        identity = identity_table(2)
        data = [-1.1, -0.3, 0, 0.1, 0.5, 1, 1.1]
        for b in data:
            for g in data:
                for r in data:
                    point = point_lut_linear(identity, (r, g, b))
                    for left, right in zip(point, (r, g, b)):
                        self.assertAlmostEqual(left, right)

    def test_identity_17(self):
        identity = identity_table(17)
        data = [-1.1, -0.3, 0, 0.1, 0.5, 1, 1.1]
        for b in data:
            for g in data:
                for r in data:
                    point = point_lut_linear(identity, (r, g, b))
                    for left, right in zip(point, (r, g, b)):
                        self.assertAlmostEqual(left, right)

    def test_identity_sizes(self):
        identity = identity_table((5, 6, 7))
        data = [-1.1, -0.3, 0, 0.1, 0.5, 1, 1.1]
        for b in data:
            for g in data:
                for r in data:
                    point = point_lut_linear(identity, (r, g, b))
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
            for l, r in zip(point_lut_linear(lut, point), res):
                self.assertAlmostEqual(l, r)


class TestTransformLut(PillowTestCase):
    def test_wrong_args(self):
        with self.assertRaisesRegexp(ValueError, "only 3-channel cubes"):
            lut = ImageFilter.Color3DLUT.generate(5, channels=4,
                callback=lambda r, g, b: (r*r, g*g, b*b, 1.0))
            result = transform_lut(lut, identity_table(3))

        with self.assertRaisesRegexp(ValueError, "linear interpolation"):
            result = transform_lut(identity_table(4), identity_table(4),
                interp=Image.CUBIC)

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
            transform_lut(identity_table(5), identity_table(10))
            self.assertEqual(w, [])

        with warnings.catch_warnings(record=True) as w:
            transform_lut(identity_table(10), identity_table(5))
            self.assertEqual(len(w), 1)
            self.assertIn('fairly slow', "{}".format(w[0].message))

    def test_identity(self):
        identity11 = identity_table(11)
        lut9 = ImageFilter.Color3DLUT.generate(9,
            lambda r, g, b: (r*r, g*g, b*b))
        lut11 = ImageFilter.Color3DLUT.generate(11,
            lambda r, g, b: (r*r, g*g, b*b))

        result = transform_lut(lut9, identity11)
        self.assertEqual(result.size, lut9.size)
        for left, right in zip(result.table, lut9.table):
            self.assertAlmostEqual(left, right)

        result = transform_lut(identity11, lut9)
        self.assertEqual(result.size, identity11.size)
        for left, right in zip(result.table, lut11.table):
            self.assertAlmostEqual(left, right, 2)

    def test_correctness(self):
        lut_in = ImageFilter.Color3DLUT.generate(11,
            lambda r, g, b: (r**1.2, g**1.2, b**1.2))
        lut_out = ImageFilter.Color3DLUT.generate(11,
            lambda r, g, b: (r**(1/1.2), g**(1/1.2), b**(1/1.2)))
        identity = identity_table(11)

        result = transform_lut(lut_in, lut_out)
        self.assertEqual(result.size, lut_in.size)
        for left, right in zip(result.table, identity.table):
            self.assertAlmostEqual(left, right, delta=0.01)

        result = transform_lut(lut_out, lut_in)
        self.assertEqual(result.size, lut_out.size)
        for left, right in zip(result.table, identity.table):
            self.assertAlmostEqual(left, right, delta=0.01)

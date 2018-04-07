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
            result = transform_lut(identity_table(3), identity_table(3),
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
        identity = identity_table(11)
        lut = ImageFilter.Color3DLUT.generate(9,
            lambda r, g, b: (r*r, g*g, b*b))
        lut7 = ImageFilter.Color3DLUT.generate(11,
            lambda r, g, b: (r*r, g*g, b*b))

        result = transform_lut(lut, identity)
        self.assertEqual(result.size, lut.size)
        for left, right in zip(result.table, lut.table):
            self.assertAlmostEqual(left, right)

        result = transform_lut(identity, lut)
        self.assertEqual(result.size, identity.size)
        for left, right in zip(result.table, lut7.table):
            self.assertAlmostEqual(left, right, 2)

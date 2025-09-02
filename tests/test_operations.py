import warnings

import numpy
import pytest
from PIL import Image, ImageFilter

from pillow_lut import (
    amplify_lut, generators, identity_table, operations, resize_lut, sample_lut_cubic,
    sample_lut_linear, transform_lut)

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
                        assert left == pytest.approx(right)

    def test_identity_17(self):
        identity = identity_table(17)
        data = [-1.1, -0.3, 0, 0.1, 0.5, 1, 1.1]
        for b in data:
            for g in data:
                for r in data:
                    point = sample_lut_linear(identity, (r, g, b))
                    for left, right in zip(point, (r, g, b)):
                        assert left == pytest.approx(right)

    def test_identity_sizes(self):
        identity = identity_table((5, 6, 7))
        data = [-1.1, -0.3, 0, 0.1, 0.5, 1, 1.1]
        for b in data:
            for g in data:
                for r in data:
                    point = sample_lut_linear(identity, (r, g, b))
                    for left, right in zip(point, (r, g, b)):
                        assert left == pytest.approx(right)

    def test_interpolation(self):
        lut = ImageFilter.Color3DLUT.generate(
            3, lambda r, g, b: (r, g * g, b * b + r))
        for point, res in [
            ((0, 0, 0), (0, 0, 0)),
            ((.3, 0, 0), (.3, 0, .3)),
            ((.6, 0, 0), (.6, 0, .6)),
            ((1, 0, 0), (1, 0, 1)),
            ((0, 0, 0), (0, 0, 0)),
            ((0, .3, 0), (0, .15, 0)),
            ((0, .6, 0), (0, .4, 0)),
            ((0, 1, 0), (0, 1, 0)),
            ((0, 0, 0), (0, 0, 0)),
            ((0, 0, .3), (0, 0, .15)),
            ((0, 0, .6), (0, 0, .4)),
            ((0, 0, 1), (0, 0, 1)),
            ((0, 0, 0), (0, 0, 0)),
            ((.3, .3, .3), (.3, .15, .45)),
            ((.6, .6, .6), (.6, .4, 1)),
            ((1, 1, 1), (1, 1, 2)),
        ]:
            for lutval, resval in zip(sample_lut_linear(lut, point), res):
                assert lutval == pytest.approx(resval)


class TestSampleLutCubic(PillowTestCase):
    def test_identity_2(self):
        identity = identity_table(2)
        with pytest.raises(ValueError, match="requires a table of size 4"):
            sample_lut_cubic(identity, (0, 0, 0))

    def test_identity_4(self):
        identity = identity_table(4)
        data = [-1.1, -0.3, 0, 0.1, 0.5, 1, 1.1]
        for b in data:
            for g in data:
                for r in data:
                    point = sample_lut_cubic(identity, (r, g, b))
                    for left, right in zip(point, (r, g, b)):
                        assert left == pytest.approx(right)

    def test_identity_17(self):
        identity = identity_table(17)
        data = [-1.1, -0.3, 0, 0.1, 0.5, 1, 1.1]
        for b in data:
            for g in data:
                for r in data:
                    point = sample_lut_cubic(identity, (r, g, b))
                    for left, right in zip(point, (r, g, b)):
                        assert left == pytest.approx(right)

    def test_identity_sizes(self):
        identity = identity_table((5, 6, 7))
        data = [-1.1, -0.3, 0, 0.1, 0.5, 1, 1.1]
        for b in data:
            for g in data:
                for r in data:
                    point = sample_lut_cubic(identity, (r, g, b))
                    for left, right in zip(point, (r, g, b)):
                        assert left == pytest.approx(right)

    def test_interpolation(self):
        lut = ImageFilter.Color3DLUT.generate(
            5, lambda r, g, b: (r, g * g, b * b + r))
        for point, res in [
            ((0, 0, 0), (0, 0, 0)),
            ((.3, 0, 0), (.3, 0, .3)),
            ((.6, 0, 0), (.6, 0, .6)),
            ((1, 0, 0), (1, 0, 1)),
            ((0, 0, 0), (0, 0, 0)),
            ((0, .3, 0), (0, .09, 0)),
            ((0, .6, 0), (0, .36, 0)),
            ((0, 1, 0), (0, 1, 0)),
            ((0, 0, 0), (0, 0, 0)),
            ((0, 0, .3), (0, 0, .09)),
            ((0, 0, .6), (0, 0, .36)),
            ((0, 0, 1), (0, 0, 1)),
            ((0, 0, 0), (0, 0, 0)),
            ((.3, .3, .3), (.3, .09, .39)),
            ((.6, .6, .6), (.6, .36, .96)),
            ((1, 1, 1), (1, 1, 2)),
        ]:
            for lutval, resval in zip(sample_lut_cubic(lut, point), res):
                assert lutval == pytest.approx(resval)


class TestResizeLut(PillowTestCase):
    identity7 = identity_table(7)
    identity9 = identity_table(9)
    lut7_in = ImageFilter.Color3DLUT.generate(
        7, lambda r, g, b: (r**1.2, g**1.2, b**1.2))
    lut7_out = ImageFilter.Color3DLUT.generate(
        7, lambda r, g, b: (r**(1/1.2), g**(1/1.2), b**(1/1.2)))
    lut9_in = ImageFilter.Color3DLUT.generate(
        9, lambda r, g, b: (r**1.2, g**1.2, b**1.2))
    lut5_4c = ImageFilter.Color3DLUT.generate(
        5, channels=4, callback=lambda r, g, b: (r*r, g*g, b*b, 1.0))

    def test_wrong_args(self):
        with pytest.raises(ValueError, match="interpolations"):
            resize_lut(identity_table(4), 5, interp=Image.NEAREST)

    def test_correct_args(self):
        result = resize_lut(identity_table((3, 4, 5), target_mode='RGB'),
                            (6, 7, 8))
        assert tuple(result.size) == (6, 7, 8)
        assert result.mode == 'RGB'
        assert result.channels == 3

        result = resize_lut(self.lut5_4c, 3)
        assert tuple(result.size) == (3, 3, 3)
        assert result.mode is None
        assert result.channels == 4

        with disable_numpy(operations):
            result = resize_lut(self.lut5_4c, 3)
        assert tuple(result.size) == (3, 3, 3)
        assert result.mode is None
        assert result.channels == 4

    def test_correctness_linear(self):
        res_numpy = resize_lut(self.lut9_in, 7)
        self.assertAlmostEqualLuts(res_numpy, self.lut7_in, 6)

        with disable_numpy(operations):
            res_native = resize_lut(self.lut9_in, 7)
        self.assertAlmostEqualLuts(res_native, res_numpy)

    def test_correctness_cubic(self):
        result = resize_lut(self.lut9_in, 7, interp=Image.BICUBIC)
        self.assertAlmostEqualLuts(result, self.lut7_in, 7)

    def test_fallback_to_linear(self):
        lut3 = ImageFilter.Color3DLUT.generate(
            (5, 5, 3), lambda r, g, b: (r**1.5, g**1.5, b**1.5))
        lut4 = ImageFilter.Color3DLUT.generate(
            (5, 5, 4), lambda r, g, b: (r**1.5, g**1.5, b**1.5))

        with warnings.catch_warnings(record=True) as w:
            cubic = resize_lut(lut4, (5, 5, 3), interp=Image.BICUBIC)
            assert len(w) == 0
        linear = resize_lut(lut4, (5, 5, 3))
        self.assertNotEqualLutTables(cubic, linear)

        with warnings.catch_warnings(record=True) as w:
            cubic = resize_lut(lut3, (5, 5, 4), interp=Image.BICUBIC)
            assert len(w) == 1
            assert 'BICUBIC interpolation' in str(w[0].message)
        linear = resize_lut(lut3, (5, 5, 4))
        self.assertEqualLuts(cubic, linear)

    def test_application(self):
        im = Image.new('RGB', (10, 10))

        lut_numpy = resize_lut(identity_table(5), 4)
        im.filter(lut_numpy)
        assert isinstance(lut_numpy.table, numpy.ndarray)

        with disable_numpy(operations):
            lut_native = resize_lut(identity_table(5), 4)
        im.filter(lut_native)
        assert isinstance(lut_native.table, list)

        with disable_numpy(generators):
            args = identity_table(5)
        assert isinstance(args.table, list)
        lut_numpy = resize_lut(args, 4)
        im.filter(lut_numpy)
        assert isinstance(lut_numpy.table, numpy.ndarray)

        args = identity_table(5)
        assert isinstance(args.table, numpy.ndarray)
        with disable_numpy(operations):
            lut_native = resize_lut(args, 4)
        im.filter(lut_native)
        assert isinstance(lut_native.table, list)


class TestTransformLut(PillowTestCase):
    identity7 = identity_table(7)
    identity9 = identity_table(9)
    lut7_in = ImageFilter.Color3DLUT.generate(
        7, lambda r, g, b: (r**1.2, g**1.2, b**1.2))
    lut7_out = ImageFilter.Color3DLUT.generate(
        7, lambda r, g, b: (r**(1/1.2), g**(1/1.2), b**(1/1.2)))
    lut9_in = ImageFilter.Color3DLUT.generate(
        9, lambda r, g, b: (r**1.2, g**1.2, b**1.2))
    lut5_4c = ImageFilter.Color3DLUT.generate(
        5, channels=4, callback=lambda r, g, b: (r*r, g*g, b*b, 1.0))

    def test_wrong_args(self):
        with pytest.raises(ValueError, match="only 3-channel cubes"):
            transform_lut(self.lut5_4c, identity_table(3))

        with pytest.raises(ValueError, match="only 3-channel cubes"):
            transform_lut(self.lut5_4c, identity_table(3), target_size=5)

        with pytest.raises(ValueError, match="interpolations"):
            transform_lut(identity_table(4), identity_table(4), interp=Image.NEAREST)

    def test_correct_args(self):
        result = transform_lut(identity_table((3, 4, 5), target_mode='RGB'),
                               identity_table((6, 7, 8), target_mode='HSV'))
        assert tuple(result.size) == (3, 4, 5)
        assert result.mode == 'HSV'
        assert result.channels == 3

        result = transform_lut(identity_table(3), self.lut5_4c)
        assert tuple(result.size) == (3, 3, 3)
        assert result.mode is None
        assert result.channels == 4

        with disable_numpy(operations):
            result = transform_lut(identity_table(3), self.lut5_4c)
        assert tuple(result.size) == (3, 3, 3)
        assert result.mode is None
        assert result.channels == 4

        result = transform_lut(identity_table(4, target_mode='RGB'),
                               identity_table(5), target_size=(6, 7, 8))
        assert tuple(result.size) == (6, 7, 8)
        assert result.mode == 'RGB'
        assert result.channels == 3

        with disable_numpy(operations):
            result = transform_lut(identity_table(4, target_mode='RGB'),
                                   identity_table(5), target_size=(6, 7, 8))
        assert tuple(result.size) == (6, 7, 8)
        assert result.mode == 'RGB'
        assert result.channels == 3

    def test_identity_linear(self):
        res_numpy = transform_lut(self.lut7_in, self.identity9)
        self.assertAlmostEqualLuts(res_numpy, self.lut7_in)

        with disable_numpy(operations):
            res_native = transform_lut(self.lut7_in, self.identity9)
        self.assertAlmostEqualLuts(res_native, res_numpy)

        res_numpy = transform_lut(self.identity9, self.lut7_in)
        self.assertAlmostEqualLuts(res_numpy, self.lut9_in, 4)

        with disable_numpy(operations):
            res_native = transform_lut(self.identity9, self.lut7_in)
        self.assertAlmostEqualLuts(res_native, res_numpy)

    def test_identity_cubic(self):
        result = transform_lut(self.lut7_in, self.identity9, interp=Image.BICUBIC)
        self.assertAlmostEqualLuts(result, self.lut7_in)

        result = transform_lut(self.identity7, self.lut9_in, interp=Image.BICUBIC)
        self.assertAlmostEqualLuts(result, self.lut7_in, 7)

    def test_correctness_linear(self):
        res_numpy = transform_lut(self.lut7_in, self.lut7_out)
        self.assertAlmostEqualLuts(res_numpy, self.identity7, 4)

        with disable_numpy(operations):
            res_native = transform_lut(self.lut7_in, self.lut7_out)
        self.assertAlmostEqualLuts(res_native, res_numpy)

        res_numpy = transform_lut(self.lut7_out, self.lut7_in)
        self.assertAlmostEqualLuts(res_numpy, self.identity7, 6)

        with disable_numpy(operations):
            res_native = transform_lut(self.lut7_out, self.lut7_in)
        self.assertAlmostEqualLuts(res_native, res_numpy)

    def test_correctness_cubic(self):
        result = transform_lut(self.lut7_in, self.lut7_out, interp=Image.BICUBIC)
        self.assertAlmostEqualLuts(result, self.identity7, 4)

        result = transform_lut(self.lut7_out, self.lut7_in, interp=Image.BICUBIC)
        self.assertAlmostEqualLuts(result, self.identity7, 7)

    def test_target_size_correctness_linear(self):
        res_numpy = transform_lut(self.lut7_out, self.lut7_in, target_size=9)
        self.assertAlmostEqualLuts(res_numpy, self.identity9, 4)

        with disable_numpy(operations):
            res_native = transform_lut(self.lut7_out, self.lut7_in,
                                       target_size=9)
        self.assertAlmostEqualLuts(res_native, res_numpy)

    def test_target_size_correctness_cubic(self):
        result = transform_lut(self.lut7_out, self.lut7_in,
                               target_size=9, interp=Image.BICUBIC)
        self.assertAlmostEqualLuts(result, self.identity9, 4)

    def test_fallback_to_linear(self):
        lut3 = ImageFilter.Color3DLUT.generate(
            (5, 5, 3), lambda r, g, b: (r**1.5, g**1.5, b**1.5))
        lut4 = ImageFilter.Color3DLUT.generate(
            (5, 5, 4), lambda r, g, b: (r**1.5, g**1.5, b**1.5))

        with warnings.catch_warnings(record=True) as w:
            cubic = transform_lut(identity_table((5, 5, 3)), lut4, interp=Image.BICUBIC)
            assert len(w) == 0
        linear = transform_lut(identity_table((5, 5, 3)), lut4)
        self.assertNotEqualLutTables(cubic, linear)

        with warnings.catch_warnings(record=True) as w:
            cubic = transform_lut(identity_table((5, 5, 4)), lut3,
                                  interp=Image.BICUBIC)
            assert len(w) == 1
            assert 'Cubic interpolation' in str(w[0].message)
        linear = transform_lut(identity_table((5, 5, 4)), lut3)
        self.assertEqualLuts(cubic, linear)

        with warnings.catch_warnings(record=True) as w:
            cubic = transform_lut(identity_table((5, 5, 3)), lut4,
                                  target_size=(5, 5, 4), interp=Image.BICUBIC)
            assert len(w) == 1
            assert 'Cubic interpolation' in str(w[0].message)
        linear = transform_lut(identity_table((5, 5, 3)), lut4,
                               target_size=(5, 5, 4))
        self.assertEqualLuts(cubic, linear)

    def test_application(self):
        im = Image.new('RGB', (10, 10))

        lut_numpy = transform_lut(identity_table(5), identity_table(5))
        im.filter(lut_numpy)
        assert isinstance(lut_numpy.table, numpy.ndarray)

        with disable_numpy(operations):
            lut_native = transform_lut(identity_table(5), identity_table(5))
        im.filter(lut_native)
        assert isinstance(lut_native.table, list)

        with disable_numpy(generators):
            args = identity_table(5), identity_table(5)
        assert isinstance(args[0].table, list)
        lut_numpy = transform_lut(*args)
        im.filter(lut_numpy)
        assert isinstance(lut_numpy.table, numpy.ndarray)

        args = identity_table(5), identity_table(5)
        assert isinstance(args[0].table, numpy.ndarray)
        with disable_numpy(operations):
            lut_native = transform_lut(*args)
        im.filter(lut_native)
        assert isinstance(lut_native.table, list)


class TestAmplifyLut(PillowTestCase):
    lut5_4c = ImageFilter.Color3DLUT.generate(
        5, channels=4, callback=lambda r, g, b: (r*r, g*g, b*b, 1.0))

    def test_correct_args(self):
        result = amplify_lut(identity_table((3, 4, 5)), -1)
        assert tuple(result.size) == (3, 4, 5)
        assert result.channels == 3

        result = amplify_lut(self.lut5_4c, 5)
        assert tuple(result.size) == (5, 5, 5)
        assert result.channels == 4

    def test_correctness(self):
        lut = ImageFilter.Color3DLUT.generate(
            5, callback=lambda r, g, b: (r + 0.1, g * 1.1, b - 0.1))
        lut_05x = ImageFilter.Color3DLUT.generate(
            5, callback=lambda r, g, b: (r + 0.05, g * 1.05, b - 0.05))
        lut_2x = ImageFilter.Color3DLUT.generate(
            5, callback=lambda r, g, b: (r + 0.2, g * 1.2, b - 0.2))
        identity = identity_table(5)

        res_numpy = amplify_lut(lut, 1.0)
        with disable_numpy(operations):
            res_native = amplify_lut(lut, 1.0)
        self.assertAlmostEqualLuts(res_numpy, lut)
        self.assertAlmostEqualLuts(res_native, res_numpy)

        res_numpy = amplify_lut(lut, 0)
        with disable_numpy(operations):
            res_native = amplify_lut(lut, 0)
        self.assertEqualLuts(res_numpy, identity)
        self.assertEqualLuts(res_native, res_numpy)

        res_numpy = amplify_lut(lut, 0.5)
        with disable_numpy(operations):
            res_native = amplify_lut(lut, 0.5)
        self.assertAlmostEqualLuts(res_numpy, lut_05x)
        self.assertAlmostEqualLuts(res_native, res_numpy)

        res_numpy = amplify_lut(lut, 2)
        with disable_numpy(operations):
            res_native = amplify_lut(lut, 2)
        self.assertAlmostEqualLuts(res_numpy, lut_2x)
        self.assertAlmostEqualLuts(res_native, res_numpy)

    def test_correctness_4c(self):
        lut = ImageFilter.Color3DLUT.generate(
            5, channels=4, callback=lambda r, g, b:
                (r + 0.1, g * 1.1, b - 0.1, r + g + b))
        lut_2x = ImageFilter.Color3DLUT.generate(
            5, channels=4, callback=lambda r, g, b:
                (r + 0.2, g * 1.2, b - 0.2, r + g + b))

        res_numpy = amplify_lut(lut, 2)
        with disable_numpy(operations):
            res_native = amplify_lut(lut, 2)
        self.assertAlmostEqualLuts(res_numpy, lut_2x)
        self.assertAlmostEqualLuts(res_native, res_numpy)

    def test_application(self):
        im = Image.new('RGB', (10, 10))

        lut_numpy = amplify_lut(identity_table(5), 2.0)
        im.filter(lut_numpy)
        assert isinstance(lut_numpy.table, numpy.ndarray)

        with disable_numpy(operations):
            lut_native = amplify_lut(identity_table(5), 2.0)
        im.filter(lut_native)
        assert isinstance(lut_native.table, list)

        with disable_numpy(generators):
            args = identity_table(5)
        assert isinstance(args.table, list)
        lut_numpy = amplify_lut(args, 2.0)
        im.filter(lut_numpy)
        assert isinstance(lut_numpy.table, numpy.ndarray)

        args = identity_table(5)
        assert isinstance(args.table, numpy.ndarray)
        with disable_numpy(operations):
            lut_native = amplify_lut(args, 2.0)
        im.filter(lut_native)
        assert isinstance(lut_native.table, list)

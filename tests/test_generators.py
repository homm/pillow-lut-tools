from __future__ import division, unicode_literals, absolute_import

from pillow_lut import ImageFilter, rgb_color_enhance
from pillow_lut import generators

from . import PillowTestCase


class TestUtils(PillowTestCase):
    def test_linear_to_rgb_stability(self):
        for x in range(0, 256):
            x /= 255.0
            self.assertAlmostEqual(
                x, generators._linear_to_srgb(generators._srgb_to_linear(x)))
            self.assertAlmostEqual(
                x, generators._srgb_to_linear(generators._linear_to_srgb(x)))

    def test_rgb_to_hsv_stability(self):
        for r in range(0, 256):
            rgb = (r / 255.0, 0.875, 0.8)
            for left, right in zip(
                rgb, generators._hsv_to_rgb(*generators._rgb_to_hsv(*rgb))
            ):
                self.assertAlmostEqual(left, right)

        for g in range(0, 256):
            rgb = (0.15, g / 255.0, 0.15)
            for left, right in zip(
                rgb, generators._hsv_to_rgb(*generators._rgb_to_hsv(*rgb))
            ):
                self.assertAlmostEqual(left, right)

        for b in range(0, 256):
            rgb = (0.15, 0.15, b / 255.0)
            for left, right in zip(
                rgb, generators._hsv_to_rgb(*generators._rgb_to_hsv(*rgb))
            ):
                self.assertAlmostEqual(left, right)


class TestRgbColorEnhance(PillowTestCase):
    unit = ImageFilter.Color3DLUT.generate(5, lambda a, b, c: (a, b, c))

    def test_wrong_args(self):
        with self.assertRaisesRegexp(ValueError, "Size should be in"):
            rgb_color_enhance(0)
        with self.assertRaisesRegexp(ValueError, "Size should be in"):
            rgb_color_enhance(66)

        with self.assertRaisesRegexp(ValueError, "Brightness should be"):
            rgb_color_enhance(3, brightness=-1.1)
        with self.assertRaisesRegexp(ValueError, "Brightness should be"):
            rgb_color_enhance(3, brightness=1.1)
        with self.assertRaisesRegexp(ValueError, "Brightness should be"):
            rgb_color_enhance(3, brightness=(0.5, 0.5, 1.1))

        with self.assertRaisesRegexp(ValueError, "Contrast should be"):
            rgb_color_enhance(3, contrast=-1.1)
        with self.assertRaisesRegexp(ValueError, "Contrast should be"):
            rgb_color_enhance(3, contrast=1.1)
        with self.assertRaisesRegexp(ValueError, "Contrast should be"):
            rgb_color_enhance(3, contrast=(0.5, 0.5, 1.1))

        with self.assertRaisesRegexp(ValueError, "Saturation should be"):
            rgb_color_enhance(3, saturation=-1.1)
        with self.assertRaisesRegexp(ValueError, "Saturation should be"):
            rgb_color_enhance(3, saturation=1.1)
        with self.assertRaisesRegexp(ValueError, "Saturation should be"):
            rgb_color_enhance(3, saturation=(0.5, 0.5, 1.1))

        with self.assertRaisesRegexp(ValueError, "Vibrance should be"):
            rgb_color_enhance(3, vibrance=-1.1)
        with self.assertRaisesRegexp(ValueError, "Vibrance should be"):
            rgb_color_enhance(3, vibrance=1.1)
        with self.assertRaisesRegexp(ValueError, "Vibrance should be"):
            rgb_color_enhance(3, vibrance=(0.5, 0.5, 1.1))

        with self.assertRaisesRegexp(ValueError, "Hue should be"):
            rgb_color_enhance(3, hue=-0.1)
        with self.assertRaisesRegexp(ValueError, "Hue should be"):
            rgb_color_enhance(3, hue=1.1)

        with self.assertRaisesRegexp(ValueError, "Gamma should be"):
            rgb_color_enhance(3, gamma=-0.1)
        with self.assertRaisesRegexp(ValueError, "Gamma should be"):
            rgb_color_enhance(3, gamma=10.1)
        with self.assertRaisesRegexp(ValueError, "Gamma should be"):
            rgb_color_enhance(3, gamma=(0.5, 0.5, 10.1))

    def test_correct_args(self):
        lut = rgb_color_enhance(5)
        self.assertTrue(isinstance(lut, ImageFilter.Color3DLUT))
        self.assertEqual(lut.table, self.unit.table)

        lut = rgb_color_enhance(5, brightness=0.1)
        self.assertNotEqual(lut.table, self.unit.table)
        lut = rgb_color_enhance(5, brightness=(-1, 0, 0))
        self.assertNotEqual(lut.table, self.unit.table)
        lut = rgb_color_enhance(5, brightness=(0, 0.1, 0))
        self.assertNotEqual(lut.table, self.unit.table)
        lut = rgb_color_enhance(5, brightness=(0, 0, 1))
        self.assertNotEqual(lut.table, self.unit.table)

        lut = rgb_color_enhance(5, contrast=0.1)
        self.assertNotEqual(lut.table, self.unit.table)
        lut = rgb_color_enhance(5, contrast=(-1, 0, 0))
        self.assertNotEqual(lut.table, self.unit.table)
        lut = rgb_color_enhance(5, contrast=(0, 0.1, 0))
        self.assertNotEqual(lut.table, self.unit.table)
        lut = rgb_color_enhance(5, contrast=(0, 0, 1))
        self.assertNotEqual(lut.table, self.unit.table)

        lut = rgb_color_enhance(5, saturation=0.1)
        self.assertNotEqual(lut.table, self.unit.table)
        lut = rgb_color_enhance(5, saturation=(-1, 0, 0))
        self.assertNotEqual(lut.table, self.unit.table)
        lut = rgb_color_enhance(5, saturation=(0, 0.1, 0))
        self.assertNotEqual(lut.table, self.unit.table)
        lut = rgb_color_enhance(5, saturation=(0, 0, 1))
        self.assertNotEqual(lut.table, self.unit.table)

        lut = rgb_color_enhance(5, vibrance=0.1)
        self.assertNotEqual(lut.table, self.unit.table)
        lut = rgb_color_enhance(5, vibrance=(-1, 0, 0))
        self.assertNotEqual(lut.table, self.unit.table)
        lut = rgb_color_enhance(5, vibrance=(0, 0.1, 0))
        self.assertNotEqual(lut.table, self.unit.table)
        lut = rgb_color_enhance(5, vibrance=(0, 0, 1))
        self.assertNotEqual(lut.table, self.unit.table)

        lut = rgb_color_enhance(5, hue=0.1)
        self.assertNotEqual(lut.table, self.unit.table)
        lut = rgb_color_enhance(5, hue=1)
        self.assertNotEqual(lut.table, self.unit.table)

        lut = rgb_color_enhance(5, gamma=1.1)
        self.assertNotEqual(lut.table, self.unit.table)
        lut = rgb_color_enhance(5, gamma=(0, 1, 1))
        self.assertNotEqual(lut.table, self.unit.table)
        lut = rgb_color_enhance(5, gamma=(1, 1.1, 1))
        self.assertNotEqual(lut.table, self.unit.table)
        lut = rgb_color_enhance(5, gamma=(1, 1, 10))
        self.assertNotEqual(lut.table, self.unit.table)

    def test_linear_space(self):
        unit = rgb_color_enhance(13)
        lut = rgb_color_enhance(13, linear=True)
        self.assertTrue(isinstance(lut, ImageFilter.Color3DLUT))
        for left, right in zip(lut.table, unit.table):
            self.assertAlmostEqual(left, right, 6)

    def test_all_args(self):
        lut = rgb_color_enhance(
            5, brightness=0.1, contrast=0.1, saturation=0.1,
            vibrance=0.1, hue=0.1, gamma=1.1, linear=True,
        )
        self.assertTrue(isinstance(lut, ImageFilter.Color3DLUT))
        self.assertNotEqual(lut.table, self.unit.table)

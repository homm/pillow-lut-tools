from __future__ import division, unicode_literals, absolute_import

from pillow_lut import generators
from pillow_lut import ImageFilter, Image, rgb_color_enhance, identity_table

from . import PillowTestCase, disable_numpy


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
    identity = identity_table(5)

    def test_wrong_args(self):
        lut_4c = ImageFilter.Color3DLUT.generate(
            3, channels=4, callback=lambda a, b, c: (a, b, c, 1))
        with self.assertRaisesRegex(ValueError, "3-channels table"):
            rgb_color_enhance(lut_4c)
        with self.assertRaisesRegex(ValueError, "Size should be in"):
            rgb_color_enhance(0)
        with self.assertRaisesRegex(ValueError, "Size should be in"):
            rgb_color_enhance(66)

        with self.assertRaisesRegex(ValueError, "Brightness should be"):
            rgb_color_enhance(3, brightness=-1.1)
        with self.assertRaisesRegex(ValueError, "Brightness should be"):
            rgb_color_enhance(3, brightness=1.1)
        with self.assertRaisesRegex(ValueError, "Brightness should be"):
            rgb_color_enhance(3, brightness=(0.5, 0.5, 1.1))

        with self.assertRaisesRegex(ValueError, "Exposure should be"):
            rgb_color_enhance(3, exposure=-5.1)
        with self.assertRaisesRegex(ValueError, "Exposure should be"):
            rgb_color_enhance(3, exposure=5.1)
        with self.assertRaisesRegex(ValueError, "Exposure should be"):
            rgb_color_enhance(3, exposure=(0.5, 0.5, 5.1))

        with self.assertRaisesRegex(ValueError, "Contrast should be"):
            rgb_color_enhance(3, contrast=-1.1)
        with self.assertRaisesRegex(ValueError, "Contrast should be"):
            rgb_color_enhance(3, contrast=5.1)
        with self.assertRaisesRegex(ValueError, "Contrast should be"):
            rgb_color_enhance(3, contrast=(0.5, 0.5, 5.1))

        with self.assertRaisesRegex(ValueError, "Warmth should be"):
            rgb_color_enhance(3, warmth=-1.1)
        with self.assertRaisesRegex(ValueError, "Warmth should be"):
            rgb_color_enhance(3, warmth=1.1)

        with self.assertRaisesRegex(ValueError, "Saturation should be"):
            rgb_color_enhance(3, saturation=-1.1)
        with self.assertRaisesRegex(ValueError, "Saturation should be"):
            rgb_color_enhance(3, saturation=5.1)
        with self.assertRaisesRegex(ValueError, "Saturation should be"):
            rgb_color_enhance(3, saturation=(0.5, 0.5, 5.1))

        with self.assertRaisesRegex(ValueError, "Vibrance should be"):
            rgb_color_enhance(3, vibrance=-1.1)
        with self.assertRaisesRegex(ValueError, "Vibrance should be"):
            rgb_color_enhance(3, vibrance=5.1)
        with self.assertRaisesRegex(ValueError, "Vibrance should be"):
            rgb_color_enhance(3, vibrance=(0.5, 0.5, 5.1))

        with self.assertRaisesRegex(ValueError, "Hue should be"):
            rgb_color_enhance(3, hue=-0.1)
        with self.assertRaisesRegex(ValueError, "Hue should be"):
            rgb_color_enhance(3, hue=1.1)

        with self.assertRaisesRegex(ValueError, "Gamma should be"):
            rgb_color_enhance(3, gamma=-0.1)
        with self.assertRaisesRegex(ValueError, "Gamma should be"):
            rgb_color_enhance(3, gamma=10.1)
        with self.assertRaisesRegex(ValueError, "Gamma should be"):
            rgb_color_enhance(3, gamma=(0.5, 0.5, 10.1))

    def test_correct_args(self):
        lut = rgb_color_enhance(5)
        self.assertTrue(isinstance(lut, ImageFilter.Color3DLUT))
        self.assertEqualLuts(lut, self.identity)

        lut = rgb_color_enhance(5, brightness=0.1)
        self.assertNotEqualLutTables(lut, self.identity)
        lut = rgb_color_enhance(5, brightness=(-1, 0, 0))
        self.assertNotEqualLutTables(lut, self.identity)
        lut = rgb_color_enhance(5, brightness=(0, 0.1, 0))
        self.assertNotEqualLutTables(lut, self.identity)
        lut = rgb_color_enhance(5, brightness=(0, 0, 1))
        self.assertNotEqualLutTables(lut, self.identity)

        lut = rgb_color_enhance(5, exposure=0.5)
        self.assertNotEqualLutTables(lut, self.identity)
        lut = rgb_color_enhance(5, exposure=(-5, 0, 0))
        self.assertNotEqualLutTables(lut, self.identity)
        lut = rgb_color_enhance(5, exposure=(0, 0.1, 0))
        self.assertNotEqualLutTables(lut, self.identity)
        lut = rgb_color_enhance(5, exposure=(0, 0, 5))
        self.assertNotEqualLutTables(lut, self.identity)

        lut = rgb_color_enhance(5, contrast=0.1)
        self.assertNotEqualLutTables(lut, self.identity)
        lut = rgb_color_enhance(5, contrast=(-1, 0, 0))
        self.assertNotEqualLutTables(lut, self.identity)
        lut = rgb_color_enhance(5, contrast=(0, 0.1, 0))
        self.assertNotEqualLutTables(lut, self.identity)
        lut = rgb_color_enhance(5, contrast=(0, 0, 5))
        self.assertNotEqualLutTables(lut, self.identity)

        lut = rgb_color_enhance(5, warmth=0.1)
        self.assertNotEqualLutTables(lut, self.identity)
        lut = rgb_color_enhance(5, warmth=-1)
        self.assertNotEqualLutTables(lut, self.identity)
        lut = rgb_color_enhance(5, warmth=1)
        self.assertNotEqualLutTables(lut, self.identity)

        lut = rgb_color_enhance(5, saturation=0.1)
        self.assertNotEqualLutTables(lut, self.identity)
        lut = rgb_color_enhance(5, saturation=(-1, 0, 0))
        self.assertNotEqualLutTables(lut, self.identity)
        lut = rgb_color_enhance(5, saturation=(0, 0.1, 0))
        self.assertNotEqualLutTables(lut, self.identity)
        lut = rgb_color_enhance(5, saturation=(0, 0, 5))
        self.assertNotEqualLutTables(lut, self.identity)

        lut = rgb_color_enhance(5, vibrance=0.1)
        self.assertNotEqualLutTables(lut, self.identity)
        lut = rgb_color_enhance(5, vibrance=(-1, 0, 0))
        self.assertNotEqualLutTables(lut, self.identity)
        lut = rgb_color_enhance(5, vibrance=(0, 0.1, 0))
        self.assertNotEqualLutTables(lut, self.identity)
        lut = rgb_color_enhance(5, vibrance=(0, 0, 5))
        self.assertNotEqualLutTables(lut, self.identity)

        lut = rgb_color_enhance(5, hue=0.1)
        self.assertNotEqualLutTables(lut, self.identity)
        lut = rgb_color_enhance(5, hue=1)
        self.assertNotEqualLutTables(lut, self.identity)

        lut = rgb_color_enhance(5, gamma=1.1)
        self.assertNotEqualLutTables(lut, self.identity)
        lut = rgb_color_enhance(5, gamma=(0, 1, 1))
        self.assertNotEqualLutTables(lut, self.identity)
        lut = rgb_color_enhance(5, gamma=(1, 1.1, 1))
        self.assertNotEqualLutTables(lut, self.identity)
        lut = rgb_color_enhance(5, gamma=(1, 1, 10))
        self.assertNotEqualLutTables(lut, self.identity)

    def test_different_dimensions(self):
        lut_ref = identity_table((4, 5, 6))

        lut_numpy = rgb_color_enhance((4, 5, 6))
        self.assertEqualLuts(lut_numpy, lut_ref)

        with disable_numpy(generators):
            lut_native = rgb_color_enhance((4, 5, 6))
        self.assertAlmostEqualLuts(lut_native, lut_ref)
        self.assertNotEqualLutTables(lut_native, lut_ref)

    def test_source_lut(self):
        source = rgb_color_enhance(5, saturation=0.5)
        lut_ref = rgb_color_enhance(5, saturation=0.5, exposure=0.3, warmth=1)

        lut_numpy = rgb_color_enhance(source, exposure=0.3, warmth=1)
        self.assertEqualLuts(lut_numpy, lut_ref)

        with disable_numpy(generators):
            lut_native = rgb_color_enhance(source, exposure=0.3, warmth=1)
        self.assertAlmostEqualLuts(lut_native, lut_ref, 14)
        self.assertNotEqualLutTables(lut_native, lut_ref)

    def test_linear_space(self):
        identity = rgb_color_enhance(13)
        lut = rgb_color_enhance(13, linear=True)
        self.assertTrue(isinstance(lut, ImageFilter.Color3DLUT))
        self.assertAlmostEqualLuts(lut, identity)
        self.assertNotEqualLutTables(lut, identity)

    def test_all_args(self):
        lut = rgb_color_enhance(
            5, brightness=0.1, exposure=-0.2, contrast=0.1, warmth=0.3,
            saturation=0.1, vibrance=0.1, hue=0.1, gamma=1.1, linear=True,
        )
        self.assertTrue(isinstance(lut, ImageFilter.Color3DLUT))
        self.assertNotEqualLutTables(lut, self.identity)

    def test_numpy_correctness(self):
        lut_numpy = rgb_color_enhance(
            13, brightness=0.1, exposure=-0.2, contrast=0.1, warmth=0.5,
            saturation=0.1, vibrance=0.1, gamma=1.1, linear=True
        )
        with disable_numpy(generators):
            lut_native = rgb_color_enhance(
                13, brightness=0.1, exposure=-0.2, contrast=0.1, warmth=0.5,
                saturation=0.1, vibrance=0.1, gamma=1.1, linear=True
            )
        self.assertAlmostEqualLuts(lut_numpy, lut_native, 10)
        self.assertNotEqualLutTables(lut_numpy, lut_native)

    def test_application(self):
        im = Image.new('RGB', (10, 10))

        lut_numpy = rgb_color_enhance(5, saturation=0.5)
        self.assertEqual(lut_numpy.table.__class__.__name__, 'ndarray')
        im.filter(lut_numpy)

        with disable_numpy(generators):
            lut_native = rgb_color_enhance(5, saturation=0.5)
        self.assertEqual(lut_native.table.__class__.__name__, 'list')
        im.filter(lut_native)

        lut_numpy = rgb_color_enhance(lut_native, saturation=0.5)
        self.assertEqual(lut_numpy.table.__class__.__name__, 'ndarray')
        im.filter(lut_numpy)

        with disable_numpy(generators):
            lut_native = rgb_color_enhance(lut_numpy, saturation=0.5)
        self.assertEqual(lut_native.table.__class__.__name__, 'list')
        im.filter(lut_native)


class TestIdentityTable(PillowTestCase):
    def test_different_dimensions(self):
        lut_ref = ImageFilter.Color3DLUT.generate((4, 5, 6),
                                                  lambda a, b, c: (a, b, c))

        lut_numpy = identity_table((4, 5, 6))
        self.assertAlmostEqualLuts(lut_numpy, lut_ref)

        with disable_numpy(generators):
            lut_native = identity_table((4, 5, 6))
        self.assertEqualLuts(lut_native, lut_ref)

    def test_application(self):
        im = Image.new('RGB', (10, 10))

        lut_numpy = identity_table(5)
        self.assertEqual(lut_numpy.table.__class__.__name__, 'ndarray')
        im.filter(lut_numpy)

        with disable_numpy(generators):
            lut_native = identity_table(5)
        self.assertEqual(lut_native.table.__class__.__name__, 'list')
        im.filter(lut_native)

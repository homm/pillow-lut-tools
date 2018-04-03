from __future__ import division, unicode_literals, absolute_import

import os
from tempfile import NamedTemporaryFile

from pillow_lut import ImageFilter, load_cube_file

from . import PillowTestCase


class TestLoadCubeFile(PillowTestCase):
    def test_minimal(self):
        lut = load_cube_file([
            "LUT_3D_SIZE 2",
            "0    0 0.031",
            "0.96 0 0.031",
            "0    1 0.031",
            "0.96 1 0.031",
            "0    0 0.931",
            "0.96 0 0.931",
            "0    1 0.931",
            "0.96 1 0.931",
        ])
        self.assertTrue(isinstance(lut, ImageFilter.Color3DLUT))
        self.assertEqual(tuple(lut.size), (2, 2, 2))
        self.assertEqual(lut.name, "Color 3D LUT")
        self.assertEqual(lut.table[:12], [
            0, 0, 0.031,  0.96, 0, 0.031,  0, 1, 0.031,  0.96, 1, 0.031])

    def test_parser(self):
        lut = load_cube_file([
            " # Comment",
            'TITLE "LUT name from file"',
            "  LUT_3D_SIZE 2 3 4",
            " SKIP THIS",
            "",
            " # Comment",
            "CHANNELS 4",
            "",
        ] + [
            " # Comment",
            "0    0 0.031 1",
            "0.96 0 0.031 1",
            "",
            "0    1 0.031 1",
            "0.96 1 0.031 1",
        ] * 6, target_mode='HSV')
        self.assertTrue(isinstance(lut, ImageFilter.Color3DLUT))
        self.assertEqual(tuple(lut.size), (2, 3, 4))
        self.assertEqual(lut.channels, 4)
        self.assertEqual(lut.name, "LUT name from file")
        self.assertEqual(lut.mode, 'HSV')
        self.assertEqual(lut.table[:12], [
            0, 0, 0.031, 1,  0.96, 0, 0.031, 1,  0, 1, 0.031, 1])

    def test_errors(self):
        with self.assertRaisesRegexp(ValueError, "No size found"):
            lut = load_cube_file([
                'TITLE "LUT name from file"',
            ] + [
                "0    0 0.031",
                "0.96 0 0.031",
            ] * 3)

        with self.assertRaisesRegexp(ValueError, "number of colors on line 3"):
            lut = load_cube_file([
                'LUT_3D_SIZE 2',
            ] + [
                "0    0 0.031",
                "0.96 0 0.031 1",
            ] * 3)

        with self.assertRaisesRegexp(ValueError, "1D LUT cube files"):
            lut = load_cube_file([
                'LUT_1D_SIZE 2',
            ] + [
                "0    0 0.031",
                "0.96 0 0.031 1",
            ])

        with self.assertRaisesRegexp(ValueError, "Not a number on line 2"):
            lut = load_cube_file([
                'LUT_3D_SIZE 2',
            ] + [
                "0  green 0.031",
                "0.96 0 0.031",
            ] * 3)

    def test_filename(self):
        with NamedTemporaryFile('w+t', delete=False) as f:
            f.write(
                "LUT_3D_SIZE 2\n"
                "0    0 0.031\n"
                "0.96 0 0.031\n"
                "0    1 0.031\n"
                "0.96 1 0.031\n"
                "0    0 0.931\n"
                "0.96 0 0.931\n"
                "0    1 0.931\n"
                "0.96 1 0.931\n"
            )

        try:
            lut = load_cube_file(f.name)
            self.assertTrue(isinstance(lut, ImageFilter.Color3DLUT))
            self.assertEqual(tuple(lut.size), (2, 2, 2))
            self.assertEqual(lut.name, "Color 3D LUT")
            self.assertEqual(lut.table[:12], [
                0, 0, 0.031,  0.96, 0, 0.031,  0, 1, 0.031,  0.96, 1, 0.031])
        finally:
            os.unlink(f.name)

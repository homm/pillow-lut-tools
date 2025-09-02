import os
from tempfile import NamedTemporaryFile

import numpy
import pytest
from PIL import Image, ImageFilter

from pillow_lut import identity_table, load_cube_file, load_hald_image, loaders

from . import PillowTestCase, disable_numpy, resource


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
        assert isinstance(lut, ImageFilter.Color3DLUT)
        assert tuple(lut.size) == (2, 2, 2)
        assert lut.name == "Color 3D LUT"
        assert lut.table[:12] == [
            0, 0, 0.031,  0.96, 0, 0.031,  0, 1, 0.031,  0.96, 1, 0.031
        ]

    def test_parser(self):
        lut = load_cube_file([
            " # Comment",
            'TITLE  "LUT name from file"',
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
        assert isinstance(lut, ImageFilter.Color3DLUT)
        assert tuple(lut.size) == (2, 3, 4)
        assert lut.channels == 4
        assert lut.name == "LUT name from file"
        assert lut.mode == 'HSV'
        assert lut.table[:12] == [
            0, 0, 0.031, 1,  0.96, 0, 0.031, 1,  0, 1, 0.031, 1
        ]

    def test_errors(self):
        with pytest.raises(ValueError, match="No size found"):
            load_cube_file([
                'TITLE "LUT name from file"',
            ] + [
                "0    0 0.031",
                "0.96 0 0.031",
            ] * 3)

        with pytest.raises(ValueError, match="number of colors on line 3"):
            load_cube_file([
                'LUT_3D_SIZE 2',
            ] + [
                "0    0 0.031",
                "0.96 0 0.031 1",
            ] * 3)

        with pytest.raises(ValueError, match="1D LUT cube files"):
            load_cube_file([
                'LUT_1D_SIZE 2',
            ] + [
                "0    0 0.031",
                "0.96 0 0.031 1",
            ])

        with pytest.raises(ValueError, match="Not a number on line 2"):
            load_cube_file([
                'LUT_3D_SIZE 2',
            ] + [
                "0  green 0.031",
                "0.96 0 0.031",
            ] * 3)

    def test_filename(self):
        with NamedTemporaryFile('w+t', delete=False) as f:
            f.write(
                "LUT_3D_SIZE\t2\n"
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
            assert isinstance(lut, ImageFilter.Color3DLUT)
            assert tuple(lut.size) == (2, 2, 2)
            assert lut.name == "Color 3D LUT"
            assert lut.table[:12] == [
                0, 0, 0.031,  0.96, 0, 0.031,  0, 1, 0.031,  0.96, 1, 0.031
            ]
        finally:
            os.unlink(f.name)

    def test_application(self):
        im = Image.new('RGB', (10, 10))

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
        im.filter(lut)
        assert isinstance(lut.table, list)


class TestLoadHaldImage(PillowTestCase):
    def test_wrong_size(self):
        with pytest.raises(ValueError, match="should be a square"):
            load_hald_image(Image.new('RGB', (8, 10)))

        with pytest.raises(ValueError, match="Can't detect hald size"):
            load_hald_image(Image.new('RGB', (7, 7)))

        with pytest.raises(ValueError, match="Can't detect hald size"):
            load_hald_image(Image.new('RGB', (729, 729)))

    def test_simple_parse(self):
        lut = load_hald_image(Image.new('RGB', (64, 64)), target_mode='HSV')
        assert isinstance(lut, ImageFilter.Color3DLUT)
        assert tuple(lut.size) == (16, 16, 16)
        assert lut.channels == 3
        assert lut.mode == 'HSV'

    def test_parse_file(self):
        lut = load_hald_image(resource('files', 'hald.6.hefe.png'))
        assert isinstance(lut, ImageFilter.Color3DLUT)
        assert tuple(lut.size) == (36, 36, 36)
        assert lut.channels == 3
        assert lut.mode is None

    def test_correctness(self):
        identity = identity_table(16)
        image = Image.open(resource('files', 'hald.4.png'))

        lut_numpy = load_hald_image(image)
        self.assertEqualLuts(lut_numpy, identity)

        with disable_numpy(loaders):
            lut_pillow = load_hald_image(image)
        self.assertEqualLuts(lut_pillow, identity)

    def test_application(self):
        im = Image.new('RGB', (10, 10))
        hald = Image.open(resource('files', 'hald.4.png'))

        lut_numpy = load_hald_image(hald)
        im.filter(lut_numpy)
        assert isinstance(lut_numpy.table, numpy.ndarray)

        with disable_numpy(loaders):
            lut_native = load_hald_image(hald)
        im.filter(lut_native)
        assert isinstance(lut_native.table, list)

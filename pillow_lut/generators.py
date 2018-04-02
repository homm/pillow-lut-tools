from __future__ import division, unicode_literals, absolute_import, print_function

from PIL import ImageFilter


def _to_linear(s):
    if s < 0.0404482362771082:
        return s / 12.92
    return pow((s + 0.055) / 1.055, 2.4)


def _to_srgb(l):
    if l < 0.00313066844250063:
        return l * 12.92
    return pow(l, 1 / 2.4) * 1.055 - 0.055


def rgb_color_enhance(size,
                      brightness=0, contrast=0, saturation=0, vibrance=0,
                      exposure=0, hue=0, sepia=0, gamma=1.0, linear=False):
    """Generates 3D color lookup table based on given values of basic
    color settings.

    :param size: Size of the table. From 2 to 65.
                 Performance can dramatically decrease wth bigger sizes.
    :param brightness: from -1.0 to 1.0.
    :param contrast: from -1.0 to 1.0.
    :param saturation: from -1.0 to 1.0.
    :param vibrance: from -1.0 to 1.0.
    :param exposure: from -1.0 to 1.0.
    :param hue: from 0 to 1.0.
    :param sepia: from 0 to 1.0.
    :param gamma: from 0 to 10.0. Default is 1.0.
    :param linear: boolean value. Convert values from sRGB to linear color space
                   before the manipulating and return after. Default is False.
                   Most arguments more sensitive in this mode.
    """

    if not -1.0 <= brightness <= 1.0:
        raise ValueError()
    if not -1.0 <= contrast <= 1.0:
        raise ValueError()
    if not -1.0 <= saturation <= 1.0:
        raise ValueError()
    if not -1.0 <= vibrance <= 1.0:
        raise ValueError()
    if not 0 <= gamma <= 10:
        raise ValueError()

    contrast = (contrast + 1)**2

    def generate(r, g, b):
        if linear:
            r = _to_linear(r)
            g = _to_linear(g)
            b = _to_linear(b)

        if brightness:
            r += brightness
            g += brightness
            b += brightness

        if contrast != 1:
            r = (r - 0.5) * contrast + 0.5
            g = (g - 0.5) * contrast + 0.5
            b = (b - 0.5) * contrast + 0.5

        if saturation:
            max_v = max(r, g, b)
            r += (r - max_v) * saturation
            g += (g - max_v) * saturation
            b += (b - max_v) * saturation

        if vibrance:
            max_v = max(r, g, b)
            avg_v = (r + g + b) / 3
            amt = abs(max_v - avg_v) * 2 * vibrance
            r += (r - max_v) * amt
            g += (g - max_v) * amt
            b += (b - max_v) * amt

        if gamma != 1:
            r = max(0, r)**gamma
            g = max(0, g)**gamma
            b = max(0, b)**gamma

        if linear:
            r = _to_srgb(r)
            g = _to_srgb(g)
            b = _to_srgb(b)

        return r, g, b

    return ImageFilter.Color3DLUT.generate(size, generate)

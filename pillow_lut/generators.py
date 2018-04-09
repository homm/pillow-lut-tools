from __future__ import division, unicode_literals, absolute_import

from . import ImageFilter


try:
    from . import _ext
except ImportError:  # pragma: no cover
    _ext = None

try:
    import numpy
except ImportError:  # pragma: no cover
    numpy = None


def _srgb_to_linear(s):
    if s < 0.0404482362771082:
        return s / 12.92
    return pow((s + 0.055) / 1.055, 2.4)


def _linear_to_srgb(l):
    if l < 0.00313066844250063:
        return l * 12.92
    return pow(l, 1 / 2.4) * 1.055 - 0.055


def _srgb_to_linear_numpy(s):
    choicelist = [s / 12.92, ((s + 0.055) / 1.055) ** 2.4]
    return numpy.select([s < 0.0404482362771082, True], choicelist)

def _linear_to_srgb_numpy(l):
    choicelist = [l * 12.92, (l ** (1 / 2.4)) * 1.055 - 0.055]
    return numpy.select([l < 0.00313066844250063, True], choicelist)


def _rgb_to_hsv(r, g, b):
    max_v = v = max(r, g, b)
    min_v = min(r, g, b)
    d = max_v - min_v
    s = 0 if max_v == 0 else d / max_v
    if max_v == min_v:
        h = 0
    elif max_v == r:
        h = (g - b) / d + (6 if g < b else 0)
    elif max_v == g:
        h = (b - r) / d + 2
    else:  # max_v == b
        h = (r - g) / d + 4
    h /= 6
    return h, s, v


def _hsv_to_rgb(h, s, v):
    i = int(h * 6)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    if i == 0: return v, t, p
    if i == 1: return q, v, p
    if i == 2: return p, v, t
    if i == 3: return p, q, v
    if i == 4: return t, p, v
    return v, p, q  # if i == 5:


def rgb_color_enhance(size,
                      brightness=0, contrast=0, saturation=0, vibrance=0,
                      # exposure=0,
                      hue=0,
                      # sepia=0,
                      gamma=1.0, linear=False, cls=ImageFilter.Color3DLUT):
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

    if brightness:
        if not isinstance(brightness, (tuple, list)):
            brightness = (brightness, brightness, brightness)
        if any(not -1.0 <= x <= 1.0 for x in brightness):
            raise ValueError("Brightness should be from -1.0 to 1.0")

    if contrast:
        if not isinstance(contrast, (tuple, list)):
            contrast = (contrast, contrast, contrast)
        if any(not -1.0 <= x <= 1.0 for x in contrast):
            raise ValueError("Contrast should be from -1.0 to 1.0")
        contrast = [(x + 1)**2 for x in contrast]

    if saturation:
        if not isinstance(saturation, (tuple, list)):
            saturation = (saturation, saturation, saturation)
        if any(not -1.0 <= x <= 1.0 for x in saturation):
            raise ValueError("Saturation should be from -1.0 to 1.0")

    if vibrance:
        if not isinstance(vibrance, (tuple, list)):
            vibrance = (vibrance, vibrance, vibrance)
        if any(not -1.0 <= x <= 1.0 for x in vibrance):
            raise ValueError("Vibrance should be from -1.0 to 1.0")

    if not 0 <= hue <= 1.0:
        raise ValueError("Hue should be from 0.0 to 1.0")

    if gamma != 1:
        if not isinstance(gamma, (tuple, list)):
            gamma = (gamma, gamma, gamma)
        if any(not 0 <= x <= 10 for x in gamma):
            raise ValueError("Gamma should be from 0.0 to 10.0")

    # if _ext:
    #     size = cls._check_size(size)
    #     table = _ext.generate_rgb_color_enhance(
    #         size,
    #         brightness or None, contrast or None, saturation or None,
    #         vibrance or None, None, hue or None, None,
    #         gamma if gamma != 1 else None, linear, linear,
    #     )
    #     return cls(size, table)

    if numpy and not hue:
        size = cls._check_size(size)
        b, g, r = numpy.mgrid[
            0 : 1 : size[2]*1j,
            0 : 1 : size[1]*1j,
            0 : 1 : size[0]*1j
        ].astype(numpy.float32)

        if linear:
            r = _srgb_to_linear_numpy(r)
            g = _srgb_to_linear_numpy(g)
            b = _srgb_to_linear_numpy(b)

        if brightness:
            r += brightness[0]
            g += brightness[1]
            b += brightness[2]

        if contrast:
            r = (r - 0.5) * contrast[0] + 0.5
            g = (g - 0.5) * contrast[1] + 0.5
            b = (b - 0.5) * contrast[2] + 0.5

        if saturation:
            max_v = numpy.maximum.reduce([r, g, b])
            r += (r - max_v) * saturation[0]
            g += (g - max_v) * saturation[1]
            b += (b - max_v) * saturation[2]

        if vibrance:
            max_v = numpy.maximum.reduce([r, g, b])
            avg_v = (r + g + b) / 3
            r += (r - max_v) * (max_v - avg_v) * vibrance[0]
            g += (g - max_v) * (max_v - avg_v) * vibrance[1]
            b += (b - max_v) * (max_v - avg_v) * vibrance[2]

        if gamma != 1:
            r = r.clip(0) ** gamma[0]
            g = g.clip(0) ** gamma[1]
            b = b.clip(0) ** gamma[2]

        if linear:
            r = _linear_to_srgb_numpy(r)
            g = _linear_to_srgb_numpy(g)
            b = _linear_to_srgb_numpy(b)

        table = numpy.stack((r, g, b), axis=-1)
        return cls(size, table.reshape(table.size))

    def generate(r, g, b):
        if linear:
            r = _srgb_to_linear(r)
            g = _srgb_to_linear(g)
            b = _srgb_to_linear(b)

        if brightness:
            r += brightness[0]
            g += brightness[1]
            b += brightness[2]

        if contrast:
            r = (r - 0.5) * contrast[0] + 0.5
            g = (g - 0.5) * contrast[1] + 0.5
            b = (b - 0.5) * contrast[2] + 0.5

        if saturation:
            max_v = max(r, g, b)
            r += (r - max_v) * saturation[0]
            g += (g - max_v) * saturation[1]
            b += (b - max_v) * saturation[2]

        if vibrance:
            max_v = max(r, g, b)
            avg_v = (r + g + b) / 3
            r += (r - max_v) * (max_v - avg_v) * vibrance[0]
            g += (g - max_v) * (max_v - avg_v) * vibrance[1]
            b += (b - max_v) * (max_v - avg_v) * vibrance[2]

        if hue:
            h, s, v = _rgb_to_hsv(r, g, b)
            r, g, b = _hsv_to_rgb((h + hue) % 1, s, v)

        if gamma != 1:
            r = max(0, r)**gamma[0]
            g = max(0, g)**gamma[1]
            b = max(0, b)**gamma[2]

        if linear:
            r = _linear_to_srgb(r)
            g = _linear_to_srgb(g)
            b = _linear_to_srgb(b)

        return r, g, b

    return cls.generate(size, generate)


def identity_table(size, target_mode=None, cls=ImageFilter.Color3DLUT):
    """Returns noop lookup table with linear distributed values.

    :param size: Size of the table. From 2 to 65.
    :param target_mode: A mode for the result image. Should have not less
                        than ``channels`` channels. Default is ``None``,
                        which means that mode wouldn't be changed.
    """
    if numpy:
        size = cls._check_size(size)
        b, g, r = numpy.mgrid[
            0 : 1 : size[2]*1j,
            0 : 1 : size[1]*1j,
            0 : 1 : size[0]*1j
        ].astype(numpy.float32)

        table = numpy.stack((r, g, b), axis=-1)
        return cls(size, table.reshape(table.size), target_mode=target_mode)

    return cls.generate(size, lambda r, g, b: (r, g, b),
                        target_mode=target_mode)

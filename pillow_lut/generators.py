from math import sin

from PIL import ImageFilter


try:
    import numpy
except ImportError:  # pragma: no cover
    numpy = None


def _srgb_to_linear(s):
    if s < 0.0404482362771082:
        return s / 12.92
    return pow((s + 0.055) / 1.055, 2.4)


def _linear_to_srgb(lin):
    if lin < 0.00313066844250063:
        return lin * 12.92
    return pow(lin, 1 / 2.4) * 1.055 - 0.055


def _srgb_to_linear_numpy(s):
    choicelist = [s / 12.92, ((s + 0.055) / 1.055) ** 2.4]
    return numpy.select([s < 0.0404482362771082, True], choicelist)


def _linear_to_srgb_numpy(lin):
    choicelist = [lin * 12.92, (lin ** (1 / 2.4)) * 1.055 - 0.055]
    return numpy.select([lin < 0.00313066844250063, True], choicelist)


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
    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    return v, p, q  # if i == 5:


def _rgb_to_yuv(r, g, b):
    y = (0.299 * r) + (0.587 * g) + (0.114 * b)
    u = (1.0 / 1.772) * (b - y)
    v = (1.0 / 1.402) * (r - y)
    return y, u, v


def _yuv_to_rgb(y, u, v):
    r = 1.402 * v + y
    g = (y - (0.299 * 1.402 / 0.587) * v - (0.114 * 1.772 / 0.587) * u)
    b = 1.772 * u + y
    return r, g, b


def rgb_color_enhance(source,
                      brightness=0, exposure=0, contrast=0, warmth=0,
                      saturation=0, vibrance=0,
                      # highlights=0, shadows=0,
                      hue=0, gamma=1.0,
                      linear=False, cls=ImageFilter.Color3DLUT):
    """Generates 3D color lookup table based on given values of basic
    color settings.

    :param source: Could be the source lookup table which will be modified,
                   or just a size of new identity table, from 2 to 65.
                   Performance can dramatically decrease wth bigger sizes.
    :param brightness: One value for all channels, or tuple of three values
                       from -1.0 to 1.0. Use ``exposure`` for better result.
    :param exposure: One value for all channels, or tuple of three values
                     from -5.0 to 5.0.
    :param contrast: One value for all channels, or tuple of three values
                     from -1.0 to 5.0.
    :param warmth: One value from -1.0 to 1.0.
    :param saturation: One value for all channels, or tuple of three values
                       from -1.0 to 5.0.
    :param vibrance: One value for all channels, or tuple of three values
                     from -1.0 to 5.0.
    :param hue: One value from 0 to 1.0.
    :param gamma: One value from 0 to 10.0. Default is 1.0.
    :param linear: boolean value. Convert values from sRGB to linear color space
                   before the manipulating and return after. Default is False.
                   Most arguments more sensitive in this mode.
    """
    source_is_lut = hasattr(source, 'table')
    if source_is_lut and source.channels != 3:
        raise ValueError("Only 3-channels table could be a source")

    if brightness:
        if not isinstance(brightness, (tuple, list)):
            brightness = (brightness, brightness, brightness)
        if any(not -1.0 <= x <= 1.0 for x in brightness):
            raise ValueError("Brightness should be from -1.0 to 1.0")

    if exposure:
        if not isinstance(exposure, (tuple, list)):
            exposure = (exposure, exposure, exposure)
        if any(not -5.0 <= x <= 5.0 for x in exposure):
            raise ValueError("Exposure should be from -5.0 to 5.0")
        exposure = [(2**x) for x in exposure]

    if contrast:
        if not isinstance(contrast, (tuple, list)):
            contrast = (contrast, contrast, contrast)
        if any(not -1.0 <= x <= 5.0 for x in contrast):
            raise ValueError("Contrast should be from -1.0 to 5.0")
        contrast = [(x + 1) for x in contrast]

    if warmth:
        if not -1.0 <= warmth <= 1.0:
            raise ValueError("Warmth should be from -1.0 to 1.0")
        if warmth < 0:
            warmth = (warmth * -0.0588, warmth * -0.1569, warmth * 0.1255)
        else:
            warmth = (warmth * 0.1765, warmth * -0.1255, warmth * 0.0902)

    if saturation:
        if not isinstance(saturation, (tuple, list)):
            saturation = (saturation, saturation, saturation)
        if any(not -1.0 <= x <= 5.0 for x in saturation):
            raise ValueError("Saturation should be from -1.0 to 5.0")

    if vibrance:
        if not isinstance(vibrance, (tuple, list)):
            vibrance = (vibrance, vibrance, vibrance)
        if any(not -1.0 <= x <= 5.0 for x in vibrance):
            raise ValueError("Vibrance should be from -1.0 to 1.0")
        vibrance = [(x * 2) for x in vibrance]

    # if highlights:
    #     if not isinstance(highlights, (tuple, list)):
    #         highlights = (highlights, highlights, highlights)
    #     if any(not -1.0 <= x <= 1.0 for x in highlights):
    #         raise ValueError("Highlights should be from -1.0 to 1.0")

    # if shadows:
    #     if not isinstance(shadows, (tuple, list)):
    #         shadows = (shadows, shadows, shadows)
    #     if any(not -1.0 <= x <= 1.0 for x in shadows):
    #         raise ValueError("Shadows should be from -1.0 to 1.0")

    if not 0 <= hue <= 1.0:
        raise ValueError("Hue should be from 0.0 to 1.0")

    if gamma != 1:
        if not isinstance(gamma, (tuple, list)):
            gamma = (gamma, gamma, gamma)
        if any(not 0 <= x <= 10 for x in gamma):
            raise ValueError("Gamma should be from 0.0 to 10.0")

    if numpy and not hue:
        if source_is_lut:
            size = source.size
            points = numpy.asarray(source.table, dtype=numpy.float32)
            r = points[0::3]
            g = points[1::3]
            b = points[2::3]
        else:
            size = cls._check_size(source)
            b, g, r = numpy.mgrid[
                0:1:size[2]*1j,
                0:1:size[1]*1j,
                0:1:size[0]*1j
            ].astype(numpy.float32)

        if linear:
            r = _srgb_to_linear_numpy(r)
            g = _srgb_to_linear_numpy(g)
            b = _srgb_to_linear_numpy(b)

        if contrast:
            r = (r - 0.5) * contrast[0] + 0.5
            g = (g - 0.5) * contrast[1] + 0.5
            b = (b - 0.5) * contrast[2] + 0.5

        if saturation:
            avg_v = r * 0.2126 + g * 0.7152 + b * 0.0722
            r += (r - avg_v) * saturation[0]
            g += (g - avg_v) * saturation[1]
            b += (b - avg_v) * saturation[2]

        if vibrance:
            max_v = numpy.maximum.reduce([r, g, b])
            avg_v = r * 0.2126 + g * 0.7152 + b * 0.0722
            r += (r - max_v) * (max_v - avg_v) * vibrance[0]
            g += (g - max_v) * (max_v - avg_v) * vibrance[1]
            b += (b - max_v) * (max_v - avg_v) * vibrance[2]

        if exposure:
            r = 1.0 - (1.0 - r).clip(0) ** exposure[0]
            g = 1.0 - (1.0 - g).clip(0) ** exposure[1]
            b = 1.0 - (1.0 - b).clip(0) ** exposure[2]

        if brightness:
            r += brightness[0]
            g += brightness[1]
            b += brightness[2]

        if warmth:
            y, u, v = _rgb_to_yuv(r, g, b)
            scale = numpy.sin(y * 3.14159)
            y += scale * warmth[0]
            u += scale * warmth[1]
            v += scale * warmth[2]
            r, g, b = _yuv_to_rgb(y, u, v)

        if gamma != 1:
            r = r.clip(0) ** gamma[0]
            g = g.clip(0) ** gamma[1]
            b = b.clip(0) ** gamma[2]

        if linear:
            r = _linear_to_srgb_numpy(r)
            g = _linear_to_srgb_numpy(g)
            b = _linear_to_srgb_numpy(b)

        table = numpy.stack((r, g, b), axis=-1)
        return cls(size, table.reshape(table.size), _copy_table=False)

    def generate(r, g, b):
        if linear:
            r = _srgb_to_linear(r)
            g = _srgb_to_linear(g)
            b = _srgb_to_linear(b)

        if contrast:
            r = (r - 0.5) * contrast[0] + 0.5
            g = (g - 0.5) * contrast[1] + 0.5
            b = (b - 0.5) * contrast[2] + 0.5

        if saturation:
            avg_v = r * 0.2126 + g * 0.7152 + b * 0.0722
            r += (r - avg_v) * saturation[0]
            g += (g - avg_v) * saturation[1]
            b += (b - avg_v) * saturation[2]

        if vibrance:
            max_v = max(r, g, b)
            avg_v = r * 0.2126 + g * 0.7152 + b * 0.0722
            r += (r - max_v) * (max_v - avg_v) * vibrance[0]
            g += (g - max_v) * (max_v - avg_v) * vibrance[1]
            b += (b - max_v) * (max_v - avg_v) * vibrance[2]

        if exposure:
            r = 1.0 - max(0, 1.0 - r) ** exposure[0]
            g = 1.0 - max(0, 1.0 - g) ** exposure[1]
            b = 1.0 - max(0, 1.0 - b) ** exposure[2]

        if brightness:
            r += brightness[0]
            g += brightness[1]
            b += brightness[2]

        if warmth:
            y, u, v = _rgb_to_yuv(r, g, b)
            scale = sin(y * 3.14159)
            y += scale * warmth[0]
            u += scale * warmth[1]
            v += scale * warmth[2]
            r, g, b = _yuv_to_rgb(y, u, v)

        if hue:
            h, s, v = _rgb_to_hsv(r, g, b)
            r, g, b = _hsv_to_rgb((h + hue) % 1, s, v)

        if gamma != 1:
            r = max(0, r) ** gamma[0]
            g = max(0, g) ** gamma[1]
            b = max(0, b) ** gamma[2]

        if linear:
            r = _linear_to_srgb(r)
            g = _linear_to_srgb(g)
            b = _linear_to_srgb(b)

        return r, g, b

    if source_is_lut:
        return source.transform(generate)
    else:
        return cls.generate(source, generate)


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
            0:1:size[2]*1j,
            0:1:size[1]*1j,
            0:1:size[0]*1j
        ].astype(numpy.float32)

        table = numpy.stack((r, g, b), axis=-1)
        return cls(size, table.reshape(table.size),
                   target_mode=target_mode, _copy_table=False)

    return cls.generate(size, lambda r, g, b: (r, g, b),
                        target_mode=target_mode)

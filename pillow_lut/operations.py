from __future__ import division, unicode_literals, absolute_import

import warnings

from . import Image, ImageFilter


def _inter_linear(d, v0, v1):
    return v0 + (v1 - v0) * d


def _inter_linear_table(d, c, table, i0, i1):
    return [
        _inter_linear(d, table[i0+i], table[i1+i])
        for i in range(c)
    ]


def _inter_linear_vector(d, c, v0, v1):
    return [
        _inter_linear(d, v0[i], v1[i])
        for i in range(c)
    ]


def _inter_cubic_inner(d, v0, v1, v2, v3):
    """
    :param d: Distance from v1 to the side of v2, from 0.0 to 1.0
    """
    # https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
    def filter_1l(x):
        a = -0.5
        return ((a + 2.0) * x - (a + 3.0)) * x*x + 1

    def filter_2l(x):
        a = -0.5
        return (((x - 5) * x + 8) * x - 4) * a

    return (v0 * filter_2l(1+d) +
            v1 * filter_1l(0+d) +
            v2 * filter_1l(1-d) +
            v3 * filter_2l(2-d))


def _inter_cubic(d, v0, v1, v2, v3):
    if d < 0:
        if d < -1.0:
            return _inter_linear(d + 1.0, v0, v1)
        return _inter_cubic_inner(d + 1.0, v0 * 2 - v1, v0, v1, v2)
    if d >= 1.0:
        if d >= 2.0:
            return _inter_linear(d - 1.0, v2, v3)
        return _inter_cubic_inner(d - 1.0, v1, v2, v3, v3 * 2 - v2)
    return _inter_cubic_inner(d, v0, v1, v2, v3)


def _inter_cubic_table(d, c, table, i0, i1, i2, i3):
    return [
        _inter_cubic(d, table[i0+i], table[i1+i], table[i2+i], table[i3+i])
        for i in range(c)
    ]


def _inter_cubic_vector(d, c, v0, v1, v2, v3):
    return [
        _inter_cubic(d, v0[i], v1[i], v2[i], v3[i])
        for i in range(c)
    ]


def _point_shift(size, point, left, right):
    size1D, size2D, size3D = size

    index1D = point[0] * (size1D - 1)
    index2D = point[1] * (size2D - 1)
    index3D = point[2] * (size3D - 1)
    idx1D = max(left, min(size1D - 1 - right, int(index1D)))
    idx2D = max(left, min(size2D - 1 - right, int(index2D)))
    idx3D = max(left, min(size3D - 1 - right, int(index3D)))
    shift1D = index1D - idx1D
    shift2D = index2D - idx2D
    shift3D = index3D - idx3D
    idx = idx1D + idx2D * size1D + idx3D * size1D * size2D
    return idx, shift1D, shift2D, shift3D


def sample_lut_linear(lut, point):
    size1D, size2D, size3D = lut.size
    c = lut.channels
    s1Dc = size1D * c
    s12Dc = size1D * size2D * c

    idx, shift1D, shift2D, shift3D = _point_shift(lut.size, point, 0, 1)
    idx *= c

    return _inter_linear_vector(shift3D, c,
        _inter_linear_vector(shift2D, c,
            _inter_linear_table(shift1D, c, lut.table,
                idx + 0, idx + c),
            _inter_linear_table(shift1D, c, lut.table,
                idx + s1Dc + 0, idx + s1Dc + c),
        ),
        _inter_linear_vector(shift2D, c,
            _inter_linear_table(shift1D, c, lut.table,
                idx + s12Dc + 0, idx + s12Dc + c),
            _inter_linear_table(shift1D, c, lut.table,
                idx + s12Dc + s1Dc + 0, idx + s12Dc + s1Dc + c),
        ),
    )


def sample_lut_cubic(lut, point):
    size1D, size2D, size3D = lut.size
    c = lut.channels
    s1Dc = size1D * c
    s12Dc = size1D * size2D * c
    c2, s1Dc2, s12Dc2 = c * 2, s1Dc * 2, s12Dc * 2

    if size1D < 4 or size2D < 4 or size3D < 4:
        raise ValueError("Cubic interpolation requires a table of size "
                         "4 in all dimensions at least. Switching to linear.")

    idx, shift1D, shift2D, shift3D = _point_shift(lut.size, point, 1, 2)
    idx *= c

    return _inter_cubic_vector(shift3D, c,
        _inter_cubic_vector(shift2D, c,
            _inter_cubic_table(shift1D, c, lut.table,
                idx-s12Dc-s1Dc-c, idx-s12Dc-s1Dc+0,
                idx-s12Dc-s1Dc+c, idx-s12Dc-s1Dc+c2),
            _inter_cubic_table(shift1D, c, lut.table,
                idx-s12Dc-c, idx-s12Dc+0, idx-s12Dc+c, idx-s12Dc+c2),
            _inter_cubic_table(shift1D, c, lut.table,
                idx-s12Dc+s1Dc-c, idx-s12Dc+s1Dc+0,
                idx-s12Dc+s1Dc+c, idx-s12Dc+s1Dc+c2),
            _inter_cubic_table(shift1D, c, lut.table,
                idx-s12Dc+s1Dc2-c, idx-s12Dc+s1Dc2+0,
                idx-s12Dc+s1Dc2+c, idx-s12Dc+s1Dc2+c2),
        ),
        _inter_cubic_vector(shift2D, c,
            _inter_cubic_table(shift1D, c, lut.table,
                idx-s1Dc-c, idx-s1Dc+0, idx-s1Dc+c, idx-s1Dc+c2),
            _inter_cubic_table(shift1D, c, lut.table,
                idx-c, idx+0, idx+c, idx+c2),
            _inter_cubic_table(shift1D, c, lut.table,
                idx+s1Dc-c, idx+s1Dc+0, idx+s1Dc+c, idx+s1Dc+c2),
            _inter_cubic_table(shift1D, c, lut.table,
                idx+s1Dc2-c, idx+s1Dc2+0, idx+s1Dc2+c, idx+s1Dc2+c2),
        ),
        _inter_cubic_vector(shift2D, c,
            _inter_cubic_table(shift1D, c, lut.table,
                idx+s12Dc-s1Dc-c, idx+s12Dc-s1Dc+0,
                idx+s12Dc-s1Dc+c, idx+s12Dc-s1Dc+c2),
            _inter_cubic_table(shift1D, c, lut.table,
                idx+s12Dc-c, idx+s12Dc+0, idx+s12Dc+c, idx+s12Dc+c2),
            _inter_cubic_table(shift1D, c, lut.table,
                idx+s12Dc+s1Dc-c, idx+s12Dc+s1Dc+0,
                idx+s12Dc+s1Dc+c, idx+s12Dc+s1Dc+c2),
            _inter_cubic_table(shift1D, c, lut.table,
                idx+s12Dc+s1Dc2-c, idx+s12Dc+s1Dc2+0,
                idx+s12Dc+s1Dc2+c, idx+s12Dc+s1Dc2+c2),
        ),
        _inter_cubic_vector(shift2D, c,
            _inter_cubic_table(shift1D, c, lut.table,
                idx+s12Dc2-s1Dc-c, idx+s12Dc2-s1Dc+0,
                idx+s12Dc2-s1Dc+c, idx+s12Dc2-s1Dc+c2),
            _inter_cubic_table(shift1D, c, lut.table,
                idx+s12Dc2-c, idx+s12Dc2+0, idx+s12Dc2+c, idx+s12Dc2+c2),
            _inter_cubic_table(shift1D, c, lut.table,
                idx+s12Dc2+s1Dc-c, idx+s12Dc2+s1Dc+0,
                idx+s12Dc2+s1Dc+c, idx+s12Dc2+s1Dc+c2),
            _inter_cubic_table(shift1D, c, lut.table,
                idx+s12Dc2+s1Dc2-c, idx+s12Dc2+s1Dc2+0,
                idx+s12Dc2+s1Dc2+c, idx+s12Dc2+s1Dc2+c2),
        ),
    )


def transform_lut(source, lut, target_size=None, interp=Image.LINEAR,
                  cls=ImageFilter.Color3DLUT):
    if source.channels != 3:
        raise ValueError("Can transform only 3-channel cubes")
    if interp == Image.LINEAR:
        sample_lut = sample_lut_linear
    elif interp == Image.CUBIC:
        sample_lut = sample_lut_cubic
    else:
        raise ValueError(
            "Only Image.LINEAR and Image.CUBIC interpolations are supported")

    if target_size:
        size1D, size2D, size3D = cls._check_size(target_size)
    else:
        size1D, size2D, size3D = source.size

    if interp == Image.CUBIC and (size1D < 4 or size2D < 4 or size3D < 4):
        sample_lut = sample_lut_linear
        interp=Image.LINEAR
        warnings.warn("Cubic interpolation requires a table of size "
                      "4 in all dimensions at least. Switching to linear.")

    if (
        (interp == Image.CUBIC and size1D * size2D * size3D >= 216) or
        (interp == Image.LINEAR and size1D * size2D * size3D >= 1000)
    ):
        warnings.warn("You are using not accelerated python version "
                      "of transform_lut, which could be fairly slow.")

    table = []
    index = 0
    for b in range(size3D):
        for g in range(size2D):
            for r in range(size1D):
                if target_size:
                    point = sample_lut(source, (r / float(size1D-1),
                                                g / float(size2D-1),
                                                b / float(size3D-1)))
                else:
                    point = (source.table[index + 0],
                             source.table[index + 1],
                             source.table[index + 2])
                    index += 3
                table.append(sample_lut(lut, point))

    return cls((size1D, size2D, size3D), table,
               channels=lut.channels, target_mode=lut.mode or source.mode)


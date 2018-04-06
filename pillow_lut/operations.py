from __future__ import division, unicode_literals, absolute_import

from . import Image, ImageFilter


def _interpolate_linear(d, v0, v1):
    return v0 + (v1 - v0) * d


def _interpolate_linear_vector(d, v0, v1):
    return [
        _interpolate_linear(d, v0, v1)
        for v0, v1 in zip(v0, v1)
    ]


def _interpolate_cubic(d, v0, v1, v2, v3):
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


def _interpolate_cubic_vector(d, v0, v1, v2, v3):
    return [
        _interpolate_cubic(d, v0, v1, v2, v3)
        for v0, v1, v2, v3 in zip(v0, v1, v2, v3)
    ]


def point_lut_linear(lut, point):
    size1D, size2D, size3D = lut.size
    c = lut.channels
    s1Dc = size1D * c
    s12Dc = size1D * size2D * c

    index1D = point[0] * (size1D - 1)
    index2D = point[1] * (size2D - 1)
    index3D = point[2] * (size3D - 1)
    idx1D = max(0, min(size1D - 2, int(index1D)))
    idx2D = max(0, min(size2D - 2, int(index2D)))
    idx3D = max(0, min(size3D - 2, int(index3D)))
    shift1D = index1D - idx1D
    shift2D = index2D - idx2D
    shift3D = index3D - idx3D
    idx = idx1D*3 + idx2D * s1Dc + idx3D * s12Dc

    return _interpolate_linear_vector(shift3D,
        _interpolate_linear_vector(shift2D,
            _interpolate_linear_vector(shift1D,
                lut.table[idx + 0:idx + c],
                lut.table[idx + c:idx + c + c]),
            _interpolate_linear_vector(shift1D,
                lut.table[idx + s1Dc + 0:idx + s1Dc + c],
                lut.table[idx + s1Dc + c:idx + s1Dc + c + c])
        ),
        _interpolate_linear_vector(shift2D,
            _interpolate_linear_vector(shift1D,
                lut.table[idx + s12Dc + 0:idx + s12Dc + c],
                lut.table[idx + s12Dc + c:idx + s12Dc + c + c]),
            _interpolate_linear_vector(shift1D,
                lut.table[idx + s12Dc + s1Dc + 0:idx + s12Dc + s1Dc + c],
                lut.table[idx + s12Dc + s1Dc + c:idx + s12Dc + s1Dc + c + c])
        )
    )


def transform_lut(source, lut, target_size=None, interp=Image.LINEAR,
                  cls=ImageFilter.Color3DLUT):
    if source.channels != 3:
        raise ValueError("Can transform only 3-channel cubes")
    if interp != Image.LINEAR:
        raise ValueError("Only linear interpolation is supported")
    size1D, size2D, size3D = source.size

    table = []
    index = 0
    for b in range(size3D):
        for g in range(size2D):
            for r in range(size1D):
                table.append(point_lut_linear(lut, (
                    source.table[index + 0],
                    source.table[index + 1],
                    source.table[index + 2],
                ), interp))
                index += 3

    return cls(source.size, table, channels=target.channels,
               target_mode=target.mode or source.mode)


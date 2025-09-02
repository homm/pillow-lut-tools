import warnings

from PIL import Image, ImageFilter


try:
    import numpy
except ImportError:  # pragma: no cover
    numpy = None


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


def _points_shift_numpy(size, points, left, right):
    size1D, size2D, size3D = size

    index1D = points[:, 0] * (size1D - 1)
    index2D = points[:, 1] * (size2D - 1)
    index3D = points[:, 2] * (size3D - 1)
    idx1D = index1D.astype(numpy.int32).clip(left, size1D - 1 - right)
    idx2D = index2D.astype(numpy.int32).clip(left, size2D - 1 - right)
    idx3D = index3D.astype(numpy.int32).clip(left, size3D - 1 - right)
    shift1D = numpy.subtract(index1D, idx1D, index1D).reshape(idx1D.shape[0], 1)
    shift2D = numpy.subtract(index2D, idx2D, index2D).reshape(idx2D.shape[0], 1)
    shift3D = numpy.subtract(index3D, idx3D, index3D).reshape(idx3D.shape[0], 1)
    idx = idx1D + idx2D * size1D + idx3D * size1D * size2D
    return idx, shift1D, shift2D, shift3D


def _sample_lut_linear_numpy(lut, points):
    s1D, s2D, s3D = lut.size
    s12D = s1D * s2D

    idx, shift1D, shift2D, shift3D = _points_shift_numpy(lut.size, points, 0, 1)
    table = numpy.asarray(lut.table, dtype=numpy.float32)
    table = table.reshape(s1D * s2D * s3D, lut.channels)

    return _inter_linear(
        shift3D,
        _inter_linear(
            shift2D,
            _inter_linear(shift1D, table[idx + 0], table[idx + 1]),
            _inter_linear(shift1D, table[idx + s1D + 0],
                          table[idx + s1D + 1]),
        ),
        _inter_linear(
            shift2D,
            _inter_linear(shift1D, table[idx + s12D + 0],
                          table[idx + s12D + 1]),
            _inter_linear(shift1D, table[idx + s12D + s1D + 0],
                          table[idx + s12D + s1D + 1]),
        ),
    )


def sample_lut_linear(lut, point):
    """Computes the new point value from given 3D lookup table
    using linear interpolation.

    :param lut: Lookup table, ``ImageFilter.Color3DLUT`` object.
    :param point: A tuple of 3 values with coordinates in the cube,
                  normalized from 0.0 to 1.0. Could be out of range.
    """
    size1D, size2D, size3D = lut.size
    c = lut.channels
    s1Dc = size1D * c
    s12Dc = size1D * size2D * c

    idx, shift1D, shift2D, shift3D = _point_shift(lut.size, point, 0, 1)
    idx *= c

    return _inter_linear_vector(
        shift3D, c,
        _inter_linear_vector(
            shift2D, c,
            _inter_linear_table(shift1D, c, lut.table, idx + 0, idx + c),
            _inter_linear_table(shift1D, c, lut.table, idx + s1Dc + 0, idx + s1Dc + c),
        ),
        _inter_linear_vector(
            shift2D, c,
            _inter_linear_table(shift1D, c, lut.table,
                                idx + s12Dc + 0, idx + s12Dc + c),
            _inter_linear_table(shift1D, c, lut.table,
                                idx + s12Dc + s1Dc + 0, idx + s12Dc + s1Dc + c),
        ),
    )


def sample_lut_cubic(lut, point):
    """Computes the new point value from given 3D lookup table
    using cubic interpolation.

    :param lut: Lookup table, ``ImageFilter.Color3DLUT`` object.
    :param point: A tuple of 3 values with coordinates in the cube,
                  normalized from 0.0 to 1.0. Could be out of range,
                  in this case, will be linear interpolated using
                  two extreme values.
    """
    size1D, size2D, size3D = lut.size
    c = lut.channels
    s1Dc = size1D * c
    s12Dc = size1D * size2D * c
    c2, s1Dc2, s12Dc2 = c * 2, s1Dc * 2, s12Dc * 2

    if size1D < 4 or size2D < 4 or size3D < 4:
        raise ValueError("BICUBIC interpolation requires a table of size "
                         "4 in all dimensions at least. Please switch to BILINEAR.")

    idx, shift1D, shift2D, shift3D = _point_shift(lut.size, point, 1, 2)
    idx *= c

    return _inter_cubic_vector(
        shift3D, c,
        _inter_cubic_vector(
            shift2D, c,
            _inter_cubic_table(
                shift1D, c, lut.table,
                idx-s12Dc-s1Dc-c, idx-s12Dc-s1Dc+0,
                idx-s12Dc-s1Dc+c, idx-s12Dc-s1Dc+c2),
            _inter_cubic_table(
                shift1D, c, lut.table,
                idx-s12Dc-c, idx-s12Dc+0, idx-s12Dc+c, idx-s12Dc+c2),
            _inter_cubic_table(
                shift1D, c, lut.table,
                idx-s12Dc+s1Dc-c, idx-s12Dc+s1Dc+0,
                idx-s12Dc+s1Dc+c, idx-s12Dc+s1Dc+c2),
            _inter_cubic_table(
                shift1D, c, lut.table,
                idx-s12Dc+s1Dc2-c, idx-s12Dc+s1Dc2+0,
                idx-s12Dc+s1Dc2+c, idx-s12Dc+s1Dc2+c2),
        ),
        _inter_cubic_vector(
            shift2D, c,
            _inter_cubic_table(
                shift1D, c, lut.table,
                idx-s1Dc-c, idx-s1Dc+0, idx-s1Dc+c, idx-s1Dc+c2),
            _inter_cubic_table(
                shift1D, c, lut.table,
                idx-c, idx+0, idx+c, idx+c2),
            _inter_cubic_table(
                shift1D, c, lut.table,
                idx+s1Dc-c, idx+s1Dc+0, idx+s1Dc+c, idx+s1Dc+c2),
            _inter_cubic_table(
                shift1D, c, lut.table,
                idx+s1Dc2-c, idx+s1Dc2+0, idx+s1Dc2+c, idx+s1Dc2+c2),
        ),
        _inter_cubic_vector(
            shift2D, c,
            _inter_cubic_table(
                shift1D, c, lut.table,
                idx+s12Dc-s1Dc-c, idx+s12Dc-s1Dc+0,
                idx+s12Dc-s1Dc+c, idx+s12Dc-s1Dc+c2),
            _inter_cubic_table(
                shift1D, c, lut.table,
                idx+s12Dc-c, idx+s12Dc+0, idx+s12Dc+c, idx+s12Dc+c2),
            _inter_cubic_table(
                shift1D, c, lut.table,
                idx+s12Dc+s1Dc-c, idx+s12Dc+s1Dc+0,
                idx+s12Dc+s1Dc+c, idx+s12Dc+s1Dc+c2),
            _inter_cubic_table(
                shift1D, c, lut.table,
                idx+s12Dc+s1Dc2-c, idx+s12Dc+s1Dc2+0,
                idx+s12Dc+s1Dc2+c, idx+s12Dc+s1Dc2+c2),
        ),
        _inter_cubic_vector(
            shift2D, c,
            _inter_cubic_table(
                shift1D, c, lut.table,
                idx+s12Dc2-s1Dc-c, idx+s12Dc2-s1Dc+0,
                idx+s12Dc2-s1Dc+c, idx+s12Dc2-s1Dc+c2),
            _inter_cubic_table(
                shift1D, c, lut.table,
                idx+s12Dc2-c, idx+s12Dc2+0, idx+s12Dc2+c, idx+s12Dc2+c2),
            _inter_cubic_table(
                shift1D, c, lut.table,
                idx+s12Dc2+s1Dc-c, idx+s12Dc2+s1Dc+0,
                idx+s12Dc2+s1Dc+c, idx+s12Dc2+s1Dc+c2),
            _inter_cubic_table(
                shift1D, c, lut.table,
                idx+s12Dc2+s1Dc2-c, idx+s12Dc2+s1Dc2+0,
                idx+s12Dc2+s1Dc2+c, idx+s12Dc2+s1Dc2+c2),
        ),
    )


def resize_lut(source, target_size, interp=Image.BILINEAR,
               cls=ImageFilter.Color3DLUT):
    """Resizes given lookup table to new size using interpolation.

    :param source: Source lookup table, ``ImageFilter.Color3DLUT`` object.
    :param target_size: Size of the resulting lookup table.
    :param interp: Interpolation type, ``Image.BILINEAR`` or ``Image.BICUBIC``.
                   BILINEAR is default. BICUBIC is dramatically slower.
    """
    size1D, size2D, size3D = cls._check_size(target_size)
    if interp == Image.BILINEAR:
        sample_lut = sample_lut_linear
    elif interp == Image.BICUBIC:
        sample_lut = sample_lut_cubic
    else:
        raise ValueError(
            "Only Image.BILINEAR and Image.BICUBIC interpolations are supported")
    if interp == Image.BICUBIC and any(s < 4 for s in source.size):
        sample_lut = sample_lut_linear
        interp = Image.BILINEAR
        warnings.warn("BICUBIC interpolation requires a table of size "
                      "4 in all dimensions at least. Switching to BILINEAR.")

    if numpy and interp == Image.BILINEAR:
        shape = (size1D * size2D * size3D, 3)
        b, g, r = numpy.mgrid[
            0:1:size3D*1j,
            0:1:size2D*1j,
            0:1:size1D*1j
        ].astype(numpy.float32)
        points = numpy.stack((r, g, b), axis=-1).reshape(shape)
        points = _sample_lut_linear_numpy(source, points)

        table = points.reshape(points.size)

    else:  # Native implementation
        table = []
        for b in range(size3D):
            for g in range(size2D):
                for r in range(size1D):
                    point = (r / (size1D-1), g / (size2D-1), b / (size3D-1))
                    table.extend(sample_lut(source, point))

    return cls((size1D, size2D, size3D), table,
               channels=source.channels, target_mode=source.mode,
               _copy_table=False)


def transform_lut(source, lut, target_size=None, interp=Image.BILINEAR,
                  cls=ImageFilter.Color3DLUT):
    """Transforms given lookup table using another table and returns the result.
    Sizes of the tables do not have to be the same. Moreover,
    you can set the result table size with ``target_size`` argument.

    :param source: Source lookup table, ``ImageFilter.Color3DLUT`` object.
    :param lut: Applied lookup table, ``ImageFilter.Color3DLUT`` object.
    :param target_size: Optional size of the resulting lookup table.
                        By default, size of the ``source`` will be used.
    :param interp: Interpolation type, ``Image.BILINEAR`` or ``Image.BICUBIC``.
                   BILINEAR is default. BICUBIC is dramatically slower.
    """
    if source.channels != 3:
        raise ValueError("Can transform only 3-channel cubes")
    if interp == Image.BILINEAR:
        sample_lut = sample_lut_linear
    elif interp == Image.BICUBIC:
        sample_lut = sample_lut_cubic
    else:
        raise ValueError(
            "Only Image.BILINEAR and Image.BICUBIC interpolations are supported")

    if target_size:
        size1D, size2D, size3D = cls._check_size(target_size)
    else:
        size1D, size2D, size3D = source.size

    if interp == Image.BICUBIC:
        small_lut = any(s < 4 for s in lut.size)
        small_source = any(s < 4 for s in source.size)
        if small_lut or (target_size and small_source):
            sample_lut = sample_lut_linear
            interp = Image.BILINEAR
            warnings.warn("Cubic interpolation requires a table of size "
                          "4 in all dimensions at least. Switching to linear.")

    if numpy and interp == Image.BILINEAR:
        shape = (size1D * size2D * size3D, source.channels)
        if target_size:
            b, g, r = numpy.mgrid[
                0:1:size3D*1j,
                0:1:size2D*1j,
                0:1:size1D*1j
            ].astype(numpy.float32)
            points = numpy.stack((r, g, b), axis=-1).reshape(shape)
            points = _sample_lut_linear_numpy(source, points)
        else:
            points = numpy.asarray(source.table, dtype=numpy.float32)
            points = points.reshape(shape)

        points = _sample_lut_linear_numpy(lut, points)
        table = points.reshape(points.size)

    else:  # Native implementation
        table = []
        index = 0
        for b in range(size3D):
            for g in range(size2D):
                for r in range(size1D):
                    if target_size:
                        point = (r / (size1D-1), g / (size2D-1), b / (size3D-1))
                        point = sample_lut(source, point)
                    else:
                        point = (source.table[index + 0],
                                 source.table[index + 1],
                                 source.table[index + 2])
                        index += 3
                    table.extend(sample_lut(lut, point))

    return cls((size1D, size2D, size3D), table,
               channels=lut.channels, target_mode=lut.mode or source.mode,
               _copy_table=False)


def amplify_lut(source, scale):
    """Amplifies given lookup table compared to identity table the same size.
    For 4-channel lookup tables the fourth channel will be unschanged.

    :param source: Source lookup table, ``ImageFilter.Color3DLUT`` object.
    :param scale: One or three floats which define the amplification strength.
                  1.0 mean no changes, 0.0 transforms to identity table.
    """
    if not isinstance(scale, (tuple, list)):
        scale = (scale, scale, scale)

    if numpy:
        size1D, size2D, size3D = source.size
        sb, sg, sr = numpy.mgrid[
            0:1:size3D*1j,
            0:1:size2D*1j,
            0:1:size1D*1j
        ].astype(numpy.float32).reshape(3, size1D * size2D * size3D)

        points = numpy.array(source.table, dtype=numpy.float32)
        points = points.reshape(size1D * size2D * size3D, source.channels)
        points[:, 0] = sr + (points[:, 0] - sr) * scale[0]
        points[:, 1] = sg + (points[:, 1] - sg) * scale[1]
        points[:, 2] = sb + (points[:, 2] - sb) * scale[2]

        return type(source)(
            source.size, points.reshape(points.size), channels=source.channels,
            target_mode=source.mode, _copy_table=False,
        )

    def transform3(sr, sg, sb, r, g, b):
        return (sr + (r - sr) * scale[0],
                sg + (g - sg) * scale[1],
                sb + (b - sb) * scale[2])

    def transform4(sr, sg, sb, r, g, b, x):
        return (sr + (r - sr) * scale[0],
                sg + (g - sg) * scale[1],
                sb + (b - sb) * scale[2],
                x)

    if source.channels == 3:
        return source.transform(transform3, with_normals=True)
    elif source.channels == 4:
        return source.transform(transform4, with_normals=True)
    else:  # pragma: no cover
        raise ValueError("The source lut should have 3 or 4 channels")

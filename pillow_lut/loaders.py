from itertools import chain

from PIL import Image, ImageFilter, ImageMath


try:
    import numpy
except ImportError:  # pragma: no cover
    numpy = None


def load_cube_file(lines, target_mode=None, cls=ImageFilter.Color3DLUT):
    """Loads 3D lookup table from .cube file format.

    :param lines: Filename or iterable list of strings with file content.
    :param target_mode: Image mode which should be after color transformation.
                        The default is None, which means mode doesn't change.
    :param cls: A class which handles the parsed file.
                Default is ``ImageFilter.Color3DLUT``.
    """
    name, size = None, None
    channels = 3
    file = None

    if isinstance(lines, str):
        file = lines = open(lines, 'rt')

    try:
        iterator = iter(lines)

        for i, line in enumerate(iterator, 1):
            line = line.strip()
            if line.startswith('TITLE'):
                name = line.split('"')[1]
                continue
            if line.startswith('LUT_3D_SIZE'):
                size = [int(x) for x in line.split()[1:]]
                if len(size) == 1:
                    size = size[0]
                continue
            if line.startswith('CHANNELS'):
                channels = int(line.split()[1])
            if line.startswith('LUT_1D_SIZE'):
                raise ValueError("1D LUT cube files aren't supported")

            try:
                float(line.partition(' ')[0])
            except ValueError:
                pass
            else:
                # Data starts
                break

        if size is None:
            raise ValueError('No size found in the file')

        table = []
        for i, line in enumerate(chain([line], iterator), i):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                pixel = [float(x) for x in line.split()]
            except ValueError:
                raise ValueError("Not a number on line {}".format(i))
            if len(pixel) != channels:
                raise ValueError(
                    "Wrong number of colors on line {}".format(i))
            table.extend(pixel)
    finally:
        if file is not None:
            file.close()

    instance = cls(size, table, channels=channels,
                   target_mode=target_mode, _copy_table=False)
    if name is not None:
        instance.name = name
    return instance


def load_hald_image(image, target_mode=None, cls=ImageFilter.Color3DLUT):
    """Loads 3D lookup table from Hald image (normally .png or .tiff files).

    :param image: Pillow RGB image or path to the file.
    :param target_mode: Image mode which should be after color transformation.
                        The default is None, which means mode doesn't change.
    :param cls: A class which handles the parsed file.
                Default is ``ImageFilter.Color3DLUT``.
    """
    if not isinstance(image, Image.Image):
        image = Image.open(image)

    if image.size[0] != image.size[1]:
        raise ValueError("Hald image should be a square")

    channels = len(image.getbands())

    for i in range(2, 9):
        if image.size[0] == i**3:
            size = i**2
            break
    else:
        raise ValueError("Can't detect hald size")

    if numpy:
        table = numpy.array(image).reshape(size**3 * channels)
        table = table.astype(numpy.float32) / 255.0
    else:
        table = []
        for color in zip(*[
            ImageMath.eval("a/255.0", a=im.convert('F')).im
            for im in image.split()
        ]):
            table.extend(color)

    return cls(size, table, target_mode=target_mode, _copy_table=False)

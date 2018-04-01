# Pillow LUT tools

Lookup tables loading, manipulating, and generating for
[Pillow][Pillow] library.

You will need to [install Pillow][install Pillow] or
[Pillow-SIMD][install Pillow-SIMD] to work. None of this packages
is included as a dependency to simplify installation of another.

## Sample code

```python
from PIL import Image
from pillow_lut import load_cube_file

lut = load_cube_file('../resources/insta - Juno.cube')
im = Image.open('../resources/pineapple.jpeg')
im = im.filter(lut)
im.save('../resources/pineapple.juno.jpeg')
```

![Original](./res/pineapple.jpeg) ![Filtered](./res/pineapple.juno.jpeg)



[Pillow]: https://pillow.readthedocs.io/
[install Pillow]: https://pillow.readthedocs.io/en/latest/installation.html#basic-installation
[install Pillow-SIMD]: https://github.com/uploadcare/pillow-simd#installation

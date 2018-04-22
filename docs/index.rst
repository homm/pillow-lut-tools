.. Pillow LUT tools documentation master file, created by
   sphinx-quickstart on Sun Apr  8 14:13:55 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Pillow LUT tools
================

This package contains tools for loading, manipulating, and generating
three-dimensional lookup tables and is designed for Pillow_ library.

Lookup tables are the powerful instrument for image color transformations
and used in many graphics and video editors.


.. toctree::
   :maxdepth: 2

.. autofunction:: pillow_lut.load_cube_file
.. autofunction:: pillow_lut.load_hald_image
.. autofunction:: pillow_lut.identity_table
.. autofunction:: pillow_lut.rgb_color_enhance
.. autofunction:: pillow_lut.sample_lut_linear
.. autofunction:: pillow_lut.sample_lut_cubic
.. autofunction:: pillow_lut.resize_lut
.. autofunction:: pillow_lut.transform_lut
.. autofunction:: pillow_lut.amplify_lut


.. _Pillow: https://pillow.readthedocs.io/
.. _install Pillow: https://pillow.readthedocs.io/en/latest/installation.html#basic-installation
.. _install Pillow-SIMD: https://github.com/uploadcare/pillow-simd#installation

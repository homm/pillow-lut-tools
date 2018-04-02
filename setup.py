#!/usr/bin/env python

from setuptools import setup

setup(
    name='pillow_lut',
    version='1.0',
    description='Lookup tables loading, manipulation and generation '
                'for Pillow library',
    author='Alex Karpinsky',
    author_email='homm86@gmail.com',
    url='https://github.com/homm/pillow-lut-tools',
    classifiers=[
        "Development Status :: 6 - Mature",
        "Topic :: Multimedia :: Graphics",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    packages=['pillow_lut'],
    setup_requires=['nose>=1.3']
)

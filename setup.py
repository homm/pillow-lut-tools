#!/usr/bin/env python

from setuptools import setup


setup(
    name='pillow_lut',
    version='1.1.0',
    description=(
        'Lookup tables loading, manipulation and generation for Pillow library'),
    author='Aleksandr Karpinskii',
    author_email='homm86@gmail.com',
    url='https://github.com/homm/pillow-lut-tools',
    classifiers=[
        "Development Status :: 6 - Mature",
        "Topic :: Multimedia :: Graphics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    packages=['pillow_lut'],
    install_requires=['pillow>=5.2.0']
)

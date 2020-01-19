#!/usr/bin/env python
### cribbed from https://github.com/maet3608/minimal-setup-py/blob/master/setup.py
from setuptools import setup, find_packages

setup(
    name='image_play',
    version='0.1.0',
    url='https://github.com/SocialScienceWithOnlineData/image_play.git',
    author='Seth Frey',
    author_email='sfrey@ucdavis.edu',
    description='Helper functions for performing image recognition on web images in a notebook environment',
    packages=find_packages(),
    license='MIT',
)

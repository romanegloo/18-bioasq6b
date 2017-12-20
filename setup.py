#!/usr/bin/env python3

from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

with open('requirements.txt') as f:
    reqs = f.read()

setup(
    name='BioAsq6B',
    version='0.1.0',
    description='pipeline for question answering in BioAsq6B task',
    long_description=readme,
    license=license,
    python_requires='>=3.5',
    packages=find_packages(),
    install_requires=reqs.strip().split('\n'),
)
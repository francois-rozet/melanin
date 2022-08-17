#!/usr/bin/env python

import setuptools

with open('README.md', 'r') as f:
    readme = f.read()

with open('requirements.txt', 'r') as f:
    required = f.read().splitlines()

setuptools.setup(
    name='melanin',
    version='0.0.3',
    packages=setuptools.find_packages(),
    description='Darken Python code exposed to sunlight',
    keywords='automation formatter pep8 black yapf',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='François Rozet',
    author_email='francois.rozet@outlook.com',
    license='MIT license',
    url='https://github.com/francois-rozet/melanin',
    project_urls={
        'Documentation': 'https://github.com/francois-rozet/melanin',
        'Source': 'https://github.com/francois-rozet/melanin',
        'Tracker': 'https://github.com/francois-rozet/melanin/issues',
    },
    classifiers=[
        'Environment :: Console',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
    ],
    install_requires=required,
    python_requires='>=3.7',
    entry_points={
        'console_scripts': [
            'tan = melanin:tan',
        ],
    },
)

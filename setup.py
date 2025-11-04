"""
OpenDPD Setup Configuration
"""
from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_long_description():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

setup(
    name='opendpd',
    version='2.0.0',
    author='Chang Gao, Yizhuo Wu, Ang Li',
    author_email='chang.gao@tudelft.nl, yizhuo.wu@tudelft.nl, a.li-2@tudelft.nl',
    description='An end-to-end learning framework for modeling power amplifiers and digital pre-distortion',
    long_description=read_long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/OpenDPD',  # Update with actual URL
    packages=find_packages(exclude=['Matlab', 'pics', 'slprj']),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'pandas>=1.3.0',
        'matplotlib>=3.4.0',
        'tqdm>=4.62.0',
        'rich>=10.0.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black>=21.0',
            'flake8>=3.9',
        ],
    },
    include_package_data=True,
    package_data={
        'opendpd': ['datasets/*/spec.json', 'datasets/*/*.csv'],
    },
    entry_points={
        'console_scripts': [
            'opendpd-cli=opendpd.cli:main',
        ],
    },
)


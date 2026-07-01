from setuptools import setup, find_packages

setup(
    name='boundary_detection',
    version='0.1.0',
    description='HII region stable boundary detection with bootstrap uncertainty',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.24,<2',
        'scipy',
        'matplotlib',
        'astropy',
        'pyyaml',
        'scikit-image',
        'astroquery',
        'watroo',
    ],
    python_requires='>=3.10',
    include_package_data=True,
)

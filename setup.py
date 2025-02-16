from setuptools import setup, find_packages

setup(
    name='QuantumInspiredKMeans',
    version='0.1',
    description='A quantum-inspired k-means clustering algorithm',
    author='Your Name',
    author_email='you@example.com',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License'
    ]
)
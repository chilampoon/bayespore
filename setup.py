from setuptools import setup, find_packages

setup(
    name='bayespore',
    version='0.0.1',
    description='Bayesian generative model for nanopore raw signals',
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'Click',
        'ont-remora',
        'pysam',
        'pod5',
        'numpy',
        'scipy>=1.11',
        'scikit-learn',
        'pyro-ppl',
        'torch',
    ],
    entry_points={
        'console_scripts': [
            'bayespore = bayespore.bin.run:bayespore'
        ],

    }
)

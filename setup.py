from setuptools import setup, find_packages

setup(
    name='pyHopperFeatures',
    version='0.1',
    author="francisco-rai",
    author_email="francisco.mendes.pv@renesas.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={
        '': ['feature_Spaces_key.csv']  # Add the CSV file to the package data
    },
    install_requires=[
        'h5py==3.9.0',
        'scipy==1.10.1',
        'numpy==1.24.3'
    ],
    entry_points={
        "console_scripts": [
            "pyHopperFeaturesDiagnostic = pyHopperFeatures.run:main"
        ]
    }
)

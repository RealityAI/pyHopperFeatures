from setuptools import setup, find_packages

setup(
    name='pyHopperFeatures',
    version='0.0.4',
    author="francisco-rai",
    author_email="francisco.mendes.pv@renesas.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    # package_data={'pyHopperFeatures': ['sample_data/**/*']
    #               },
    # package_data={'pyHopperFeatures': ['*.csv', 'sample_data/*.csv']},
    package_data={'pyHopperFeatures': ['matlab_files']},
    include_package_data=True,
    install_requires=[
        'h5py==3.9.0',
        'scipy==1.10.1',
        'numpy==1.24.3'
    ],
    entry_points={
        "console_scripts": [
            "pyHopperFeatures = pyHopperFeatures.run:main"
        ]
    }
)

from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop

class CustomInstall(install):
    def run(self):
        install.run(self)
        self.do_custom_install()

    def do_custom_install(self):
        import shutil
        import os

        # Define the source and destination directories
        src_dir = os.path.join(self.install_libbase, 'pyHopperFeatures', 'sample_data')
        dst_dir = os.path.join(self.install_libbase, 'pyHopperFeatures')

        # Copy the entire "sample_data" folder to the destination
        shutil.copytree(src_dir, os.path.join(dst_dir, 'sample_data'))

class CustomDevelop(develop):
    def run(self):
        develop.run(self)
        self.do_custom_install()

    def do_custom_install(self):
        import shutil
        import os

        # Define the source and destination directories
        src_dir = os.path.join(self.install_libbase, 'pyHopperFeatures', 'sample_data')
        dst_dir = os.path.join(self.install_libbase, 'pyHopperFeatures')

        # Copy the entire "sample_data" folder to the destination
        shutil.copytree(src_dir, os.path.join(dst_dir, 'sample_data'))

setup(
    name='pyHopperFeatures',
    version='0.4',
    author="francisco-rai",
    author_email="francisco.mendes.pv@renesas.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        'h5py==3.9.0',
        'scipy==1.10.1',
        'numpy==1.24.3'
    ],
    entry_points={
        "console_scripts": [
            "pyHopperFeatures = pyHopperFeatures.run:main"
        ]
    },
    cmdclass={'install': CustomInstall, 'develop': CustomDevelop}
)

from setuptools import setup, find_packages

setup(
    name="mjaigym",
    version="0.2.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "gym",
        "joblib",
        "tqdm",
        "dataclasses",
        "tensorboard",
        "torch",
        "pyyaml",
    ],
    include_package_data=True,
    package_data={
        "mjaigym": ["shanten.so"],
    },
)

from setuptools import setup, find_packages

setup(
    name="mjaigym",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "gym",
        "joblib",
        "tqdm",
        "dataclass",
        "tensorboard",
        "pytorch",
    ],
    include_package_data=True,
    package_data={
        "mjaigym": ["shanten.so"],
    },
)

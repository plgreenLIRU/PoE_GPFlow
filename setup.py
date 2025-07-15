from setuptools import setup, find_packages

setup(
    name="PoE_GPFlow",
    version="1.2.0",
    packages=find_packages(),
    install_requires=["gpflow"],
    entry_points={
        "console_scripts": [
        ],
    },
)

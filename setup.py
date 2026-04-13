from setuptools import setup, find_packages

setup(
    name="mini-google",
    version="1.0.0",
    description="Production-grade distributed search engine",
    packages=find_packages(),
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "mini-google=main:cli",
        ],
    },
)

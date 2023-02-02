import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()

setup(
    name="qpsolversbench",
    version="0.0.1",
    description="Evaluation of quadratic solvers using qusolvers",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/microprediction/qpsolversbench",
    author="microprediction",
    author_email="peter.cotton@microprediction.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    packages=["qpsolersbench","qpsolversbench.volumetrics"],
    test_suite='pytest',
    tests_require=['qpsolvers'],
    include_package_data=True,
    install_requires=["wheel"],
    entry_points={
        "console_scripts": [
            "qpsolversbench=qpsolversbench.__main__:main",
        ]
    },
)
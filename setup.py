from setuptools import setup, find_packages

setup(
    name="letsloop",
    version="0.1.0",
    author="Jackson Kunde",
    author_email="jacksonkunde@gmail.com",
    description="A library for transformer architectures with looping mechanisms.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jacksonkunde/letsloop",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        "transformers>=4.46.3",
        "torch==2.5.1"
    ],
    extras_require={
        "examples": ["datasets>=3.1.0"],  # Dependencies for example scripts
    },
    python_requires=">=3.10",
    classifiers=[],
    include_package_data=True,
    zip_safe=False,
)

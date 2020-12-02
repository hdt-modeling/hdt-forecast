import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hdt_utils", # Replace with your own username
    version="0.0.1",
    author="Lifeng Wei, James Sharpnack",
    author_email="jsharpna@gmail.com",
    description="Utilities for backtesting, extracting Covid-19 data, evaluating performance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hdt-modeling/covid-19-forecast",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires = ["covidcast",
                        "requests_cache",
                        "pandas",
                        "scipy",
                        "tqdm",
                        "numpy",
                        "censusgeocode",
                        ],
)


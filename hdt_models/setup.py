import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hdt_models", # Replace with your own username
    version="0.0.1",
    author="James, Sharpnack, Stephen Sheng, Shitong Wei",
    author_email="stevexsheng@gmail.com",
    description="Model",
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
    install_requires=[
        "matplotlib",
        "scipy",
        "numpy",
        "pytest",
    ]
)

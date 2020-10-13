import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="effmap",  # Replace with your own username
    version="0.0.5",
    author="Ivan Okhotnikov",
    author_email="ivan.okhotnikov@outlook.com",
    description="Custom regressor and HST object",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ivanokhotnikov/effmap/tree/master",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    include_package_data=True
)

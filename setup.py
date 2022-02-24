import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="quanta-SL",  # Replace with your own username
    version="0.1",
    author="Varun Sundar",
    author_email="vsundar4@wisc.edu",
    description="Quanta Structured Light",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/varun19299/quanta-SL",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)

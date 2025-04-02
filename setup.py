from setuptools import setup, find_packages

setup(
    name="monkey_genericutils",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "opencv-python",
        "numpy",
        "matplotlib",
        "scipy",
        "pandas",
        "vidgear",
        "progressbar2",
    ],
    author="Andres Mendez",
    author_email="your.email@example.com",
    description="A package for processing and analyzing monkey videos",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/monkey_genericutils",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
) 
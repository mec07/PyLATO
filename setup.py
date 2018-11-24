from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='pylato',
    version='1.0.0',
    author='Marc Coury',
    author_email='marc.coury@gmail.com',
    description="PyLATO: Noncollinear Magnetic Tight Binding code",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mec07/PyLATO",
    packages=find_packages(),
)

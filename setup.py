from setuptools import setup
import pathlib
curr_dir = pathlib.Path(__file__).absolute().parent

with open(curr_dir / "version.txt") as f:
    version = f.read().strip()

setup(
    version=version,
)

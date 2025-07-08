from setuptools import setup, find_packages

def readme():
    with open('README.md') as f:
        return f.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("version.txt") as f:
    version = f.read().strip()

setup(
    name='gorgo',
    version=version,
    description='Gorgo probabilistic programming language',
    keywords = [],
    packages=find_packages(),
    test_suite='nose.collector',
    tests_require=['nose'],
    zip_safe=False,
    install_requires=requirements,
)

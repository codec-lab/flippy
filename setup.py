from setuptools import setup, find_packages

def readme():
    with open('README.md') as f:
        return f.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name='gorgo',
    version='0.0.1',
    description='Gorgo probabilistic programming language',
    keywords = [],
    packages=find_packages(),
    test_suite='nose.collector',
    tests_require=['nose'],
    zip_safe=False,
    install_requires=requirements,
)

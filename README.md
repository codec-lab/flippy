# Gorgo probablistic programming language

🚧 This project is a work in progress 🚧

Gorgo is a lightweight probabilistic programming language for cognitive modeling
that is written entirely in Python. It prioritizes ease of use, universality,
and maintainability. It takes inspiration from the design and implementation
of [WebPPL](https://dippl.org/).

## Getting started

In OSX, you can set up a virtual environment and install gorgo with dependencies:
```
$ python3 -m venv venv
$ source venv/bin/activate
(venv) $ pip install git+https://github.com/markkho/gorgo.git
```

To run the tests (this requires installing pytest):
```
(venv) $ pytest
```

## Examples

### Simple discrete variables


```python
from gorgo import infer, condition, flip

@infer
def model(p):
    x = flip(p)
    y = flip(p)
    condition(x >= y)
    return x + y
```

```python
model(0.5)
```

<table><thead><tr><th></th><th>Element</th><th>Probability</th></tr></thead><tbody><tr><td><b>0</b></td><td>2</td><td>0.333</td></tr><tr><td><b>1</b></td><td>1</td><td>0.333</td></tr><tr><td><b>2</b></td><td>0</td><td>0.333</td></tr></tbody></table>

```python
model(.7)
```

<table><thead><tr><th></th><th>Element</th><th>Probability</th></tr></thead><tbody><tr><td><b>0</b></td><td>2</td><td>0.620</td></tr><tr><td><b>1</b></td><td>1</td><td>0.266</td></tr><tr><td><b>2</b></td><td>0</td><td>0.114</td></tr></tbody></table>

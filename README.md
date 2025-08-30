<p align="center">
    <img src="https://raw.githubusercontent.com/codec-lab/flippy/refs/heads/main/flippy-wide.svg" alt="FlipPy Logo" width="65%"/>
</p>

# FlipPy: Pythonic Probabilistic Programming

FlipPy lets you specify probabilistic programs in Python syntax
while seamlessly interacting with the rest of Python.

Documentation and tutorials can be found [here](https://codec-lab.github.io/flippy/flippy.html).

## Quick start

FlipPy can be installed with `pip`:

```bash
pip install flippy-lang
```

The core functionality of FlipPy does not require any dependendencies,
so the above command will only install FlipPy. To install the dependencies required
for full functionality, use:

```bash
pip install flippy-lang[full]
```

## Example: Sum of Bernoullis

FlipPy lets you specify probablistic programs using standard Python syntax.
Here is a simple example involving the sum of two Bernoulli random variables:

```python
from flippy import infer, flip

@infer
def fn():
    x = flip(0.5)
    y = flip(0.5)
    return x + y

fn() # Distribution({0: 0.25, 1: 0.5, 2: 0.25})
```

## Tests

To run the tests (this requires installing pytest):
```
(venv) $ pytest
```

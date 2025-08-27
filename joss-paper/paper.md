---
title: 'FlipPy: Pythonic Probabilistic Programming'
tags:
  - Python
  - probablistic programming
  - cognitive modeling
  - Bayesian inference
  - reinforcement learning
authors:
  - name: Mark K. Ho
    orcid: 0000-0002-1454-4768
    equal-contrib: true
    affiliation: 1
    corresponding: true
  - name: Carlos G. Correa
    orcid: 0000-0001-9138-7818
    equal-contrib: true
    affiliation: 1
affiliations:
 - name: New York University
   index: 1
date: 14 August 2025
bibliography: paper.bib

---

# Summary

Probabilistic programming languages provide a user-friendly interface for
specifying complex probabilistic models and inference algorithms over those models.
Within psychology and cognitive science, probabilistic programming languages
have been used to formally characterize
human concept learning, social reasoning, intuitive physics, and planning,
among many other higher-level cognitive phenomena [@griffiths2024].
Meanwhile, Python is a widely used, general-purpose
programming language that is commonly taught and increasingly used by students
in psychology, neuroscience, and computer science. While several probabilistic
programming frameworks currently exist in the scientific Python eco-system,
these require beginners to learn new framework-specific syntax for specifying models
and not all of them are universal (i.e., allow specification and inference over
any computable distribution).

# Statement of need

`FlipPy` is a package for specifying probabilistic programs directly in
Python syntax and allows users to write any computable distribution.
Existing Python-based probabilistic programming libraries tend to be optimized
for specific use cases in machine learning and require users to learn specialized
syntax, which can be challenging for beginners (e.g., [@abril2023pymc,@bingham2019pyro]).
The API of `FlipPy` is intentionally beginner-friendly and heavily inspired by that of
`WebPPL` [@dippl], a Javascript-based probabilistic programming language that
is widely used in computational cognitive science.
Like `WebPPL` and the LISP-based `Church` [@goodman2012church],
`FlipPy` lets users specify probabilistic models as programs in a deterministic "host language"
(Python) that is augmented with `sample` and `observe` statements.
A custom interpreter treats these statements as
sampling and conditioning events, which is sufficient for universality [@van2018introduction].
This interpreter is based on a continuation-passing style transform, allowing program execution to be forked when samples are taken, and resumed or halted to facilitate caching [@ritchie2016c3].
Importantly, `FlipPy` itself is entirely Python-based: the codebase is implemented in Python
and it performs the probabilistic execution necessary for inference in Python.
This means that `FlipPy` can seamlessly interoperate with other Python code before,
during, and after a user performs inference, including in Jupyter notebooks [@kluyver2016jupyter].

`FlipPy` has been designed to facilitate rapid prototyping and "hackability"
while also being as accessible as possible for users
who have only basic familiarity with programming in Python
(e.g., behavioral scientists who are new to computational modeling).
Because the specification language is Python itself,
the programs are highly readable and models can be expressed using syntactic constructs
like branching, function calls, for loops,
etc. [@van2007python]. This makes the library especially valuable for teaching
abstract probabilistic concepts in a concrete,
iterative manner by starting with simpler models and adding complexity.
Earlier versions of `FlipPy` have been used
in undergraduate- and graduate-level courses on computational cognitive science.

# Example Usage

The following is a simple program that samples (with `flip`)
and observes (with `condition`) from Bernoulli distributions.

```python
from flippy import infer, condition, flip

@infer
def model(p):
    x = flip(p)
    y = flip(p)
    condition(x >= y)
    return x + y

model(0.5)
```

||Element|Probability|
|---|---|---|
|0|2|0.333|
|1|1|0.333|
|2|0|0.333|


# Research projects using `FlipPy`

As noted, `FlipPy` has so far been primarily used for teaching, but
the authors and their colleagues are using the library in several
ongoing projects related to decision-making and social cognition. For example,
@zhang2025learning used `FlipPy` to model interactions between pragmatic
reasoning and hierarchical Bayesian inference during the interpretation of
generic utterances.

# Acknowledgements

We acknowledge support from Thomas Griffiths during the genesis of this project.

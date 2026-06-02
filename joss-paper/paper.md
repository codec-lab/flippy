---
title: 'FlipPy: Pythonic Probabilistic Programming'
tags:
  - Python
  - probabilistic programming
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
date: 30 May 2026
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
programming frameworks currently exist in the scientific Python ecosystem,
these require beginners to learn new framework-specific syntax for specifying models
and not all of them are universal (i.e., allow specification and inference over
any computable distribution).

# Statement of need
<!-- A section that clearly illustrates the research purpose of the software and places it in the context of related work. This should clearly state what problems the software is designed to solve, who the target audience is, and its relation to other work. -->

`FlipPy` is a package for specifying probabilistic programs directly in
Python syntax and allows users to express any computable distribution.
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

# State of the field
<!-- State of the Field (~200 words): A description of how this software compares to other commonly-used packages in the research area. If related tools exist, provide a clear "build vs. contribute" justification explaining your unique scholarly contribution and why existing alternatives are insufficient. -->

Probabilistic programming languages (PPLs) have been used as modeling tools in cognitive
science for over a decade [@goodman2015concepts]. One approach to PPL design
takes an existing "host language" (e.g., Python) and extends its syntax and interpreter
to support probabilistic constructs, such as `sample`, `observe`, and `infer` statements.
The extended language can be used to specify any computable distribution [@van2018introduction].
Prior libraries in cognitive science that follow this strategy include `Church`
[@goodman2012church], built on LISP, and `WebPPL` [@dippl], built on JavaScript.

Host-language PPLs inherit many of the strengths of
general-purpose languages: they are universal, syntactically familiar,
and interoperable with deterministic code. However, this generality trades off
against inference efficiency, motivating tools that commit to a particular
model-specification API or syntax so that dedicated inference algorithms can be
used. For example, `Stan` [@carpenter2017stan] and `PyMC` [@abril2023pymc]
target hierarchical Bayesian models with Hamiltonian MCMC [@betancourt2017conceptual],
`Pyro` [@bingham2019pyro] and related libraries focus on
variational inference [@blei2017variational], and `Roulette` [@moy2025roulette]
focuses on exact, discrete inference.
A similar approach is also taken by `memo` [@chandra2025domain],
a highly optimized, domain-specific PPL for reasoning about reasoning that is also
used for cognitive modeling.

`FlipPy` adopts the host-language philosophy of `Church` and `WebPPL`
but targets Python. This emphasis on generality means `FlipPy` is currently
less optimized for particular inference algorithms than other languages.
That said, the separation of concerns in `FlipPy`'s architecture
(described below) and the flexible nature of Python makes it possible to
write custom inference backends that are optimized for specific model classes.
Nonetheless, at the user-facing layer, the language is intended to be
beginner-friendly. Indeed, `FlipPy`'s
API is modeled on that of `WebPPL`, which is widely used in tutorials
and courses on computational cognitive science.
`FlipPy` therefore stakes out a distinct position along the expressivity–efficiency
tradeoff while remaining accessible to newcomers.

# Software Design
<!-- Software Design (~200 words): An explanation of the trade-offs you weighed, the design/architecture you chose, and why it matters for your research application. This should demonstrate meaningful design thinking beyond a superficial code structure description. -->

`FlipPy`'s design emphasizes expressivity, usability, and interoperability with Python.
From a beginning user's perspective, a `FlipPy` model is written as a Python
function that calls `sample` and `observe` methods of custom `Distribution` objects
(the "universal PPL" paradigm described in @van2018introduction).
The `infer` function is then used to compile the function into a new function
that returns posterior `Distribution`s over the return values of the original function
(see **Example Usage**). Users can specify what inference algorithm to use via `infer`
(e.g., exact enumeration, Metropolis-Hastings, etc.). These algorithms subclass
the `InferenceAlgorithm` abstract base class and use
a `ProgramState` abstraction that allows inference algorithms
to inspect, modify, and resume a program's execution at each `sample` and `observe`
site. The design also makes it straightforward for
advanced users to write custom `Distribution` and `InferenceAlgorithm` implementations.

Under the hood, the `ProgramState` abstraction uses a custom interpreter
that can store and re-run __program continuations__ that represent
the remainder of a program's computation after each `sample` and `observe` site. This is
accomplished in two steps:
First, the source code of a Python function is statically transformed
at the AST level into continuation-passing style (CPS) Python [@might2010continuation] (see `transforms.py`).
Second, a `CPSInterpreter` object (see `interpreter.py`) generates `ProgramState`s
that can execute the CPS-transformed code using a trampoline-based execution loop [@ganz1999trampolined],
allowing program execution to be forked when samples are taken, and resumed or halted to facilitate caching and other optimizations [@ritchie2016c3].
Inference algorithms do not touch the CPS layer and only interact with `ProgramState`s,
which keeps model specification, inference strategy, and non-deterministic execution
as independent concerns.

The CPS-transformed code is Python and executed in the Python
interpreter, as opposed to compiled and executed using a custom backend, which makes it possible to interoperate seamlessly with
the rest of the Python ecosystem.
For exsample, we can call into arbitrary deterministic Python libraries
(the `keep_deterministic` decorator exempts code from the CPS transform)
as well as run `FlipPy` inference within Jupyter notebooks [@kluyver2016jupyter].

The transformation pipeline, interpreter, and program state interfaces are all
extensively unit tested. Inference algorithm correctness is verified
by comparing against known analytical posteriors as well as
via differential testing (e.g., comparing the output of
exact enumeration against a sampling algorithm).

## Example Usage

The following is a simple program that samples (with `flip`)
and observes (with `condition`).

```python
from flippy import infer
from flippy.distributions import Bernoulli
# from flippy import flip, condition

# FlipPy comes packaged with flip and condition functions but
# we implement them here explicitly for clarity
def flip(p):
    return Bernoulli(p).sample()

def condition(event):
    return Bernoulli(True).observe(event)

# Model is written in Python syntax and inference is done using
# the infer decorator
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


# Research Impact Statement
<!-- Research Impact Statement (~200 words): Evidence of realized impact (publications, external use, integrations) or credible near-term significance (benchmarks, reproducible materials, community-readiness signals). The evidence should be compelling and specific, not aspirational. -->

`FlipPy` has so far been primarily used for teaching,
including undergraduate- and graduate-level computational cognitive science courses at
Stevens Institute of Technology and New York University.
Additionally, the authors and their colleagues are using the library in several
ongoing projects related to decision-making and social cognition. For example,
@zhang2025learning used `FlipPy` to model interactions between pragmatic
reasoning and hierarchical Bayesian inference during the interpretation of
generic utterances. @yang2026spontaneous have also used `FlipPy` for modeling hierarchical
problem solving in complex tasks.

The core features of the library are documented ([documentation link](https://codec-lab.github.io/flippy/flippy.html)),
and we have published a suite of publicly available tutorials online
([tutorial link](https://codec-lab.github.io/flippy-tutorials/)).
As of this writing, this includes an introductory tutorial that introduces the
basic constructs of the library as well as tutorials for
recursive social modeling, language of thought models,
hidden Markov models, Bayesian non-parametrics, intuitive physics,
and sequential decision-making.

# AI usage disclosure
<!-- AI Usage Disclosure: Transparent disclosure of any use of generative AI in the software creation, documentation, or paper authoring. If no AI tools were used, state this explicitly. If AI tools were used, describe how they were used and how the quality and correctness of AI-generated content was verified. -->

Generative AI was not used for developing `FlipPy` through version 0.1.4. This
includes the core functionality (e.g., the CPS transformation pipeline,
CPS interpreter, and program state interface), the built-in inference
algorithms, and related tests. Ongoing development uses generative AI for writing code.
Correctness of all AI-generated code is verified via unit tests of specific components
and comparison of inference outputs against known analytical solutions as well as
against solutions from human-written implementations.
Quality is maintained by strict human review of all committed code.
Initial versions of documentation, tutorials, and the current paper did not
use generative AI. Generative AI is used to review drafts and suggest edits for
these materials, but all changes are reviewed and implemented by humans.

# Acknowledgements

We acknowledge support from Thomas Griffiths during the genesis of this project.

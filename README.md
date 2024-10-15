# Gorgo probablistic programming language

🚧 This project is a work in progress 🚧

Gorgo is a lightweight probabilistic programming language for cognitive modeling
that is written entirely in Python. It prioritizes ease of use, universality, 
and maintainability. It takes inspiration from the design and implementation
of [WebPPL](https://dippl.org/).  

## Getting started

### Using a virtual environment
- Set up a [virtual environment](https://docs.python.org/3/library/venv.html)
- In the virtual environment, run: `pip install git+https://github.com/markkho/gorgo.git`


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




<table><thead><tr><th></th><th>Element</th><th>Probability</th></tr></thead><tbody><tr><td><b>0</b></td><td>0</td><td>0.333</td></tr><tr><td><b>1</b></td><td>1</td><td>0.333</td></tr><tr><td><b>2</b></td><td>2</td><td>0.333</td></tr></tbody></table>




```python
model(.7)
```




<table><thead><tr><th></th><th>Element</th><th>Probability</th></tr></thead><tbody><tr><td><b>0</b></td><td>2</td><td>0.620</td></tr><tr><td><b>1</b></td><td>1</td><td>0.266</td></tr><tr><td><b>2</b></td><td>0</td><td>0.114</td></tr></tbody></table>



### Medical Diagnosis Example


```python
@infer
def model():
    smokes = flip(.2)
    lungDisease = flip(0.001) or (smokes and flip(0.1))
    cold = flip(0.02)
    
    cough = (cold and flip(0.5)) or (lungDisease and flip(0.5)) or flip(0.001)
    fever = (cold and flip(0.3)) or flip(0.01)
    chestPain = (lungDisease and flip(0.2)) or flip(0.01)
    shortnessOfBreath = (lungDisease and flip(0.2)) or flip(0.01)

    condition(cough and chestPain and shortnessOfBreath)

    return {'smokes': smokes}

model()
```




<table><thead><tr><th>smokes</th><th>Probability</th></tr></thead><tbody><tr><td>True</td><td>0.960</td></tr><tr><td>False</td><td>0.040</td></tr></tbody></table>



### Recursive distributions


```python
# note we need to bound the number of states to avoid infinite recursion
@infer(max_states=100)
def geometric(p):
    if flip(p):
        return 1 + geometric(p)
    return 0
res = geometric(.9)
res.plot()
```




    <Axes: ylabel='Probability'>




    
![png](README_files/README_9_1.png)
    



```python
@infer
def truncated_geometric(p, n):
    if n == 0:
        return 0
    if flip(p):
        return 1 + truncated_geometric(p, n - 1)
    return 0
truncated_geometric(.7, 10).plot()
```




    <Axes: ylabel='Probability'>




    
![png](README_files/README_10_1.png)
    


### Continuous Distributions


```python
from gorgo.distributions import Normal

@infer(
    method="LikelihoodWeighting",
    samples=1000,
    seed=1234
)
def normal_model(obs, hyper_mu, hyper_sigma, sigma):
    mu = Normal(hyper_mu, hyper_sigma).sample()
    [Normal(mu, sigma).observe(o) for o in obs]
    return mu

obs = (-.75, -.72, -.57, -.82, -.69, -.77, -.74, -.76, -.8, -.79)
res = normal_model(obs, hyper_mu=0, hyper_sigma=1, sigma=1)
res.plot()
```




    <Axes: ylabel='Probability'>




    
![png](README_files/README_12_1.png)
    



```python
from gorgo.distributions import Uniform

@infer(
    method="MetropolisHastings",
    samples=500,
    burn_in=500,
    thinning=5,
    seed=1234
)
def mixture_model(data):
    p = Uniform(0, 1).sample()
    mu1 = Normal(0, 5).sample()
    mu2 = Normal(0, 5).sample()
    cluster = [flip(p) for _ in data]
    [
        Normal(mu1, 1).observe(data[i]) if cluster[i] else Normal(mu2, 1).observe(data[i])
        for i in range(len(data))
    ]
    return dict(p=p, mu1=mu1, mu2=mu2)
    # return cluster
```


```python
import matplotlib.pyplot as plt

data = (
    -3.39, -0.74, -1.74, -3.84, -1.87,  # cluster 1; mu1 = -2, sigma1 = 1
    0.12, 1.18, -0.32, 1.3, 2.26 # cluster 2; mu2 = 1, sigma2 = 1
)
res = mixture_model(data)

fig, ax = plt.subplots()
res.marginalize(lambda x: x["mu1"]).plot(ax)
res.marginalize(lambda x: x["mu2"]).plot(ax)
```




    <Axes: ylabel='Probability'>




    
![png](README_files/README_14_1.png)
    


To create this README file, run:
```
jupyter nbconvert --execute --to markdown README.ipynb
```

---
author: Chen Li
title: Presuppositions of Physics Simulations
date: 2024-09-01
tags:
  - physics
math: true
---

I haven't done a simulation project, but these are my presuppositions of simulation. Let's see how they would change over time (haha pun intended).

## §1 Simulation in General

We divide space into several grids, but we also have Adaptive Mesh Refinement, see [[2312.05438] _Adaptive mesh refinement in binary black holes simulations_](https://arxiv.org/abs/2312.05438). Also, there are [Adaptive step size](https://en.wikipedia.org/wiki/Adaptive_step_size).

Butterfly Effect: [[2401.13381]_Spontaneous stochasticity amplifies even thermal noise to the largest scales of turbulence in a few eddy turnover times_](https://arxiv.org/abs/2401.13881)

There are 3 kinds of simulation:
1. radiative processes, spectrum.
2. dark matter halos, n-body/test particle, gravity.
3. source population, luminosity distribution, redshift. neutrino in the universe.

## §2 Language

[n-body - Which programs are fastest? (Benchmarks Game)](https://benchmarksgame-team.pages.debian.net/benchmarksgame/performance/nbody.html), like that's even a question.

## §3 Memory Usage

In particle n-body simulation, because of the shape of the tensor `[num_particles, num_properties]` (`num_properties` refers to $x$, $y$, $z$, $v_x$, $v_y$, $v_z$, spin, etc) does not change, the memory usage is almost const.

Also, if we encode time in this tensor, we should put it in the first dimension such that `[time_steps, num_particles, num_properties]`. Because now if we want to take a picture from the video, we just do `x[time_step]`. In terms of semetic, this feels natural. In terms of meaning, this shows that, time is just a set of tensors.

<!-- Can we use `torch.nn.Linear` to do the simulation? -->

## §4 Framework

### §4.1 Run

Only use [functional programming](https://en.wikipedia.org/wiki/Functional_programming), instead of [object-oriented programming](https://en.wikipedia.org/wiki/Object-oriented_programming). Because the latter one is not necessary and the previous one, without state and so on, is faster.

```
def one_step():
    ...

def save():
    ...# save as `.parquet`?

for i in range():
    one_step()# the next step is determined by the previous step, so GPU is not much useful here
    save()
```

### §4.2 Plot

Plot the animation of the simulation, and plot the variation of derived properties such kinetic energy, potential energy and so on:

```
def plot_animation():
    ....

def plot_kinetic_energy():
    ...

def plot_potential_energy():
    ...
```
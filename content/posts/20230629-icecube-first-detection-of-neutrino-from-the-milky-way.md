---
author: "Chen Li"
title: "IceCube: First Detection of Neutrinos from the Milky Way"
date: "2023-06-29"
tags: 
- Physics
math: true
---

There's a GW background and there are neutrinos emitted from the milky way! What a great day!

Summary of links: [their twitter with wonderful pictures](https://twitter.com/uw_icecube/status/1674478436038107136), [news on astrobites](https://astrobites.org/2023/06/29/neutrinos-from-our-backyard-icecube-sees-the-milky-way-in-neutrinos/), [news on IceCube's official website](https://icecube.wisc.edu/news/press-releases/2023/06/our-galaxy-seen-through-a-new-lens-neutrinos-detected-by-icecube/), [news conference on YouTube](https://www.youtube.com/watch?v=35YUzuhadOs), [the paper](https://arxiv.org/abs/2307.04427) and [data release](https://icecube.wisc.edu/data-releases/2023/06/observation-of-high-energy-neutrinos-from-the-galactic-plane/). And later this paper about extended sources is published: [[2307.07576] _Search for Extended Sources of Neutrino Emission in the Galactic Plane with IceCube_](https://arxiv.org/abs/2307.07576).

And here's my notes for the first detection paper.

## §1 Two Types of Events

|track|cascade|
|-|-|
|$\mu$ from interaction of cosmic-rays and atmosphere; $\mu$ from interaction of $\nu_{\mu}$ and nuclei|interaction of $\nu_e$ and $\nu_{\tau}$ and nuclei; scattering interactions of all $3$ neutrino flavor|
|clear path|more spherical in shape|
|easier to trace back to source|harder to trace back to source|
|harder to be distinguished from muons produced by cosmic rays in the atmosphere|easier to be distinguished from atmospheric background|
||energy is contained within the instrumented volume, which provides a more complete measure of the neutrino energy|

## §2 Difficulties

### §2.1 Order of Magnitude

|number of events|the ratio of astrophysical neutrinos and background (atmospheric muons/neutrinos)|dataset|
|-|-|-|
|$2.7 \space \text{kHz}$|$10^{-8}$|$59592$ events from May 2011 to May 2021|

The result is $4.48 \sigma$ level of significance on average, for different template see Table 1 of the paper.

### §2.2 Location, Location, Location

Muons from the Southern Hemisphere (above IceCube can penetrate several kms deep into the ice, while the muons from the Northern Hemisphere (below IceCube) are absorbed during passage through Earth. Thus IceCube is the most sensitive to astrophysical sources in the __Northern sky__. However, the Galactic Center, as well as the bulk of the neutrino emission expected from the Galactic plane, is located in the Southern sky. This problem is solved by choosing cascade events instead of track events.

## §3 Method

### §3.1 Deep Learning

1. CNN: choose the cascade event. __It's reasonable to use convolutional core to distinguish two different types of event__. Which is generally use to detect cats (or a certain object) in a dataset of a lot of different pictures.

2. NN with symmetries (that smoothly approximates a Monte Carlo simulation): refine event properties.

### §3.2 Template

1. _diffuse_ emission from the entire galaxy.
- Interactions between cosmic rays (high-energy protons and heavier nuclei) from the Milky Way and galactic gas/dust produce $\pi$. An accelerated hadron will interact with other hadron or $\gamma$-ray:
        $$N+N',\gamma \rightarrow X+ \begin{cases} \pi^+ \rightarrow \mu^+ \nu_{\mu} \rightarrow e^+ \nu_{\mu} \nu_{\mu} \nu_e \left( + \text{c.c.} \right) \\\ \pi^0 \rightarrow \gamma \gamma \end{cases} \tag{1}$$
        We expect both types to be produced together. 
- Template: a background-only model for the expected flux of neutrinos over the entire Galactic plane, based on $\gamma$-ray observation. $3$ template in total:

    One made from extrapolating Fermi-LAT $\gamma$-ray observations of the Milky Way (called $\pi^0$. power law, $E^{-2.7}$). More evenly distributed along the Galactic plane.

    Two alternative maps identified as KRA-gamma ($\text{KRA}^5_{\gamma}$, $\text{KRA}_{\gamma}^{50}$. Harder neutrino spectrum, roughly a $E^{-2.5}$ power law, include a spectral cutoff at the highest energies, $5$ and $50 \space \text{PeV}$ respectively). More concentrated from the Galactic Center.

2. _Individual_ sources in our galaxy.

    Stacking: stack the signals from all the possible sources, such as supernova remnants (SNR), pulsar wind nebula (PWN), unidentified (UNID) Galactic source.

## §4 Open Questions

1. Why it's diffuse?
2. Which model is the best?
3. Do individual sources exist?
4. How to report cascade events live?

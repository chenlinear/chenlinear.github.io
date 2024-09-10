---
author: "Chen Li"
title: "Machine Learning Notes: einops"
date: "2023-12-18"
tags: 
- programming
math: true
---

Frankly speaking, when I saw Einstein notation in Classical Mechanics, I'm not used to it, especially when it's not explicitly said in the context that Einstein notation is used here. I just feel like we can save the trouble by writing more $\sum$.

Anyway, [`einops`](https://github.com/arogozhnikov/einops/) do the following things right:
1. Notation: `output_tensor = rearrange(input_tensor, 't b c -> b c t')`
2. API: provide package-specific APIs, e.g. `torch`, `tensorflow`, `jax`.

There are more and more similar packages:
- [`einop`](https://github.com/cgarciae/einop)
- [`einx`](https://github.com/fferflo/einx)
- [`einshape`](https://github.com/google-deepmind/einshape/)
---
author: "Chen Li"
title: "New Paper on arXiv & New Repository: ISeeCube"
date: "2023-08-28"
tags: 
- physics
math: true
---

The arXiv link is [2308.13285v1](https://arxiv.org/abs/2308.13285v1) and the Repository is [ISeeCube](https://github.com/ChenLi2049/ISeeCube). Here's what I learned and what to do next.

## What I Learned

- Test the code by `print()` everything.
- Input and output are the most important thing. Test these while reading the code.
- Read official package documentation for examples.
- ChatGPT is good at analyzing error report and giving examples, but not so much at writing code, especially when there's already a lot of code.

## To Do

- Train the model with VMF Loss. So that the distribution of azimuthal and zenithal error is in better shape.
- Try different kinds of Embedding. In NLP, a random Embedding `nn.Embedding` will get nice results too. Because the structure is big enough.
- Learn the relationship between Graph and [Sparse Matrix](https://en.wikipedia.org/wiki/Sparse_matrix), see [_Matrices and graphs_ - by Tivadar Danka - The Palindrome](https://thepalindrome.org/p/matrices-and-graphs). Learn GNN and [`GraphNeT`](https://github.com/graphnet-team/graphnet).
- Learn [why Transformers are CNNs](https://arxiv.org/abs/1911.03584), [GNNs](https://towardsdatascience.com/transformers-are-graph-neural-networks-bca9f75412aa), [RNNs](https://arxiv.org/abs/2006.16236).
- Clean the dataset. Why there are events with pulses nearly $110000$?
- model $\rightarrow$ simulated data $\rightarrow$ reconstruction, and then, real data $\rightarrow$ reconstruction. This workflow probably can be improved by using self-supervised learning (see [[2304.12210] _A Cookbook of Self-Supervised Learning_](https://arxiv.org/abs/2304.12210)) directly on real data and then fine-tuning it on simulated data with labels. So that the model is closer to real data than simulated data. This is just an idea, which probably can be tested on the public dataset.
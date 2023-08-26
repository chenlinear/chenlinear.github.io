---
author: "Chen Li"
title: "Gravity Spy 2.0"
date: "2023-08-10"
tags: 
- Physics
---

You can classify some patterns on [Gravity Spy](https://www.zooniverse.org/projects/zooniverse/gravity-spy) and it's fun. Their GitHub repo is [gravityspy-plus](https://github.com/haorenzhi/gravityspy-plus/tree/main), and their Wiki is [Gravity Spy 2.0 Wiki](https://gswiki.ischool.syr.edu/). The Kaggle dataset is [Gravity Spy (Gravitational waves)](https://www.kaggle.com/datasets/tentotheminus9/gravity-spy-gravitational-waves). Not to be confused with [G2Net](https://www.kaggle.com/competitions/g2net-gravitational-wave-detection), the format of the data is quite different.

I took a look of those patterns and I got the idea that we can use Transfer Learning from ordinary Machine Learning models for Computer Vision, because the format is RGB image and it's a classic classification job. Then I look it up on arXiv, there are some papers on this subject. The Fig. 10 of [[2303.13917] _Convolutional Neural Networks for the classification of glitches in gravitational-wave data streams_](https://arxiv.org/abs/2303.13917) is crazy, I have never seen so many "1"s in the diagonal line of a Confusion Matrix.
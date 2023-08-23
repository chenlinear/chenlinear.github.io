---
author: "Chen Li"
title: "Contamination of LLM"
date: "2023-06-13"
tags: 
- Programming
math: true
---

This is the dark side of _the Force_.

In [one of my previous posts](https://chenli2049.github.io/posts/20230321-robin/), I talked about Ted Chiang's idea on LLMs. At that time his idea only seems plausible, but now that more papers are published, I want to talk about how LLMs are contaminating the source material.

1. Training LLM with LLM-produced material will produce terrible results.[^1] This is $\text{Ted Chiang's idea}^n$.

2. And the contamination can be divided into two parts in general:
    
    1. Unintentional: 
        1. Code. This is what I'm most worried about.
        2. Will modify an article with ChatGPT make it worse?
    
    2. Intentional: 
        1. Automated YouTube, TikTok videos, with the help of ChatGPT, music-generator, mid-journey, ppt-generator, etc.
        2. Automated Reddit or other discussion board robots, including posts and replies.
        3. Automated Twitter robots, though I would say blue check is still definitely not a good idea.

We are taking some measures, but they are probably not enough:

1. The co-founder of Wikipedia, Jimmy Wales, recently talked about regulation for the use of ChatGPT in Wikipedia in [this podcast episode with Lex Fridman](https://www.youtube.com/watch?v=diJp4zoQPqo). I really like the part where he said that you, the person, should always be the last barrier. 

	Yes, for Wikipedia, you should do the fact check, and for code, you should run it on your computer. (The latter one is what Stack Overflow has been debating about I think.)

2. Try not to use ChatGPT to write articles. I personally would not use ChatGPT to write my posts. Modifying academic papers with ChatGPT is ok I guess, but if you write academic papers by ChatGPT, [there will be consequences](https://english.elpais.com/science-tech/2023-04-02/one-of-the-worlds-most-cited-scientists-rafael-luque-suspended-without-pay-for-13-years.html).

[^1]: See [_The Curse of Recursion: Training on Generated Data Makes Models Forget_](https://arxiv.org/abs/2305.17493).
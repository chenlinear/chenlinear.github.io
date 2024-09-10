---
author: "Chen Li"
title: "arXiv in HTML"
date: "2023-12-21"
tags: 
- physics
---

[arXiv now offers papers in HTML format](https://blog.arxiv.org/2023/12/21/accessibility-update-arxiv-now-offers-papers-in-html-format/). So basically [ar5iv](https://chenli2049.github.io/posts/20230318-ar5iv/) is now native and applied to the latest papers.

## HTML: Better & Worse

Here are several reasons why HTML is better:

- You can just save the link, instead of downloading pdf.

- In Markdown, you can directly link the pictures in a paper. For example:
    
    ```Markdown
    ![](https://browse.arxiv.org/html/2308.13285v4/x1.png)
    ```
    
    ![](https://browse.arxiv.org/html/2308.13285v4/x1.png)

- You can use [`immersive-translate`](https://chenli2049.github.io/posts/20230602-immersive-translate/).

Here are several reasons why HTML is worse:

- Dependency errors with $\LaTeX$. But you can avoid it by not using rare $\LaTeX$ packages.

- PDFs give you sense of feeling of where the certain stuff is. For example, "the concept of A is in the lower left of page 3". But you can avoid it by taking notes with Markdown.

## Mission Accomplished

[ar5iv](https://ar5iv.labs.arxiv.org/) states that

>- __Goal__: incremental improvement until worthy of native arXiv adoption.

I'm really glad this goal is kind of accomplished.
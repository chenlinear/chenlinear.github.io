---
author: "Chen Li"
title: "New Repository: Slidy"
date: "2023-04-23"
tags: 
- CS
---

I just created a new repository for slide, which is called [Slidy](https://github.com/ChenLi2049/Slidy). Well, technically I didn't create it, I just moved stuff around.

I use it for several reasons:

1. Microsoft PPT is just painful to use. And [Marp](https://marp.app/) is too much for me to handle.
2. Even though I use [Obsidian](https://obsidian.md/) most often, I have to say that its slide function is not so convenient, I want to change to other windows when giving a presentation.
3. I want it to be offline, because I don't have access to the internet all the time. And I solved it by adding the KaTeX package in it and adding few commands in the Slidy.html file:
```html
    <link rel="stylesheet" href="./layouts/katex/katex.min.css">
    <script defer src="./layouts/katex/katex.min.js"></script>
    <script defer src="./layouts/katex/contrib/auto-render.min.js" onload="renderMathInElement(document.body);"></script>
    <script>
        document.addEventListener("DOMContentLoaded", function() {
            renderMathInElement(document.body, {
                delimiters: [
                    {left: "$$", right: "$$", display: true},
                    {left: "$", right: "$", display: false}
                ]
            });
        });
    </script>
```

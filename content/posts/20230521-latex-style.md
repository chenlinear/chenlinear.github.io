---
author: "Chen Li"
title: "LaTeX Style"
date: "2023-05-21"
tags: 
- Programming
math: true
---

If you don't know what it is, I recommend you should just play with it and don't care about the style. 

But I'd like to say [TinyTeX](https://github.com/rstudio/tinytex) (~500 MB) or [Overleaf](https://www.overleaf.com/) (0 MB) are better choices than [TeX Live](https://www.tug.org/texlive/) (~5 GB). I normally write in Markdown first, and then use Overleaf. If you're using JupyterLab, [jupyterlab-latex](https://github.com/jupyterlab/jupyterlab-latex) is basically the local version of Overleaf.

## §1 General

- About the name: $\LaTeX$ (`$\LaTeX$`) is great, LaTeX or latex is acceptable, lAtEx is brutal.

- You should never start from scratch. Finding a template and modifying the template is so much easier. Starting from scratch is [mission impossible](https://www.imdb.com/title/tt0117060/).

- The folder structure should look like this so that it's more manageable:

    ```
    ├── data
    │   └── i-tried.cls
    ├── figures
    │   ├── figure1-amazing-plot.png
    │   ├── figure2-accurate-figure.svg
    │   └── figure3-beautiful-plot.pdf
    ├── main.bbl
    ├── main.tex
    └── pages
        ├── abstract.tex
        ├── appendix.tex
        ├── chapter1.tex
        ├── chapter2.tex
        ├── chapter3.tex
        └── chapter4.tex
    ```

    This might be different for arXiv submission, check out this guide: [Submit TeX/LaTeX - arXiv info](https://info.arxiv.org/help/submit_tex.html), this template: [arxiv-style](https://github.com/kourgeorge/arxiv-style) and two python tools: [arxiv-converter](https://github.com/sdatkinson/arxiv-converter), [flatex](https://github.com/johnjosephhorton/flatex). Or just download the source of a paper you like and modify it.

## §2 This and That

These are strange rules, and keep them.

### §2.1 Equations

I normally use [Supported Functions · KaTeX](https://katex.org/docs/supported.html) or [LaTeX数学公式语法](https://www.wolai.com/wolai/egjDbHiAfGfJmwR972fcEW) to find how to write a certain symbol or equation.

#### §2.1.1 When a Sentence End with an Equation

Use `\,` for the end of an equation, when followed by ".".

For example: The Energy is given by $$E_0=mc^2 \space . \tag{1}$$

```LaTeX
The Energy is given by \[E_0=mc^2 \, . \tag{1}\]
```

(In the source code of this article, I have to use `\space` rather than `\,`. Because the latter one would compile as "$\,$".)

#### §2.1.2 Number and Series Number

Always use `\( \)` for number. For example: There are $24$ elements in total.

```LaTeX
There are \(24\) elements in total.
```

And for series number, for example: This is the $n^{\textrm{th}}$ element.

```LaTeX
This is the \(n^{\textrm{th}}\) element.
```

#### §2.1.3 Unit

1. Use `\,` between number and unit. 
2. Use `\mathrm{}` for the unit.

For example: $3 \times 10^8 \space \mathrm{m/s}$

```LaTeX
\(3 \times 10^8 \, \mathrm{m/s}\)
```

#### §2.1.4 Integral

1. `\mathrm{d} x` for $\mathrm{d} x$.
2. `\,` before $\mathrm{d}x$.
3. `\!` between $\int$ and what's to be intergraded.

For example: $\int_0^{\pi} f(x) \space \mathrm{d} x = 1$

```LaTex
\(\int_0^{\pi} \! f(x) \, \mathrm{d} x = 1\)
```

#### §2.1.5  Transpose

For example: $\mathbf{A}^\mathsf{T}$

```LaTeX
\(\mathbf{A}^\mathsf{T}\)
```

#### §2.1.6  bracket

For example: $\lparen \rparen$, $\lbrack \rbrack$, $\lbrace \rbrace$

```LaTeX
\(\lparen \rparen\)
\(\lbrack \rbrack\)
\(\lbrace \rbrace\)
```

So that the bracket would get a proper length.

And for $\bra{\phi}$, $\ket{\psi}$, $\braket{\phi\vert\psi}$ in Quantum Physics:

```LaTeX
\(\bra{\phi}\)
\(\ket{\psi}\)
\(\braket{\phi\vert\psi}\)
```


### §2.2 Text

#### §2.2.1 "." "!" "?"

1. "." would be considered as the end of a sentence, but sometimes it's not right, for example: "e.g." and "i.e.". That's when we use `\ ` :

	```LaTeX
    Neutron is composed of quarks, i.e.\ it's not fundamental.
    ```

2. After a capital Letter, the "." would be considered in the middle of a name, for example: "F. Scott Fitzgerald". But sometimes we don't want that, for example: "Then I asked ChatGPT. It's of no use either." That's when we use `\@. ` to tell $\LaTeX$ it is the end of a sentence:

    ```LaTeX
    Then I asked ChatGPT\@. It's of no use either.
    ```

It is the same for "!" and "?".

#### §2.2.2 Hyphens, En-dashes and Em-dashes

Hyphens, for example: 5-year-old

```LaTeX
5-year-old
```

En-dashes, for example: See pages 200-201.

```LaTeX
See pages 200--201.
```

Em-dashes, for example: Most newspapers — and all that follow AP style — insert a space before and after the em-dash.

```LaTeX
Most newspapers --- and all that follow AP style --- insert a space before and after the em-dash.
```

#### §2.2.3 Quotation

""
```LaTeX
`` ''
```

I understand, but this is just weird.

#### §2.2.4 People's name

Use `~` instead of space to avoid a name being broken into two parts to the next line:

```LaTeX
F.~Scott~Fitzgerald is known as a writer.
```

It is the same if you don't want to break a phrase. e.g. rank $1$

```LaTeX
rank~\(1\)
```

### §2.3 Table

[LaTeX table generator](https://www.tablesgenerator.com/) can generate tables.

### §2.4 Code

To quote a package, for example, $\texttt{GraphNeT}$:

```LaTeX
\texttt{NumPy}
```
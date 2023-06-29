---
author: "Chen Li"
title: "Group Theory Calculations"
date: "2023-06-15"
tags: 
- Physics
math: true
---

In this article I will summarize some simple examples as a quick introduction to Group Theory.

## §1 Finite Group

If you wanna know about the full finite group, see [Classification of finite simple groups](https://en.wikipedia.org/wiki/Classification_of_finite_simple_groups). The __FULL__ list is one of the greatest human achievements.

I think the best way to learn finite group is to start from various examples.

Here's some general rules:

- Some rules to calculate permutations: 
    - $(12)=(21)$, $(123)=(231)=(312)$
    - $(12)(23)=(123)$
    - $(12)^{-1}=(21)=(12)$, $(123)^{-1}=(321)$, $((12)(34))^{-1}=(43)(21)$. The structure of the permutation is the same, the numbers in the permutation goes backwards.
    - $(12)(12)=e$
    - To calculate $((12)(34))(123)((12)(34))^{-1}$, first exchange $1$ and $2$ for $(123)$, than exchange $3$ and $4$ for $(123)$. $(123)\xrightarrow{\text{exchange 1 and 2}}(213)\xrightarrow{\text{exchange 3 and 4}}(214)$
- For a $S_n$ group, permutations with the same structure form a conjugate class.
- The number of elements in a conjugate class is a factor of the order of the group.
- The order of the subgroup is a factor of the order of the group.
- An invariant subgroup must consist of several complete conjugate classes.

### §1.1 C2

In the notation of permutation, $$C_2 = \lbrace e, (12) \rbrace \tag{1.1.1}$$It has $2$ conjugate classes: $$\begin{aligned} \zeta_1&=\lbrace e \rbrace \\\ \zeta_2&=\lbrace (12) \rbrace \end{aligned} \tag{1.1.2}$$thus only has $2$ trivial invariant subgroups: $$\begin{aligned} H_1 &= \lbrace e\rbrace \\\ H_2&=\lbrace e, (12) \rbrace (C_2 \space \text{itself}) \end{aligned} \tag{1.1.3}$$thus has $2$ quotient groups, which are trivial: $$\begin{aligned} C_2 / H_1 &= \lbrace H_1, (12)H_1 \rbrace \cong C_2 \\\ C_2 / H_2 &= \lbrace H_2\rbrace \end{aligned}  \tag{1.1.4}$$

In the character table, the first row is always all $1$, and the first column is always the dimension $n_{\mu}$ of the irreducible representation (irr. rep.) of the group, which satisfies $$n_G = \sum n_{\mu}^2 \tag{1.1.5}$$where $n_G$ is the order of original group, in this case $C_2$. So the character table is:

|Conjugate Class:|$\zeta_1$|$\zeta_2$|
|-|-:|-:|
|1|$1$|$1$|
|2|$1$|$x$|

According to orthonormality, the $1^{\text{st}}$ row and the $2^{\text{nd}}$ row satisfy (the coefficient ahead is $\sqrt{\frac{\text{the number of elements in the conjugate class}}{n_G}}$ that each row have the same value):$$\begin{pmatrix} \sqrt{\frac{1}{2}}\cdot1 & \sqrt{\frac{1}{2}}\cdot1 \end{pmatrix} \begin{pmatrix} \sqrt{\frac{1}{2}}\cdot1 \\\ \sqrt{\frac{1}{2}}\cdot x \end{pmatrix} = 0 \tag{1.1.6}$$so $x=-1$, thus complete the character table:

|Conjugate Class:|$\zeta_1$|$\zeta_2$|
|-|-:|-:|
|1|$1$|$1$|
|2|$1$|$-1$|

### §1.2 C3

$$C_3 = \lbrace e, (123), (321) \rbrace \tag{1.2.1}$$It has $3$ conjugate classes: $$\begin{aligned} \zeta_1&=\lbrace e \rbrace \\\ \zeta_2&=\lbrace (123) \rbrace \\\ \zeta_3&=\lbrace (321) \rbrace \end{aligned} \tag{1.2.2}$$thus only has $2$ trivial invariant subgroups: $$\begin{aligned} H_1 &= \lbrace e\rbrace \\\ H_2&=\lbrace e, (123), (321) \rbrace (C_3 \space \text{itself}) \end{aligned} \tag{1.2.3}$$thus has $2$ quotient groups, which are trivial: $$\begin{aligned} C_3 / H_1 &= \lbrace H_1, (123)H_1, (321)H_1 \rbrace \cong C_3 \\\ C_3 / H_2 &= \lbrace H_2\rbrace \end{aligned}  \tag{1.2.4}$$

According to Eq. (1.1.5), the character table is:

|Conjugate Class:|$\zeta_1$|$\zeta_2$|$\zeta_3$|
|-|-:|-:|-:|
|1|$1$|$1$|$1$|
|2|$1$|$x$|$y$|
|3|$1$|$z$|$w$|

According to orthonormality, the character table is:

|Conjugate Class:|$\zeta_1$|$\zeta_2$|$\zeta_3$|
|-|-:|-:|-:|
|1|$1$|$1$|$1$|
|2|$1$|$e^{i \frac{2 \pi}{3}}$|$e^{i \frac{4 \pi}{3}}$|
|3|$1$|$e^{i \frac{4 \pi}{3}}$|$e^{i \frac{2 \pi}{3}}$|

### §1.3 S3

$$S_3 = \lbrace e, (12), (13), (23), (123), (321) \rbrace \tag{1.3.1}$$It has $3$ conjugate classes: $$\begin{aligned} \zeta_1&=\lbrace e \rbrace \\\ \zeta_2&=\lbrace (12), (13), (23) \rbrace \\\ \zeta_3&=\lbrace (123), (321) \rbrace \end{aligned} \tag{1.3.2}$$thus only has $1$ nontrivial invariant subgroup, composed of $\zeta_1$ and $\zeta_3$: $$H_1 = \lbrace e, (123), (321) \rbrace \tag{1.3.3}$$thus has $1$ quotient group: $$S_3 / H_1 = \lbrace H_1, (12)H_1 \rbrace \cong C_2 \tag{1.3.4}$$

According to Eq. (1.1.5), the character table is:

|Conjugate Class:|$\zeta_1$|$\zeta_2$|$\zeta_3$|
|-|-:|-:|-:|
|1|$1$|$1$|$1$|
|2|$1$|$x$|$y$|
|3|$2$|$z$|$w$|

Because $H_1$ is composed of $\zeta_1$ and $\zeta_3$, and $S_3 / H_1 \cong C_2$, the column of $\zeta_1$ and $\zeta_3$ is the same as the column of $\zeta_1$ in the character table of $C_2$, when having the same dimension of irr. rep.; the column of $\zeta_2$ is the same as the column of $\zeta_2$ in the character table of $C_2$, when having the same dimension of irr. rep. Thus the character table is :

|Conjugate Class:|$\zeta_1$|$\zeta_2$|$\zeta_3$|
|-|-:|-:|-:|
|1|$1$|$1$|$1$|
|2|$1$|$-1$|$1$|
|3|$2$|$z$|$w$|

According to orthonormality, the character table is:

|Conjugate Class:|$\zeta_1$|$\zeta_2$|$\zeta_3$|
|-|-:|-:|-:|
|1|$1$|$1$|$1$|
|2|$1$|$-1$|$1$|
|3|$2$|$0$|$-1$|

### §1.4 S4

$$\begin{aligned}S_4 = \lbrace &e, \\\ &(12), (13), (14), (23), (24), (34), \\\ &(12)(34), (13)(24), (14)(23), \\\ &(123), (124), (132), (134), (142), (143), (234), (243), \\\ &(1234), (1243), (1324), (1342), (1423), (1432) \rbrace \end{aligned}\tag{1.4.1}$$It has $5$ conjugate classes: $$\begin{aligned} \zeta_1&=\lbrace e \rbrace \\\ \zeta_2&=\lbrace (12), (13), (14), (23), (24), (34) \rbrace \\\ \zeta_3&=\lbrace (12)(34), (13)(24), (14)(23) \rbrace \\\ \zeta_4&=\lbrace (123), (124), (132), (134), (142), (143), (234), (243) \rbrace \\\ \zeta_5&=\lbrace (1234), (1243), (1324), (1342), (1423), (1432) \rbrace
\end{aligned} \tag{1.4.2}$$thus has $2$ nontrivial invariant subgroups: $$\begin{aligned} H_1 = \lbrace &e, (12)(34), (13)(24), (14)(23) \rbrace \cong V_4\\\ H_2=\lbrace &e, \\\ &(123), (321), (124), (421), (234), (432), \\\ &(12)(34), (13)(24), (14)(23) \rbrace \cong A_4 \end{aligned} \tag{1.4.3}$$thus has $2$ quotient groups: $$\begin{aligned} S_4 / V_4 &= \lbrace V_4, (12)V_4, (13)V_4, (23)V_4, (123)V_4, (321)V_4 \rbrace \cong S_3 \\\ S_4 / A_4 &\cong C_2 \end{aligned}  \tag{1.4.4}$$

According to Eq. (1.1.5) and quotient group, the character table is:

|Conjugate Class:|$\zeta_1$|$\zeta_2$|$\zeta_3$|$\zeta_4$|$\zeta_5$|
|-|-:|-:|-:|-:|-:|
|1|$1$|$1$|$1$|$1$|$1$|
|2|$1$|$-1$|$1$|$1$|$x$|
|3|$2$|$0$|$2$|$-1$|$y$|
|4|$3$|||||
|5|$3$|||||

According to orthonormality, the character table is:

|Conjugate Class:|$\zeta_1$|$\zeta_2$|$\zeta_3$|$\zeta_4$|$\zeta_5$|
|-|-:|-:|-:|-:|-:|
|1|$1$|$1$|$1$|$1$|$1$|
|2|$1$|$-1$|$1$|$1$|$-1$|
|3|$2$|$0$|$2$|$-1$|$0$|
|4|$3$|||||
|5|$3$|||||

According to orthonormality, the character table is:

|Conjugate Class:|$\zeta_1$|$\zeta_2$|$\zeta_3$|$\zeta_4$|$\zeta_5$|
|-|-:|-:|-:|-:|-:|
|1|$1$|$1$|$1$|$1$|$1$|
|2|$1$|$-1$|$1$|$1$|$-1$|
|3|$2$|$0$|$2$|$-1$|$0$|
|4|$3$|$1$|$-1$|$0$|$-1$|
|5|$3$|$-1$|$-1$|$0$|$1$|

### §1.5 D2

$$D_2 = \lbrace e, (13), (24), (13)(24) \rbrace \tag{1.5.1}$$It has $4$ conjugate classes: $$\begin{aligned} \zeta_1&=\lbrace e \rbrace \\\ \zeta_2&=\lbrace (13) \rbrace \\\ \zeta_3&=\lbrace (24) \rbrace \\\ \zeta_4&=\lbrace (13)(24) \rbrace\end{aligned} \tag{1.5.2}$$thus has $3$ nontrivial invariant subgroups: $$\begin{aligned} H_1 &= \lbrace e, (13)\rbrace \\\ H_2&=\lbrace e, (24) \rbrace \\\ H_3&=\lbrace e, (13)(24) \rbrace \end{aligned} \tag{1.5.3}$$thus has $3$ quotient groups: $$\begin{aligned} D_2 / H_1 &= \lbrace H_1, (13)H_1 \rbrace \cong C_2 \\\ D_2 / H_2 &= \lbrace H_2, (24)H_2 \rbrace \cong C_2 \\\ D_2 / H_3 &= \lbrace H_3, (13)(24)H_3 \rbrace \cong C_2\end{aligned}  \tag{1.5.4}$$

According to Eq. (1.1.5) and quotient groups, the character table is:

|Conjugate Class:|$\zeta_1$|$\zeta_2$|$\zeta_3$|$\zeta_4$|
|-|-:|-:|-:|-:|
|1|$1$|$1$|$1$|$1$|
|2|$1$|$1$|$-1$|$-1$|
|3|$1$|$-1$|$1$|$-1$|
|4|$1$|$-1$|$-1$|$1$|

### §1.6 D3

$D_3 \cong S_3$, see §1.3.

### §1.7 D4

$$\begin{aligned}D_4 = \lbrace &e, \\\ &(13), (24), \\\ &(12)(34), (14)(23), \\\ &(1234), (13)(24), (4321) \rbrace \end{aligned}\tag{1.7.1}$$where $(13)(24)=(1234)^2$, $(4321)=(1234)^3$. It has $5$ conjugate classes: $$\begin{aligned} \zeta_1&=\lbrace e \rbrace \\\ \zeta_2&=\lbrace (13)(24) \rbrace \\\ \zeta_3&=\lbrace (13), (24) \rbrace \\\ \zeta_4&=\lbrace (12)(34), (14)(23) \rbrace \\\ \zeta_5&=\lbrace (1234), (4321) \rbrace \end{aligned} \tag{1.7.2}$$thus has $4$ nontrivial invariant subgroups: $$\begin{aligned} H_1 &= \lbrace e, (13)(24)\rbrace \\\ H_2&=\lbrace e, (13), (24), (13)(24) \rbrace \\\ H_3&=\lbrace e, (12)(34), (14)(23), (13)(24) \rbrace \\\ H_4&=\lbrace e, (1234), (4321), (13)(24) \rbrace \end{aligned} \tag{1.7.3}$$thus has $4$ quotient groups: $$\begin{aligned} D_4 / H_1 &= \lbrace H_1, (13)H_1, (12)(34)H_1, (1234)H_1 \rbrace \cong D_2 \\\ D_4 / H_2 &= \lbrace H_2, (12)(34)H_2\rbrace \cong C_2 \\\ D_4 / H_3 &= \lbrace H_3, (13)H_3\rbrace \cong C_2 \\\ D_4 / H_4 &= \lbrace H_4, (13)H_4\rbrace \cong C_2 \end{aligned}  \tag{1.7.4}$$

According to Eq. (1.1.5) and quotient groups, the character table is:

|Conjugate Class:|$\zeta_1$|$\zeta_2$|$\zeta_3$|$\zeta_4$|$\zeta_5$|
|-|-:|-:|-:|-:|-:|
|1|$1$|$1$|$1$|$1$|$1$|
|2|$1$|$1$|$1$|$-1$|$-1$|
|3|$1$|$1$|$-1$|$1$|$-1$|
|4|$1$|$1$|$-1$|$-1$|$1$|
|5|$2$|$x$|$y$|$z$|$w$|

According to orthonormality, the character table is:

|Conjugate Class:|$\zeta_1$|$\zeta_2$|$\zeta_3$|$\zeta_4$|$\zeta_5$|
|-|-:|-:|-:|-:|-:|
|1|$1$|$1$|$1$|$1$|$1$|
|2|$1$|$1$|$1$|$-1$|$-1$|
|3|$1$|$1$|$-1$|$1$|$-1$|
|4|$1$|$1$|$-1$|$-1$|$1$|
|5|$2$|$-2$|$0$|$0$|$0$|

### §1.8 D5

$$\begin{aligned} D_5 = \lbrace &e, \\\ &(12345), (13524), (42531), (54321), \\\ &(25)(34), (13)(45), (15)(24), (12)(35), (14)(23) \rbrace \end{aligned} \tag{1.8.1}$$where $(13524)=(12345)^2$, $(42531)=(12345)^3$, $(54321)=(12345)^4$. It has $4$ conjugate classes: $$\begin{aligned} \zeta_1&=\lbrace e \rbrace \\\ \zeta_2&=\lbrace (25)(34), (13)(45), (15)(24), (12)(35), (14)(23) \rbrace \\\ \zeta_3&=\lbrace (12345), (54321) \rbrace \\\ \zeta_4&=\lbrace (13524), (42531) \rbrace \end{aligned} \tag{1.8.2}$$thus has $1$ nontrivial invariant subgroup: $$ H_1 = \lbrace e, (12345), (13524), (42531), (54321)\rbrace \cong C_5\tag{1.8.3}$$thus has $1$ quotient group: $$D_5 / C_5 = \lbrace C_5, (25)(34)C_5 \rbrace \cong C_2\tag{1.8.4}$$

According to Eq. (1.1.5) and quotient groups, the character table is:

|Conjugate Class:|$\zeta_1$|$\zeta_2$|$\zeta_3$|$\zeta_4$|
|-|-:|-:|-:|-:|
|1|$1$|$1$|$1$|$1$|
|2|$1$|$-1$|$1$|$1$|
|3|$2$|$x$|$y$|$z$|
|4|$2$|$r$|$s$|$t$|

According to orthonormality, the character table is:

|Conjugate Class:|$\zeta_1$|$\zeta_2$|$\zeta_3$|$\zeta_4$|
|-|-:|-:|-:|-:|
|1|$1$|$1$|$1$|$1$|
|2|$1$|$-1$|$1$|$1$|
|3|$2$|$0$|$2\cos(\frac{2 \pi}{5})$|$2\cos(\frac{4 \pi}{5})$|
|4|$2$|$0$|$2\cos(\frac{4 \pi}{5})$|$2\cos(\frac{2 \pi}{5})$|

By the way, $2\cos(\frac{2 \pi}{5}) \approx 0.618$, is the Golden Ratio.

### §1.9 A4

$$\begin{aligned}A_4 = \lbrace &e, \\\ &(123), (321), (124), (421), (134), (431), (234), (432), \\\ &(12)(34), (13)(24), (14)(23) \rbrace \end{aligned} \tag{1.9.1}$$It has $4$ conjugate classes: $$\begin{aligned} \zeta_1&=\lbrace e \rbrace \\\ \zeta_2&=\lbrace (12)(34), (13)(24), (14)(23) \rbrace \\\ \zeta_3&=\lbrace (123), (421), (134), (432) \rbrace \\\ \zeta_4&=\lbrace (321), (124), (431), (234) \rbrace \end{aligned} \tag{1.9.2}$$thus has $1$ nontrivial invariant subgroup: $$H_1 = \lbrace e, (12)(34), (13)(24), (14)(23)\rbrace \cong V_4\tag{1.9.3}$$thus has $1$ quotient group: $$A_4 / V_4 = \lbrace V_4, (123)V_4, (321)V_4 \rbrace \cong C_3 \tag{1.9.4}$$

According to Eq. (1.1.5) and quotient groups, the character table is:

|Conjugate Class:|$\zeta_1$|$\zeta_2$|$\zeta_3$|$\zeta_4$|
|-|-:|-:|-:|-:|
|1|$1$|$1$|$1$|$1$|
|2|$1$|$x$|$e^{i \frac{2 \pi}{3}}$|$e^{i \frac{4 \pi}{3}}$|
|3|$1$|$y$|$e^{i \frac{4 \pi}{3}}$|$e^{i \frac{2 \pi}{3}}$|
|4|$3$||||

According to orthonormality, the character table is:

|Conjugate Class:|$\zeta_1$|$\zeta_2$|$\zeta_3$|$\zeta_4$|
|-|-:|-:|-:|-:|
|1|$1$|$1$|$1$|$1$|
|2|$1$|$1$|$e^{i \frac{2 \pi}{3}}$|$e^{i \frac{4 \pi}{3}}$|
|3|$1$|$1$|$e^{i \frac{4 \pi}{3}}$|$e^{i \frac{2 \pi}{3}}$|
|4|$3$|$x$|$y$|$z$|

According to orthonormality, the character table is:

|Conjugate Class:|$\zeta_1$|$\zeta_2$|$\zeta_3$|$\zeta_4$|
|-|-:|-:|-:|-:|
|1|$1$|$1$|$1$|$1$|
|2|$1$|$1$|$e^{i \frac{2 \pi}{3}}$|$e^{i \frac{4 \pi}{3}}$|
|3|$1$|$1$|$e^{i \frac{4 \pi}{3}}$|$e^{i \frac{2 \pi}{3}}$|
|4|$3$|$-1$|$0$|$0$|

## §2 Use normal Yang Tableaux to Construct the Irr. Rep. Matrices of S3

Drawing the Yang Tableaux with html is so simple and elegant:

```html
<table>
    <tr>
        <td>1</td>
        <td>2</td>
    </tr>
    <tr>
        <td>3</td>
    </tr>
</table>
```

1. normal Yang tableaux $\Theta_1$:

<table>
    <tr>
        <td>1</td>
        <td>2</td>
        <td>3</td>
    </tr>
</table>

According to [Hook length formula](https://en.wikipedia.org/wiki/Hook_length_formula), the irr. rep. is one-dimensional.

We have$$\begin{aligned} s_1 &= e+(12)+(23)+(13)+(123)+(321) \\\ a_1 &= e \\\ e_1 &= e+(12)+(23)+(13)+(123)+(321) \end{aligned} \tag{2.1}$$Define $$\ket{\nu} = \ket{e}+\ket{(12)}+\ket{(23)}+\ket{(13)}+\ket{(123)}+\ket{(321)} \tag{2.2}$$Thus we have $$p\ket{\nu}=\ket{\nu}\tag{2.3}$$where $p=e, (12), (23), (13), (123), (321)$. So $$p=1 \tag{2.4}$$

2. normal Yang tableaux $\Theta_2$:

<table>
    <tr>
        <td>1</td>
    </tr>
    <tr>
        <td>2</td>
    </tr>
    <tr>
        <td>3</td>
    </tr>
</table>

According to [Hook length formula](https://en.wikipedia.org/wiki/Hook_length_formula), the irr. rep. is one-dimensional.

We have$$\begin{aligned} s_2 &= e \\\ a_2 &= e-(12)-(23)-(13)+(123)+(321) \\\ e_2 &= e-(12)-(23)-(13)+(123)+(321) \end{aligned} \tag{2.5}$$Define $$\ket{\nu} = \ket{e}-\ket{(12)}-\ket{(23)}-\ket{(13)}+\ket{(123)}+\ket{(321)} \tag{2.6}$$Thus when $p$ is an odd permutation, $$p\ket{\nu}=(12)\ket{\nu}=(23)\ket{\nu}=(13)\ket{\nu}=-\ket{\nu} \tag{2.7}$$when $p$ is a even permutation, $$p\ket{\nu}=e\ket{\nu}=(123)\ket{\nu}=(321)\ket{\nu}=\ket{\nu} \tag{2.8}$$So $$p=(-1)^p \tag{2.9}$$

3. normal Yang tableaux $\Theta_3$:

<table>
    <tr>
        <td>1</td>
        <td>2</td>
    </tr>
    <tr>
        <td>3</td>
    </tr>
</table>

According to [Hook length formula](https://en.wikipedia.org/wiki/Hook_length_formula), the irr. rep. is two-dimensional.

We have$$\begin{aligned} s_3 &= e+(12) \\\ a_3 &= e-(13) \\\ e_3 &= e+(12)-(13)-(12)(13) \\\ &=e+(12)-(13)-(321) \end{aligned} \tag{2.10}$$
Define $$\ket{\nu_1} = \ket{e}+\ket{(12)}-\ket{(13)}-\ket{(321)} \tag{2.11}$$Thus$$\begin{aligned} e\ket{\nu_1}&=\ket{\nu_1} \\\ (12)\ket{\nu_1}&=\ket{(12)}+\ket{e}-\ket{(321)}-\ket{(13)}&=\ket{\nu_1}\\\ (23)\ket{\nu_1}&=\ket{(23)}+\ket{(321)}-\ket{(123)}-\ket{(12)}&=\ket{\nu_2}\\\ (13)\ket{\nu_1}&=\ket{(13)}+\ket{(123)}-\ket{e}-\ket{(23)}&=-\ket{\nu_1}-\ket{\nu_2}\\\ (123)\ket{\nu_1}&=\ket{(123)}+\ket{(13)}-\ket{(23)}-\ket{e}&=-\ket{\nu_1}-\ket{\nu_2}\\\ (321)\ket{\nu_1}&=\ket{(321)}+\ket{(23)}-\ket{(12)}-\ket{(123)}&=\ket{\nu_2}\\\ e\ket{\nu_2}&=\ket{\nu_2}\\\ (12)\ket{\nu_2}&=\ket{(123)}+\ket{(13)}-\ket{(23)}-\ket{e}&=-\ket{\nu_1}-\ket{\nu_2}\\\ (23)\ket{\nu_2}&=\ket{e}+\ket{(12)}-\ket{(13)}-\ket{(321)}&=\ket{\nu_1}\\\ (13)\ket{\nu_2}&=\ket{(321)}+\ket{(23)}-\ket{(12)}-\ket{(123)}&=\ket{\nu_2}\\\ (123)\ket{\nu_2}&=\ket{(12)}+\ket{e}-\ket{(321)}-\ket{(13)}&=\ket{\nu_1}\\\ (321)\ket{\nu_2}&=\ket{(13)}+\ket{(123)}-\ket{e}-\ket{(23)}&=-\ket{\nu_1}-\ket{\nu_2} \end{aligned} \tag{2.12}$$So $$\begin{aligned} D(e) &= \begin{pmatrix} 1 & 0 \\\ 0 & 1 \end{pmatrix} \\\ D(12) &= \begin{pmatrix} 1 & -1 \\\ 0 & -1 \end{pmatrix}, D(23) = \begin{pmatrix} 0 & 1 \\\ 1 & 0 \end{pmatrix}, D(13) = \begin{pmatrix} -1 & 0 \\\ -1 & 1 \end{pmatrix} \\\ D(123) &= \begin{pmatrix} -1 & 1 \\\ -1 & 0 \end{pmatrix}, D(321) = \begin{pmatrix} 0 & -1 \\\ 1 & -1 \end{pmatrix} \\\ \end{aligned} \tag{2.13}$$

## §3 Irr. Rep. of SO(3) Lie Algebra

### §3.1 j=1/2

When $j = \frac{1}{2}$, $\frac{3}{2}$, $\frac{5}{2}$, …, it's a double-valued representation. Let's take $j=\frac{1}{2}$ for example.

Define $$\ket{\frac{1}{2},\frac{1}{2}}=\begin{pmatrix} 1 \\\ 0 \end{pmatrix}, \ket{\frac{1}{2},\frac{1}{2}}=\begin{pmatrix} 0 \\\ 1 \end{pmatrix} \tag{3.1.1}$$According to $$\begin{aligned} J_3 \ket{j,m} &= m \ket{j,m} \\\ J_{\pm} \ket{j,m} &= \sqrt{j(j+1)-m(m \pm 1)} \ket{j,m \pm 1}\end{aligned} \tag{3.1.2}$$we have $$J_3=\begin{pmatrix} \frac{1}{2} & 0 \\\ 0 & -\frac{1}{2} \end{pmatrix}, J_+=\begin{pmatrix} 0 & 1 \\\ 0 & 0 \end{pmatrix}, J_-=\begin{pmatrix} 0 & 0 \\\ 1 & 0 \end{pmatrix} \tag{3.1.3}$$According to $$J_{\pm}=J_1 \pm i J_2 \tag{3.1.4}$$we have $$J_1=\begin{pmatrix} 0 & \frac{1}{2} \\\ \frac{1}{2} & 0 \end{pmatrix}, J_2=\begin{pmatrix} 0 & -\frac{i}{2} \\\ \frac{i}{2} & 0 \end{pmatrix}, J_3=\begin{pmatrix} \frac{1}{2} & 0 \\\ 0 & -\frac{1}{2} \end{pmatrix} \tag{3.1.5}$$According to $$J_k=\frac{\sigma_k}{2}, k=1,2,3 \tag{3.1.6}$$we have $$\sigma_3=\begin{pmatrix} 0 & 1 \\\ 1 & 0 \end{pmatrix}, \sigma_2=\begin{pmatrix} 0 & -i \\\ i & 0 \end{pmatrix}, \sigma_3=\begin{pmatrix} 1 & 0 \\\ 0 & 1 \end{pmatrix} \tag{3.1.7}$$So $$\begin{aligned} d^{\frac{1}{2}}(\beta) &=e^{-i \beta J_2} \\\ &= e^{-\frac{i \beta \sigma_2}{2}} \\\ &=E+(-\frac{i \beta}{2})\sigma_2+\frac{1}{2!}(-\frac{i \beta}{2})^{2}\sigma_2^2+\frac{1}{3!}(-\frac{i \beta}{2})^{3}\sigma_2^3+ \cdots \\\ &=E\cos{\frac{\beta}{2}}-i \sigma_2 \sin{\frac{\beta}{2}} \\\ &= \begin{pmatrix} \cos{\frac{\beta}{2}} & -\sin{\frac{\beta}{2}} \\\ \sin{\frac{\beta}{2}} & \cos{\frac{\beta}{2}} \end{pmatrix}\end{aligned} \tag{3.1.8}$$So $$D^{\frac{1}{2}}(\alpha, \beta, \gamma)=\begin{pmatrix} e^{-\frac{i\alpha}{2}}\cos{\frac{\beta}{2}}e^{-\frac{i\gamma}{2}} & -e^{-\frac{i\alpha}{2}}\sin{\frac{\beta}{2}}e^{\frac{i\gamma}{2}} \\\ e^{\frac{i\alpha}{2}}\sin{\frac{\beta}{2}}e^{-\frac{i\gamma}{2}} & e^{\frac{i\alpha}{2}}\cos{\frac{\beta}{2}}e^{\frac{i\gamma}{2}}\end{pmatrix} \tag{3.1.9}$$So $$\begin{aligned} D[R_n(2n \pi)] &= D[R] e^{in \pi \sigma_2} D[R]^{-1} \\\ &= D[R] (-E)^n D[R]^{-1} \\\ &=(-1)^n E \end{aligned} \tag{3.1.10}$$Thus it's a double-valued representation.

### §3.2 j=1

When $j = 0$, $1$, $2$, …, it's a irreducible representation. Let's take $j=1$ for example.

$$J_3=\begin{pmatrix} 1 & 0 & 0 \\\ 0 & 0 & 0 \\\ 0 & 0 & -1 \end{pmatrix}, J_+=\begin{pmatrix} 0 & \sqrt{2} & 0 \\\ 0 & 0 & \sqrt{2} \\\ 0 & 0 & 0 \end{pmatrix}, J_-=\begin{pmatrix} 0 & 0 & 0 \\\ \sqrt{2} & 0 & 0 \\\ 0 & \sqrt{2} & 0 \end{pmatrix} \tag{3.2.1}$$And$$e^{-i \beta J_2} = e^{\beta K} =E+\sin{\beta} K+(1-\cos{\beta})K^2 \tag{3.2.2}$$

The rest is left as an exercise for the reader.

Eventually, $$d^1(\beta)= \begin{pmatrix} \frac{1+\cos{\beta}}{2} & -\frac{\sin{\beta}}{\sqrt{2}} & \frac{1-\cos{\beta}}{2} \\\ \frac{\sin{\beta}}{\sqrt{2}} & \cos{\beta} & -\frac{\sin{\beta}}{\sqrt{2}} \\\ \frac{1-\cos{\beta}}{2} & \frac{\sin{\beta}}{\sqrt{2}} & \frac{1+\cos{\beta}}{2}  \end{pmatrix} \tag{3.2.3}$$

## §4 Decomposing the Direct Product of Irr. Rep. of SU(2) into the Direct Sum of Irr. Rep.

Here's an example of $j_1 = \frac{1}{2}$, $j_2=1$. You can check [Table of Clebsch–Gordan coefficients](https://en.wikipedia.org/wiki/Table_of_Clebsch%E2%80%93Gordan_coefficients) to make sure it's right.

1. $$\ket{\frac{3}{2}, \frac{3}{2}}=\ket{\frac{1}{2}, \frac{1}{2}} \otimes \ket{1, 1} \tag{4.1}$$

2. $$J_-\ket{\frac{3}{2}, \frac{3}{2}}=\sqrt{\frac{3}{2}(\frac{3}{2}+1)-\frac{3}{2}(\frac{3}{2}-1)}\ket{\frac{3}{2}, \frac{1}{2}}=\sqrt{3}\ket{\frac{3}{2}, \frac{1}{2}} \tag{4.2}$$
    $$\begin{aligned}J_-(\ket{\frac{1}{2}, \frac{1}{2}} \otimes \ket{1, 1})&=(J_-\ket{\frac{1}{2}, \frac{1}{2}})\otimes\ket{1, 1}+\ket{\frac{1}{2}, \frac{1}{2}} \otimes J_-(\ket{1, 1}) \\\ &=\ket{\frac{1}{2}, -\frac{1}{2}} \otimes \ket{1, 1}+\sqrt{2}\ket{\frac{1}{2}, \frac{1}{2}} \otimes \ket{1, 0} \end{aligned} \tag{4.3}$$
    So $$\ket{\frac{3}{2}, \frac{1}{2}}=\sqrt{\frac{1}{3}}\ket{\frac{1}{2}, -\frac{1}{2}} \otimes \ket{1, 1}+\sqrt{\frac{2}{3}}\ket{\frac{1}{2}, \frac{1}{2}} \otimes \ket{1, 0} \tag{4.4}$$

3. Similarly, $$\ket{\frac{3}{2}, -\frac{1}{2}}=\sqrt{\frac{2}{3}}\ket{\frac{1}{2}, -\frac{1}{2}} \otimes \ket{1, 0}+\sqrt{\frac{1}{3}}\ket{\frac{1}{2}, \frac{1}{2}} \otimes \ket{1, -1} \tag{4.5}$$

4. Similarly, $$\ket{\frac{3}{2}, -\frac{3}{2}}=\ket{\frac{1}{2}, -\frac{1}{2}} \otimes \ket{1, -1} \tag{4.6}$$

5. Because $\ket{\frac{3}{2}, \frac{1}{2}}$ and $\ket{\frac{1}{2}, \frac{1}{2}}$ are orthogonal, and by normalization, we have $$\ket{\frac{1}{2}, \frac{1}{2}} = \sqrt{\frac{2}{3}}\ket{\frac{1}{2}, -\frac{1}{2}} \otimes \ket{1, 1}-\sqrt{\frac{1}{3}}\ket{\frac{1}{2}, \frac{1}{2}} \otimes \ket{1, 0} \tag{4.7}$$

6. Similarly, $$\ket{\frac{1}{2}, -\frac{1}{2}} = \sqrt{\frac{1}{3}}\ket{\frac{1}{2}, -\frac{1}{2}} \otimes \ket{1, 0}-\sqrt{\frac{2}{3}}\ket{\frac{1}{2}, \frac{1}{2}} \otimes \ket{1, -1} \tag{4.8}$$

Thus complete the Clebsch–Gordan coefficients for $j_1 = \frac{1}{2}$, $j_2=1$.

## §5 Box-Weight Diagram of SU(3) Irr. Rep.

### §5.1 Box-Weight Diagram

We have $$\begin{cases} \vec{\alpha^1} = (\frac{1}{2},\frac{\sqrt{3}}{2}) \\\ \vec{\alpha^2} = (\frac{1}{2},-\frac{\sqrt{3}}{2}) \end{cases} \tag{5.1.1}$$and according to the definition of Cartan matrix $$A_{ij} = \frac{2 \vec{\alpha^i} \cdot \vec{\alpha^j}}{|\vec{\alpha^j}|^2} \tag{5.1.2}$$we have:$$A = \begin{pmatrix} 2 & -1 \\\ -1 & 2 \end{pmatrix} \tag{5.1.3}$$

When drawing the box-weight diagram, there are some rules to follow:

- When minus the number on the left of $\boxed{x \space y}$, the $x$, 
    1. minus the $1^{\text{st}}$ row of eq. (5.1.3);
    2. draw a line down-right from $x$ to the $x'$ in the next box;
    3. until it's $-x$.
- When minus the number on the right of $\boxed{x \space y}$, the $y$,
    1. minus the $2^{\text{nd}}$ row of eq. (5.1.3);
    2. draw a line down-left from $y$ to the $y'$ in the next box;
    3. until it's $-y$.
- If you start from $\boxed{a \space b}$ on the top, the final result should be $\boxed{-b \space -a}$ in the bottom.

For example, when the highest weight is $3$ and $0$, or $\boxed{3 \space 0}$, the box-weight diagram should look like this:

![20230615-group-theory-calculations-highest-weight-3-and-0](20230615-group-theory-calculations-highest-weight-3-and-0.jpg)

### §5.2 All Possible Roots

According to $$\frac{2 \vec{\alpha^j} \cdot \vec{\mu^k}}{|\vec{\alpha^j}|^2}=\delta_{jk} \tag{5.2.1}$$we have $$\begin{cases} \vec{\mu^1}=(\frac{1}{2}, \frac{\sqrt{3}}{6}) \\\ \vec{\mu^2}=(\frac{1}{2}, -\frac{\sqrt{3}}{6}) \end{cases} \tag{5.2.2}$$

Starting from $3 \vec{\mu^1}$, from box-weight diagram, we have $$\begin{aligned} &3 \vec{\mu^1}-\vec{\alpha^1}, \\\ &3 \vec{\mu^1}-\vec{\alpha^1}-\vec{\alpha^2}, 3 \vec{\mu^1}-2\vec{\alpha^1}, \\\ &3 \vec{\mu^1}-2\vec{\alpha^1}-\vec{\alpha^2}, 3 \vec{\mu^1}-3\vec{\alpha^1}, \\\ &3 \vec{\mu^1}-2\vec{\alpha^1}-2\vec{\alpha^2}, 3 \vec{\mu^1}-3\vec{\alpha^1}-\vec{\alpha^2}, \\\ & 3 \vec{\mu^1}-3\vec{\alpha^1}-2\vec{\alpha^2}, \\\ &3 \vec{\mu^1}-3\vec{\alpha^1}-3\vec{\alpha^2} \end{aligned} \tag{5.2.3}$$ 
(Or you can do it more easily by using the fact that the figure below is symmetric with respect to $H_2$ axis.)

Drawing on the $H_1 - H_2$ plane, we have:

![20230615-group-theory-calculations-highest-weight-3-and-0-H_1-H_2](20230615-group-theory-calculations-highest-weight-3-and-0-H_1-H_2.png)
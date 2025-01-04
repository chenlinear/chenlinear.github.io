---
author: "Chen Li"
title: "Markdown Style"
date: "2023-05-21"
tags: 
- programming
---

If you don't know what it is, I recommend you should just play with it and don't care about the style.

[Obsidian](https://obsidian.md/) is the text editor that I'm using. I like [workspaces](https://help.obsidian.md/Plugins/Workspaces) and [double link](https://help.obsidian.md/Getting+started/Link+notes) feature.

## §1 General

- [_Markdown Guide_](https://www.markdownguide.org/) is good as a quick introduction.

- The thing about Markdown style is that every text editor has its own accent. I suggest using the universal ones, not the ones that a certain text editor created. Also, the best way to read this article is to copy and paste what's in the code block, and see if it fits your text editor.

## §2 This and That

### §2.1 Use Blank Lines!

Wrong:

```markdown
use blank lines!
use blank lines!
```

Correct:

```markdown
use blank lines!

use blank lines!
```

See [_Let Your Markdown Breathe! - Remember to insert blank lines between different Markdown elements_](https://yihui.org/en/2021/06/markdown-breath/).

### §2.2 Only Use `_` for Bold and Italic

```markdown
Bold _and_ Italic

Bold __and__ Italic

Bold ___and___ Italic
```

### §2.3 In the Middle of a Sentence, use `*` for Bold and Italic

```markdown
Bold*and*Italic

Bold**and**Italic

Bold***and***Italic
```

But to tell you the truth, I haven't seen **any**one write like this.

### §2.4 Only use `-` for Unordered Lists

As I said, there are a lot of Markdown accent, but `-` is universal. Also, this is in agreement with §2.2, so that you can only use the `-`/`_` key on the keyboard.

```markdown
- unordered lists
```

### §2.5 Four Spaces in Lists

This is a fun coincidence with Python.

```markdown
- First paragraph

    Still first paragraph

- Second paragraph
```

Or:

```markdown
1. First paragraph
 
	Still first paragraph

2. Second paragraph
```

To do list too:

```markdown
- [ ] hurry
 
	- [ ] up,

- [ ] now!
```

### §2.6 Quote and Code Block in List

Again, four spaces. But in Obsidian, these are shown correct in reading mode, shown werid in editing mode. See [_Code blocks nested in lists render in editor as if at top-level_](https://forum.obsidian.md/t/code-blocks-nested-in-lists-render-in-editor-as-if-at-top-level/870/10).

```markdown
- Quote:

    >This is the quote.
```

````markdown
- Code Block:

    ```python
    x = x + 1
    ```
````

By the way, when you write verbatim text that contains three backticks (ie. code block in code block), you have to use four backticks to wrap the text. See [_The Two Surprisingly Hard Things about the Otherwise Simple Markdown - Writing three backticks verbatim, and understanding nested lists_](https://yihui.org/en/2018/11/hard-markdown/).

### §2.7 Equations

Different Markdown accent handle equations differently, see [_混乱的 Markdown 世界_](https://yihui.org/cn/2017/08/markdown-flavors/). And for Obsidian:

```markdown
1. display equations: $$\begin{aligned} E_0&=mc^2 \\\ E^2&=(mc^2)^2+(pc)^2 \end{aligned} \tag{1}$$where $c = 3 \times 10^8 \space \mathrm{m/s}$

2. or inline equations: $\vec{F} = m \ddot{\vec{x}}$.
```

### §2.8 Two Pictures Alignment

Use table (compared with inserting html `<>` and all that, this is easier to read):

```markdown
|![name](link)|![name](link)|
|-|-|
```

Most text editors show the picture with modified proper size. But for some text editors, when the picture is too big, you should reduce pixel size of the picture before showing it in the table. This is the case for [remark](https://github.com/gnab/remark) and [DrSlidelove](https://github.com/chenlinear/DrSlidelove). But you can solve it by putting it in a table.

## §3 使用中文时

### §3.1 空格

[_中英文混排中的空格_](https://yihui.org/cn/2017/04/space)和[_盘古之白_](https://yihui.org/cn/2017/05/pangu/)说在英文单词两边各空一空格。我的建议是在表示代码时也是如此，例如：

```markdown
我们可以 `from scipy import interpolate` 来解决这个 interpolation 问题。
```

### §3.2 引用诗词

[_如何在 Markdown 中引用诗词或歌词_](https://yihui.org/cn/2018/07/quote-poem/)的引用方式在 Obsidian 可能不是最好的解决方法，可以直接每行右边没有空格，见 [_Blockquotes with Multiple Paragraphs_](https://www.markdownguide.org/basic-syntax/#blockquotes-with-multiple-paragraphs)：

```markdown
>缺月昏昏漏未央，一灯明灭照秋床。
>病身最觉风露早，归梦不知山水长。
>坐感岁时歌慷慨，起看天地色凄凉。
>鸣蝉更乱行人耳，正抱疏桐叶半黄。
```

但是在我的个人网站中应该每行右边两个空格：

```markdown
>缺月昏昏漏未央，一灯明灭照秋床。  
>病身最觉风露早，归梦不知山水长。  
>坐感岁时歌慷慨，起看天地色凄凉。  
>鸣蝉更乱行人耳，正抱疏桐叶半黄。  
```

才能显示正确：

>缺月昏昏漏未央，一灯明灭照秋床。  
>病身最觉风露早，归梦不知山水长。  
>坐感岁时歌慷慨，起看天地色凄凉。  
>鸣蝉更乱行人耳，正抱疏桐叶半黄。  

---
author: "Chen Li"
title: "Python Style"
date: "2023-05-21"
tags: 
- programming
---

If you don't know what it is, I recommend you should just play with it and don't care about the style.

But I'd like to say [Anaconda](https://www.anaconda.com/) is a better choice than just [Python](https://www.python.org/) itself. You can manage different packages in different environments with Anaconda.

This article is about what I find interesting in [_PEP 8 – Style Guide for Python Code_](https://peps.python.org/pep-0008/)[^1], [_PEP 257 – Docstring Conventions_](https://peps.python.org/pep-0257/), [_Style guide — numpydoc v1.6.0rc1.dev0 Manual_](https://numpydoc.readthedocs.io/en/latest/format.html) and [SPT3G](https://github.com/CMB-S4/spt3g_software/blob/master/doc/styleguide.rst). This is not a guide, I recommend [_Google Style Guides_](https://google.github.io/styleguide/)[^2] as the guide.

## §1 General

>Readability counts. — [PEP 20](https://peps.python.org/pep-0020 "PEP 20 – The Zen of Python")

- Always break your code apart by defining functions, classes, or create a new .py file and `from myproject.backend.hgwells import time_machine`. Don't write code like a long staircase of indentation, that would drive people crazy.

![Joker2019](https://irs.www.warnerbros.com/gallery-v2-jpeg/movies/node/91131/edit/joker_joaquin_phoenix_03.jpg)

- Comments are of equal importance as code. It's for your own good, because you can't remember everything.

## §2 This and That

If your project has a style guide already, always follow that rather than this one.

### §2.1 Naming

Naming is a big deal, the biggest probably. And I generally follow these rules from SPT3G.

- Function, Variable: `lower_with_under`
- Class: `CapWords`
- Constant: `CAPS_WITH_UNDER`, and use [`scipy.constants`](https://docs.scipy.org/doc/scipy/reference/constants.html).
- Non-public class methods and instance variables should begin with a single underscore (this is baked into the way python implements classes).

See [this table](https://google.github.io/styleguide/pyguide.html#3164-guidelines-derived-from-guidos-recommendations) for the full naming convention.

### §2.2 Four Spaces for Indentation

Also according to SPT3G:

```python
CONSTANT_VALUE = 10

def function_name(variable_name, other):
    do_some_things

class MyClass:
    CONSTANCE_INSTANCE_VARIABLE = 100
    def __init__(self, input, other_input):
        self.instance_variable = input + other_input

    def class_method(self, input):
        do_things
```

### §2.3 Comments and Doc Strings

You should assume the person who is reading the comments are more familiar with python than you, they just don't understand what your code is doing:

```python
# wrong:
x = x + 1                 # Increment x
# correct:
x = x + 1                 # Compensate for border
```

Also according to SPT3G, for classes, modules and functions, write doc strings:

```python
def get_fft_scale_fac(res=None, n1=None, n2=None,
                      apod_mask=None, parent=None):
    """
    Returns a scale factor to convert a 2d fft of a map into
    sqrt(C_l) normalized units by scaling it by the sky area.
    In particular it returns:
      np.mean(apod_mask**2)**0.5 * (n1 * n2)**0.5 / reso_rad

    For forward transforms: fft2(mp)  / scale_fac
    For inverse transforms: ifft2(mp) * scale_fac

    Arguments
    ---------
    res : float, default=None
        Resolution of the map in G3Units.
        Must be specified if parent is None.
    n1 : int, default=None
        Number of map pixels in x direction.
        Must be specified if parent is None.
    n2 : int, default=None
        Number of map pixels in y direction.
        Must be specified if parent is None.
    apod_mask : FlatSkyMap or ndarray, default=None
        An apodization mask.
    parent : FlatSkyMap, default=None
        Parent Map object containing resolution and shape info.
        Must be specified if res, n1, and n2 are None.

    Returns
    -------
    float
        FFT normalization factor

    See Also
    -------
    ft_to_map
    """
```

By the way, `'` and `"` are the same thing in code.

###  §2.4 Maximum Line Length is 80 Characters

If you need to wrap multiple lines, you should wrap them on the same level of syntax structure:

```python
bridgekeeper.answer(
     name="Arthur", quest=questlib.find(owner="Arthur", perilous=True))

 answer = (a_long_line().of_chained_methods()
           .that_eventually_provides().an_answer())

 if (
     config is None
     or 'editor.language' not in config
     or config['editor.language'].use_spaces is False
 ):
   use_tabs()
```

And follow the tradition from mathematics:

```python
income = (gross_wages
          + taxable_interest
          + (dividends - qualified_dividends)
          - ira_deduction
          - student_loan_interest)
```

###  §2.5 Import Order

1. Python future import statements.
2. Python standard library imports.
3. [Third-party](https://pypi.org/) module or package imports.
4. Code repository sub-package. Absolute imports are recommended.

For example:

```python
from __future__ import annotations

import os
import sys

import numpy as np
import tensorflow as tf
from scipy import constants

from myproject.backend.hgwells import time_machine
```

[^1]: 我和 ChatGPT 合著的中文版见 [PEP-8-ZH](https://github.com/ChenLi2049/PEP-8-ZH).
[^2]: 中文版见 [_Google 开源项目风格指南_](https://zh-google-styleguide.readthedocs.io/en/latest/).
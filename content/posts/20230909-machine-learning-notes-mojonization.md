---
author: "Chen Li"
title: "Machine Learning Notes: Mojonization"
date: "2023-09-09"
tags: 
- Programming
---

__mojonization__: _n._ the migration and translation from Python to Mojo, the superset of Python. (Yeah I made up this word, as far as I know. [_Cross Mojonization_](http://darkbluejacket.blogspot.com/2015/09/the-glory-of-cross-mojonization.html) is a totally different thing.)

This article summarizes this process and is my notes / cheat sheet. Note that Mojo is relatively new and some rules might be changing rapidly, thus this article can be outdated easily. So please always check the [official documents](https://docs.modular.com/).

The following content is based on what happened in a `.ipynb` file, to run a `.mojo` or `.🔥` file, see [_Build and run Mojo source files_](https://docs.modular.com/mojo/manual/get-started/hello-world.html#build-and-run-mojo-source-files).

## §1 Mojo Language Basics

This is my notes from [_Mojo language basics_](https://docs.modular.com/mojo/manual/basics/) and [_Mojo programming manual_](https://docs.modular.com/mojo/programming-manual.html).

- Mojo’s standard library has not yet grown a standard list or dictionary type and [list or dict comprehensions is not supported](https://docs.modular.com/mojo/roadmap.html#no-list-or-dict-comprehensions). See [_Roadmap & sharp edges_](https://docs.modular.com/mojo/roadmap.html).
- See [_Mojo modules_](https://docs.modular.com/mojo/lib.html) for all the APIs.
- You can use `%%python` at the top of a notebook cell and write normal Python code. Variables, functions, and imports defined in a Python cell are available for access in subsequent Mojo cells.

### §1.1 `var` & `let`

You can declare variables with `var` to create a mutable value, or with `let` to create an immutable value. Although types aren't required, it's recommended to do so. You can check all the types in Mojo at [Modular Docs - int](https://docs.modular.com/mojo/stdlib/builtin/int.html) or [Modular Docs - dtype](https://docs.modular.com/mojo/stdlib/builtin/dtype.html). For example:
```python
var a = 1
var b: Int = 1
let c = 1
let d: Int = 1
let e: Float64 = 1.0
```

Always do these declaration at the beginning of code, because:
- Redefining implicit variables is not supported (variables without a `let` or `var` in front). If you’d like to redefine a variable across notebook cells, you must introduce the variable with `var` (`let` variables are immutable). 
- You can’t use global variables inside functions—they’re only visible to other global variables.

### §1.2 `fn`

`fn` is similar to `def` in Python. And Mojo supports `def` too. Types are required for arguments and return values for a `fn` function.

#### §1.2.1 Immutable Arguments: `borrowed`

- If the arguments of the function are immutable:
    ```python
    fn add(borrowed x: Int, borrowed y: Int) -> Int:
        return x + y
    
    c = add(1, 2)
    print(c)
    ```
    will get:
    ```Shell
    3
    ```

#### §1.2.2 Mutable Arguments: `inout`

- If the arguments are mutable:
    ```python
    fn add_inout(inout x: Int, inout y: Int) -> Int:
        x += 1
        y += 1
        return x + y
    
    var a = 1
    var b = 2
    c = add_inout(a, b)
    print(a)
    print(b)
    print(c)
    ```
    will get (__Note that `a` and `b` has changed__):
    ```Shell
    2
    3
    5
    ```

#### §1.2.3 Transfer Arguments: `owned` & `^`

- If the arguments are immutable but you still want to modify it without affecting variables outside the function (__This is normally what would happen in Python that global variables (variables outside the function) won't change.__)
    ```python
    fn add_owned(owned x: Int, owned y: Int) -> Int:
        x += 1
        y += 1
        return x + y
    
    let a = 1 # or var a = 1
    let b = 2 # or var b = 2
    c = add_owned(a, b)
    print(a)
    print(b)
    print(c)
    ```
    will get:
    ```Shell
    1
    2
    5
    ```
    or
    ```python
    fn add_owned(owned x: Int, owned y: Int) -> Int:
        x += 1
        y += 1
        return x + y
    
    c = add_owned(1, 2)
    print(c)
    ```
    will get:
    ```Shell
    5
    ```

- `^` will destroy local variable (variable inside the `fn`):
    ```python
    fn add_owned(owned x: Int, owned y: Int) -> Int:
        x += 1
        y += 1
        return x+y
    
    fn main():
        let a: Int = 1 # or var a = 1, this will get a warning to suggest using let because a is never mutated
        let b: Int = 2
        let c: Int # late initialization
        c = add_owned(a^, b^)
        print(a)
        print(b)
        print(c)
    
    main()
    ```
    will get an error at `print(a)` because `a` has been destroyed. And
    ```python
    fn add_owned(owned x: Int, owned y: Int) -> Int:
        x += 1
        y += 1
        return x+y
    
    fn main():
        let a: Int = 1 # or var a = 1, this will get a warning to suggest using let because a is never mutated
        let b: Int = 2
        let c: Int # late initialization
        c = add_owned(a^, b^)
        # print(a)
        # print(b)
        print(c)
    
    main()
    ```
    will get:
    ```Shell
    5
    ```

In summary, in practice I will probably:
- use Transfer Arguments (`owned` and `^`)
- start by writing `var` and then change it into `let` according to the warning.

### §1.3 `struct`

`struct` is similar to `class` in Python:
```python
struct MyPair:
    var first: Int
    var second: Int

    fn __init__(inout self, first: Int, second: Int):
        self.first = first
        self.second = second
    
    fn dump(self):
        print(self.first, self.second)

let mine = MyPair(2, 4)
mine.dump()
```
will get:
```Shell
2 4
```

### §1.4 Python packages

```python
from python import Python

let np = Python.import_module("numpy")
let pd = Python.import_module("pandas")
let plt = Python.import_module("matplotlib.pyplot")

ar = np.arange(15).reshape(3, 5)
print(ar)
print(ar.shape)
```
will get:
```Shell
[[ 0  1  2  3  4]
 [ 5  6  7  8  9]
 [10 11 12 13 14]]
(3, 5)
```

## §2 Other Documents

[_Mojo programming manual_](https://docs.modular.com/mojo/programming-manual.html) explains why Mojo is designed in this way. I will look it up when facing bugs. I learned C++ years ago and I haven't really learned Rust. I'm not familiar with Lifetime, Pointer, etc.

[_Modular AI Engine_](https://docs.modular.com/engine/) is about Modular AI Engine, which is not publicly available yet.
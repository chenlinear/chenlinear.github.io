---
author: "Chen Li"
title: "List of a Folder"
date: "2023-04-23"
tags: 
- programming
---

## §1 path

In Windows, to get the list of all the files in a folder:

1. Create a .txt file.
2. Open it with notepad. Copy and paste the following commands:

    ```bash
    @echo off
    dir %1 /s/b > %~n1.txt
    ```

    Or, if you don't want the list of what's in the subdirectory:

    ```bash
    @echo off
    dir %1 /b > %~n1.txt
    ```

3. Rename it getList.bat ("getList" could be anything.)
4. Drag the target folder to this .bat file.

## §2 path & size

Only the code in step 2. is different:

```bash
@echo off
( for /r %1 %%i in (*) do @echo %%~fi ^(%%~zi bytes^)) > %~n1.txt
```

And, to convert bytes to GB, use this python code:

```python
import re

# open original.txt, create new.txt
with open('original.txt', 'r') as f1, open('new.txt', 'w') as f2:
    for line in f1:
        # use regular expressions to match file size information
        match = re.search(r'(\d+)\sbytes', line)
        if match:
            # get the file size information
            size_bytes = int(match.group(1))
            # convert bit to GB
            size_gb = size_bytes / 1024**3
            # write to new.txt
            new_line = re.sub(r'(\d+)\sbytes', f'{size_gb:.2f} GB', line)
            f2.write(new_line)
        else:
            # if file size information is not matched, write the original line
            f2.write(line)
```
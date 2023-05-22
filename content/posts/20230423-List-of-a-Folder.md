---
author: "Chen Li"
title: "List of a Folder"
date: "2023-04-23"
tags: 
- CS
---

In Windows, to get the list of all the files in a folder:

1. Create a .txt file.
2. Open it with notepad. Copy and paste the following commands:
```propmt
@echo off
dir %1 /s/b > %~n1.txt
```
Or, if you don't want the list of what's in the subdirectory:
```prompt
@echo off
dir %1 /b > %~n1.txt
```
3. Rename it getList.bat ("getList" could be anything.)
4. Drag the target folder to this .bat file.
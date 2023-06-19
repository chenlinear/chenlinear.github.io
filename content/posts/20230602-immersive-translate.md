---
author: "Chen Li"
title: "immersive-translate"
date: "2023-06-02"
tags: 
- CS
---

[immersive-translate](https://github.com/immersive-translate/immersive-translate) is a great translating extension, which has a nice format, can open .epub file and translate .epub file directly.

1. There's a bug, when translating [Feedly](feedly.com), the translated content in the side bar would convert with the original content. According to [高级自定义配置](https://immersive-translate.owenyoung.com/advanced#user-rules), I used:

    ```
    [
      {
        "matches": "feedly.com",
        "excludeSelectors": [
          "nav",
          "footer"
        ]
      }
    ]
    ```

    Learning a little html is really useful.

2. For the translation box, I used purple quotation-style.

    ```
      "translationTheme": "blockquote",
      "translationThemePatterns": {
        "blockquote": {
          "borderColor": "#c325ef",
          "zoom": "80"
        },
        "paper": {
          "zoom": "80"
        }
    ```

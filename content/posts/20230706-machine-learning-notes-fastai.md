---
author: "Chen Li"
title: "Machine Learning Notes: fastai"
date: "2023-07-06"
tags: 
- Programming
---

Compared with `tensorflow`, `mxnet`, `paddle` or pure `numpy` (just for the fun of it), `torch` is probably the easiest Machine Learning package, and to get it even easier, let's take a look at `fastai`.

By the way, I subscribed GitHub Trending by RSS and the other day I got these two at the same time. Machine Learning in `numpy` is really cool, but the second one is like, why? ... these two target markets do NOT overlap.

![ml-in-np-python-in-excel](20230706-machine-learning-notes-fastai-ml-np-py-excel.png)

Back to the topic, here's my notes from [fastai](https://docs.fast.ai/), [fastbook (github.com)](https://github.com/fastai/fastbook) and [_Practical Deep Learning for Coders_](https://course.fast.ai/). The main idea of `fastai` is to make Machine Learning accessible to every individual so that it can be applied to various subjects, and to do so:

- Use Google Colab, Kaggle, Paperspace, Hugging face.
- Only requires one GPU (or at least try to).
- fastai starts with higher architecture (see [fastai - _Quick start_](https://docs.fast.ai/quick_start.html)) and digs deeper so that it's more customizable. Nonetheless, some higher architectures are useful that you will see it in almost any project that uses `fastai`, for example, `DataLoaders` and `Learners`.

## Tabular Data

Random Forest is baseline, sometimes even the best method. See [sklearn.ensemble.RandomForestClassifier — scikit-learn 1.3.0 documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).

## Transfer Learning for CV

See [_The best vision models for fine-tuning_ | Kaggle](https://www.kaggle.com/code/jhoward/the-best-vision-models-for-fine-tuning).

## Two Categories

Sometimes predicting two categories at the same time has these two advantages:
- Parallel computing, which saves more time and computing sources.
- The categories might help each other. For example, predicting the fish and the boat may produce better results than only predicting the fish.

## `Learners`

```python
from fastai.vision.all import Learners
learn = Learners(...)
```

See [fastai - _Pytorch to fastai details_](https://docs.fast.ai/examples/migrating_pytorch_verbose.html). Here's some useful tricks:

- To find the best learning rate, see [[1506.01186] _Cyclical Learning Rates for Training Neural Networks_](https://arxiv.org/abs/1506.01186):

    ```python
    learn.lr_find(suggest_funcs=(slide, valley))
    ```

- To test the model with a dummy:

    ```python
    test_df = ...
    test_dl = learn.dls.test_dl(test_df)
    preds, _ = learn.get_preds(dl=test_dl)
    ```
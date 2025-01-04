---
author: "Chen Li"
title: "Machine Learning Notes: Workflow & Tips"
date: "2023-07-01"
tags: 
- programming
---

(Please refer to [_Wow It Fits! — Secondhand Machine Learning_](https://chenlinear.github.io/posts/20231011-wow-it-fits-secondhand-machine-learning/).)

Here are my notes from [Zero to Mastery Learn PyTorch for Deep Learning](https://www.learnpytorch.io/). For cheatsheet, see [_PyTorch Cheatsheet_ - Zero to Mastery Learn PyTorch for Deep Learning](https://www.learnpytorch.io/pytorch_cheatsheet/) or [_Create a training/testing loop_](https://www.learnpytorch.io/pytorch_cheatsheet/#create-a-trainingtesting-loop) or [_PyTorch documentation_](https://pytorch.org/docs/stable/index.html).

## §1 Workflow

![](https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/03-pytorch-computer-vision-workflow.png)

Most of the time it's necessary to subclass the classes mentioned above, check [_PyTorch documentation_](https://pytorch.org/docs/stable/index.html).

## §2 Tensor Error

See [_The Three Most Common Errors in PyTorch_ - Zero to Mastery Learn PyTorch for Deep Learning](https://www.learnpytorch.io/pytorch_most_common_errors/).

1. __Shape__, e.g. [H, W, C] (usually used in `numpy` or `matplotlib.pyplot`) or [C, H, W] (usually used in `torch`) or [batch_size, C, H, W].

![](https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/00-pytorch-different-tensor-dimensions.png)

2. __Device__

3. __Type__

## §3 Jupyter Notebook

- markdown.
- markdown titles.
- print the output of a function.

## §4 [`torchinfo`](https://github.com/TylerYep/torchinfo) & [`torch.utils.tensorboard`](https://pytorch.org/docs/stable/tensorboard.html)

To use `torchinfo` to check the structure of a model:

```python
from torchinfo import summary

summary(model=VisionCNN)

# or

summary(model=VisionCNN,
        input_size=(32, 3, 224, 224), # (batch_size, C, H, W)
        col_names=["input_size", "output_size", "num_params"],
        col_width=20,# column width
        row_settings=["var_names"])
```

To use `torch.nn.tensorboard`, see [_07. PyTorch Experiment Tracking_ - Zero to Mastery Learn PyTorch for Deep Learning](https://www.learnpytorch.io/07_pytorch_experiment_tracking/). Here's an example of `train` function, the log of which is like a simple version of `torch.nn.tensorboard`:

```python
from typing import Dict, List
from tqdm.auto import tqdm

# Add writer parameter to train()
def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device, 
          writer: torch.utils.tensorboard.writer.SummaryWriter # new parameter to take in a writer
          ) -> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Stores metrics to specified writer log_dir if present.

    Args:
      model: A PyTorch model to be trained and tested.
      train_dataloader: A DataLoader instance for the model to be trained on.
      test_dataloader: A DataLoader instance for the model to be tested on.
      optimizer: A PyTorch optimizer to help minimize the loss function.
      loss_fn: A PyTorch loss function to calculate loss on both datasets.
      epochs: An integer indicating how many epochs to train for.
      device: A target device to compute on (e.g. "cuda" or "cpu").
      writer: A SummaryWriter() instance to log model results to.

    Returns:
      A dictionary of training and testing loss as well as training and
      testing accuracy metrics. Each metric has a value in a list for 
      each epoch.
      In the form: {train_loss: [...],
                train_acc: [...],
                test_loss: [...],
                test_acc: [...]} 
      For example if training for epochs=2: 
              {train_loss: [2.0616, 1.0537],
                train_acc: [0.3945, 0.3945],
                test_loss: [1.2641, 1.5706],
                test_acc: [0.3400, 0.2973]} 
    """
    # Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
    }

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
        test_loss, test_acc = test_step(model=model,
          dataloader=test_dataloader,
          loss_fn=loss_fn,
          device=device)

        # Print out what's happening
        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)


        ### New: Use the writer parameter to track experiments ###
        # See if there's a writer, if so, log to it
        if writer:
            # Add results to SummaryWriter
            writer.add_scalars(main_tag="Loss", 
                               tag_scalar_dict={"train_loss": train_loss,
                                                "test_loss": test_loss},
                               global_step=epoch)
            writer.add_scalars(main_tag="Accuracy", 
                               tag_scalar_dict={"train_acc": train_acc,
                                                "test_acc": test_acc}, 
                               global_step=epoch)

            # Close the writer
            writer.close()
        else:
            pass
    ### End new ###

    # Return the filled results at the end of the epochs
    return results
```

## §5 More

In terms of "vibe", here's what I find interesting in the course that I'll try to take to my future communication and teaching (if possible):

- It's like a cooking show, try to keep it fun and interactive. The way to learn Machine Learning is to do different projects, and you will get better each time.
- Bugs are fun and valuable, deal with them with positivity. When facing a problem while programming, use search engines, Wikipedia and official documents (in this case [_PyTorch documentation_](https://pytorch.org/docs/stable/index.html)).
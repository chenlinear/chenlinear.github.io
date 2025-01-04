---
author: "Chen Li"
title: "Machine Learning Notes: torch.nn"
date: "2023-06-14"
tags: 
- programming
math: true
---

(Please refer to [_Wow It Fits! — Secondhand Machine Learning_](https://chenlinear.github.io/posts/20231011-wow-it-fits-secondhand-machine-learning/).)

This is a quick introduction to torch or how to build a neural network without writing the source code. For the purpose of each layer, see [torch.nn](https://pytorch.org/docs/stable/nn.html) and [_Dive into Deep Learning_](https://d2l.ai/). Basically, after CNN, parts of the picture is highlighted and the number of channels (RGB $\rightarrow$ many more) can be different (see [CNN Explainer](https://poloclub.github.io/cnn-explainer/)).

In the following code, first `import` the required packages:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

## nn.Conv2d

```python
x = torch.randn(1, 3, 28, 28)
print(x.shape)

conv2d = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3)
x = conv2d(x)
print(x.shape)
```

will get:

```bash
torch.Size([1, 3, 28, 28])
torch.Size([1, 12, 26, 26])
```

## nn.MaxPool2d

```python
x = torch.randn(1, 3, 28, 28)
print(x.shape)

pool = nn.MaxPool2d(kernel_size=2)
x = pool(x)
print(x.shape)
```

will get:

```bash
torch.Size([1, 3, 28, 28])
torch.Size([1, 3, 14, 14])
```

## nn.BatchNorm2d

See Fig.2 of [[1803.08494] _Group Normalization_](https://arxiv.org/abs/1803.08494).

```python
batchnorm = nn.BatchNorm2d(3)
x = torch.randn(1, 3, 3, 3)
print(x)

x = batchnorm(x)
print(x)
```

will get:

```bash
tensor([[[[-2.3440,  0.5965,  1.2750],
          [-0.8684,  1.4049, -1.5865],
          [ 0.0760,  0.3308,  1.4631]],

         [[-0.7283,  0.6978,  1.1655],
          [-0.2090,  1.4090,  1.0433],
          [-0.1820,  0.0191,  1.8880]],

         [[ 2.5024,  0.5590,  1.6343],
          [-0.2351, -0.8212, -1.0195],
          [-0.6536,  0.2503,  0.1578]]]])
tensor([[[[-1.8478,  0.4327,  0.9589],
          [-0.7034,  1.0596, -1.2603],
          [ 0.0290,  0.2266,  1.1048]],

         [[-1.5610,  0.1576,  0.7212],
          [-0.9351,  1.0146,  0.5739],
          [-0.9027, -0.6603,  1.5918]],

         [[ 2.0339,  0.2682,  1.2452],
          [-0.4533, -0.9858, -1.1660],
          [-0.8336, -0.0122, -0.0964]]]], grad_fn=<NativeBatchNormBackward0>)
```

## nn.Linear

For fully connected layer.

```python
linear = nn.Linear(3, 12)
x = torch.randn(128, 3)
x = linear(x)
print(x.shape)
```

will get:

```bash
torch.Size([128, 12])
```

## nn.Dropout

For fully connected layer. Using the samples in the Bernoulli distribution, some elements of the input tensor are randomly zeroed with probability $p$. To use it:

```python
dropout = nn.Dropout(p=0.5, inplace=False)
x = dropout(x)
```

`x` can be a tensor in any shape.

## nn.ReLU or F.relu

Activation function, $\text{ReLU}(x)=\max{(0,x)}$, to use it：

```python
x = nn.ReLU(x)
```

or:

```python
x = F.relu(x)
```

`x` can be a tensor in any shape.

## nn.RNN

```python
input_size = 10
hidden_size = 20
num_layers = 2
seq_length = 5
batch_size = 3
rnn = nn.RNN(input_size, hidden_size, num_layers)
input_data = torch.randn(seq_length, batch_size, input_size)
output, hidden_state = rnn(input_data)
print(output.shape)
```

will get:

```bash
torch.Size([5, 3, 20])
```

## nn.Module

Construct a block of layers. It could be the entire model or just a block of the entire model or loss function, etc.

```python
class MyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        # define every layer
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        # define forward propagation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x
```

## nn.Sequential

Compared with `nn.Module`, `nn.Sequential` can add the layers more easily and don't have to define forward propagation. This is more useful when building a simple neural network.

For example, to construct a model with a 2d convolution layer, a ReLU layer and a 2d convolution layer:

```python
model = nn.Sequential(
    nn.Conv2d(1, 20, 5),
    nn.ReLU(),
    nn.Conv2d(20, 64, 5),
)

x = torch.randn(1, 1, 30, 30)
y = model(x)
print(y.shape)
```

will get:

```bash
torch.Size([1, 64, 22, 22])
```

## The Smallest Framework

In practice, the whole project often looks like this:

```
├── data
│   ├── processed
│   └── raw
├── src
│   ├── __init__.py
│   ├── config.json
│   ├── loss.py
│   ├── models.py
│   └── utils.py
├── LICENSE
├── predict.py
├── prepare_dataset.py
├── README.md
├── requirements.txt
└── train.py
```

But this is just a simple model, so it's just one file.

```python
class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Define every layer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Define forward propagation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = MyNetwork()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

num_epochs = 10
inputs = torch.randn(64, 3, 32, 32)
labels = torch.randint(0, 10, (64,))

# Train the model
for epoch in range(num_epochs):
    # Forward propagation
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    # Backward propagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the training process
    if (epoch + 1) % 2 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

# Predict
with torch.no_grad():
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    print(outputs.shape)
```

will get:

```bash
Epoch [2/10], Loss: 2.2880
Epoch [4/10], Loss: 2.2841
Epoch [6/10], Loss: 2.2805
Epoch [8/10], Loss: 2.2769
Epoch [10/10], Loss: 2.2734
torch.Size([64, 10])
```

Start with torch's example [mnist](https://github.com/pytorch/examples/blob/main/mnist) (or [_What is torch.nn really?_](https://pytorch.org/tutorials/beginner/nn_tutorial.html)) is a great idea.
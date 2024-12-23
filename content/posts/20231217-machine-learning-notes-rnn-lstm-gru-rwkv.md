---
author: "Chen Li"
title: "Machine Learning Notes: RNN, LSTM, GRU, RWKV"
date: "2023-12-17"
tags: 
- programming
math: true
tabsets: true
---

"Recurrent" means that, hidden state $h_t$ is a function of the current input $x_t$ and the last hidden state $h_{t-1}$:$$h_t=f(x_t, h_{t-1}; \theta)$$where $\theta$ is all the trainable parameters. We iterate over each word (sub-word) in the entire sequence.

## §1 RNN

[`nn.RNNCell`](https://pytorch.org/docs/stable/generated/torch.nn.RNNCell.html), [`nn.RNN`](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)

RNN (or vanilla RNN) is composed of 2 Linear layers and an activation function: $$h_t = \tanh(x_t W_{ih}^T + b_{ih} + h_{t-1}W_{hh}^T + b_{hh})$$

Note that in the figure below each square represents the same parameters.

![RNNScratch](20231217-machine-learning-notes-rnn-lstm-gru-rwkv-rnnscratch.svg)

<div class="tabset"></div>

- `RNNScratch`
    
    ```python
    class RNNScratch(nn.Module):
        def __init__(self, input_size, hidden_size, nonlinearity='tanh', bias=True):
            super().__init__()
            self.w_ih = nn.Linear(input_size, hidden_size, bias=bias)
            self.w_hh = nn.Linear(hidden_size, hidden_size, bias=bias)
            if nonlinearity == 'tanh':
                self.act = nn.Tanh()
            elif nonlinearity == 'relu':
                self.act = nn.ReLU()
    
        def forward(self, input, h0):
            output = torch.Tensor([])
            hn = h0
    
            # Iterate over the sequence of input
            for x in input:
                # print('x shape:', x.shape)# [batch_size, input_size]
                hn = self.act(self.w_ih(x) + self.w_hh(hn))
                # print('hn shape:', hn.shape)# [D * num_layers, batch_size, hidden_size]
                output = torch.cat((output, hn), dim=0)
                # print('output shape:', output.shape)# [seq_length, batch_size, D * hidden_size]
    
            # return, same as `torch.nn.RNN`
            return output, hn
    ```

- testing
    
    ```python
    rnn_scratch = RNNScratch(input_size=10, hidden_size=20)
    
    input = torch.randn(5, 16, 10)# [seq_length, batch_size, input_size]
    h0 = torch.zeros(1, 16, 20)# [D * num_layers, batch_size, hidden_size]
    output, hn = rnn_scratch(input, h0)
    print('------')
    print('output shape:', output.shape)# [seq_length, batch_size, D * hidden_size]
    print('hn shape:', hn.shape)# same as h0
    print(torch.all(output[4:5]==hn))# y_{t-1} == h_t
    ```
    
    will get:
    
    ```bash
    x shape: torch.Size([16, 10])
    hn shape: torch.Size([1, 16, 20])
    output shape: torch.Size([1, 16, 20])
    x shape: torch.Size([16, 10])
    hn shape: torch.Size([1, 16, 20])
    output shape: torch.Size([2, 16, 20])
    x shape: torch.Size([16, 10])
    hn shape: torch.Size([1, 16, 20])
    output shape: torch.Size([3, 16, 20])
    x shape: torch.Size([16, 10])
    hn shape: torch.Size([1, 16, 20])
    output shape: torch.Size([4, 16, 20])
    x shape: torch.Size([16, 10])
    hn shape: torch.Size([1, 16, 20])
    output shape: torch.Size([5, 16, 20])
    ------
    output shape: torch.Size([5, 16, 20])
    hn shape: torch.Size([1, 16, 20])
    tensor(True)
    ```

- `nn.RNN`
    
    ```python
    rnn = nn.RNN(input_size=10, hidden_size=20)
    
    input = torch.randn(5, 16, 10)# [seq_length, batch_size, input_size]
    h0 = torch.zeros(1, 16, 20)# [D * num_layers, batch_size, hidden_size]
    output, hn = rnn(input, h0)
    print(output.shape)# [seq_length, batch_size, D * hidden_size]
    print(hn.shape)# same as h0
    print(torch.all(output[4:5]==hn))# y_{t-1} == h_t
    ```
    
    will get:
    
    ```bash
    torch.Size([5, 16, 20])
    torch.Size([1, 16, 20])
    tensor(True)
    ```

### §1.1 `num_layers`

![RNNDeep](20231217-machine-learning-notes-rnn-lstm-gru-rwkv-rnndeep.svg)

<div class="tabset"></div>

- `RNNDeep`
    
    ```python
    class RNNDeep(nn.Module):
        def __init__(self, input_size, hidden_size, nonlinearity='tanh', bias=True, num_layers=1):
            super().__init__()
            self.num_layers = num_layers
            self.rnns = nn.ModuleList([
                RNNScratch(
                    input_size if i==0 else hidden_size,
                    hidden_size,
                    nonlinearity,
                    bias
                )
                for i in range(num_layers)
            ])
    
        def forward(self, input, h0):
            output = input
            hn = h0
    
            # Iterate over rnns
            for i in range(self.num_layers):
                # print(hn[i:i+1].shape)# [1, batch_size, hidden_size]
                output, hn[i:i+1] = self.rnns[i](output, hn[i:i+1])
    
            # return, same as `torch.nn.RNN`
            return output, hn
    ```

- testing
    
    ```python
    rnn_deep = RNNDeep(input_size=10, hidden_size=20, num_layers=3)
    
    input = torch.randn(5, 16, 10)# [seq_length, batch_size, input_size]
    h0 = torch.zeros(3, 16, 20)# [D * num_layers, batch_size, hidden_size]
    output, hn = rnn_deep(input, h0)
    print(output.shape)# [seq_length, batch_size, D * hidden_size]
    print(hn.shape)# same as h0
    ```
    
    will get:
    
    ```bash
    torch.Size([5, 16, 20])
    torch.Size([3, 16, 20])
    ```

- `nn.RNN`
    
    ```python
    rnn = nn.RNN(input_size=10, hidden_size=20, num_layers=3)
    
    input = torch.randn(5, 16, 10)# [seq_length, batch_size, input_size]
    h0 = torch.zeros(3, 16, 20)# [D * num_layers, batch_size, hidden_size]
    output, hn = rnn(input, h0)
    print(output.shape)# [seq_length, batch_size, D * hidden_size]
    print(hn.shape)# same as h0
    ```
    
    will get:
    
    ```bash
    torch.Size([5, 16, 20])
    torch.Size([3, 16, 20])
    ```

### §1.2 `bidirectional`

`bidirecitonal=True` $\to D=2$

![RNNBidirectional](20231217-machine-learning-notes-rnn-lstm-gru-rwkv-rnnbidirectional.svg)

<div class="tabset"></div>

- `RNNBidirectional`
    
    ```python
    class RNNBidirectional(nn.Module):
        def __init__(self, input_size, hidden_size, nonlinearity='tanh', bias=True, num_layers=1):
            super().__init__()
            self.num_layers = num_layers# number of layers on one side
            self.rnn_f = RNNDeep(input_size, hidden_size, nonlinearity, bias, num_layers)
            self.rnn_b = RNNDeep(input_size, hidden_size, nonlinearity, bias, num_layers)
    
        def forward(self, input, h0):
            seq_length = input.shape[0]
    
            # hn_f, hn_b = h0[:self.num_layers], h0[self.num_layers:]
            hn_f = torch.cat(
                [h0[2*i:2*i+1] for i in range(self.num_layers)],
                dim=0
            )
            hn_b = torch.cat(
                [h0[2*i+1:2*i+2] for i in range(self.num_layers)],
                dim=0
            )
    
            output_f, hn_f = self.rnn_f(input, hn_f)
            output_b, hn_b = self.rnn_b(torch.flip(input, dims=[0]), hn_b)
            output_b = torch.flip(output_b, dims=[0])
    
            output = torch.Tensor([])
            # concat every y_i and z_i
            for i in range(seq_length):
                output_i = torch.cat((output_f[i:i+1], output_b[i:i+1]), dim=2)
                output = torch.cat((output, output_i), dim=0)
    
            hn = torch.Tensor([])
            # concat h_n, (h_n)', (h_n)'', ...
            for i in range(self.num_layers):
                hn_i = torch.cat((hn_f[i:i+1], hn_b[i:i+1]), dim=0)
                hn = torch.cat((hn, hn_i), dim=0)
    
            return output, hn
    ```

- testing
    
    ```python
    rnn_bidirectional = RNNBidirectional(input_size=10, hidden_size=20, num_layers=3)
    
    input = torch.randn(5, 16, 10)# [seq_length, batch_size, input_size]
    h0 = torch.zeros(6, 16, 20)# [D * num_layers, batch_size, hidden_size]
    output, hn = rnn_bidirectional(input, h0)
    print(output.shape)# [seq_length, batch_size, D * hidden_size]
    print(hn.shape)# same as h0
    print(torch.all(output[4:5, :, :20]==hn[4:5]))# ouput[seq_length-1:seq_length, :, :hidden_size] == hn[D * num_layers-2:D * num_layers-1]
    print(torch.all(output[0:1, :, 20:]==hn[5:6]))# ouput[0:1, :, hidden_size:] == hn[D * num_layers-1:D * num_layers]
    ```
    
    will get:
    
    ```bash
    torch.Size([5, 16, 40])
    torch.Size([6, 16, 20])
    tensor(True)
    tensor(True)
    ```

- `nn.RNN`
    
    ```python
    rnn = nn.RNN(input_size=10, hidden_size=20, num_layers=3, bidirectional=True)
    
    input = torch.randn(5, 16, 10)# [seq_length, batch_size, input_size]
    h0 = torch.zeros(6, 16, 20)# [D * num_layers, batch_size, hidden_size]
    output, hn = rnn(input, h0)
    print(output.shape)# [seq_length, batch_size, D * hidden_size]
    print(hn.shape)# same as h0
    print(torch.all(output[4:5, :, :20]==hn[4:5]))# ouput[seq_length-1:seq_length, :, :hidden_size] == hn[D * num_layers-2:D * num_layers-1]
    print(torch.all(output[0:1, :, 20:]==hn[5:6]))# ouput[0:1, :, hidden_size:] == hn[D * num_layers-1:D * num_layers]
    ```
    
    will get:
    
    ```bash
    torch.Size([5, 16, 40])
    torch.Size([6, 16, 20])
    tensor(True)
    tensor(True)
    ```

So we introduced two ways to stack up the layers: adding layers is like parallel connection in circuit; bidirectional is like series connection. In the following we will not explicitly write how to do these two ways, because (a) the code is pretty much the same; (b) `torch.nn.RNN`, `torch.nn.LSTM` and `torch.nn.GRU` do these implementations in C++ and CUDA, thus faster.

## §2 LSTM

[`nn.LSTMCell`](https://pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html), [`nn.LSTM`](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)

LSTM introduces input gate $i_t$, forget gate $f_t$, cell gate $g_t$ and output gate $o_t$, which are functions of the current input $x_t$ and the last hidden state $h_{t-1}$: $$\begin{aligned} i_t &= \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\\ f_t &= \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\\ g_t &= \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\\ o_t &= \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \end{aligned}$$then the current cell state $c_t$ and the current hidden state $h_t$ are:$$\begin{aligned} c_t &= f_t \odot c_{t-1} + i_t \odot g_t \\\ h_t &= o_t \odot \tanh(c_t) \end{aligned}$$where $\sigma$ is the [sigmoid function](https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html) $\sigma(x)=\frac{1}{1+e^{-x}} \in (0, 1)$ and $\odot$ is the [Hadamard product](https://en.wikipedia.org/wiki/Hadamard_product_(matrices)). I find [this post](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) and [this summary paper](https://arxiv.org/abs/2410.01201) explaining the design motivation really well. By using sigmoid, the mechanism of these gates is that they are close to masks that marks what to forget and what to remember.

<div class="tabset"></div>

- `LSTMScratch`
    
    ```python
    class LSTMScratch(nn.Module):
        def __init__(self, input_size, hidden_size, bias=True):
            super().__init__()
            self.w_ii = nn.Linear(input_size, hidden_size, bias=bias)
            self.w_hi = nn.Linear(hidden_size, hidden_size, bias=bias)
            self.w_if = nn.Linear(input_size, hidden_size, bias=bias)
            self.w_hf = nn.Linear(hidden_size, hidden_size, bias=bias)
            self.w_ig = nn.Linear(input_size, hidden_size, bias=bias)
            self.w_hg = nn.Linear(hidden_size, hidden_size, bias=bias)
            self.w_io = nn.Linear(input_size, hidden_size, bias=bias)
            self.w_ho = nn.Linear(hidden_size, hidden_size, bias=bias)
            self.sigmoid = nn.Sigmoid()
            self.tanh = nn.Tanh()
    
        def forward(self, input, h0, c0):
            output = torch.Tensor([])
            hn, cn = h0, c0
    
            # Iterate over the sequence of input
            for x in input:
                # print('x shape: ', x.shape)# [batch_size, input_size]
                i_t = self.sigmoid(self.w_ii(x) + self.w_hi(hn))
                f_t = self.sigmoid(self.w_if(x) + self.w_hf(hn))
                g_t = self.tanh(self.w_ig(x) + self.w_hg(hn))
                o_t = self.sigmoid(self.w_io(x) + self.w_ho(hn))
                cn = f_t * cn + i_t * g_t
                hn = o_t * self.tanh(cn)
                output = torch.cat((output, hn), dim=0)# only hn is in the output
    
            # return, same as `torch.nn.LSTM`
            return output, (hn, cn)
    ```

- testing
    
    ```python
    lstm_scratch = LSTMScratch(input_size=10, hidden_size=20)
    
    input = torch.randn(5, 16, 10)# [seq_length, batch_size, input_size]
    h0 = torch.zeros(1, 16, 20)# [D * num_layers, batch_size, hidden_size]
    c0 = torch.zeros(1, 16, 20)# [D * num_layers, batch_size, hidden_size]
    output, (hn, cn) = lstm_scratch(input, h0, c0)
    print(output.shape)# [seq_length, batch_size, hidden_size]
    print(hn.shape)# same as h0
    print(cn.shape)# same as c0
    ```
    
    will get:
    
    ```bash
    torch.Size([5, 16, 20])
    torch.Size([1, 16, 20])
    torch.Size([1, 16, 20])
    ```

- `nn.LSTM`
    
    ```python
    lstm = nn.LSTM(input_size=10, hidden_size=20)
    
    input = torch.randn(5, 16, 10)# [seq_length, batch_size, input_size]
    h0 = torch.zeros(1, 16, 20)# [D * num_layers, batch_size, hidden_size]
    c0 = torch.zeros(1, 16, 20)# [D * num_layers, batch_size, hidden_size]
    output, (hn, cn) = lstm(input, (h0, c0))
    print(output.shape)# [seq_length, batch_size, hidden_size]
    print(hn.shape)# same as h0
    print(cn.shape)# same as c0
    ```
    
    will get:
    
    ```bash
    torch.Size([5, 16, 20])
    torch.Size([1, 16, 20])
    torch.Size([1, 16, 20])
    ```

### §2.1 `proj_size`

<div class="tabset"></div>

- `LSTMProj`
    
    ```python
    class LSTMProj(nn.Module):
        def __init__(self, input_size, hidden_size, proj_size, bias=True):
            super().__init__()
            self.w_ii = nn.Linear(input_size, hidden_size, bias=bias)
            self.w_hi = nn.Linear(proj_size, hidden_size, bias=bias)# proj_size
            self.w_if = nn.Linear(input_size, hidden_size, bias=bias)
            self.w_hf = nn.Linear(proj_size, hidden_size, bias=bias)# proj_size
            self.w_ig = nn.Linear(input_size, hidden_size, bias=bias)
            self.w_hg = nn.Linear(proj_size, hidden_size, bias=bias)# proj_size
            self.w_io = nn.Linear(input_size, hidden_size, bias=bias)
            self.w_ho = nn.Linear(proj_size, hidden_size, bias=bias)# proj_size
            self.sigmoid = nn.Sigmoid()
            self.tanh = nn.Tanh()
            self.w_hr = nn.Linear(hidden_size, proj_size, bias=bias)
    
        def forward(self, input, h0, c0):
            output = torch.Tensor([])
            hn, cn = h0, c0
    
            # Iterate over the sequence of input
            for x in input:
                # print('x shape: ', x.shape)# [batch_size, input_size]
                i_t = self.sigmoid(self.w_ii(x) + self.w_hi(hn))
                f_t = self.sigmoid(self.w_if(x) + self.w_hf(hn))
                g_t = self.tanh(self.w_ig(x) + self.w_hg(hn))
                o_t = self.sigmoid(self.w_io(x) + self.w_ho(hn))
                cn = f_t * cn + i_t * g_t
                hn = o_t * self.tanh(cn)
                hn = self.w_hr(hn)
                output = torch.cat((output, hn), dim=0)# only hn is in the output
    
            # return, same as `torch.nn.LSTM`
            return output, (hn, cn)
    ```

- testing
    
    ```python
    lstm_proj = LSTMProj(input_size=10, hidden_size=20, proj_size=15)
    
    input = torch.randn(5, 16, 10)# [seq_length, batch_size, input_size]
    h0 = torch.zeros(1, 16, 15)# [D * num_layers, batch_size, proj_size]
    c0 = torch.zeros(1, 16, 20)# [D * num_layers, batch_size, hidden_size]
    output, (hn, cn) = lstm_proj(input, h0, c0)
    print(output.shape)# [seq_length, batch_size, proj_size]
    print(hn.shape)# same as h0
    print(cn.shape)# same as c0
    ```
    
    will get:
    
    ```bash
    torch.Size([5, 16, 15])
    torch.Size([1, 16, 15])
    torch.Size([1, 16, 20])
    ```

- `nn.LSTM`
    
    ```python
    lstm = nn.LSTM(input_size=10, hidden_size=20, proj_size=15)
    
    input = torch.randn(5, 16, 10)# [seq_length, batch_size, input_size]
    h0 = torch.zeros(1, 16, 15)# [D * num_layers, batch_size, proj_size]
    c0 = torch.zeros(1, 16, 20)# [D * num_layers, batch_size, hidden_size]
    output, (hn, cn) = lstm(input, (h0, c0))
    print(output.shape)# [seq_length, batch_size, proj_size]
    print(hn.shape)# same as h0
    print(cn.shape)# same as c0
    ```
    
    will get:
    
    ```bash
    torch.Size([5, 16, 15])
    torch.Size([1, 16, 15])
    torch.Size([1, 16, 20])
    /usr/local/lib/python3.10/dist-packages/torch/nn/modules/rnn.py:879: UserWarning: LSTM with projections is not supported with oneDNN. Using default implementation. (Triggered internally at ../aten/src/ATen/native/RNN.cpp:1492.)
      result = _VF.lstm(input, hx, self._flat_weights, self.bias, self.num_layers,
    ```
    
    (the warning is about [`oneDNN`](https://github.com/oneapi-src/oneDNN).)
    

## §3 GRU

[`nn.GRUCell`](https://pytorch.org/docs/stable/generated/torch.nn.GRUCell.html), [`nn.GRU`](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html)

GRU introduces reset gate $r_t$, update gate $z_t$ and new gate $n_t$, which are functions of the current input $x_t$ and the last hidden state $h_{t-1}$$$\begin{aligned} r_t &= \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{t-1} + b_{hr}) \\\ z_t &= \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{t-1} + b_{hz}) \\\ n_t &= \tanh(W_{in} x_t + b_{in} + r_t \odot (W_{hn} h_{t-1}+ b_{hn})) \end{aligned}$$then the current hidden state $h_t$ is$$h_t = (1 - z_t) \odot n_t + z_t \odot h_{t-1}$$

<div class="tabset"></div>

- `GRUScratch`
    
    ```python
    class GRUScratch(nn.Module):
        def __init__(self, input_size, hidden_size, bias=True):
            super().__init__()
            self.w_ir = nn.Linear(input_size, hidden_size, bias=bias)
            self.w_hr = nn.Linear(hidden_size, hidden_size, bias=bias)
            self.w_iz = nn.Linear(input_size, hidden_size, bias=bias)
            self.w_hz = nn.Linear(hidden_size, hidden_size, bias=bias)
            self.w_in = nn.Linear(input_size, hidden_size, bias=bias)
            self.w_hn = nn.Linear(hidden_size, hidden_size, bias=bias)
            self.sigmoid = nn.Sigmoid()
            self.tanh = nn.Tanh()
    
        def forward(self, input, h0):
            output = torch.Tensor([])
            hn = h0
    
            # Iterate over the sequence of input
            for x in input:
                # print('x shape: ', x.shape)# [batch_size, input_size]
                r_t = self.sigmoid(self.w_ir(x) + self.w_hr(hn))
                z_t = self.sigmoid(self.w_iz(x) + self.w_hz(hn))
                n_t = self.tanh(self.w_in(x) + r_t * self.w_hn(hn))
                hn = (1 - z_t) * n_t + z_t * hn
                output = torch.cat((output, hn), dim=0)
    
            # return, same as `torch.nn.LSTM`
            return output, hn
    ```

- testing
    
    ```python
    gru_scratch = GRUScratch(input_size=10, hidden_size=20)
    
    input = torch.randn(5, 16, 10)# [seq_length, batch_size, input_size]
    h0 = torch.zeros(1, 16, 20)# [D * num_layers, batch_size, hidden_size]
    output, hn = gru_scratch(input, h0)
    print(output.shape)# [seq_length, batch_size, hidden_size]
    print(hn.shape)# same as h0
    ```
    
    will get:
    
    ```bash
    torch.Size([5, 16, 20])
    torch.Size([1, 16, 20])
    ```

- `nn.GRU`
    
    ```python
    gru = nn.GRU(input_size=10, hidden_size=20)
    
    input = torch.randn(5, 16, 10)# [seq_length, batch_size, input_size]
    h0 = torch.zeros(1, 16, 20)# [D * num_layers, batch_size, hidden_size]
    output, hn = gru(input, h0)
    print(output.shape)# [seq_length, batch_size, hidden_size]
    print(hn.shape)# same as h0
    ```
    
    will get:
    
    ```bash
    torch.Size([5, 16, 20])
    torch.Size([1, 16, 20])
    ```

## §4 RWKV: The RNN Strikes Back

| [[2305.13048] _RWKV: Reinventing RNNs for the Transformer Era_](https://arxiv.org/abs/2305.13048) | [RWKV-LM (GitHub)](https://github.com/BlinkDL/RWKV-LM) | [nanoRWKV (GitHub)](https://github.com/BlinkDL/nanoRWKV) | [rwkv-cpp-accelerated (GitHub)](https://github.com/harrisonvanderbyl/rwkv-cpp-accelerated) |

Receptance Weighted Key Value (RWKV) combines the efficient parallelizable training of transformers with the efficient inference of RNNs. Generally speaking it's composed of several layers of Time Mix module and Channel Mix module. It can be considered as a convolutional network across an entire one-dimensional sequence (because $r_t$, $k_t$, $v_t$ does not contain non-linearity thus are weighted sum), which is the same thing we will see in SSM.

- The Time Mix module is linear projections ($W$) of linear combinations ($\mu$ and $(1-\mu)$) of the current input $x_t$ and the last input $x_{t-1}$, $wkv_t$ is weighted sum over the entire past sequence:$$\begin{aligned} r_t &= W_r (\mu_r \odot x_t + (1-\mu_r) \odot x_{t-1}) \\\ k_t &= W_k (\mu_k \odot x_t + (1-\mu_k) \odot x_{t-1}) \\\ v_t &= W_v (\mu_v \odot x_t + (1-\mu_v) \odot x_{t-1}) \\\ wkv_t &= \frac{ \sum_{i=1}^{t-1} e^{-(t-1-i)w+k_i} \odot v_i + e^{u+k_t} \odot v_t }{\sum_{i=1}^{t-1} e^{-(t-1-i)w+k_i} + e^{u+k_t}} \\\ o_t &= W_o (\sigma(r_t) \odot wkv_t) \end{aligned}$$

- The Channel Mix module is $$\begin{aligned} r_t &= W_r (\mu_r \odot x_t + (1 - \mu_r) \odot x_{t-1} ) \\\ k_t &= W_k (\mu_k \odot x_t + (1 - \mu_k) \odot x_{t-1} ) \\\ o_t &= \sigma(r_t) \odot (W_v \max({k_t}, 0)^2) \end{aligned}$$

The usage of $x_t$ and $x_{t-1}$ is the "Token shift" in Fig.3 of the paper:

![fig3_of_2305.13048](https://pbs.twimg.com/media/FwyDRLvaQAEBYok?format=png)

[RWKV-v6](https://twitter.com/BlinkDL_AI/status/1735258602473197721) looks scary...

![RWKV-v6](https://pbs.twimg.com/media/GBTgK7TbYAAeeCN?format=jpg)

## §5 RNNLanguageModel

The detailed code to train a language model is largely from [nanoGPT](https://github.com/karpathy/nanoGPT). Please refer to [gpt-fast](https://github.com/pytorch-labs/gpt-fast/) for training on a larger (like, way larger) dataset.

```python
class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_size=20, emb_size=10, hidden_size=20, window_size=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.pos_embedding = nn.Embedding(window_size, emb_size)
        self.rnn = RNNScratch(emb_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, index, targets=None):
        # index, targets shape: [batch_size, seq_length]
        batch_size, seq_length = index.shape
        # embedding
        embedding = self.embedding(index)# [batch_size, seq_length, emb_size]
        pos_embedding = self.pos_embedding(torch.arange(seq_length))# [seq_length, emb_size]
        x = embedding + pos_embedding# [batch_size, seq_length, emb_size]
        # RNN
        x = x.permute(1, 0, 2)# [seq_length, batch_size, emb_size]
        h0 = torch.zeros(1, batch_size, self.hidden_size)# [D * num_layers, batch_size, hidden_size]
        x, _ = self.rnn(x, h0) # x shape: [seq_length, batch_size, D * hidden_size]
        x = x.permute(1, 0, 2)# [batch_size, seq_length, D * hidden_size]
        # project out
        x = self.layer_norm(x)# [batch_size, seq_length, D * hidden_size]
        logits = self.proj(x)# [batch_size, seq_length, vocab_size]

        if targets is None:
            loss = None
        else:
            batch_size, seq_length, vocab_size = logits.shape
            logits = logits.view(batch_size*seq_length, vocab_size)
            targets = targets.view(batch_size*seq_length)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, index, max_new_tokens):
        # index shape [batch_size, seq_length]
        for _ in range(max_new_tokens):
            # crop index to the last window_size tokens
            index_cond = index[:, -window_size:]
            # get the predictions
            logits, loss = self(index_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # [batch_size, vocab_size]
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # [batch_size, vocab_size]
            # sample from the distribution
            index_next = torch.multinomial(probs, num_samples=1) # [batch_size, 1]
            # append sampled index to the running sequence
            index = torch.cat((index, index_next), dim=1) # [batch_size, seq_length+1]
        return index
```

### §4.1 Dataset

```python
text = '''
All work and no play makes Jack a doll boy. 
All work and no play makes Jack a doll boy. 
All work and no play makes Jack a doll boy. 
All work and no play makes Jack a doll boy. 
All work and no play makes Jack a doll boy. 
All work and no play makes Jack a doll boy. 
All work and no play makes Jack a doll boy. 
All work and no play makes Jack a doll boy. 
All work and no play makes Jack a doll boy. 
All work and no play makes Jack a doll boy. 
'''

# https://docs.python.org/3/library/re.html#raw-string-notation

# text = 'All work and no play makes Jack a doll boy. \nAll work and no play makes Jack a doll boy. \nAll work and no play makes Jack a doll boy. \nAll work and no play makes Jack a doll boy. \nAll work and no play makes Jack a doll boy. \nAll work and no play makes Jack a doll boy. \nAll work and no play makes Jack a doll boy. \nAll work and no play makes Jack a doll boy. \nAll work and no play makes Jack a doll boy. \nAll work and no play makes Jack a doll boy. \n'

# text = r'All work and no play makes Jack a doll boy. \nAll work and no play makes Jack a doll boy. \nAll work and no play makes Jack a doll boy. \nAll work and no play makes Jack a doll boy. \nAll work and no play makes Jack a doll boy. \nAll work and no play makes Jack a doll boy. \nAll work and no play makes Jack a doll boy. \nAll work and no play makes Jack a doll boy. \nAll work and no play makes Jack a doll boy. \nAll work and no play makes Jack a doll boy. \n'

print(text[:100])
```

will get:

```bash
All work and no play makes Jack a doll boy. 
All work and no play makes Jack a doll boy. 
All work a
```

Here's a character-level tokenizer, please refer to [`tiktoken`](https://github.com/openai/tiktoken) or [`minbpe`](https://github.com/karpathy/minbpe/) (BPE is combining words to get pharses) for a sub-word-level tokenizer.

```python
characters = sorted(list(set(text)))
vocab_size = len(characters)

print(''.join(characters))
print(vocab_size)
```

will get:

```bash

.AJabcdeklmnoprswy
20
```

```python
string_to_int = { character:integer for integer,character in enumerate(characters) }
int_to_string = { interger:character for interger,character in enumerate(characters) }
encode = lambda strings: [string_to_int[string] for string in strings] # string -> a list of integers
decode = lambda ints: ''.join([int_to_string[integer] for integer in ints]) # a list of integers -> string

print(encode('cJ'))
print(decode([7, 4]))
```

will get:

```bash
[7, 4]
cJ
```

```python
data = torch.Tensor(encode(text)).to(torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]
```

```python
batch_size = 4
window_size = 8

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    if split == 'train':
        data = train_data
    else:
        data = val_data

    ix = torch.randint(len(data) - window_size, (batch_size,))
    x = torch.stack([data[i:i+window_size] for i in ix])# [batch_size, window_size]
    y = torch.stack([data[i+1:i+window_size+1] for i in ix])# [batch_size, window_size]
    return x, y
```

### §4.2 Train and Predict

```python
eval_iters = 200

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
```

```python
torch.manual_seed(2001)

model = RNNLanguageModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# train
max_iters = 2000
eval_interval = 100
for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```

will get:

```bash
step 0: train loss 3.0177, val loss 3.0082
step 100: train loss 2.2342, val loss 2.1759
step 200: train loss 1.5963, val loss 1.5902
step 300: train loss 1.1176, val loss 1.1208
step 400: train loss 0.7405, val loss 0.7353
step 500: train loss 0.5035, val loss 0.4992
step 600: train loss 0.3738, val loss 0.3645
step 700: train loss 0.3333, val loss 0.3219
step 800: train loss 0.2738, val loss 0.2673
step 900: train loss 0.2465, val loss 0.2517
step 1000: train loss 0.2248, val loss 0.2286
step 1100: train loss 0.2163, val loss 0.2102
step 1200: train loss 0.1975, val loss 0.2043
step 1300: train loss 0.1981, val loss 0.1930
step 1400: train loss 0.1873, val loss 0.1965
step 1500: train loss 0.1792, val loss 0.1813
step 1600: train loss 0.1752, val loss 0.1775
step 1700: train loss 0.1713, val loss 0.1809
step 1800: train loss 0.1673, val loss 0.1762
step 1900: train loss 0.1762, val loss 0.1739
step 1999: train loss 0.1697, val loss 0.1724
```

```python
# generate from the trained model
context = torch.zeros((1, 1)).to(torch.long)
# context = torch.unsqueeze(torch.Tensor(encode('cJ')).to(torch.long), dim=0)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
```

will get Jack Torrance played by Jack Nicholson:

```bash

All boy. 
All work and no plakes Jack a doll boy. 
All work and no play. 
All work and no play makesl play makes Jack a doll boy. 
All work and no play makes Jack a doll boy.. 
All work and no play makes Jack a doll boy. 
All work and no play makes Jack a doll boy. 
All work and no play makes Jack a Jack a doll boy. 
All work and no play makes Jack a doll boy. 
All work amakes Jack a doll boy. 
All work ank a doll boy. 
Alak a d no playes Jack a doll boy. 
All work and no play makes Jack a doll
```

By the way, I guess the reasons Transformers outperform RNNs are:
- Transformers don't use `for` loops thus are more parallel-processing-friendly. (There is this new architecture with Transformers that the same single Transformer layer is repeated several times, which is like using Transformer layer as RNN layer.)
- Transformers can look into the entire sequence equally while RNNs focus more on the current input of the entire sequence.

On the other hand, the advantages of RNNs are:
- The scaling of memory usage is linear (Transformers are quadratically, please refer to Tab.1 of [[2305.13048] _RWKV: Reinventing RNNs for the Transformer Era_](https://arxiv.org/abs/2305.13048)).
- The memory usage is constant, because `hn` is passed down.
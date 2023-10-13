---
author: "Chen Li"
title: "Machine Learning Notes: GraphNeT"
date: "2023-09-10"
tags: 
- Programming
math: true
---

Technically this is only part of the entire blog post. My `.ipynb` are available at GitHub repository [blog-graphnet](https://github.com/ChenLi2049/blog-graphnet), see §3.

## §1 Graph & Matrix

Graphs can be represented by adjacency matrix, which can be normalized into [Frobenius normal form](https://en.wikipedia.org/wiki/Frobenius_normal_form), see [_Matrices and graphs_ - by Tivadar Danka - The Palindrome](https://thepalindrome.org/p/matrices-and-graphs).

![](https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F485ec4b4-6869-43bb-b17c-e2151a428dbe_1920x1080.png)

Often, large part of the matrix is empty, which is called [Sparse matrix](https://en.wikipedia.org/wiki/Sparse_matrix).

## §2 GNN

See [_A Gentle Introduction to Graph Neural Networks_ - Distill.pub](https://distill.pub/2021/gnn-intro/).

After GNN, part of the nodes are highlighted and the number of features of each nodes can be different. This is in consistence with CNN, where parts of the picture is highlighted and the number of channels (RGB $\rightarrow$ many more) can be different (see [CNN Explainer](https://poloclub.github.io/cnn-explainer/)).

For [Graph Attention Networks (GAT)](https://arxiv.org/abs/1710.10903), see [Graph Attention Networks (GAT) (labml.ai)](https://nn.labml.ai/graphs/gat/index.html) and [Graph Attention Networks v2 (GATv2) (labml.ai)](https://nn.labml.ai/graphs/gatv2/index.html).

## §3 `GraphNeT`

This section is about GitHub repository [`GraphNeT`](https://github.com/graphnet-team/graphnet).

See their [getting started document](https://github.com/graphnet-team/graphnet/blob/main/GETTING_STARTED.md), especially [the example](https://github.com/graphnet-team/graphnet/blob/main/GETTING_STARTED.md#example-energy-reconstruction-using-modelconfig). In the following article I will focus on the model rather than how to load the data or how to train the model with `pytorch-lightning`, and I will test these models with dummies. Because I just understand things better in this way and I haven't used these models in practice.

And my notes are in GitHub repository [blog-graphnet](https://github.com/ChenLi2049/blog-graphnet). _In order to be actually used, the models in §3.2 and §3.3 need to be followed by a Fully Connected Layer._

### §3.1 Shape of the Data

See [4. The `Dataset` and `DataLoader` classes](https://github.com/graphnet-team/graphnet/blob/main/GETTING_STARTED.md#4-the-dataset-and-dataloader-classes) of the getting started document:

>- `graph.x`: Node feature matrix with shape `[num_nodes, num_features]`  
>- `graph.edge_index`: Graph connectivity in [COO format](https://pytorch.org/docs/stable/sparse.html#sparse-coo-docs) with shape `[2, num_edges]` and type `torch.long` (by default this will be `None`, i.e., the nodes will all be disconnected).

### §3.2 `ConvNet`

[`graphnet.models.gnn.ConvNet`](https://github.com/graphnet-team/graphnet/blob/main/src/graphnet/models/gnn/convnet.py) is based on [`torch_geometric.nn.TAGConv`](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.TAGConv.html). For the structure of the model, see Fig.3 of [[2107.12187] _Reconstruction of Neutrino Events in IceCube using Graph Neural Networks_](https://arxiv.org/abs/2107.12187).

In line 46:
```python
self.conv1 = TAGConv(self.nb_inputs, self.nb_intermediate, 2)
```
This means that Number of hops $K=2$. When $K=1$, GNN only considers the information of directly adjacent nodes. When $K=2$, GNN takes into account the information of first-order and second-order adjacent nodes. When $K=3$, GNN considers the information of first-order, second-order, and third-order adjacent nodes, and so on. Of course, after few `TAGConv` layers, the output can gather the information of more than second-order adjacent of the original graph.

In [2107.12187], it's stated that:

>In the simplest way, each pulse can be described by following quantities: the hit DOM, and therefore the position of the pulse, collected charge, and time recorded. ... We can then represent each event by a graph, with the nodes representing the pulses in an abstract 5-dimensional (or 12-dimensional for the Upgrade) space.

Therefore, the event is an input matrix in the shape of `[number_of_pulses, 5]`, where `5` is x, y, z, time, charge.

### §3.3 `DynEdge`

#### §3.3.1 `DynEdgeConv`

The basic layer of `DynEdge` is [`graphnet.models.components.layers.DynEdgeConv`](https://github.com/graphnet-team/graphnet/blob/main/src/graphnet/models/components/layers.py), which is based on [`torch_geometric.nn.EdgeConv`](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.EdgeConv.html).

#### §3.3.2 `DynEdge`

[`graphnet.models.gnn.DynEdge`](https://github.com/graphnet-team/graphnet/blob/main/src/graphnet/models/gnn/dynedge.py)

#### §3.3.3 `DynEdgeJINST`

[`graphnet.models.gnn.DynEdgeJINST`](https://github.com/graphnet-team/graphnet/blob/main/src/graphnet/models/gnn/dynedge_jinst.py) is the model in Fig.2 of [[2209.03042] _Graph Neural Networks for Low-Energy Event Classification & Reconstruction in IceCube_](https://arxiv.org/abs/2209.03042). `[n, 6]` in the figure means the number of nodes is `n` and the number of features is `6`, while `[1, n_outputs]` means the prediction of this event has `n_outputs` features (azimuth, zenith, energy, etc.)

#### §3.3.4 `DynEdgeTITO`

[`graphnet.models.gnn.DynEdgeTITO`](https://github.com/graphnet-team/graphnet/blob/main/src/graphnet/models/gnn/dynedge_kaggle_tito.py) is from [1st Place Solution | Kaggle](https://www.kaggle.com/competitions/icecube-neutrinos-in-deep-ice/discussion/402976).

An modification for [`torch_geometric.nn.EdgeConv`](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.EdgeConv.html) is $$x_i^{\prime} = \sum_{j \in \mathcal{N}(i)} h_{\mathbf{\Theta}}(x_i , x_j - x_i)$$ to $$x_i^{\prime} = \sum_{j \in \mathcal{N}(i)} h_{\mathbf{\Theta}}(x_i , x_j - x_i , x_j)$$

And this solution also uses custom VMF Loss (not in [`graphnet.models.gnn.DynEdgeTITO`](https://github.com/graphnet-team/graphnet/blob/main/src/graphnet/models/gnn/dynedge_kaggle_tito.py) of course): $$\text{VMF}=- \kappa \cos{\theta}+C(\kappa)$$ to $$\text{VMF}= - \theta - \kappa \cos{\theta}+C(\kappa)$$, where $\theta$ is the angle between truth and prediction, $\kappa$ is the length of the 3D prediction.

### §3.4 `Model` & `StandardModel`

[`graphnet.models.Model`](https://github.com/graphnet-team/graphnet/blob/main/src/graphnet/models/model.py) and [`graphnet.models.StandardModel`](https://github.com/graphnet-team/graphnet/blob/main/src/graphnet/models/standard_model.py) are [wrappers](https://en.wikipedia.org/wiki/Wrapper_function) choosing models and goals.
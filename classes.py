{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "classes",
      "version": "0.3.2",
      "provenance": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/MicroprocessorX069/Stroke-Prediction-/blob/master/classes.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "metadata": {
        "id": "qw6UbBzbj6ev",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "import torch\n",
        "from torch.utils.data import Dataset, DataLoader\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import torch.nn as nn\n",
        "import torch.nn.functional as F\n",
        "import os"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "Kx6VlFLvjtAY",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "class TabularDataset(Dataset):\n",
        "  def __init__(self, data, cat_cols=None, output_col=None):\n",
        "    \n",
        "    self.n=data.shape[0]\n",
        "    \n",
        "    if output_col: \n",
        "      self.y=data[output_col].astype(np.int64).values.reshape(-1,1)\n",
        "    else: \n",
        "      self.y=np.zeros((self.n,1)) #all output =0 if output is not specified\n",
        "    \n",
        "    self.cat_cols=cat_cols if cat_cols else []\n",
        "    self.cont_cols=[col for col in data.columns if col not in self.cat_cols\n",
        "                   + [output_col]]\n",
        "    \n",
        "    if self.cont_cols:\n",
        "      self.cont_x=data[self.cont_cols].astype(np.float32).values\n",
        "    else:\n",
        "      self.cont_x=np.zeros((self.n,1))\n",
        "    \n",
        "    if self.cat_cols:\n",
        "      self.cat_x=data[cat_cols].astype(np.int64).values\n",
        "    else:\n",
        "      self.cat_x=np.zeros((self.n,1))\n",
        "      \n",
        "  def __len__(self):\n",
        "    return self.n\n",
        "  \n",
        "  def __getitem__(self,idx):\n",
        "    return [self.y[idx],self.cont_x[idx],self.cat_x[idx]]"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "j8bRkR8zj4dr",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        "# embedding emb i.e. categorical features\n",
        "class FeedForwardNN(nn.Module):\n",
        "  def __init__(self,emb_dims,no_of_cont, lin_layer_sizes,\n",
        "              output_size,emb_dropout, lin_layer_dropouts):\n",
        "    '''\n",
        "    #emb_dims: list of two tuples\n",
        "    #tuple1: no. of unqie values for that categorical variable\n",
        "    #tuple 2: shape of that features data\n",
        "    \n",
        "    #no_of_cont: number of continuous features\n",
        "    \n",
        "    lin_layer_sizes: list of integers\n",
        "    no. of nodes in each linear layer in the network\n",
        "    \n",
        "    output_size=Integer\n",
        "    size of final output\n",
        "    \n",
        "    emb_dropout: float\n",
        "    dropout used after embedding layers\n",
        "    \n",
        "    lin_layer_dropouts: \n",
        "    dropout after each linear layer\n",
        "    \n",
        "    '''\n",
        "    super().__init__()\n",
        "    \n",
        "    #Embedding layers or layers with categorical layers\n",
        "    self.emb_layers=nn.ModuleList([nn.Embedding(x,y)\n",
        "                                  for x,y in emb_dims])\n",
        "    # no of categorical features\n",
        "    no_of_embs=sum([y for x,y in emb_dims])\n",
        "    self.no_of_embs=no_of_embs\n",
        "    self.no_of_cont=no_of_cont\n",
        "    \n",
        "    #Linear Layers\n",
        "    first_lin_layer=nn.Linear(self.no_of_embs +self.no_of_cont,\n",
        "                             lin_layer_sizes[0])\n",
        "    self.lin_layers=nn.ModuleList([first_lin_layer]+\n",
        "                                 [nn.Linear(lin_layer_sizes[i],lin_layer_sizes[i+1])\n",
        "                                 for i in range(len(lin_layer_sizes)-1)])\n",
        "    \n",
        "    for lin_layer in self.lin_layers:\n",
        "      nn.init.kaiming_normal_(lin_layer.weight.data)\n",
        "     \n",
        "    #Output Layer\n",
        "    self.output_layer=nn.Linear(lin_layer_sizes[-1],\n",
        "                                               output_size)\n",
        "    nn.init.kaiming_normal_(self.output_layer.weight.data)\n",
        "\n",
        "    #BatchNorm Layer on continuous variables\n",
        "    self.first_bn_layer=nn.BatchNorm1d(self.no_of_cont)\n",
        "    self.bn_layers=nn.ModuleList([self.first_bn_layer]+\n",
        "                                [nn.BatchNorm1d(size) \n",
        "                                for size in lin_layer_sizes])\n",
        "    \n",
        "    #Dropout Layer\n",
        "    self.emb_dropout_layer=nn.Dropout(emb_dropout)\n",
        "    self.dropout_layers=nn.ModuleList([nn.Dropout(dropout)\n",
        "                                     for dropout in lin_layer_dropouts])\n",
        "    \n",
        "  def forward(self,cont_data,cat_data):\n",
        "    if self.no_of_embs!=0:\n",
        "      x = [emb_layer(cat_data)]"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "metadata": {
        "id": "zXSZ08rlbErJ",
        "colab_type": "code",
        "colab": {}
      },
      "cell_type": "code",
      "source": [
        ""
      ],
      "execution_count": 0,
      "outputs": []
    }
  ]
}
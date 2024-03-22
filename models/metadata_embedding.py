from torch import nn
from einops import rearrange
import torch
import numpy

from ldm.modules.diffusionmodules.util import SinusoidalEmbedding,create_condition_vector

import numpy as np
import torch

import torch.nn as nn

class MetadataMLP(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(MetadataMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, embedding_dim)
        # self.activation = nn.SiLU()
        # self.fc2=nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        out = self.fc1(x)
        # out = self.activation(out)
        # out = self.fc2(out)
        return out

class metadata_embeddings(nn.Module):
    def __init__(self, max_value,embedding_dim,max_period,metadata_dim):
        super().__init__()
        self.sinusoidal_embedding = SinusoidalEmbedding(max_value, embedding_dim)
        self.mlp_models = nn.ModuleList([MetadataMLP(embedding_dim, embedding_dim*4) for _ in range(metadata_dim)])
        self.max_period = max_period
        self.embedding_dim = embedding_dim
        self.metadata_dim = metadata_dim
        self.max_value=max_value


    def forward(self, metadata=None):
        while len(metadata)==1:
            metadata=metadata[0]
        if metadata.dim()==1:
            metadata=metadata.unsqueeze(0)
        embedded_metadata = self.sinusoidal_embedding(metadata)
        condition_vector = create_condition_vector(embedded_metadata, self.mlp_models, self.embedding_dim)
        return condition_vector
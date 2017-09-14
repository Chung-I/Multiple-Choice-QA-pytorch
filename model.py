import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pickle
import pdb
import argparse

class TextCNN(nn.Module):
  def __init__(self, args):
    super(TextCNN, self).__init__()
    self.args = args
    with open(args.embed_file, "rb") as f:
      embeddings = torch.from_numpy(np.load(f))
    self.embed = nn.Embedding(embeddings.size(0), embeddings.size(1))
    self.embed.weight.data.copy_(embeddings)
    #self.embed.weight.requires_grad = False
    #if args.cuda:
    #  self.embed = self.embed.cuda()
    self.convs1 = nn.ModuleList([nn.Conv2d(1, args.hidden_size, (k, args.embedding_size))
            for k in args.kernel_sizes])
    self.dropout = nn.Dropout(args.dropout)
    final_size = len(args.kernel_sizes)*args.hidden_size
    self.fc1 = nn.Bilinear(final_size, final_size, 1)

  def forward(self, dialogue, options):
    x = self.embed(dialogue)
    x = x.unsqueeze(1)
    x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
    x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
    x = torch.cat(x, 1)
    x = self.dropout(x)
    logits = []
    for option in options:
        y = self.embed(options)
        y = y.unsqueeze(1)
        y = [F.relu(conv(y)).squeeze(3) for conv in self.convs1]
        y = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in y]
        y = torch.cat(y, 1)
        y = self.dropout(y)
        logit = self.fc1(x,y)
        logits.append(logit)
    logits = torch.from_numpy(np.array(logits))
    return logits

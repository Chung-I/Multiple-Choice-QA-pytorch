import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pickle
import pdb
import argparse
import onmt

class Seq2seqModel(nn.Module):
  def __init__(self, args):
    super(Seq2seqModel, self).__init__()
    encoder = onmt.Models.Encoder(args)
    decoder = onmt.Models.Decoder(args)
    generator = nn.Sequential(
      nn.Linear(args.rnn_size, args.vocab_size),
      nn.LogSoftmax())

    model = onmt.Models.NMTModel(encoder, decoder)
    model.generator = generator
    self.model = model
  def forward(self, src):
    return self.model(src)

class MultiChoiceQAModel(nn.Module):
  def __init__(self, args):
    super(MultiChoiceQAModel, self).__init__()
    self.args = args
    if args.pre_word_vecs_enc is not None:
      pretrained = torch.load(args.pre_word_vecs_enc)
      self.embed = nn.Embedding(pretrained.size(0), pretrained.size(1))
      self.embed.weight.data.copy_(pretrained)
      if not args.update_embedding:
        self.embed.weight.requires_grad = False
    else:
      self.embed = nn.Embedding(args.vocab_size,
                                args.word_vec_size,
                                padding_idx=onmt.Constants.PAD)
    self.seq2seq_model = Seq2seqModel(args)


  def forward(self, dialogue, options, answer):
    def lstm_state_flatten(state):
      h, c = state
      h = h.transpose(0,1).transpose(1,2).contiguous().view(h.size(1),-1)
      c = c.transpose(0,1).transpose(1,2).contiguous().view(c.size(1),-1)
      #h = h.view(h.size(0), -1)
      #c = c.view(c.size(0), -1)
      return torch.cat((h,c), -1)
    dialogue = self.embed(dialogue)
    answer = self.embed(answer)
    old_size = options.size()
    options = self.embed(options.view(options.size(0), -1))
    response, resp_vec = self.seq2seq_model([dialogue, answer]) # resp_vec.size: [batch_size, rnn_size]
    opts_vec, _ = self.seq2seq_model.model.encoder(options) #opts_vec.size: [batch_size*num_options, rnn_size]
    opts_vec = lstm_state_flatten(opts_vec)
    resp_vec = lstm_state_flatten(resp_vec)
    opts_vec = opts_vec.view(old_size[1], old_size[2], -1).contiguous()
    resp_vec = resp_vec.unsqueeze(1).contiguous()
    sums = opts_vec * resp_vec
    scores = torch.sum(sums, -1)
    return scores, response

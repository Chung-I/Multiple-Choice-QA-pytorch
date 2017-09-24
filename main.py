import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

from tensorpack.dataflow import ConcatData, BatchData, LocallyShuffleData
from model import MultiChoiceQAModel

import onmt
from GenLCData import GenLCData

import pickle
import csv
import argparse
import glob
import random
import time
import math

torch.backends.cudnn.benchmark=True

parser = argparse.ArgumentParser(description='Multiple Choice QA Model')

## Data options

parser.add_argument('--vocab-file', default="vocab.txt",
                            help='location of embedding file.')
parser.add_argument('--valid-file', default="../training_data/all_no_TC.txt",
                            help='location of embedding file.')
parser.add_argument('--ckpt', default='model.pt',
                            help='saved checkpoint name')

## Model options

parser.add_argument('--layers', type=int, default=2, metavar='N',
        help='layers of encoder and decoder in seq2seq model (default: 2)')
parser.add_argument('--word-vec-size', type=int, default=64, metavar='N',
                            help='embedding size for each word/character (default: 512)')
parser.add_argument('--rnn-size', type=int, default=64, metavar='N',
                            help='rnn hidden size (default: 512)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                            help='input batch size for testing (default: 1000)')
parser.add_argument('-input_feed', type=int, default=1,
                    help="""Feed the context vector at each time step as
                    additional input (via concatenation with the word
                    embeddings) to the decoder.""")
## Optimization options

parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                            help='input batch size for training (default: 64)')
parser.add_argument('-max_generator_batches', type=int, default=32,
                    help="""Maximum batches of words in a sequence to run
                    the generator on in parallel. Higher is faster, but uses
                    more memory.""")
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                            help='number of epochs to train (default: 10)')
parser.add_argument('-param_init', type=float, default=0.1,
                    help="""Parameters are initialized over uniform distribution
                    with support (-param_init, param_init)""")
parser.add_argument('-optim', default='sgd',
                    help="Optimization method. [sgd|adagrad|adadelta|adam]")
parser.add_argument('-max_grad_norm', type=float, default=5,
                    help="""If the norm of the gradient vector exceeds this,
                    renormalize it to have the norm equal to max_grad_norm""")
parser.add_argument('--dropout', type=float, default=0.5,
                            help='dropout rate (default: 0.5)')
parser.add_argument('--steps-per-epoch', type=int, default=10000, metavar='N',
                            help='train this many steps for each epoch')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                            help='how many batches to wait before logging training status')

#learning rate

parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                            help='learning rate (default: 0.01)')
parser.add_argument('-lr_decay', type=float, default=0.5,
                    help="""If update_learning_rate, decay learning rate by
                    this much if (i) perplexity does not decrease on the
                    validation set or (ii) epoch has gone past
                    start_decay_at""")
parser.add_argument('-start_decay_at', type=int, default=200,
                    help="""Start decaying every epoch after and including this
                    epoch""")

parser.add_argument('--no-cuda', action='store_true', default=False,
                            help='disables CUDA training')
parser.add_argument('--update-embedding', action='store_true', default=False,
                            help='whether to update pretrained embedding')
parser.add_argument('--brnn', action='store_true',
                            help='bidirection rnn')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
parser.add_argument('--mode', default='train',
                            help='either train or test')
#parser.add_argument('--embed_file', default='/data/groups/make_speech_lab_great_again/skip_thoughts/zh_novels/data/exp_vocab/embeddings.npy',
#                            help='location of embedding file.')
#pretrained word vectors
parser.add_argument('--pre_word_vecs_enc', default=None,
                            help='location of embedding file.')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
with open(args.vocab_file) as f:
  args.vocab_size = len(f.readlines())

torch.manual_seed(args.seed)
if args.cuda:
  torch.cuda.manual_seed(args.seed)

model = MultiChoiceQAModel(args)

if args.cuda:
  model = model.cuda()
optimizer = onmt.Optim(
    args.optim, args.lr, args.max_grad_norm,
    lr_decay=args.lr_decay,
    start_decay_at=args.start_decay_at
)
optimizer.set_parameters(model.parameters())

datasets = []
for f in glob.glob("/data/users/iLikeNLP/AIContest/ChatBotCourse/subtitle/preprocess/chinese/*.srt"):
    datasets.append(GenLCData(f, args.vocab_file))
train_loader = ConcatData(datasets)
#train_loader = LocallyShuffleData(train_loader, args.batch_size*100)
train_loader = BatchData(train_loader, args.batch_size)
train_data = train_loader.get_data()
valid_loader = GenLCData(args.valid_file, args.vocab_file)
valid_loader = BatchData(valid_loader, args.batch_size)
valid_data = valid_loader.get_data()
#test_loader = Loader("test.pkl", args.test_batch_size, False)

def to_cuda_var(tensor, use_cuda):
    tensor = np.array(tensor, dtype=np.int64)
    tensor = np.rollaxis(tensor,-1)
    tensor = torch.from_numpy(tensor)
    if use_cuda:
      return Variable(tensor.cuda())
    else:
      return Variable(tensor)

def seq2seqLoss(outputs, targets, model):
  def NMTCriterion(vocabSize):
    weight = torch.ones(vocabSize)
    weight[onmt.Constants.PAD] = 0
    crit = nn.NLLLoss(weight, size_average=False)
    if args.cuda:
        crit.cuda()
    return crit

  def memoryEfficientLoss(outputs, targets, generator, crit, eval=False):
    # compute generations one piece at a time
    num_correct, loss = 0, 0
    outputs = Variable(outputs.data, requires_grad=(not eval), volatile=eval)

    batch_size = outputs.size(1)
    outputs_split = torch.split(outputs, args.max_generator_batches)
    targets_split = torch.split(targets, args.max_generator_batches)
    for i, (out_t, targ_t) in enumerate(zip(outputs_split, targets_split)):
      out_t = out_t.view(-1, out_t.size(2))
      scores_t = generator(out_t)
      loss_t = crit(scores_t, targ_t.view(-1))
      pred_t = scores_t.max(1)[1]
      num_correct_t = pred_t.data.eq(targ_t.data).masked_select(targ_t.ne(onmt.Constants.PAD).data).sum()
      num_correct += num_correct_t
      loss += loss_t.data[0]
      if not eval:
        loss_t.div(batch_size).backward()

    grad_output = None if outputs.grad is None else outputs.grad.data
    return loss, grad_output, num_correct

  criterion = NMTCriterion(args.vocab_size)
  loss, gradOutput, num_correct = memoryEfficientLoss(
          outputs, targets, model.seq2seq_model.model.generator, criterion, not model.training)

  outputs.backward(gradOutput)

  # update the parameters
  num_words = targets.data.ne(onmt.Constants.PAD).sum()

  return loss, num_correct, num_words

def train(epoch):
  for p in model.parameters():
    p.data.uniform_(-args.param_init, args.param_init)
  model.train()
  report_num_correct = 0
  report_tgt_words = 0
  report_seq2seq_loss = 0
  report_class_loss = 0
  start = time.time()
  for batch_idx, (dialogue, options, answer_id) in zip(range(args.steps_per_epoch), train_data):
    answer = options[range(len(answer_id)), answer_id]
    dialogue = to_cuda_var(dialogue, args.cuda)
    options = to_cuda_var(options, args.cuda)
    answer_id = to_cuda_var(answer_id, args.cuda)
    answer = to_cuda_var(answer, args.cuda)
    optimizer.zero_grad()
    scores, response = model(dialogue, options, answer)
    batch_size = scores.size(1)
    class_loss = F.cross_entropy(scores, answer_id)
    class_loss.div(batch_size).backward(retain_graph=True)
    seq2seq_loss, num_correct, num_words = seq2seqLoss(response, answer, model)

    optimizer.step()

    report_num_correct += num_correct
    report_tgt_words += num_words
    report_seq2seq_loss += seq2seq_loss
    report_class_loss += class_loss

    if batch_idx % args.log_interval == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\t acc: {%6.2f}; ppl: {%6.2f}; class loss: {}; {%3.0f} tgt tok/s'.format(
        epoch, batch_idx, args.steps_per_epoch, batch_idx / args.steps_per_epoch,
          report_num_correct / report_tgt_words * 100,
          math.exp(report_seq2seq_loss / report_tgt_words),
          report_class_loss / args.log_interval,
          report_tgt_words/(time.time()-start)))
        report_num_correct = report_tgt_words = report_seq2seq_loss = report_class_loss = 0
        start = time.time()


def validate(epoch):
  model.eval()
  report_class_loss = 0
  report_seq2seq_loss = 0
  report_num_word_correct = 0
  report_num_words = 0
  report_num_class_correct = 0
  for batch_idx, (dialogue, options, answer_id) in zip(range(100), valid_data):
    answer = options[range(len(answer_id)), answer_id]
    dialogue = to_cuda_var(dialogue, args.cuda)
    options = to_cuda_var(options, args.cuda)
    answer_id = to_cuda_var(answer_id, args.cuda)
    answer = to_cuda_var(answer, args.cuda)
    scores, response = model(dialogue, options, answer)
    batch_size = scores.size(0)
    report_class_loss += F.cross_entropy(scores, answer_id)
    seq2seq_loss, num_correct, num_words = seq2seqLoss(response, answer, model)
    report_seq2seq_loss += seq2seq_loss
    pred = scores.data.max(1)[1]
    report_num_class_correct += pred.eq(answer_id.data).cpu().sum()
    report_num_word_correct += num_correct
    report_num_words = num_words

  data_len = args.batch_size * (batch_idx + 1)
  print('\nValidation set: seq2seq accuracy: %6.2f, classification accuracy: {}/{} ({:.0f}%)\n'.format(report_num_word_correct / report_num_words, report_num_class_correct, data_len, 100. * report_num_class_correct / data_len))

#def test():
#  model.eval()
#  preds = []
#  for sent1, sent2 in test_loader:
#    if args.cuda:
#      sent1, sent2 = sent1.cuda(), sent2.cuda()
#    sent1, sent2, = Variable(sent1), Variable(sent2)
#    output = model(sent1, sent2)
#    pdb.set_trace()
#    pred = output.data.max(1)[1]
#    preds.append(pred)
#  preds = torch.cat(preds)
#  return preds

if args.mode == "train":
  for epoch in range(1, args.epochs + 1):
    train(epoch)
    validate(epoch)
    d = model.state_dict()
    #state_dict_wo_embeddings = {i:d[i] for i in d if i!='embed.weight'}
    #torch.save(state_dict_wo_embeddings, args.ckpt)
    torch.save(d, args.ckpt)

elif args.mode == "test":
  pass
  #with open(args.embed_file, "rb") as f:
  #  embedding = torch.from_numpy(np.load(f))
  #state_dict = torch.load(args.ckpt)
  ##state_dict['embed.weight'] = embedding
  #model.load_state_dict(state_dict)
  #predictions = test().cpu().numpy()
  #with open("pred.csv", "w") as f:
  #  f.write("Id,Relation\n")

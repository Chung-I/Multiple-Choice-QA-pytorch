import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pickle
import pdb
import csv
import argparse
from model import TextCNN
import random
from trainLCData import LCData
from tensorpack import ConcatData

torch.backends.cudnn.benchmark=True

parser = argparse.ArgumentParser(description='PyTorch Classifier')
parser.add_argument('--hidden-size', type=int, default=128, metavar='N',
                            help='input batch size for training (default: 64)')
parser.add_argument('--embedding-size', type=int, default=512, metavar='N',
                            help='input batch size for training (default: 64)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                            help='input batch size for training (default: 64)')
parser.add_argument('--dropout', type=float, default=0.5,
                            help='dropout rate (default: 0.5)')
parser.add_argument('--num_classes', type=float, default=4,
                            help='number of classes (default: 4)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                            help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                            help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                            help='learning rate (default: 0.01)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                            help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
parser.add_argument('--mode', default='train',
                            help='either train or test')
#parser.add_argument('--embed_file', default='/data/groups/make_speech_lab_great_again/skip_thoughts/zh_novels/data/exp_vocab/embeddings.npy',
#                            help='location of embedding file.')
parser.add_argument('--embed_file', default='/data/users/iLikeNLP/word2vec/giga.npy',
                            help='location of embedding file.')
parser.add_argument('--ckpt', default='model.pt',
                            help='either train or test')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                            help='how many batches to wait before logging training status')
args = parser.parse_args()
args.kernel_sizes = [2,3,4,5]
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
        torch.cuda.manual_seed(args.seed)

model = TextCNN(args)

if args.cuda:
  model = model.cuda()
parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.Adam(parameters, lr=args.lr)

batch_size = 64
datasets = []
for f in glob.glob("/data/users/iLikeNLP/AIContest/ChatBotCourse/subtitle/preprocess/chinese/*.srt"):
    datasets.append(LCData(f, has_label=True))
train_loader = ConcatData(datasets)
train_data = train_loader.get_data()
valid_loader = LCData("../training_data/all_no_TC.txt", True)
valid_data = valid_loader.get_data()
#test_loader = Loader("test.pkl", args.test_batch_size, False)

def train(epoch):
  model.train()
  for i in range(10000):
    dialogue, *options, answer_id = next(train_data)
    if args.cuda:
      sent1, sent2, target = sent1.cuda(), sent2.cuda(), target.cuda()
    sent1, sent2, target = Variable(sent1), Variable(sent2),Variable(target)
    optimizer.zero_grad()
    output = model(sent1, sent2)
    loss = F.cross_entropy(output, target)
    loss.backward()
    optimizer.step()
    #if batch_idx % args.log_interval == 0:
    #  print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    #    epoch, batch_idx * len(data), train_loader.data_len,
    #    100. * batch_ivalidtrain_loader.batch_len, loss.data[0]))

def validate(epoch):
  model.eval()
  valid_loss = 0
  correct = 0
  for dialogue, *options, answer_id in valid_data:
    if args.cuda:
      sent1, sent2, target = sent1.cuda(), sent2.cuda(), target.cuda()
    sent1, sent2, target = Variable(sent1), Variable(sent2),Variable(target)
    output = model(sent1, sent2)
    valid_loss += F.cross_entropy(output, target).data[0]
    pred = output.data.max(1)[1]
    correct += pred.eq(target.data).cpu().sum()

  valid_loss = valid_loss
  valid_loss /= valid_loader.batch_size # loss function already averages over batch size
  print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
      valid_loss, correct, valid_loader.data_len,
      100. * correct / valid_loader.data_len))

def test():
  model.eval()
  preds = []
  for sent1, sent2 in test_loader:
    if args.cuda:
      sent1, sent2 = sent1.cuda(), sent2.cuda()
    sent1, sent2, = Variable(sent1), Variable(sent2)
    output = model(sent1, sent2)
    pdb.set_trace()
    pred = output.data.max(1)[1]
    preds.append(pred)
  preds = torch.cat(preds)
  return preds

if args.mode == "train":
  for epoch in range(1, args.epochs + 1):
    train(epoch)
    validate(epoch)
    d = model.state_dict()
    #state_dict_wo_embeddings = {i:d[i] for i in d if i!='embed.weight'}
    #torch.save(state_dict_wo_embeddings, args.ckpt)
    torch.save(d, args.ckpt)

elif args.mode == "test":
  with open(args.embed_file, "rb") as f:
    embedding = torch.from_numpy(np.load(f))
  state_dict = torch.load(args.ckpt)
  #state_dict['embed.weight'] = embedding
  model.load_state_dict(state_dict)
  predictions = test().cpu().numpy()
  with open("pred.csv", "w") as f:
    f.write("Id,Relation\n")

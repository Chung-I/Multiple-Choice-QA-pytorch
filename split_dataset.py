import pickle
import numpy as np
import random
import sys

with open(sys.argv[1], "rb") as f:
  dataset = pickle.load(f)
percent = float(sys.argv[2])
split = int(len(dataset) * percent)
print("total data: {0}".format(len(dataset)))
print("split: {0}".format(split))
random.shuffle(dataset)
train_dataset = dataset[:split]
valid_dataset = dataset[split:]
with open(sys.argv[3], "wb") as f:
  pickle.dump(train_dataset, f)
with open(sys.argv[4], "wb") as f:
  pickle.dump(valid_dataset, f)

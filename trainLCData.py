from tensorpack import RNGDataFlow, BatchData
import csv
import os
import re
import random

_WORD_SPLIT = re.compile(r"([\u4e00-\u9fa5\uff0c\u3002]|[a-zA-Z0-9_]+)")

_PAD = "_PAD"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _EOS, _UNK]

PAD_ID = 0
EOS_ID = 1
UNK_ID = 2

_DIGIT_RE = re.compile(r"\d")

def initialize_vocabulary(vocabulary_path):
  if os.path.exists(vocabulary_path):
    rev_vocab = []
    with open(vocabulary_path) as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)

def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(_WORD_SPLIT.split(space_separated_fragment))
    return [w for w in words if w]

def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                                            tokenizer=None, normalize_digits=True):
    if not os.path.exists(vocabulary_path):
        print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
        vocab = {}
        with open(data_path) as f:
            counter = 0
            for line in f:
                counter += 1
                if counter % 100000 == 0:
                    print("    processing line %d" % counter)
                if tokenizer:
                    tokens = tokenizer(line)
                else:
                    tokens = basic_tokenizer(line)
                for w in tokens:
                    word = _DIGIT_RE.sub("0", w) if normalize_digits else w
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1
            vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
            if len(vocab_list) > max_vocabulary_size:
                vocab_list = vocab_list[:max_vocabulary_size]
            with open(vocabulary_path, "w") as vocab_file:
                for w in vocab_list:
                    vocab_file.write(w + "\n")

def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=True):

  if tokenizer:
    words = tokenizer(sentence)
  else:
    words = basic_tokenizer(sentence)
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]
  # Normalize digits by 0 before looking words up in the vocabulary.
  return [vocabulary.get(_DIGIT_RE.sub("0", w), UNK_ID) for w in words]


class LCData(RNGDataFlow):
    def __init__(self, csv_path, has_label=False, vocab_path=None):
        self.csv_path = csv_path
        self.has_label = has_label
        if vocab_path is None:
            vocab_path = "vocab.txt"
            create_vocabulary(vocab_path, csv_path, 20000)
        self.vocab_path = vocab_path

    def get_data(self):
        vocab, rev_vocab = initialize_vocabulary(self.vocab_path)
        with open(self.csv_path) as csvfile:
            lines = csvfile.read().splitlines()
            for idx, row in enumerate(lines):
                if idx % 3 == 0:
                    continue
                num_dialogues = random.randint(1,3)
                num_responses = random.randint(1,3)
                dialogue = _EOS.join(lines[idx : idx + num_dialogues])
                response = _EOS.join(lines[idx + num_dialogues : idx + num_dialogues + num_responses])
                options = [response]
                for num in range(5):
                    mean = random.choice(list(range(-200,-20,30)) + list(range(20,200,30)))
                    deviation = random.randint(-2, 2)
                    num_responses = random.randint(1,3)
                    option = _EOS.join(lines[idx + mean + deviation : idx + mean + deviation + num_responses])
                    options.append(option)
                indexed_options = list(enumerate(options))
                random.shuffle(indexed_options)
                answer_id = [idx for idx, option in enumerate(indexed_options) if option[0] == 0][0]
                options = [i[1] for i in indexed_options]
                dialogue = sentence_to_token_ids(dialogue, vocab)
                options = [sentence_to_token_ids(opt, vocab) for opt in options]
                yield [dialogue, *options, answer_id]



if __name__ == '__main__':
    ds = LCData("data/all_no_TC.txt", has_label=False)
    ds = BatchData(ds, 64, use_list=True)
    b = ds.get_data()
    print(next(b))

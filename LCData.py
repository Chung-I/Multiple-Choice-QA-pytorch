from tensorpack.dataflow import RNGDataFlow, BatchData
import csv
import os
import re
import numpy as np

_WORD_SPLIT = re.compile(r"([\u4e00-\u9fa5\uff0c\u3002]|[a-zA-Z0-9_]+)")

_PAD = onmt.Constants.PAD_WORD
_EOS = onmt.Constants.EOS_WORD
_UNK = onmt.Constants.UNK_WORD
_START_VOCAB = [_PAD, _EOS, _UNK]

PAD_ID = onmt.Constants.PAD
EOS_ID = onmt.Constants.EOS
UNK_ID = onmt.Constants.UNK

MAX_SEQ_LEN = 60

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

class LEN_TOO_LONG_ERR(BaseException):
    pass

def pad(sentence, seq_len):
    if len(sentence) > seq_len:
        raise LEN_TOO_LONG_ERR 
    return sentence + [PAD_ID] * (seq_len - len(sentence))

def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=True):

  if tokenizer:
    words = tokenizer(sentence)
  else:
    words = basic_tokenizer(sentence)
  if not normalize_digits:
    return pad([vocabulary.get(w, UNK_ID) for w in words], MAX_SEQ_LEN)
  # Normalize digits by 0 before looking words up in the vocabulary.
  return pad([vocabulary.get(_DIGIT_RE.sub("0", w), UNK_ID) for w in words], MAX_SEQ_LEN)


class LCData(RNGDataFlow):
    def __init__(self, csv_path, has_answer=False, vocab_path=None):
        self.csv_path = csv_path
        self.has_answer = has_answer
        if vocab_path is None:
            vocab_path = "vocab.txt"
            create_vocabulary(vocab_path, csv_path, 20000)
        self.vocab_path = vocab_path

    def get_data(self):
        vocab, rev_vocab = initialize_vocabulary(self.vocab_path)
        with open(self.csv_path) as csvfile:
            csvfile.readline() # bye header
            reader = csv.reader(csvfile)
            num_discard = 0
            for row in reader:
                if self.has_answer:
                    idx, dialogue, options, answer = row
                else:
                    idx, dialogue, options = row
                    answer = -1
                options = options.split("\t")
                try:
                    dialogue = sentence_to_token_ids(dialogue, vocab)
                    options = [sentence_to_token_ids(opt, vocab) for opt in options]
                    dialogue = np.array(dialogue, dtype=np.int32)
                    options = np.array(options, dtype=np.int32)
                    yield [dialogue, options, answer]
                except LEN_TOO_LONG_ERR:
                    num_discard += 1
                    print("sequence length too long; number of sentences thrown away: {}".format(num_discard))



if __name__ == '__main__':
    a = LCData("data/testing_data.csv", has_answer=False)
    print(a.vocab_path)
    a = BatchData(a, 64)
    print(a.vocab_path)
    b = a.get_data()
    c = next(b)
    print(c[0].shape)
    print(c[1].shape)

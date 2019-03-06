import os
import json
import nltk
from nltk.tokenize import RegexpTokenizer
import pickle

from collections import Counter, defaultdict

from utils.vocab import Vocabulary

def get_annotations(json_root, split='train'):    
    path = os.path.join(json_root, 'captions_{}2014.json'.format(split))
    with open(path, 'r') as f:
        data = json.load(f)
        anns = data.get('annotations')
    return anns


def build_vocab(anns, threshold=4):
    """Build a simple vocabulary wrapper."""

    counter = Counter()

    for i, ann in enumerate(anns):
        # print('Processing {}/{}...'.format(i+1, len(anns)))
        caption = ann.get('caption')
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(caption.lower())
        counter.update(tokens)

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab


def extract_caps(anns):    
    data = defaultdict(list)
    for i, ann in enumerate(anns):        
        # print('Processing {}/{}...'.format(i+1, len(anns)))
        tokenizer = RegexpTokenizer(r'\w+')
        data[str(ann.get('image_id'))].append(tokenizer.tokenize(ann.get('caption').lower()))
    return data


def parse_exist_vocab(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    idx2word = data[2]
    word2idx = data[3]

    vocab = Vocabulary()
    vocab.set_content(word2idx=word2idx, idx2word=idx2word)
    return vocab

if __name__ == '__main__':
    nltk.download('punkt')

    json_root = '../data/coco/annotations/'
    vocab_path = '../data/coco/captions/vocab.pickle'

    train_caps_path = '../data/coco/captions/train_cap.pickle'
    val_caps_path = '../data/coco/captions/val_cap.pickle'

    train_anns = get_annotations(json_root, split='train')
    val_anns = get_annotations(json_root, split='val')

    print('=== Building Vocabulary ===')
    # vocab = build_vocab(train_anns+val_anns, threshold=4)
    # with open(vocab_path, 'wb') as f:
    #     pickle.dump(vocab, f)
    
    vocab = parse_exist_vocab('/home/andgitisaac/Downloads/coco/captions.pickle')
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)

    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))


    # print('=== Extracting Train Captions ===')
    # train_caps = extract_caps(train_anns)
    # with open(train_caps_path, 'wb') as f:
    #     pickle.dump(train_caps, f)

    # print("Total Train Caption size: {}".format(len(train_caps)))
    # print("Saved the Caption wrapper to '{}'".format(train_caps_path))

    # print('=== Extracting Val Captions ===')
    # val_caps = extract_caps(val_anns)
    # with open(val_caps_path, 'wb') as f:
    #     pickle.dump(val_caps, f)

    # print("Total Val Caption size: {}".format(len(val_caps)))
    # print("Saved the Caption wrapper to '{}'".format(val_caps_path))

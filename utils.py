import pandas as pd
from dataset import PreProcessDataset, RunDataset
from torch.utils.data import random_split
import torch
import spacy
from whatthelang import WhatTheLang
from tqdm import tqdm
import numpy as np
from pickle5 import pickle

def apply_transforms(x):
    return x

def get_word_embeddings():
    vecs = dict()
    languages = ["de", "fr", "it", "en", "da"]
    dir = "/home/marcelbraasch/PycharmProjects/MultiTextClassifier/Data/"
    for lang in tqdm(languages):
        curr_vec = dict()
        with open(dir + f"wiki.multi.{lang}.vec", encoding="utf-8") as f:
            for line in f:
                # split into name and vector
                name = line.split(" ")[0]
                vec = line.split(" ")[1:]
                vec[-1] = vec[-1].replace("\n", "")
                # convert to numpy array
                v = np.ones((300,))
                for i, val in enumerate(vec):
                    v[i] = float(val)
                # add to dict
                curr_vec[name] = v
            vecs[lang] = curr_vec
    return vecs

def get_tokenizers():
    tokenizers = {
        "de": spacy.load("de_core_news_md"),
        "fr": spacy.load("fr_core_news_md"),
        "it": spacy.load("it_core_news_md"),
        "da": spacy.load("da_core_news_md"),
        "en": spacy.load("en_core_web_md"),
    }
    return tokenizers

def get_label_map(data):
    label_map = dict()
    counter = 0
    for author in data["author"]:
        if author not in label_map:
            label_map[author] = counter
            counter += 1
    return label_map

def get_data_as_df():
    train_path = "/home/marcelbraasch/PycharmProjects/MultiTextClassifier/Data/classifier_data_train.json"
    eval_path = "/home/marcelbraasch/PycharmProjects/MultiTextClassifier/Data/classifier_data_eval.json"
    train = pd.read_json(train_path, lines=True)
    eval = pd.read_json(eval_path, lines=True)
    data = pd.concat([train, eval], axis=0)
    return data

def split_dataset(ratio=None):
    """Utility function to split given data set into
        train, test, eval partitions."""
    ratio = [.8, .1, .1] if ratio == None else ratio
    train_r, val_r, test_r = ratio
    data = None
    with open("data.pickle", mode="rb") as handle:
        data = pickle.load(handle)
    dataset = RunDataset(data=data)
    train_size, val_size, test_size = int(train_r * len(dataset)), int(val_r * len(dataset)), int(test_r * len(dataset))
    diff = len(dataset) - train_size - val_size - test_size
    split = random_split(dataset,
                        [int(train_r * len(dataset)) + diff,
                         int(val_r * len(dataset)),
                         int(test_r * len(dataset))])
    train, test, val = split
    return train, test, val

def create_training_samples():

    wtl = WhatTheLang()
    data = get_data_as_df()
    label_map = get_label_map(data)
    tokenizers = get_tokenizers()
    word_embeddings = get_word_embeddings()

    ready_data = []
    for author, text, lang in tqdm(list(zip(data["author"], data["text"], data["lang"]))):
        # Get correct language, if inconsistent correct by hand
        #pred_lang = wtl.predict_lang(text)
        if lang not in ["de", "fr", "en", "it", "da"]:
            continue
        # Get correct tokenizer and embedder
        tokenizer = tokenizers[lang]
        embedder = word_embeddings[lang]
        # Vectorize each word
        tokenized_words = apply_transforms([str(word) for word in tokenizer(text)])
        document = None
        good, bad = 0, 0
        for word in tokenized_words:
            if word in embedder:
                embedding = torch.tensor(embedder[word]).reshape(1, 300)
                good +=1
            else: # this is an unknown word
                embedding = torch.ones(1, 300)
                bad += 1
            if not document == None:
                document = torch.cat([document, embedding], 0)
            else:
                document = embedding
        # cut off all docs > 100:
        document = document[:100]
        # pad with zeros
        #print(f"Good-Ratio: {good/(good+bad)}")
        ready_data.append({
            "response": label_map[author],
            "document": document
        })
    return ready_data

def pickle_data():
    data = create_training_samples()
    with open("data.pickle", mode="wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def unpickle_data():
    data = None
    with open("data.pickle", mode="rb") as handle:
        data = pickle.load(handle)
    return data
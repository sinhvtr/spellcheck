# Importing pytorch and the library for TPU execution
import torch
from transformers import AutoModel, AutoTokenizer

# Importing stock ml libraries
import numpy as np
import pandas as pd
import transformers
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

# Importing local libraries
from data_utils import CustomDataset, SentenceGetter
from model import PhoBertBaseSP

# Importing system libraries
import pickle
import tqdm
import sys

# Preparing for GPU usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

data_path = sys.argv[1] # default data/mispell_data_10k.txt
dataset = pd.read_csv(data_path, delimiter='\t', header=1, names=['sentence_idx', 'word', 'tag', 'type'])
getter = SentenceGetter(dataset)

# Creating new lists and dicts that will be used at a later stage for reference and processing
tags_vals = list(set(dataset["tag"].values))
tag2idx = {t: i for i, t in enumerate(tags_vals)}
sentences = [' '.join([str(s[0]) for s in sent]) for sent in getter.sentences]
labels = [[s[1] for s in sent] for sent in getter.sentences]
labels = [[tag2idx.get(l) for l in lab] for lab in labels]
print(tags_vals)
print(sentences[3])
print(labels[3])

# Hyperparams
MAX_LEN = 200
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 2e-05
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
config = {
    "max_len": MAX_LEN,
    "train_batch_size": TRAIN_BATCH_SIZE,
    "valid_batch_size": VALID_BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE
}
pickle.dump( config, open( "trained_models/config_5ep_bs8_data10k_maxlen200.p", "wb" ) )

# Creating the dataset and dataloader for the neural network

train_percent = 0.8
train_size = int(train_percent*len(sentences))
# train_size = 80
# test_size = 20
# train_dataset=df.sample(frac=train_size,random_state=200).reset_index(drop=True)
# test_dataset=df.drop(train_dataset.index).reset_index(drop=True)
train_sentences = sentences[0:train_size]
train_labels = labels[0:train_size]

test_sentences = sentences[train_size:]
test_labels = labels[train_size:]

# train_sentences = sentences[0:train_size]
# train_labels = labels[0:train_size]

# test_sentences = sentences[train_size:train_size + test_size]
# test_labels = labels[train_size:train_size + test_size]

print("FULL Dataset: {}".format(len(sentences)))
print("TRAIN Dataset: {}".format(len(train_sentences)))
print("TEST Dataset: {}".format(len(test_sentences)))

training_set = CustomDataset(tokenizer, train_sentences, train_labels, MAX_LEN)
testing_set = CustomDataset(tokenizer, test_sentences, test_labels, MAX_LEN)

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 5
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)

model = PhoBertBaseSP()
model.to(device)

optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

def train(epoch):
    model.train()
    for _, data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        targets = data['tags'].to(device, dtype = torch.long)

        loss = model(ids, mask, labels = targets)[0]
        if _%2000==0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

PATH = "trained_models/sp_base_gpu_10k_5ep_udts"

for epoch in range(EPOCHS):
    print("Training epoch", epoch)
    train(epoch)
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            }, PATH)

torch.save(model, "trained_models/sp_base_gpu_10k_5ep_udts")
pickle.dump( tags_vals, open( "trained_models/sp_base_gpu_10k_5ep_udts.p", "wb" ) )



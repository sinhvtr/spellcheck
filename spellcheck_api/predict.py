# Importing pytorch and the library for TPU execution

import torch
import transformers
from transformers import AutoModel, AutoTokenizer, AutoConfig, DistilBertTokenizer
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from model import PhoBertBaseSP
from data_utils import CustomDataset

# Importing ml libraries

import numpy as np
import pandas as pd
import pickle

import time

# Importing Vietnamese NLP toolkits
# from vncorenlp import VnCoreNLP
# use VnCoreNLP to preprocess raw text
# rdrsegmenter = VnCoreNLP("/home/local/Zalo/spellcheck_baomoi/VnCoreNLP/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')
from underthesea import sent_tokenize

MAX_LEN = 200

def splitTextBySentences(text):
  text2sentences = sent_tokenize(text)
  print(text2sentences)
  # labels = []
  # for sentence in text2sentences:
  #   label_sent = [0 for i in range(len(sentence.split()))]
  labels = [[0 for i in range(len(sentence.split()))] for sentence in text2sentences]
  print(labels)
  return text2sentences, labels

def splitTextByMaxLen(text, max_len=MAX_LEN):
  # split text by MAX_LEN/2
  text_array = text.split()
  # text_array = rdrsegmenter.tokenize(text)
  label_array = [0 for i in range(len(text_array))]
  chunks, chunk_size = len(text_array), int(max_len/2)
  # print(chunks)
  text_chunked_array = [ text_array[i:i+chunk_size] for i in range(0, chunks, chunk_size) ]
  text_chunked = [' '.join(token) for token in text_chunked_array]
  label_chunked = [ label_array[i:i+chunk_size] for i in range(0, chunks, chunk_size) ]
  return text_chunked, label_chunked

def spellCheck(text):  
  # split text to batches of (sub_text, labels)
  print("Split text to chunks")
  # text_array, label_array = splitTextByMaxLen(text, MAX_LEN)
  sentences, labels = splitTextBySentences(text)
  print("--- %s seconds ---" % (time.time() - start_time))  
  # exit()
  # print(text_array, label_array)
  
  # tokenize input text to ids
  # text = [text]
  # text2ids = tokenizer.encode(text[0])
  # for token in text2ids:
  #   print(tokenizer.decode(token))
  # ids2tokens = tokenizer.decode(text2ids)

  # assign 0 to all labels
  # pseudo_labels = [[0 for i in range(len(text2ids))]]


  # Preparing for GPU usage
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print('Using device:', device)

  new_model = torch.load('trained_models/sp_base_gpu_8ep_new_tokenize',map_location='cpu')
  new_model.to(device)
  print("Model loaded")

  tags_vals = pickle.load(open('trained_models/tags_vals_gpu.p', 'rb'))
  print('Tags loaded')

  # AutoTokenizer.from_pretrained("bert-base-cased").save_pretrained("./trained_models/")
  # tokenizer = DistilBertTokenizer.from_pretrained("./trained_models/tokenizer/tokenizer_config.json") # throws exception
  # tokenizer = AutoTokenizer.from_pretrained('./trained_models/tokenizer/')

  # Load & save pretrained tokenizer model to local
  tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base', use_fast=False)
  # tokenizer_local.save_pretrained('./trained_models/tokenizer/')

  # Load tokenizer model from local
  # tokenizer = AutoTokenizer.from_pretrained('./trained_models/tokenizer')

  test_set = CustomDataset(tokenizer, sentences, labels, MAX_LEN)
  test_params = {'batch_size': 1,
                  'shuffle': False,
                  'num_workers': 0
                  }
                  
  test_loader = DataLoader(test_set, **test_params)
  print("--- %s seconds ---" % (time.time() - start_time)) 
  predictions, true_labels = [], []
  print("Begin check spelling")
  predict_text, predict_mispell = '', []
  for _, data in enumerate(test_loader, 0):
    # predict_text_chunk, predict_mispell_chunk = [], []
    # ids2tokens = tokenizer.decode(text2ids)
    
    ids = data['ids'].to(device, dtype = torch.long)
    mask = data['mask'].to(device, dtype = torch.long)
    targets = data['tags'].to(device, dtype = torch.long)

    output = new_model(ids, mask, labels=targets)
    loss, logits = output[:2]
    logits = logits.detach().cpu().numpy()
    predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
    pred = [list(p) for p in np.argmax(logits, axis=2)]

    for id in ids:
      # print('id', id)
      predict_text_chunk = tokenizer.decode(id)

    predict_text_chunk_removed_pad = predict_text_chunk.replace('<pad>', '')
    predict_text_chunk_removed_pad = predict_text_chunk_removed_pad.replace('<s>', '')
    predict_text_chunk_removed_pad = predict_text_chunk_removed_pad.replace('</s>', '')
    predict_text_chunk_removed_pad = predict_text_chunk_removed_pad.rstrip()
    print(predict_text_chunk_removed_pad)
    predict_text = predict_text + predict_text_chunk_removed_pad
    # predict_text.replace('<s>', '')
    # predict_text.replace('</s>', '')
    # print(pred)
    # print(len((predict_text_chunk_removed_pad.split())))
    pred = pred[0]
    pred = [tags_vals[p] for p in pred[:len(predict_text_chunk_removed_pad.split())]]
    print(pred)
    predict_mispell = predict_mispell + pred
  # print(predict_text)
  # print(predict_mispell)
  print("--- %s seconds ---" % (time.time() - start_time)) 
  return predict_text, predict_mispell

start_time = time.time()

text = "M??? c??o bu???c Iran ?????ng sau c??c v??? t???n c??ng t??u ch??? d???u ??? V??ng V???nh. C?? quan An ninh qu???c gia M??? cho r???ng m???t s??? nh??m c?? li??n quan t???i Iran c?? th??? ???? t???n c??ng 4 t??u ch??? d???u ??? UAE, thay v?? l???c l?????ng v?? trang Iran tr???c ti???p h??nh ?????ng, theo m???t quan ch???c M??? ho???t ?????ng trong l??nh v???c ?????nh gi?? t??i s???n cho bi???t. V??? quan ch???c n??y cho bi???t, c??c th??? ph???m g??y ra v??? vi???c, kh??? n??ng cao c?? c??c phi???n qu??n Houthi ??? Yemen v?? l???c l?????ng phi???n qu??n Shi'ite ???????c Iran ???ng h?? , song Washington hi???n ch??a c?? b???ng. ch???ng x??c th???c n??o v??? th??? ph???m ph?? ho???i 4 t??u ch??? d???u, trong ???? c?? 2 t??u c???a Saudi Arabia g???n c???ng Fujairah, n???m ngo??i eo bi???n Hormuz. "

sp_result = spellCheck(text)
# print(sp_result)
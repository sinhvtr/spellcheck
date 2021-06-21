from flask import Flask, render_template, request
import torch
import transformers
from data_utils import CustomDataset
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoModel, AutoTokenizer
from model import PhoBertBaseSP

# Importing ml libraries
import numpy as np
import pandas as pd
import pickle
import time
import logging
import json
import re

# Importing Vietnamese NLP toolkits
# from vncorenlp import VnCoreNLP
# use VnCoreNLP to preprocess raw text
# rdrsegmenter = VnCoreNLP("/home/local/Zalo/spellcheck_baomoi/VnCoreNLP/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')
from underthesea import sent_tokenize, word_tokenize

loggers = [logging.getLogger(name) for name in logging.Logger.manager.loggerDict]
for logger in loggers:
    logger.setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.INFO)

# Preparing for GPU/CPU usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info('Using device: {}'.format(device))

# Load model
new_model = torch.load('trained_models/zte_trained_models/sp_base_gpu_10k_5ep_udts',map_location='cpu')
# new_model = torch.load('trained_models/sp_base_gpu_8ep_new_tokenize',map_location='cpu')
new_model.to(device)
logging.info("Model loaded: {}".format(new_model))
tags_vals = pickle.load(open('trained_models/zte_trained_models/sp_base_gpu_10k_5ep_udts.p', 'rb'))
logging.info('Tags loaded')

# Load rdrsegmenter from VnCoreNLP
# from vncorenlp import VnCoreNLP
# rdrsegmenter = VnCoreNLP("/VnCoreNLP/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m') 

MAX_LEN = 200
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False, proxies={'http': '10.60.28.99:81'})

def splitTextBySentences(text):
  text2sentences = sent_tokenize(text)
  for i in range(len(text2sentences)):
    text2sentences[i] = word_tokenize(text2sentences[i])
    text2sentences[i] = ' '.join(text2sentences[i])
    text2sentences[i] = text2sentences[i].rstrip()
  labels = [[0 for i in range(len(sentence.split()))] for sentence in text2sentences]
  return text2sentences, labels

def splitTextByParagraph(text, max_len=MAX_LEN):
  # annotator = VnCoreNLP(address="http://127.0.0.1", port=9000) 
  pattern = ">>.+>>" # remove hyperlinks between ">>"

  # split by max len (default 200)
  sub_paragraphs = []
  sub_paragraphs.append('')
  j = 0
  # print(len(sub_paragraphs[j].split()))

  sentences = sent_tokenize(text) # sentence tokenize
  for i in range(len(sentences)):
      # print(i)
      sentences[i] = re.sub(pattern,"",sentences[i])
      sentences[i] = sentences[i].replace(".. ",". ")
      sentences[i] = re.sub(r'([?!,.]+)',r' \1 ', sentences[i])
      sentences[i] = word_tokenize(sentences[i])
      sentences[i] = ' '.join(sentences[i])
      # print(sentences[i])
      # print(len(sub_paragraphs[j].split()))

      # if current sub_paragraph has len < MAX_LEN, concat new sentence
      if len(sub_paragraphs[j].split()) + len(sentences[i].split()) < MAX_LEN:
          sub_paragraphs[j] = sub_paragraphs[j] + ' ' + sentences[i]
          
      else:
          j = j + 1
          sub_paragraphs.append('')
  labels = [[0 for i in range(len(sentence.split()))] for sentence in sub_paragraphs]
  return sub_paragraphs, labels

def splitTextByMaxLen(text, max_len=MAX_LEN):
  # split text by MAX_LEN/2
  text_array = text.split()
  # text_array = rdrsegmenter.tokenize(text) 
  label_array = [0 for i in range(len(text_array))]
  chunks, chunk_size = len(text_array), int(max_len-10)
  logging.info('Total length: {}'.format(chunks))
  logging.info('Max len: {}'.format(chunk_size))
  
  text_chunked_array = [ text_array[i:i+chunk_size] for i in range(0, chunks, chunk_size) ]
  if len(text_chunked_array) > 1:
    text_chunked = [' '.join(token) for token in text_chunked_array]  
    # logging.info('Chunk: {}'.format(text_chunked))
  else:
    text_chunked = text_chunked_array
  label_chunked = [ label_array[i:i+chunk_size] for i in range(0, chunks, chunk_size) ]
  return text_chunked, label_chunked

def spellCheck(text):  
  
  # split text to batches of (sub_text, labels)
  # text_array, label_array = splitTextByMaxLen(text, MAX_LEN)
  text_array, label_array = splitTextByParagraph(text, MAX_LEN)
  # print(text_array)
  # print(label_array)
  # exit()
  # text_array, label_array = splitTextBySentences(text)
  original_text = ''.join(text_array)
  logging.info(text_array)
  # logging.info('Chunk: {}'.format(text_array))
  # create dataset & dataloader for prediction
  test_set = CustomDataset(tokenizer, text_array, label_array, MAX_LEN)
  test_params = {'batch_size': 1,
                  'shuffle': False,
                  'num_workers': 16
                  }
                  
  test_loader = DataLoader(test_set, **test_params)

  # begin predict
  predictions, true_labels = [], []
  logging.info("Begin check spelling")
  predict_text, predict_mispell = '', []
  for chunk, data in enumerate(test_loader):    
    start_time_chunk = time.time()
    # logging.info("Chunk No. {}".format(chunk))
    # logging.info("Predict Text: {}".format(predict_text))

    ids = data['ids'].to(device, dtype = torch.long)
    for token in ids:
      # logging.info('Token: {}, decode: {}'.format(token, tokenizer.decode(token)))
      predict_text_chunk = tokenizer.decode(token)
    # logging.info(predict_text_chunk)
    mask = data['mask'].to(device, dtype = torch.long)
    targets = data['tags'].to(device, dtype = torch.long)

    output = new_model(ids, mask, labels=targets)
    loss, logits = output[:2]
    smax_tensor = torch.nn.functional.softmax(logits, dim=2)
    # logging.info(logits)
    # logging.info(smax_tensor)
    # exit()
    smax_np = smax_tensor.detach().numpy()
    # prob_np = np.max(smax_np, axis=2)
    # logging.info(prob_np.shape)
    # logging.info(prob_np)
    logits_np = logits.detach().cpu().numpy()
    
    # predictions.extend([list(p) for p in np.argmax(logits, axis=2)])

    # prediction
    pred = [list(p) for p in np.argmax(logits_np, axis=2)]
    # logging.info(len(pred[0]))
    # logging.info(pred)
    # extract text chunks
    # for id in ids:
    #   predict_text_chunk = tokenizer.decode(id)

    # process the string to recover original text
    predict_text_chunk_removed_pad = predict_text_chunk.replace('<pad>', '')
    predict_text_chunk_removed_pad = predict_text_chunk_removed_pad.replace('<s>', '')
    predict_text_chunk_removed_pad = predict_text_chunk_removed_pad.replace('</s>', '')
    predict_text_chunk_removed_pad = predict_text_chunk_removed_pad.rstrip()
    logging.info('Text len: {}'.format(len(predict_text_chunk_removed_pad.split())))
    logging.info('Text chunk: {}'.format(predict_text_chunk_removed_pad))
    logging.info('Predict time: %s seconds ---' % (time.time() - start_time_chunk))
    predict_text = predict_text + predict_text_chunk_removed_pad
  
    # combine predictions among chunks
    pred = pred[0]
    pred = [tags_vals[p] for p in pred[:len(predict_text_chunk_removed_pad.split())]]
    predict_mispell = predict_mispell + pred

  return predict_text, predict_mispell

def spellCheck_old(sentence):  
  tokens = word_tokenize(sentence)
  test_sent = [sentence]
  test_sent_tokens = tokenizer.encode(test_sent[0])

  test_label = [[0 for i in range(len(test_sent_tokens))]]
  # test_set = CustomDataset(tokenizer, test_sent, max_len=MAX_LEN)
  test_set = CustomDataset(tokenizer, test_sent, test_label, MAX_LEN)
  test_params = {'batch_size': 1,
                  'shuffle': True,
                  'num_workers': 0
                  }
                  
  test_loader = DataLoader(test_set, **test_params)

  predictions, true_labels = [], []
  for _, data in enumerate(test_loader, 0):
    ids2tokens = tokenizer.decode(test_sent_tokens)
    ids = data['ids'].to(device, dtype = torch.long)
    mask = data['mask'].to(device, dtype = torch.long)
    targets = data['tags'].to(device, dtype = torch.long)
    output = new_model(ids, mask, labels=targets)
    loss, logits = output[:2]
    logits = logits.detach().cpu().numpy()
    predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
    prediction = [tags_vals[p] for p in predictions[0][:len(tokens)]]

  return prediction

app = Flask(__name__)

@app.route('/')
def index():
    base_url = request.base_url
    logging.info('Host: {}, port: {}'.format(request.environ['REMOTE_ADDR'], request.environ['REMOTE_PORT']))
    return render_template('spellcheck.html', url=base_url)

@app.route('/spellcheck_result',methods = ['POST', 'GET'])
def result():
  start_time = time.time()

  base_url = request.base_url 
  home_url = base_url.split("/")[0] + "/" + base_url.split("/")[1]
  if request.method == 'POST':
    result = request.form
    
    form_request = list(result.items())
    text = form_request[0][1]
    # logging.info('Input text: {}'.format(text))
    sp_text, sp_check = spellCheck(text)
    # logging.info('Input tokens: {}'.format(sp_text))
    # logging.info('Predict: {}'.format(sp_check))
    text_token = list(sp_text.split())
    # text_token = word_tokenize(sp_text)
    # logging.info("Spell Check results: %s" % sp_check)
    logging.info("Total time: --- %s seconds ---" % (time.time() - start_time))

    return render_template("spellcheck_result.html", text = text, text_token = text_token, sp_check = sp_check, base_url = home_url)

@app.route('/check',methods = ['POST', 'GET'])
def api_result():
  start_time = time.time()

  base_url = request.base_url 
  if request.method == 'POST':
    post_data = request.form
    post_data_items = list(post_data.items())
    input_text = post_data_items[0][1]
    
    result_array = []
    sp_tokens, sp_check = spellCheck(input_text)    
    text_token = list(sp_tokens.split())
    for i in range(len(text_token)):
      if sp_check[i] == "C":
        result_array.append(text_token[i])
      else:
        typo = dict()
        typo[text_token[i]] = []
        result_array.append(typo)
  response = {'result': result_array}
  logging.info("--- %s seconds ---" % (time.time() - start_time))
  return response


@app.route('/check_plain',methods = ['POST', 'GET'])
def api_plain_result():
  start_time = time.time()

  if request.method == 'POST':
    input_text = request.get_data()
    input_text = input_text.decode("utf8")
    logging.info('Input: {}'.format(input_text))
    
    result_array = []
    sp_tokens, sp_check = spellCheck(input_text)    
    text_token = list(sp_tokens.split())
    for i in range(len(text_token)):
      if sp_check[i] == "C":
        result_array.append(text_token[i])
      else:
        typo = dict()
        typo[text_token[i]] = []
        result_array.append(typo)
  # response = jsonify(result_array)
  response = json.dumps(result_array)
  # response = {result_array}
  # response = {'result': result_array}
  logging.info("--- %s seconds ---" % (time.time() - start_time))
  return response

if __name__ == '__main__':
  # app.run(debug = True, host="127.0.0.1", port=9488)
  # app.run(debug = True, host="0.0.0.0", port=9488)
  app.run(debug = True)
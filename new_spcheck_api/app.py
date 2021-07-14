from flask import Flask, render_template, request

import time
import logging
import json
import requests
import re
from datetime import datetime

from underthesea import sent_tokenize, word_tokenize, ner

url = "https://nlp.laban.vn/wiki/spelling_checker_api/"
proxies = {"http": "http://10.30.11.17:8081"}

loggers = [logging.getLogger(name) for name in logging.Logger.manager.loggerDict]
for logger in loggers:
    logger.setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.INFO)

# split text and return tokens along with their offset position in text
def split_span(s):
    for match in re.finditer(r"\S+", s):
        span = match.span()
        yield match.group(0), span[0], span[1] - 1

def get_ner(text):
    ner_analyzing = ner(text)
    name_entities = []
    for token in ner_analyzing:
        if token[3] != 'O':
            name_entities.append(token[0])
    return name_entities

def get_spcheck(text):
    payload={'text': text,
        'app_type': 'baomoi'}
    files=[

    ]
    headers = {
    'Cache-Control': 'no-cache',
    'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8'
    }

    # response = requests.request("POST", url, headers=headers, data=payload, files=files, proxies=proxies)
    response = requests.post(url, headers=headers, data=payload, files=files, proxies=proxies)
    result = response.json()['result']

    for paragraph in result:
    	for sentence in sent_tokenize(paragraph['text']):
            if sentence[0].islower():
                mistake_pos = paragraph['text'].find(sentence)
                mistake_text = sentence.split()[0]
                suggest = mistake_text.capitalize()
                mistake = {
                    'text': mistake_text,
                    'score': 1,
                    'start_offset': mistake_pos,
                    'suggest': [[suggest, 1]]
                }
                paragraph['mistakes'].append(mistake)
                paragraph['suggested_text'] = paragraph['suggested_text'][:mistake_pos] + suggest + paragraph['suggested_text'][mistake_pos+len(mistake_text):]
    return result

app = Flask(__name__)

@app.route('/')
def index():
    base_url = request.base_url
    logging.info('Host: {}, port: {}'.format(request.environ['REMOTE_ADDR'], request.environ['REMOTE_PORT']))
    return render_template('new_spellcheck.html', url=base_url)

@app.route('/spcheck_result',methods = ['POST'])
def spcheck_result():
    start_time = time.time()

    base_url = request.base_url 
    home_url = base_url.split("/")[0] + "/" + base_url.split("/")[1]
    if request.method == 'POST':
        result = request.form
        
        form_request = list(result.items())
        text = form_request[0][1]
        # logging.info('Input text: {}'.format(text))
        # logging.info("Input text: %s" % text)

        result = get_spcheck(text)

        mistakes_text, suggested_text = [], []
        for paragraph in result:
            mistake_text = paragraph['text']
            if len(paragraph['mistakes']) > 0:
                for mistake in paragraph['mistakes']:
                    mistake_pos = mistake['start_offset']
                    mistake_len = len(mistake['text'])

        logging.info("Total time: --- %s seconds ---" % (time.time() - start_time))
        
        return render_template("new_spcheck_result.html", original_text = text, mistakes = mistakes_text, suggested_text = suggested_text, base_url = home_url)


@app.route('/check_plain',methods = ['POST', 'GET'])
def api_plain_result():
    start_time = time.time()

    logging.info('System time: {}'.format(datetime.fromtimestamp(start_time)))

    all_tokens = []
    mistake_count = 0

    if request.method in ['POST', 'GET']:
        input_text = request.get_data()
        input_text = input_text.decode("utf8")
        # logging.info('Input: {}'.format(input_text))
        
        result_array = []
        sp_check = get_spcheck(input_text)
        # for paragraph in sp_check:
        #     logging.info("Typos: %s " % paragraph)    
        
        
        
        for paragraph in sp_check:
            para_tokens = split_span(paragraph['text'])
            mistake_count = mistake_count + len(paragraph['mistakes'])
            mistake_positions = [mistake['start_offset'] for mistake in paragraph['mistakes']]
            
            for token in para_tokens:
                if token[1] not in mistake_positions:
                    all_tokens.append(token[0])
                else:
                    typo = dict()
                    typo[token[0]] = []
                    all_tokens.append(typo)
    text_length = len(all_tokens)
    response = json.dumps(all_tokens)
    
    logging.info("---Check spelling: %d tokens ---" % text_length)
    logging.info("---Number of typos: %d tokens ---" % mistake_count)
    logging.info("---Total time: %s seconds ---" %  (time.time() - start_time))
    
    return response

if __name__ == '__main__':
    # app.run(debug = True, host="127.0.0.1", port=9488)
    app.run(debug = True, host="0.0.0.0", port=9489)
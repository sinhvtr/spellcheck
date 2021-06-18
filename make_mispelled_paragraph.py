import os
import sys
# from vncorenlp import VnCoreNLP
from underthesea import sent_tokenize, word_tokenize
import random
import numpy as np
import re
import csv
from tqdm import tqdm
import string

# Constants
MAX_LEN = 200
s1 = u'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
s0 = u'AAAAEEEIIOOOOUUYaaaaeeeiioooouuyAaDdIiUuOoUuAaAaAaAaAaAaAaAaAaAaAaAaEeEeEeEeEeEeEeEeIiIiOoOoOoOoOoOoOoOoOoOoOoOoUuUuUuUuUuUuUuYyYyYyYy'

s3 = u'ẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẾếỀềỂểỄễỆệỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỨứỪừỬửỮữỰự'
s2 = u'ÂâÂâÂâÂâÂâĂăĂăĂăĂăĂăÊêÊêÊêÊêÊêÔôÔôÔôÔôÔôƠơƠơƠơƠơƠơƯưƯưƯưƯưƯư'
alphabet = 'abcdefghijklmnopqrstuvwxyz'

def remove_accents(input_str):
    s = ''
    for c in input_str:
        if c in s1:
            s += s0[s1.index(c)]
        else:
            s += c
    return s

def preprocess_article(input_article):
    # annotator = VnCoreNLP(address="http://127.0.0.1", port=9000) 
    pattern = ">>.+>>" # remove hyperlinks between ">>"

    # split by max len (default 200)
    sub_paragraphs = []
    sub_paragraphs.append('')
    j = 0
    # print(len(sub_paragraphs[j].split()))

    sentences = sent_tokenize(input_article) # sentence tokenize
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

    return sub_paragraphs

def remove_special_char(input_article):
    # annotator = VnCoreNLP(address="http://127.0.0.1", port=9000) 
    pattern = ">>.+>>" # remove hyperlinks between ">>"

    sentences = sent_tokenize(input_article) # sentence tokenize
    for i in range(len(sentences)):
        sentences[i] = re.sub(pattern,"",sentences[i])
        sentences[i] = sentences[i].replace(".. ",". ")
        sentences[i] = re.sub(r'([?!,.]+)',r' \1 ', sentences[i]) 

    return sentences

def read_raw_text(input_file):
    with open(input_file, 'r') as fr:
        text = fr.read()
    return text

def generate_typos(token,
                   no_typo_prob=0.7,
                   asccents_prob=0.5,
                   lowercase_prob=0.5,
                   swap_char_prob=0.1,
                   add_chars_prob=0.1,
                   remove_chars_prob=0.1
                   ):
    if random.random() < no_typo_prob:
        # print("No typo prob")
        return token
    if random.random() < asccents_prob:
        if random.random() < 0.5:
            # print("asccents_prob < 0.5")
            token = remove_accents(token)
            # print(token)
        else:
            # print("asccents_prob >= 0.5")
            new_chars = []
            for cc in token:
                if cc in s3 and random.random() < 0.7:
                    cc = s2[s3.index(cc)]
                if cc in s1 and random.random() < 0.5:
                    cc = s0[s1.index(cc)]
                new_chars.append(cc)
            token = "".join(new_chars)
            # print(token)
    if random.random() < lowercase_prob:
        # print("lowercase_prob")bsxh
        token = token.lower()
        # print(token)
    if random.random() < swap_char_prob:
        chars = list(token)
        n_swap = min(len(chars), np.random.poisson(0.5) + 1)
        index = np.random.choice(
            np.arange(len(chars)), size=n_swap, replace=False)
        swap_index = index[np.random.permutation(index.shape[0])]
        swap_dict = {ii: jj for ii, jj in zip(index, swap_index)}
        chars = [chars[ii] if ii not in index else chars[swap_dict[ii]]
                 for ii in range(len(chars))]
        token = "".join(chars)
    if random.random() < remove_chars_prob:
        # print("remove_chars_prob")
        n_remove = min(len(token), np.random.poisson(0.005) + 1)
        for _ in range(n_remove):
            pos = np.random.choice(np.arange(len(token)), size=1)[0]
            token = token[:pos] + token[pos+1:]
        # print(token)
    if random.random() < add_chars_prob:
        # print("add_chars_prob")
        n_add = min(len(token), np.random.poisson(0.05) + 1)
        adding_chars = np.random.choice(
            list(alphabet), size=n_add, replace=True)
        for cc in adding_chars:
            pos = np.random.choice(np.arange(len(token)), size=1)[0]
            token = "".join([token[:pos], cc, token[pos:]])
        # print(token)
    # print(token)
    return token

def generate_typos_and_pos(token,
                   no_typo_prob=0.7,
                   asccents_prob=0.5,
                   lowercase_prob=0.5,
                   swap_char_prob=0.1,
                   add_chars_prob=0.1,
                   remove_chars_prob=0.1
                   ):
    
    position = "C"
    typo_type = "c"

    # If token is a punctuation, just skip and assign "C"
    if token in string.punctuation:
        return token, position, typo_type

    if random.random() < no_typo_prob:
        # print("No typo prob")
        return token, position, typo_type

    if random.random() < asccents_prob:
        if random.random() < 0.5:
            # print("asccents_prob < 0.5")
            token = remove_accents(token)
            position = "TYPO"
            typo_type = "remove_accents"
        else:
            # print("asccents_prob >= 0.5")
            new_chars = []
            for cc in token:
                if cc in s3 and random.random() < 0.7:
                    cc = s2[s3.index(cc)]
                    position = "TYPO"
                    typo_type = "remove_accents"
                if cc in s1 and random.random() < 0.5:
                    cc = s0[s1.index(cc)]
                    position = "TYPO"
                    typo_type = "remove_accents"
                new_chars.append(cc)
            token = "".join(new_chars)
            
    # if random.random() < lowercase_prob:
    #     print("lowercase_prob")
    #     token = token.lower()
    #     # print(token)
    if random.random() < swap_char_prob:
        # print("swap_char_prob")
        chars = list(token)
        n_swap = min(len(chars), np.random.poisson(0.5) + 1)
        index = np.random.choice(
            np.arange(len(chars)), size=n_swap, replace=False)
        swap_index = index[np.random.permutation(index.shape[0])]
        swap_dict = {ii: jj for ii, jj in zip(index, swap_index)}
        chars = [chars[ii] if ii not in index else chars[swap_dict[ii]]
                 for ii in range(len(chars))]
        token = "".join(chars)
        position = "TYPO"
        typo_type = "swap_char_prob"
    if random.random() < remove_chars_prob:
        # print("remove_chars_prob")
        n_remove = min(len(token), np.random.poisson(0.005) + 1)
        for _ in range(n_remove):
            pos = np.random.choice(np.arange(len(token)), size=1)[0]
            token = token[:pos] + token[pos+1:]
        position = "TYPO"
        typo_type = "remove_chars_prob"
    if random.random() < add_chars_prob:
        # print("add_chars_prob")
        n_add = min(len(token), np.random.poisson(0.05) + 1)
        adding_chars = np.random.choice(
            list(alphabet), size=n_add, replace=True)
        for cc in adding_chars:
            pos = np.random.choice(np.arange(len(token)), size=1)[0]
            token = "".join([token[:pos], cc, token[pos:]])
        position = "TYPO"
        typo_type = "add_chars_prob"

    # print(token, position, typo_type)
    return token, position, typo_type

def generate_typos_for_text(texts):
    new_texts = []
    if len(texts) > 0:
        for s in texts:
            new_s = " ".join([generate_typos(t) for t in s.split()])
            new_texts.append(new_s)
    return new_texts

def generate_typos_and_pos_for_text(sentence):
    new_texts = []
    typo_positions = []
    typo_types = []
    
    if len(sentence) > 0:
        for s in sentence:
            new_tokens = []
            for t in s.split():
                # print(t)
                new_t, typo_label, typo_type = generate_typos_and_pos(t)
                new_tokens.append(new_t)
                typo_positions.append(typo_label)
                typo_types.append(typo_type)
            new_sentence = " ".join(new_tokens)

            new_texts.append(new_sentence)
    return new_texts, typo_positions, typo_types

def main():
    input_file = sys.argv[1] # default = mispell_data_10k.txt
    output_file = sys.argv[2] # default = mispelled_data_10k.txt
    print("Reading input file")
    text = read_raw_text(input_file)
    articles = text.split("\n")
    print("Number of articles: ", len(articles))
    
    # sentences_1 = preprocess_article(articles[0])
    # for s in sentences_1:
    #     print(s)
    # exit()

    with open(output_file, "w") as csvfile:
        csv_writer =  csv.writer(csvfile, delimiter='\t')
        header = ['sentence_idx', 'word', 'tag', 'type']
        csv_writer.writerow(header)
        sentence_idx = 0

        for a in tqdm(articles, desc="Processing ..."):
            # preprocessed_sentences = remove_special_char(a)
            preprocessed_sentences = preprocess_article(a)
            # print("Done splitting text.")
            for sentence in preprocessed_sentences:       
                # if sentence is too short (<=10), skip
                if len(sentence.split()) > 9: 
                    sentence_idx = sentence_idx + 1
                    # print(sentence)
                    typo_sentence, pos, typo_type = generate_typos_and_pos_for_text([sentence])
                    if len(typo_sentence[0].split()) == len(pos):
                        for i in range(len(pos)):
                            # print(sentence_idx, )
                            csv_writer.writerow([sentence_idx, typo_sentence[0].split()[i], pos[i], typo_type[i]] )

                    # f_original.write(sentence)
                    # f_original.write("\n")
                    # f_typos.write(generate_typos_for_text([sentence])[0])
                    # f_typos.write("\n")

    print("Done. Created %s mispelled sentences." % sentence_idx) 
    # f_original.close()                 
    # f_typos.close()  

if __name__=="__main__":
    main()
import os
import sys
import csv

import random

def toggle_case(token):
    if token[0].isupper:
        return token[:1].lower() + token[1:]
    else:
        return token.title()
    
def create_ners_typos(original_ners, error_rate=0.5):
    ners_labels = []
    
    for original_ner in original_ners:
        # print('********')
        original_ner = original_ner.strip()
        # print(original_ner)
        label = 0
        new_ner = ''
        if random.random() < error_rate:
            ner_tokens = original_ner.split(' ')
            # print(ner_tokens)
            
            if len(ner_tokens) > 1:
                for token in ner_tokens:
                    print(token)
                    new_token = token
                    if random.random() < error_rate:
                        new_token = toggle_case(token)
                        label = 1
                    new_ner = new_ner + ' ' + new_token
            else:
                new_ner = original_ner
        else:
            new_ner = original_ner
        # print(new_ner)
        new_ner = new_ner.strip()
        ners_labels.append((new_ner, label))
    return ners_labels

def main():
    input_file = sys.argv[1] # default = mispell_data_10k.txt
    output_file = sys.argv[2] # default = mispelled_data_10k.txt
    
    print("Reading input file")
    with open(input_file, 'r') as fr:
        original_ners = fr.readlines()
    
    print('Total name entities: ', len(original_ners))
    number_of_ners_to_process = 100
    print('Processing ...', number_of_ners_to_process)
    ners_labels = create_ners_typos(original_ners[:number_of_ners_to_process])
    print(ners_labels)
    with open(output_file, "w") as csvfile:
        csv_writer =  csv.writer(csvfile, delimiter='\t')
        header = ['ner', 'label']
        csv_writer.writerow(header)
        for ners_label in ners_labels:
            csv_writer.writerow([ners_label[0], ners_label[1]])
    print("Done. Created %s mispelled sentences." % len(ners_labels)) 

if __name__=="__main__":
    main()
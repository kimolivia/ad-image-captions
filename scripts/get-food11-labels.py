import os
import csv
import random
import re


dir_names = [
    './data/Food-11/training',
    './data/Food-11/validation',
    './data/Food-11/evaluation']
csv_fns = [
    './data/Food-11/train-index.csv',
    './data/Food-11/validate-index.csv',
    './data/Food-11/evaluate-index.csv' 
]

def write_names(dir_name, csv_fn):
    '''Write file names in folder to csv.'''
    img_names  = os.listdir(dir_name)
    with open(csv_fn, 'w+') as f:
        f.write('img_name,label\n')
        for i in range(len(img_names)):
            img_name = img_names[i]
            pat = r'[0-9]{1,2}(?=_)'
            m = re.match(pat, img_name)
            if m:
                img_label = m.group()
            else:
                img_label = '-1'

            row = f'{img_name},{img_label}\n'
            f.write(row)

[write_names(dn, cf) for dn, cf in zip(dir_names, csv_fns)]
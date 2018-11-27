'''
Create an index for all the tweets with images

Output a csv './output/tweet-imgs-index2.csv'
'img_name', 'label', 'tweet', 'link'
'''

import numpy as np
import pandas as pd
import os
import re

root_folder = './output/tweet_info/'
twt_csvs  = os.listdir(root_folder)
img_dfs = []

def extract_fastfood(file_name):
    '''Get name of fastfood'''
    pat = r'[A-Za-z]+(?=-)'
    m = re.match(pat, file_name)
    if m:
        ff = m.group()
    else:
        ff = ''
    return ff

for twt_csv in twt_csvs:
    ff = extract_fastfood(twt_csv)
    print(ff)
    new_df = pd.read_csv(root_folder + twt_csv)
    new_df.columns = ['id', 'tweet', 'link']
    sub_df = new_df[new_df.link.notnull()]
    img_names = ['{}-i{}.jpg'.format(ff,x) for x in sub_df['id'].tolist()]
    sub_df['img_name'] = img_names
    sub_df['label'] = ['-1' for _ in range(len(img_names))]
    img_dfs.append(sub_df[['img_name', 'label', 'tweet', 'link']])

big_img_df = pd.concat(img_dfs)
file_loc = './output/tweet-imgs-index2.csv'
big_img_df.to_csv(file_loc, index=False, mode='w+')
# print(big_img_df.sample(frac=0.1))


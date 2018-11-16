import os
import csv
import random
img_names  = os.listdir('./output/imgs')
cats =  [
    'bread', 'dairy-product', 'dessert', 'egg', 'fried-food',
    'meat', 'noodles/pasta', 'rice', 'seafood', 'soup',
    'vegetable', 'fruit']

rand_cats = [cats[random.randint(0, len(cats) - 1)] for i in range(len(img_names))]
print(len(img_names), len(rand_cats))
print(img_names[:5])
print(rand_cats[:5])


csv_filename = './output/dummy-nnet-out.csv'

# write tweet info into a csv
with open(csv_filename, 'w') as f:
    f.write('img_names,rand_cats\n')
    for i in range(len(img_names)):
        row = '{},{}\n'.format(img_names[i], rand_cats[i])
        f.write(row)

import os
import random 
path = '/root/HSK/Water_Surface_Object_detection/dataset/images'
all_file = os.listdir(path)
random.shuffle(all_file)
train_ratio = 0.8
with open('./train.txt', 'w') as f:
    for i in range(int(len(all_file) * train_ratio)):
        img_path = path + '/'+ all_file[i]
        f.writelines(img_path + '\n')

with open('./val.txt', 'w') as f:
    for i in range(int(len(all_file) * train_ratio), len(all_file) - 1):
        img_path = path + '/'+ all_file[i]
        f.writelines(img_path + '\n')

print('done!')



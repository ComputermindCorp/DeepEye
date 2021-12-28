import os
import re
import csv
import numpy as np
import random
#from keras.utils import np_utils


# sort
def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

#output predict result csv file
def output_classification_result_csv(output_path, files, label, pred, all_pred):
    header = ['file' , 'label' , 'predict']

    correct_cnt = 0
    with open(output_path,'w') as fw:
        writer = csv.writer(fw)
        writer.writerow(header)

        for f, l, p, ap in zip(files, label , pred, all_pred):
            body = '"{}",{},{}'.format(f,l,p)

            for index in range(all_pred.shape[1]):
                body = '{},{}'.format(body, ap[index])

            body = body + '\n'
            fw.writelines(body)

            if l == p:
                correct_cnt += 1

        acc = float(correct_cnt) / float(len(label))
        fw.writelines(',Accuracy,{:.3%}'.format(acc))

# get dataset data
def get_data_list(input_file, classes):
    with open(input_file,'r') as f:
        reader =csv.reader(f)
        data_list = []

        for row in reader:
            if len(row) == 0:
                continue

            temp = row[0].split(" ")
            file_path = row[0][:-(len(temp[len(temp)-1]) + 1)]
            
            #if os.path.exists(temp[0]):
            if os.path.exists(file_path):

                label = np.zeros(classes)
                #label[int(temp[1])] = 1
                label[int(temp[len(temp)-1])] = 1
                label = label.reshape([1,-1])
                                
                #label = np.asarray(int(temp[1]))
                #label = np_utils.to_categorical(label, classes)
                
                #data_list.append( [temp[0], label] )
                data_list.append( [file_path, label] )
                
        data_list = random.sample(data_list, len(data_list))
    if len(data_list) == 0:
        data_list = None
    return data_list

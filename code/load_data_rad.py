from kanji_and_radicals import *
import os
import re
import sys
import numpy as np
import utils

dataset_location = "##REDACTED" # Insert path to ETL9G directory

data = {}
n_img = 0
i_added = 0
count_rad = {rad:0 for rad in radk_dict}
min_rad = 200
dataset_size = 31717

def load_dataset(db_dir, verbouse = True):
    for item in os.listdir(db_dir):
        if verbouse: print(item)
        if not re.findall("ETL9G", item)\
            or re.findall("ETL.*INFO", item)\
            or re.findall("jpg", item):
            if verbouse: print(item + " skipped!")
            continue
        elif os.path.isfile(item):
            done = load_file(item)
            if done: return

def load_file(filename, forceadd = False):
    global i_added
    block_size = 8199
    num_records = os.path.getsize(filename)//8199
    raw_records = np.fromfile(filename, dtype=np.uint8)

    for i in range(num_records):
        jis = utils.convert_jis208(raw_records[block_size*i+2:block_size*i+4])
        if jis not in jis_to_kj:
            continue
        add = True if forceadd else False
        for radical in krad_dict[jis_to_kj[jis]]:
            if count_rad[radical] < min_rad:
                add = True
                break

        if add:
            for radical in krad_dict[jis_to_kj[jis]]:
                count_rad[radical] += 1
            data["codes"][i_added] = jis
            data["images"][i_added] = utils.decode_image(raw_records[block_size*i+64:block_size*i+8192], 4)
            i_added += 1
        else:
            continue





def load():
    global data
    data = {"codes":np.zeros((dataset_size + 8424*4,), dtype=np.object),
            "images":np.zeros((dataset_size + 8424*4,127*128))}


    print("Loading training data")
    load_dataset(dataset_location)
    print("Training data Loaded!")

    print("Loading validation data")
    load_file("ETL9G_22", forceadd = True)
    load_file("ETL9G_14", forceadd = True)

    print("Loading testing data")
    load_file("ETL9G_03", forceadd = True)
    load_file("ETL9G_12", forceadd = True)
    print("Data Loaded!")

    x_train = data["images"][:dataset_size]
    y_train = make_onehot(data["codes"][:dataset_size], rad=True)
    x_val = data["images"][dataset_size:dataset_size+8424*2]
    y_val = make_onehot(data["codes"][dataset_size:dataset_size+8424*2], rad=True)
    x_test = data["images"][dataset_size+8424*2:dataset_size+8424*4]
    y_test = make_onehot(data["codes"][dataset_size+8424*2:dataset_size+8424*4], rad=True)

    return data, x_train, y_train, x_val, y_val, x_test, y_test

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

def load_dataset(db_dir, verbouse = True):
    for item in os.listdir(db_dir):
        if verbouse: print(item)
        if not re.findall("ETL9G", item)\
            or re.findall("\.", item):
            if verbouse: print(item + " skipped!")
            continue
        elif os.path.isfile(item):
            done = load_file(item)
            if done: return

def load_file(filename):
    global i_added
    block_size = 8199
    num_records = os.path.getsize(filename)//8199
    raw_records = np.fromfile(filename, dtype=np.uint8)

    for i in range(num_records):
        if i_added == data["images"].shape[0]:
            return True # Enough images loaded
        jis = utils.convert_jis208(raw_records[block_size*i+2:block_size*i+4])
        if jis not in jis_to_kj:
            continue

        data["codes"][i_added] = jis
        data["images"][i_added] = utils.decode_image(raw_records[block_size*i+64:block_size*i+8192], 4)
        i_added += 1



def load(max_jis=100):
    global data
    # Calculate sizes of each dataset
    n_train = max_jis*2106 # Number of kanji
    n_val = n_test = 10*2106 # Number of images per file, or 4 images per kanji
    n_img = n_train + 2*n_val
    if n_img > 50*4*2106:
        n_img = 50*4*2106
        n_train = n_img - 2*n_val
    global data

    data = {"codes":np.zeros((n_img,), dtype=np.object),
            "images":np.zeros((n_img,127*128))}

    print("Loading ETL9G")
    load_dataset(dataset_location)
    print("ETL9G Loaded!")

    x_val = data["images"][:n_val]
    x_test = data["images"][n_val:n_val+n_test]
    y_val = make_onehot(data["codes"][:n_val])
    y_test = make_onehot(data["codes"][n_val:n_val+n_test])
    x_train = data["images"][n_val+n_test:n_val+n_test+n_train]
    y_train = make_onehot(data["codes"][n_val+n_test:n_val+n_test+n_train])

    return data, x_train, y_train, x_val, y_val, x_test, y_test

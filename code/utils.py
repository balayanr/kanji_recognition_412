"""
Utilities needed to load and decode ETL datasets
"""

import numpy as np
from definitions import *


# Image decoder for 4 and 1 bit images
def decode_image(raw_image, bpc):
    img = np.zeros((len(raw_image)*8//bpc,), dtype = np.uint8)
    i = 0
    for block in raw_image:
        for mask in pixel_masks[bpc]:
            img[i] = (block & pixel_masks[bpc][mask]["mask"]) \
                            >> pixel_masks[bpc][mask]["bitshift"]
            i += 1
    return (img * conversion_factor[bpc])


# Decodes raw 6 bit record to 8bit characters
def decode_record(raw_data):
    data = np.zeros((len(raw_data)*8//6,), dtype = np.uint8)
    for i in range(len(raw_data)//3):
        block = join_bits(raw_data[3*i:3*i+3], 8)
        for mask in pixel_masks[6]:
            data[i*4+mask] = block & pixel_masks[6][mask]["mask"]\
                                  >> pixel_masks[6][mask]["bitshift"]
    return data


# Converts an array of numbers into a single int
def join_bits(string, active_bits):
    out_char = string[0]
    for char in string[1:]:
        out_char = (out_char << active_bits) | char
    return out_char


# Takes an array of T56-encoded characters and decodes them
def decode_t56code(string):
    decoded = [None] * len(string)
    for i, char in enumerate(string):
        decoded[i] = T56CODE[char]
    return ''.join(decoded)


# Converts a 2-byte integer into a proper jis x 0208 code
def convert_jis208(code):
    return '-'.join([str(code[0] - 0x20).zfill(2), str(code[1]  - 0x20).zfill(2)])

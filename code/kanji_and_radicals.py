"""
Preparation of kanji and radical datasets.
"""
FAKERADS = {'鼓', '香', '鼻'}
import numpy as np
import sys

joyo_list_location = '../data/joyo_kanji.txt'
kradfile_location = '../data/kradfile-u.gz'
radkfile_location = '../data/radkfile-u-jis208.txt'
jis_to_kj = "../data/jis208_to_kj.txt"


# Load files
with open(joyo_list_location, encoding="utf-8") as joyo_file:
    joyo_kanji = set(joyo_file.readlines()[0])
    joyo_kanji.remove('\n')

with open(kradfile_location, encoding="utf-8") as kradfile:
    krad_dict = {}
    for kanji in kradfile.readlines():
        if kanji[0] == '#' or kanji[0] not in joyo_kanji: continue
        kanji = kanji.split()
        krad_dict[kanji[0]] = set(kanji[2:]) - FAKERADS

with open(radkfile_location, encoding="utf-8") as radkfile:
    radk_dict = {}
    for radical in radkfile.readlines():
        if radical[0] == '#' or radical[0] in FAKERADS: continue
        # radical = radical.split()
        radk_dict[radical[0]] = list(joyo_kanji.intersection(list(radical[4:-1])))

with open(jis_to_kj, encoding="utf-8") as j2kj:
    jis_to_kj = {}
    kj_to_jis = {}
    for line in j2kj.readlines():
        line = line.split(" ")
        try:
            int(line[0])
        except:
            continue
        if int(line[0]) < 16: continue
        jis = line[0] + '-' + line[1]
        kj = line[-1][0]
        if kj not in joyo_kanji: continue

        jis_to_kj[jis] = kj
        kj_to_jis[kj] = jis


# Process data:

# Cross-reference the dictionaries, there was a radical and a kanji missing (before removal of non-joyo kanji, didn't check after)
for kanji in krad_dict:
    for radical in krad_dict[kanji]:
        if radical not in radk_dict:
            radk_dict[radical] = {kanji}
        if kanji not in radk_dict[radical]:
            radk_dict[radical].add(kanji)

for radical in radk_dict:
    for kanji in radk_dict[radical]:
        if kanji not in krad_dict:
            krad_dict[kanji] = {radical}
        if radical not in krad_dict[kanji]:
            krad_dict[kanji].add(radical)


# Remove useless radicals
radcount = {rad:len(radk_dict[rad]) for rad in radk_dict}
for rad in radcount:
    if radcount[rad] == 0:
        del radk_dict[rad]
radcount = {rad:len(radk_dict[rad]) for rad in radk_dict}


# Enumerate radicals and create a stable mapping for them
rad_num = {}
num_rad = {}
for i, radical in enumerate(radk_dict.keys()):
    rad_num[radical] = i
    num_rad[i] = radical




# Enumerate jis codes and create a stable mapping for them
jis_num = {}
num_jis = {}
for i, jis in enumerate(jis_to_kj.keys()):
    jis_num[jis] = i
    num_jis[i] = jis


# Create one-hot encoding for a list of kanji
def make_onehot(jis_list, rad = False):
    if rad:
        onehot = np.zeros((len(jis_list), len(rad_num)))
        for i, jis in enumerate(jis_list):
            for radical in krad_dict[jis_to_kj[jis]]:
                onehot[i, rad_num[radical]] = 1
        return onehot
    else:
        onehot = np.zeros((len(jis_list), len(jis_to_kj)))
        for i, jis in enumerate(jis_list):
            onehot[i, jis_num[jis]] = 1
        return onehot

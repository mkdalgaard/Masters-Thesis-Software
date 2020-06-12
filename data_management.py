# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 03:30:03 2020

@author: Martin Kamp Dalgaard
"""

import pandas as pd
import pickle

# Demographics
demo_header = pd.read_csv("data/raw/demographic_header.csv")
demo_data = pd.read_csv("data/raw/demographic.csv", index_col=False,
                        header=None, names=list(demo_header.columns))

# Actual data
data_header = pd.read_csv("data/raw/data_header.csv")
data = {}

for i in range(1, 21):
    if i < 10:
        subj_numb = str(0) + str(i)
    else:
        subj_numb = str(i)
    data["subject_%s" %i] = pd.read_csv("data/raw/subject_%s.csv" %subj_numb,
                                        index_col=False, header=None,
                                        names=list(data_header.columns))

# Saving merged data
with open('data/merged_data.pickle', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Truncating each session to only contain 5120 samples (or less)
sess_dic = {}
for subject in range(1, 21):
    sess_dic[str(subject)] = {}
    df = data["subject_%s" %subject]
    sub_idxs = list(df[(df["EyesClosed"]==1) | (df["EyesOpened"]==1)].index)
    sub_idxs.append(len(df))
    sess_lens = []
    for i in range(len(sub_idxs)-1):
        sess_lens.append(sub_idxs[i+1] - sub_idxs[i])
    j = 0
    for sess_len in sess_lens:
        sess_start = sub_idxs[j]
        if sess_len > 5120: # Three sessions are below 5120 samples
            sess_end = sess_start + 5120
        else:
            sess_end = sess_start + sess_len
        sess_dic[str(subject)][str(j+1)] = df.iloc[sess_start:sess_end,:]
        j += 1

# Sanity check
for subject in range(1, 21):
    print("\nSubject: %d" %subject)
    for key in sess_dic[str(subject)].keys():
        print("Length: %d. Eyes closed: %d. Eyes opened: %d."
              %(len(sess_dic[str(subject)][key]),
                sess_dic[str(subject)][key]["EyesClosed"].iloc[0],
                sess_dic[str(subject)][key]["EyesOpened"].iloc[0]))

trunc_data = {}
for subject in range(1, 21):
    trunc_data["subject_%s" %subject] = pd.concat(sess_dic[str(subject)],
                                                      ignore_index=True)

with open('data/truncated_data.pickle', 'wb') as handle:
    pickle.dump(trunc_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

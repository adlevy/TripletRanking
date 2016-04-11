import os
import sys
import math
import sidekit
import fuel
import h5py
from fuel.datasets.hdf5 import H5PYDataset
import numpy as np

from collections import OrderedDict
from fuel.datasets import IndexableDataset
from fuel.schemes import (SequentialScheme, ShuffledScheme, SequentialExampleScheme, ShuffledExampleScheme)
from fuel.schemes import ConstantScheme
from fuel.transformers import Mapping, Batch, Padding, Filter, Unpack
from fuel.streams import DataStream

import sympy

""" ------------------ Set a few parameters here --------------------"""
Test = False

""" ---------------------------------------------------------------- """


# LOAD STATSERVER OF I-VECTORS
ivss = sidekit.StatServer("iv_sre04050608_m_training_tandem.h5")
mu = ivss.get_mean_stat1()
std = ivss.get_total_covariance_stat1()
ivss.whiten_stat1(mu, std)
ivss.norm_stat1()

# GET LIST OF UNIQUE SPEAKERS WITH NUMBER OF SESSIONS FOR EACH
unique_spk = set(ivss.modelset.tolist())
speakers = []
sess = []
for spk in unique_spk:
    speakers.append((spk, (ivss.modelset == spk).sum()))
    sess.append((ivss.modelset == spk).sum())

# COMPUUTE THE NUMBER OF POSSIBLE UNIQUE TARGET TRIPLETS
positive_example = int(0)
for spk, sess in speakers:
    positive_example += int(sympy.binomial(sess, 2))

# GENERATE ALL POSSIBLE PERMUTATIONS OF TWO I-VECTORS FROM A SAME SPEAKER. FOR EACH COUPLE, CREATE 10 TRIPLETS WITH NEGATIVE EXAMPLES
positive_example = int(0)
for spk, sess in speakers:
    positive_example += sess * (sess - 1)

iv_example = []
iv_positive = []
iv_negative = []

for spk in unique_spk:
    target_iv = ivss.get_model_stat1(spk)
    tmp_example = target_iv.repeat(target_iv.shape[0]-1, axis=0)
    tmp = []
    for i in range(target_iv.shape[0]):
        tmp.append(np.delete(target_iv, i, 0))
    tmp_positive = np.vstack(tmp)
    iv_example.append(tmp_example.repeat(10, axis=0))
    iv_positive.append(tmp_positive.repeat(10, axis=0))
    iv_negative.append(ivss.stat1[ivss.modelset != spk , :][np.random.choice((ivss.modelset != spk).sum(), iv_example[-1].shape[0]), :])
    
example = np.vstack(iv_example)
positive = np.vstack(iv_positive)
negative = np.vstack(iv_negative)



# Create the HDF5 file
with h5py.File('~larcher/expe/triplet_ranking/sre04050608SWB_dataset_norm.hdf5', mode='w') as f:
    
    # Store example
    iv_ex = f.create_dataset("iv_example", data=example.astype('float32'))
    iv_ex.dims[0].label = 'sessions'
    iv_ex.dims[1].label = 'vectsize'

    # Store positive examples
    iv_pos = f.create_dataset("iv_positive", data=positive.astype('float32'))
    iv_pos.dims[0].label = 'sessions'
    iv_pos.dims[1].label = 'vectsize'

    # Store negative examples
    iv_neg = f.create_dataset("iv_negative", data=negative.astype('float32'))
    iv_neg.dims[0].label = 'sessions'
    iv_neg.dims[1].label = 'vectsize'

    train_part = int(iv_ex.shape[0] * 0.9)
    split_dict = {
        'train': {'iv_example': (0, train_part), 'iv_positive': (0, train_part), 'iv_negative': (0, train_part)},
        'validation': {'iv_example': (train_part, iv_ex.shape[0]), 'iv_positive': (train_part, iv_pos.shape[0]), 'iv_negative': (train_part, iv_neg.shape[0])}}
    f.attrs['split'] = H5PYDataset.create_split_array(split_dict)


if Test:

    # TEST THE NEWLY CREATED DATASET
    train_set = H5PYDataset('sre04050608SWB_dataset.hdf5', which_sets=('train',))
    cv_set = H5PYDataset('sre04050608SWB_dataset.hdf5', which_sets=('validation',))



    handle = train_set.open()
    iv = train_set.get_data(handle, slice(0, 9))
    train_set.close(handle)

    scheme = ShuffledScheme(examples=train_set.num_examples, batch_size=4)


    # Test d'un datastream
    data_stream = DataStream(dataset=train_set, iteration_scheme=scheme)
                          

    for idx, data in enumerate(data_stream.get_epoch_iterator()):
        #print(data[0].shape, data[1].shape)
        #print(data[0].shape)
        if idx == 2:
            print(data[0][0].shape)
            print(data[1][0].shape)



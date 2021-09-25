import numpy as np
import random
import tensorflow as tf
from ase.db import connect
from base64 import b64encode, b64decode
import pickle
import joblib
import torch
import os
import schnetpack as spk
from schnetpack.datasets import QM9
from megnet.activations import softplus2
from typing import Dict, List, Callable
from dscribe.descriptors import SOAP
from sklearn.decomposition import PCA

species = ["H", "C", "O", "N"]
atomic_numbers = [1, 8]
rcut = 6.0
nmax = 8
lmax = 6

def mae_loss(n1, n2):
    err, total_err = 0, 0
    for i in range(len(n1)):
        err = abs(n1[i] - n2[i])
        total_err += err
    return total_err/len(n1)

for ran in [0]:
    RANDOM_SEED = ran
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTHONHASHSEED'] = 'RANDOM_SEED'
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    tf.random.set_seed(RANDOM_SEED)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation, LSTM, RepeatVector
    from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
    from tensorflow.keras import optimizers

    qm9data_withot_F = QM9('./qm9_without_F.db', download=False, load_only=[QM9.G], remove_uncharacterized=True)
    qm9data = QM9('./qm9.db', download=False, load_only=[QM9.G], remove_uncharacterized=True)
    atomrefs = qm9data.get_atomref(QM9.G)
    print('G of hyrogen:', format(atomrefs[QM9.G][1][0]), 'eV')
    print('G of carbon:', format(atomrefs[QM9.G][6][0]), 'eV')
    print('G of nitrogen:', format(atomrefs[QM9.G][7][0]), 'eV')
    print('G of oxygen:', format(atomrefs[QM9.G][8][0]), 'eV')
    G_H = '{:.10f}'.format(atomrefs[QM9.G][1][0])
    G_C = '{:.10f}'.format(atomrefs[QM9.G][6][0])
    G_N = '{:.10f}'.format(atomrefs[QM9.G][7][0])
    G_O = '{:.10f}'.format(atomrefs[QM9.G][8][0])

    db = connect('qm9_without_F.db')
    rows = list(db.select(sort='id'))
    #rows = list(db.select('id<200'))

    # load data
    with open('/data2/deh/cm_data.lst', 'rb') as fp:
        print('start to load data.')
        dataset = joblib.load(fp)
    print(dataset[0])
    print(len(dataset), len(dataset[0]))
    data = dataset

    # process label
    label = []
    for row in rows:
        at, props = qm9data_withot_F.get_properties(idx=row.id - 1)
        at_num = at.numbers
        H_num = 0
        C_num = 0
        N_num = 0
        O_num = 0
        for i in range(len(at_num)):
            if at_num[i] == 1:
                H_num += 1
            elif at_num[i] == 6:
                C_num += 1
            elif at_num[i] == 7:
                N_num += 1
            else:
                O_num += 1
        pp = props[QM9.G].cpu().numpy()[0] - float(G_H) * H_num - float(G_C) * C_num - float(G_N) * N_num - float(
            G_O) * O_num
        label.append(pp)

    data = np.array(data)
    pca = PCA(n_components=99)
    pca_data = pca.fit_transform(data)

    print('data process finish!')
    print('start to cat data')
    print(len(data), len(data[0]))

    data_arr = np.array(pca_data, dtype=np.float32)
    label_arr = np.array(label, dtype=np.float32)

    # 回调函数
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=2)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)

    input_shape = (99,)
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=input_shape))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.build(99,)
    adam = optimizers.Adam(lr=0.01)
    model.compile(loss='mae', optimizer=adam, metrics=['mae'])

    data_train = data_arr[:int(len(data_arr) * 0.8)]
    data_val = data_arr[int(len(data_arr) * 0.8): int(len(data_arr) * 0.9)]
    data_test = data_arr[int(len(data_arr) * 0.9):]
    target_train = label_arr[:int(len(data_arr) * 0.8)]
    target_val = label_arr[int(len(data_arr) * 0.8): int(len(data_arr) * 0.9)]
    target_test = label_arr[int(len(data_arr) * 0.9):]

    model.fit(data_train, target_train, shuffle=True, batch_size=32, epochs=1000, verbose=1,
              callbacks=[reduce_lr, early_stopping], validation_data=(data_val, target_val))

    print(data_test[0])
    prediction = model.predict(data_test)

    test_e = mae_loss(prediction, target_test)

    print(test_e)
    print('Training End!')

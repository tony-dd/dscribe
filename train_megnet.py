import tensorflow as tf
from ase import Atom
import numpy as np
from ase.db import connect
import pickle
import schnetpack as spk
from schnetpack.datasets import QM9
from megnet.models import MEGNetModel
from megnet.data.graph import GaussianDistance
from megnet.data.crystal import CrystalGraph
from megnet.utils.preprocessing import StandardScaler
from pymatgen.io.ase import AseAtomsAdaptor
aaa = AseAtomsAdaptor()

with tf.device('/gpu:0'):
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
    # rows = list(db.select('id<101'))
    structures, targets = [], []
    for row in rows:
        atoms = row.toatoms()
        struct = aaa.get_molecule(atoms, cls=None)
        at, props = qm9data_withot_F.get_properties(idx=row.id - 1)
        structures.append(struct)
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
        targets.append(pp)

    train_structures = structures[:int(len(structures) * 0.9)]
    vali_structures = structures[int(len(structures) * 0.9):]
    train_targets = targets[:int(len(structures) * 0.9)]
    vali_targets = targets[int(len(structures) * 0.9):]

    gc = CrystalGraph(bond_converter=GaussianDistance(
        np.linspace(0, 5, 100), 0.5), cutoff=4)
    model = MEGNetModel(100, 2, graph_converter=gc)

    INTENSIVE = False
    scaler = StandardScaler.from_training_data(train_structures, train_targets, is_intensive=INTENSIVE)
    model.target_scaler = scaler

    model.train(train_structures, train_targets, vali_structures, vali_targets, epochs=2000, verbose=2)


    def mae_loss(n1, n2):
        err, total_err = 0, 0
        for i in range(len(n1)):
            err = abs(n1[i] - n2[i])
            total_err += err
        return total_err / len(n1)


    predicted_tests = []
    for i in vali_structures_structures:
        predicted_tests.append(model.predict_structure(i).ravel()[0])

    print(predicted_tests[:10])
    print(test_targets[:10])

    print('mae:', mae_loss(predicted_tests, vali_targets))

    model.save_model('test.hdf5')
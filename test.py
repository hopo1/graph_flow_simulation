"""
Script for testing models defined in model_def.py
"""

import tensorflow as tf
import pickle

from impl.common import NodeType, get_init_grp, rollout
from impl.dataset import load_dataset

from model_def import models

data_path = 'data/airfoil'
data_test = 'test'

mod = "named12n"

selected_id = 49  # correct ids m-15 188, m-12 31, m-12n 49
m_name = mod + '-' + str(selected_id)

params = models[mod]["params"]
targets = tuple([x["field"] for x in params])

md = models[mod]
model = md["model"](**md["args"])
bsize = 1
mx_iter = 1

dtTest = load_dataset(data_path, data_test)
chck_root = "models/" + md["path"] + "/"
chck_name = mod

chck = tf.train.Checkpoint(module=model)

kk = iter(dtTest).next()

loss_mask = tf.logical_or(tf.equal(kk['node_type'][0, :, 0], NodeType.NORMAL),
                          tf.equal(kk['node_type'][0, :, 0], NodeType.OUTFLOW))
loss_mask = tf.reshape(tf.concat([loss_mask for _ in range(md['args']['node_feat_cnt'])], -1),
                       [-1, md['args']['node_feat_cnt']])


def make_loss(model, c_name, mx_iter=600, init=0):
    chck = tf.train.Checkpoint(module=model)
    chck.restore(chck_root + c_name)
    error = tf.zeros((601, 5233, 2))
    itTest = iter(dtTest)
    for val in itTest:
        rr = rollout(model, get_init_grp(val, targets, init), mx_iter, loss_mask)
        pred = tf.concat([x.nodes[..., :model.learn_features] for x in rr], axis=0)
        correct = tf.concat([val[x] for x in targets], axis=2)
        pred = tf.reshape(pred, [mx_iter + 1, *correct.shape[1:]])
        error += (pred - correct[init:mx_iter + +init + 1]) ** 2
    return error


rs = {}

print("start_count", flush=True)

for i in [0]:
    rs[i] = make_loss(model, m_name, mx_iter=600)
    print(i, "done", flush=True)

with open("results/" + m_name + "_test.pickle", 'wb') as x:
    pickle.dump(rs, x)

print("Results save to", "results/" + m_name + "_test.pickle")

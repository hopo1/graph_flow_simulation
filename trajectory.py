# -*- coding: utf-8 -*-
"""visualize.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1na9QeKK3ZY3D8ieR_I5aw9HNeMEskZqn
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
try:
    import google.colab

    IN_COLAB = True
except:
    IN_COLAB = False
print(IN_COLAB)

if IN_COLAB and 'drive' not in os.listdir("/content"):
    from google.colab import drive

    drive.mount('/content/drive')

if IN_COLAB:
    os.chdir('/content/drive/MyDrive/bakalarka')

import tensorflow as tf

import pickle
#import matplotlib.pyplot as plt
#import plotly.graph_objects as go
#from plotly.subplots import make_subplots

from graph_nets import utils_tf

from impl.common import NodeType, triangles_to_edges
from impl.dataset import load_dataset, add_targets

from impl.EncodeProcessDecode import EncodeProcessDecode as EncodeProcessDecodeNamedMultinet

data_path = 'data/airfoil'
data_train = 'small' if IN_COLAB else 'train'
data_valid = 'small' if IN_COLAB else 'valid'
data_test = 'small' if IN_COLAB else 'test'

mod = "named12n"
models = {"named": {"model": EncodeProcessDecodeNamedMultinet,
                    "args": {"steps": 15,
                             "learn_features": 2,
                             "n_layers": 2,
                             "lat_size": 128,
                             "edge_feat_cnt": 3,
                             "node_feat_cnt": 5},
                    "params": [dict(noise=0.02, gamma=1.0, field='velocity', history=False,
                                    size=2, batch=2)],
                    "path": "namednet"},
          "named18": {"model": EncodeProcessDecodeNamedMultinet,
                      "args": {"steps": 18,
                               "learn_features": 2,
                               "n_layers": 2,
                               "lat_size": 128,
                               "edge_feat_cnt": 3,
                               "node_feat_cnt": 5},
                      "params": [dict(noise=0.02, gamma=1.0, field='velocity', history=False,
                                      size=2, batch=2)],
                      "path": "namednet18"},
          "named12": {"model": EncodeProcessDecodeNamedMultinet,
                      "args": {"steps": 12,
                               "learn_features": 2,
                               "n_layers": 2,
                               "lat_size": 128,
                               "edge_feat_cnt": 3,
                               "node_feat_cnt": 5},
                      "params": [dict(noise=0.02, gamma=1.0, field='velocity', history=False,
                                      size=2, batch=2)],
                      "path": "namednet12"
                      },
          "named-full": {"model": EncodeProcessDecodeNamedMultinet,
                         "args": {"steps": 12,
                                  "learn_features": 4,
                                  "n_layers": 2,
                                  "lat_size": 128,
                                  "edge_feat_cnt": 3,
                                  "node_feat_cnt": 7},
                         "params": [dict(noise=0.02, gamma=1.0, field='velocity', history=False,
                                         size=2, batch=2),
                                    dict(noise=0.02, gamma=1.0, field='pressure', history=False,
                                         size=1, batch=2),
                                    dict(noise=0.02, gamma=1.0, field='density', history=False,
                                         size=1, batch=2)],
                         "path": "fullnet"},
          "named12n": {"model": EncodeProcessDecodeNamedMultinet,
                       "args": {"steps": 12,
                                "learn_features": 2,
                                "n_layers": 2,
                                "lat_size": 128,
                                "edge_feat_cnt": 3,
                                "node_feat_cnt": 5},
                       "params": [dict(noise=1.0, gamma=1.0, field='velocity', history=False,
                                       size=2, batch=2)],
          "path": "namednet12n"}}

params = models[mod]["params"]
targets = tuple([x["field"] for x in params])

dt2 = load_dataset(data_path, data_train)
dt2 = add_targets(dt2, targets, add_history=params[0]['history'])
qq = iter(dt2).next()

NodeTypeCnt = tf.unique(tf.reshape(qq['node_type'][0], qq['node_type'][0].shape[:1])).y.shape[0]


@tf.function
def toGraphsTuple(d, targets=('velocity',)):
    send, recive = triangles_to_edges(d['cells'])
    rel_pos = (tf.gather(d['mesh_pos'], send) - tf.gather(d['mesh_pos'], recive))
    nodes_unique = tf.unique_with_counts(tf.reshape(d["node_type"], [-1]))
    one_hot = tf.one_hot(nodes_unique.idx, NodeTypeCnt, dtype=tf.float32)
    dd = {
        "nodes": tf.concat([*[d[x] for x in targets], one_hot], 1),
        # on change update loss function ^
        "senders": send,
        "receivers": recive,
        "edges": tf.concat([
            rel_pos,
            tf.norm(rel_pos, axis=-1, keepdims=True)], 1)
    }
    return utils_tf.data_dicts_to_graphs_tuple([dd])


def get_init_grp(qq, targets=('velocity',), st=0):
    qqq = {}
    for i, j in qq.items():
        qqq[i] = j[st]
    return toGraphsTuple(qqq, targets)


#@tf.function
def rollout(model, grp_, length, loss_mask):
    res = [grp_, ]
    for i in range(length):
        grp2_ = model(grp_, False)
        grp_ = grp_.replace(nodes=tf.where(loss_mask, grp2_.nodes, grp_.nodes))
        res.append(grp_)
    return res


md = models[mod]
model = md["model"](**md["args"])
bsize = 1
mx_iter = 1

dtTrain = load_dataset(data_path, data_train)
dtValid = load_dataset(data_path, data_valid)
#dtTrain = add_batch(prepare(split_dataset(add_targets(dtTrain, targets, False)), 100 if IN_COLAB else 1000), bsize)
dtTest = load_dataset(data_path, data_test)
chck_root = "models/" + md["path"] + "/"
chck_name = mod

chck = tf.train.Checkpoint(module=model)

itTrain = iter(dtTrain)
itValid = iter(dtValid)

kk = itValid.next()

loss_mask = tf.logical_or(tf.equal(kk['node_type'][0, :, 0], NodeType.NORMAL),
                          tf.equal(kk['node_type'][0, :, 0], NodeType.OUTFLOW))
loss_mask = tf.reshape(tf.concat([loss_mask for _ in range(md['args']['node_feat_cnt'])], -1),
                       [-1, md['args']['node_feat_cnt']])


def make_traj(model, c_name, mx_iter=600, init=0):
    chck = tf.train.Checkpoint(module=model)
    chck.restore(chck_root + c_name)
    res = []
    itTest = iter(dtTrain)
    ctr = 0
    for val in itTest:
        rr = rollout(model, get_init_grp(val, targets, init), mx_iter, loss_mask)
        res.append(rr)
        ctr+=1
        if ctr>10:
          break
    return res


rs = {}

if mod == 'named':
    m_name = 'named-188'
if mod == 'named12':
    m_name = 'named12-31'

if mod == 'named12n':
    m_name = 'named12n-49'

print("start_count",flush=True)

for i in [0]: #, 5, 10, 20, 30, 50, 100]:
    rs = make_traj(model, m_name, mx_iter=600 - i, init=i)
    print(i, "done", flush=True)

with open("results/" + m_name + "_test_track.pickle",'wb') as x:
    pickle.dump(rs, x)
"""
Script for evaluating different iteration of model for models defined in model_def.py
"""
import os
import pickle

import tensorflow as tf

import sys
import re

from impl.common import NodeType, rollout, get_init_grp, toGraphsTuple
from impl.dataset import load_dataset, add_batch, prepare, split_dataset, add_targets
from model_def import models

data_path = 'data/airfoil'
data_valid = 'valid'
data_train = 'train'

mod = "named12n"

bsize = 10
mx_iter = 21

params = models[mod]["params"]
targets = tuple([x["field"] for x in params])

md = models[mod]
model = md["model"](**md["args"])

dtTrain = load_dataset(data_path, data_train)
dtValid = load_dataset(data_path, data_valid)
dtTrain = add_batch(prepare(split_dataset(add_targets(dtTrain, targets, False)), 1000), bsize)

chck_root = "models/" + md["path"] + "/"
chck_name = mod

chck = tf.train.Checkpoint(module=model)

latest = tf.train.latest_checkpoint(chck_root)
if latest is not None:
    print("loading", latest)
    chck.restore(latest)

models_ = set([re.match('\w+-\w*-?\d+', x)[0] for x in filter(lambda x: x != "checkpoint", os.listdir(chck_root))])

itTrain = iter(dtTrain)
itValid = iter(dtValid)

kk = itValid.next()

loss_mask = tf.logical_or(tf.equal(kk['node_type'][0, :, 0], NodeType.NORMAL),
                          tf.equal(kk['node_type'][0, :, 0], NodeType.OUTFLOW))
loss_mask = tf.reshape(tf.concat([loss_mask for _ in range(md['args']['node_feat_cnt'])], -1),
                       [-1, md['args']['node_feat_cnt']])

chck = tf.train.Checkpoint(module=model)
poss_steps = [x for x in [1, 10, 20, 50, 100, 200, 600] if x <= mx_iter]
print(poss_steps)
stats = {}
losses = {}
print("start eval")
sys.stdout.flush()
for mid in models_:
    chck.restore(chck_root + mid)
    tr = itTrain.next()
    gr = toGraphsTuple(tr, targets)
    loss = model.loss(gr, tr, targets) / bsize
    losses[mid] = loss
    itValid = iter(dtValid)
    eval = {x: 0 for x in poss_steps}
    for val in itValid:
        rr = rollout(model, get_init_grp(val, targets), mx_iter, loss_mask)
        pred = tf.concat([x.nodes[..., :model.learn_features] for x in rr], axis=0)
        correct = tf.concat([val[x] for x in targets], axis=2)
        pred = tf.reshape(pred, [mx_iter + 1, *correct.shape[1:]])
        error = tf.reduce_mean((pred - correct[:mx_iter + 1]) ** 2, axis=-1)
        for x in poss_steps:
            eval[x] += tf.reduce_mean(error[1:x + 1])
    stats[mid] = eval
    print(mid)
    sys.stdout.flush()

ref = [{"name": k,
        "i": int(re.search("\d+$", k)[0]) * 40000,
        "loss": losses[k].numpy(),
        "stat": {i: j.numpy() for i, j in v.items()}} for k, v in stats.items()]

with open("results/" + mod + ".pickle", 'wb') as x:
    pickle.dump(ref, x)

print("Results saved to:", "results/" + mod + ".pickle")

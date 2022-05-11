# -*- coding: utf-8 -*-
"""graph_lib.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1kT6JQ-uOB7X9H1CUbss55QI_6HCNFIUP

## Setup
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
try:
    import google.colab

    IN_COLAB = True
except:
    IN_COLAB = False
print(IN_COLAB)

#!pip install graph_nets  dm-sonnet tensorflow_probability

if IN_COLAB and 'drive' not in os.listdir("/content"):
    from google.colab import drive

    drive.mount('/content/drive')

if IN_COLAB:
    os.chdir('/content/drive/MyDrive/bakalarka')

import tensorflow as tf

import time
import sys
import re

from graph_nets import utils_tf

from EncodeProcessDecodeBasic import EncodeProcessDecode
from EncodeProcessDecodeNorm import EncodeProcessDecode as EncodeProcessDecodeNorm
from EncodeProcessDecodeAddNorm import EncodeProcessDecode as EncodeProcessDecodeAddNorm
from EncodeProcessDecodeNoLib import EncodeProcessDecode as EncodeProcessDecodeNoLib
from EncodeProcessDecodeMultinet import EncodeProcessDecode as EncodeProcessDecodeMultinet
from EncodeProcessDecodeNamedMultinet import EncodeProcessDecode as EncodeProcessDecodeNamedMultinet

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

from common import NodeType

"""## Data Loading"""

data_path = 'data/airfoil'
data_train = 'small' if IN_COLAB else 'train'
data_valid = 'small' if IN_COLAB else 'valid'
data_test = 'small' if IN_COLAB else 'test'

from dataset import load_dataset, split_dataset, prepare, add_noises, add_targets, add_batch, triangles_to_edges

ds = load_dataset(data_path, data_train)
ds = add_targets(ds, targets, add_history=params[0]['history'])

ds = split_dataset(ds)
for param in params:
    ds = add_noises(ds, noise_field=param['field'],
                    noise_scale=param['noise'],
                    noise_gamma=param['gamma'])
ds = prepare(ds, 100 if IN_COLAB else 10000)

ds = add_batch(ds, params[0]['batch'])

"""## Prepare for learning"""

itr = iter(ds)
d = itr.next()

NodeTypeCnt = tf.unique(tf.reshape(d['node_type'], d['node_type'].shape[:1])).y.shape[0]

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

dd = toGraphsTuple(d, targets)
LINEAR_FEATURES = dd.nodes.shape[1] - NodeTypeCnt

dd

aa = itr.next()

"""## Model and loss function

### Model hyperparameters
"""

from AdamModLr import Adam

lr = 1e-4
# opt = snt.optimizers.Adam(lr)
opt = Adam(lr)
steps = 15
#model = EncodeProcessDecode(steps, LINEAR_FEATURES, [128] * 2, [128] * 2 + [4])
md = models[mod]
model = md["model"](**md["args"])
initial_learning_rate = lr
decay_rate = 0.1
decay_steps = 5e6
learning_increase = 1e-6

"""## Training"""

def decayed_learning_rate(step):
    return initial_learning_rate * decay_rate ** (step / decay_steps) + learning_increase

@tf.function
def update_step(data, targets=('velocity')):
    print("Tracing!")
    grp = toGraphsTuple(data, targets)
    with tf.GradientTape() as tape:
        los = model.loss(grp, data, targets=targets)  #change to loss

    gradients = tape.gradient(los, model.trainable_variables)

    opt.apply(gradients, model.trainable_variables)
    return los

update_step(aa, targets);

chck_root = "models/" + md['path'] + '/'
chck_name = mod

chck = tf.train.Checkpoint(module=model)

latest = tf.train.latest_checkpoint(chck_root)
if latest is not None:
    print("loading", latest)
    chck.restore(latest)

if latest is None and not isinstance(model, EncodeProcessDecode):
    print("Acumulating")
    for i in range(1000):
        data = itr.next()
        grp = toGraphsTuple(data, targets)
        model.loss(grp, data,targets=targets)

colab_train = True
start = 0
sm_print = 500
save_itr = 40000
if not IN_COLAB or colab_train:
    if latest is not None:
        start = int(re.findall('\d+$', latest)[0]) * save_itr
    t = time.time()
    print("training")
    sys.stdout.flush()
    for i in range(start, int(1e7)+1):
        a = itr.next()
        m = update_step(a,targets)
        if i % sm_print == 0:
            opt.learning_rate.assign(decayed_learning_rate(i))
            print("i", i, "mse:", m.numpy())
            if i and i % save_itr == 0:
                sys.stdout.flush()
                chck.save(chck_root + chck_name)
    print(time.time() - t)

dt2 = load_dataset(data_path, data_train)
dt2 = add_targets(dt2, targets, add_history=params[0]['history'])
qq = iter(dt2).next()
qqq = {}
for i, j in qq.items():
    qqq[i] = j[0]
grp_ = toGraphsTuple(qqq, targets)

tf.reduce_min(qqq['density']), tf.reduce_max(qqq['density']),tf.reduce_min(qqq['velocity'][0]), tf.reduce_max(qqq['velocity'][0]),tf.reduce_min(qqq['pressure']), tf.reduce_max(qqq['pressure'])

if "m" in locals():
    m.numpy()

"""## Vizualization"""

res = [grp_, ]
loss_mask = tf.logical_or(tf.equal(qqq['node_type'][:, 0], NodeType.NORMAL),
                          tf.equal(qqq['node_type'][:, 0], NodeType.OUTFLOW))
feat = grp_.nodes.shape
loss_mask = tf.reshape(tf.concat([loss_mask for _ in range(grp_.nodes.shape[1])], -1), [-1, grp_.nodes.shape[1]])
for i in range(600):
    grp2_ = model(grp_, False)
    grp_ = grp_.replace(nodes=tf.where(loss_mask, grp2_.nodes, grp_.nodes))
    res.append(grp_)

@tf.function
def toGraphsTupleOld(d):
    send, recive = triangles_to_edges(d['cells'])
    rel_pos = (tf.gather(d['mesh_pos'], send) - tf.gather(d['mesh_pos'], recive))
    nodes_unique = tf.unique_with_counts(tf.reshape(d["node_type"], [-1]))
    dd = {
        #"nodes": tf.concat([d["velocity"],d["pressure"],d["density"],tf.cast(d["node_type"],tf.float32),d["mesh_pos"]],1),
        "nodes": tf.concat([d['velocity'], tf.one_hot(tf.reshape(d["node_type"], [-1]), NodeTypeCnt, dtype=tf.float32)],
                           1),  # on change update loss function ^
        "senders": send,
        "receivers": recive,
        "edges": tf.concat([
            rel_pos,
            tf.norm(rel_pos, axis=-1, keepdims=True)], 1)
    }
    return utils_tf.data_dicts_to_graphs_tuple([dd])

# fix mistake in data preparation
if chck_root == "models/new":
    grp_ = toGraphsTupleOld(qqq)

grp_.nodes.shape[1]

res = [grp_, ]
loss_mask = tf.logical_or(tf.equal(qqq['node_type'][:, 0], NodeType.NORMAL),
                          tf.equal(qqq['node_type'][:, 0], NodeType.OUTFLOW))
loss_mask = tf.reshape(tf.concat([loss_mask for _ in range(grp_.nodes.shape[1])], -1), [-1, grp_.nodes.shape[1]])
for i in range(600):
    grp2_ = model(grp_, False)
    grp_ = grp_.replace(nodes=tf.where(loss_mask, grp2_.nodes, grp_.nodes))
    res.append(grp_)

grp_.nodes.shape, grp2_.nodes.shape, loss_mask.shape

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import tri as mtri

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
skip = 5
num_steps = len(res)
num_frames = num_steps // skip
# compute bounds
bounds = []
bb_min, bb_max = tf.reduce_min(qq['velocity'][:, 0]), tf.reduce_max(qq['velocity'][:, 0])


def animate(num):
    global t
    step = (num * skip) % num_steps
    traj = (num * skip) // num_steps
    ax.cla()
    ax.set_xlim(-1, 2)
    ax.set_ylim(-1.5, 1.5)
    ax.set_autoscale_on(False)
    vmin, vmax = bb_min, bb_max
    pos = qqq['mesh_pos']
    faces = qqq['cells']
    velocity = res[step].nodes[..., :2].numpy()
    triang = mtri.Triangulation(pos[:, 0], pos[:, 1], faces)
    t = ax.tripcolor(triang, velocity[:, 0], vmin=vmin, vmax=vmax)
    ax.triplot(triang, 'ko-', ms=0.5, lw=0.3)
    ax.set_title('Trajectory %d Step %d' % (traj, step))
    return fig,


animate(0)
plt.colorbar(t)

anim = FuncAnimation(fig, animate, frames=num_frames, interval=200)
from IPython.display import HTML

HTML(anim.to_html5_video())

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
skip = 5
num_steps = len(res)
num_frames = num_steps // skip
# compute bounds
bounds = []

bb_min, bb_max = tf.reduce_min(qq['velocity'][:, 0]), tf.reduce_max(qq['velocity'][:, 0])


def animate(num):
    global t
    step = (num * skip) % num_steps
    traj = (num * skip) // num_steps
    ax.cla()
    ax.set_xlim(-1, 2)
    ax.set_ylim(-1.5, 1.5)
    ax.set_autoscale_on(False)
    vmin, vmax = bb_min, bb_max
    pos = qq['mesh_pos'][0]
    faces = qq['cells'][0]
    velocity = qq['velocity'][step]
    triang = mtri.Triangulation(pos[:, 0], pos[:, 1], faces)
    t = ax.tripcolor(triang, velocity[:, 0], vmin=vmin, vmax=vmax)
    ax.triplot(triang, 'ko-', ms=0.5, lw=0.3)
    ax.set_title('Trajectory %d Step %d' % (traj, step))
    return fig,


animate(0)
plt.colorbar(t)

anim = FuncAnimation(fig, animate, frames=num_frames, interval=200)
from IPython.display import HTML

HTML(anim.to_html5_video())

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
skip = 5
num_steps = len(res)
num_frames = num_steps // skip
# compute bounds
bounds = []

bb_min, bb_max = 0, tf.reduce_max(qq['velocity'][:, 0])


def animate(num):
    global t
    step = (num * skip) % num_steps
    traj = (num * skip) // num_steps
    ax.cla()
    ax.set_xlim(-1, 2)
    ax.set_ylim(-1.5, 1.5)
    ax.set_autoscale_on(False)
    vmin, vmax = bb_min, bb_max
    pos = qq['mesh_pos'][0]
    faces = qq['cells'][0]
    velocity = tf.math.abs(qq['velocity'][step] - res[step].nodes[..., :2])
    triang = mtri.Triangulation(pos[:, 0], pos[:, 1], faces)
    t = ax.tripcolor(triang, velocity[:, 0], vmin=vmin, vmax=vmax, cmap='Reds')
    ax.triplot(triang, 'ko-', ms=0.5, lw=0.3)
    ax.set_title('Trajectory %d Step %d' % (traj, step))
    return fig,


animate(0)
plt.colorbar(t)

anim = FuncAnimation(fig, animate, frames=num_frames, interval=200)
from IPython.display import HTML

HTML(anim.to_html5_video())

bb_min, bb_max

model._node_norm._std_with_epsilon(), model._node_norm._mean()

qq['target|velocity'].shape, qq['velocity'].shape

r = tf.reduce_sum((qq['velocity'] - qq['target|velocity']) ** 2)
r

tt = [{i: qq[i][x] for i in qq.keys()} for x in range(599)]

lss = []
for i in tt:
    lss.append(model.loss(toGraphsTuple(i), i))

lss = [x.numpy() for x in lss]

plt.plot(lss)

1e7 / (1000 * 600)

ee = [x.shape.as_list() for x in model.trainable_variables]
sum([(y[0] if len(y) == 1 else y[0] * y[1]) for y in ee])


"""
Script for training models defined in model_def.py
"""

import tensorflow as tf

import time
import sys
import re

from model_def import models
from impl.AdamModLr import Adam
from impl.common import toGraphsTuple

# Learning settings

model_name = "named12n"

lr = 1e-4
initial_learning_rate = lr
decay_rate = 0.1
decay_steps = 5e6
learning_increase = 1e-6

data_path = 'data/airfoil'
data_train = 'train'
print_loss_after = 500
save_model_after = 40000
opt = Adam(lr)

md = models[model_name]
model = md["model"](**md["args"])

params = models[model_name]["params"]
targets = tuple([x["field"] for x in params])

from impl.dataset import load_dataset, split_dataset, prepare, add_noises, add_targets, add_batch

ds = load_dataset(data_path, data_train)
ds = add_targets(ds, targets, add_history=params[0]['history'])

ds = split_dataset(ds)
for param in params:
    ds = add_noises(ds, noise_field=param['field'],
                    noise_scale=param['noise'],
                    noise_gamma=param['gamma'])
ds = prepare(ds, 10000)

ds = add_batch(ds, params[0]['batch'])

itr = iter(ds)

def decayed_learning_rate(step):
    return initial_learning_rate * decay_rate ** (step / decay_steps) + learning_increase


@tf.function
def update_step(data, targets=('velocity')):
    print("Tracing!")
    grp = toGraphsTuple(data, targets)
    with tf.GradientTape() as tape:
        los = model.loss(grp, data, targets=targets)  # change to loss

    gradients = tape.gradient(los, model.trainable_variables)

    opt.apply(gradients, model.trainable_variables)
    return los


chck_root = "models/" + md['path'] + '/'
chck_name = model_name

chck = tf.train.Checkpoint(module=model)

latest = tf.train.latest_checkpoint(chck_root)
if latest is not None:
    print("loading", latest)
    chck.restore(latest)

if latest is None:
    print("Acumulating")
    for i in range(1000):
        data = itr.next()
        grp = toGraphsTuple(data, targets)
        model.loss(grp, data, targets=targets)

start = 0
if latest is not None:
    start = int(re.findall('\d+$', latest)[0]) * save_model_after
t = time.time()
print("training", flush=True)
for i in range(start, int(1e7) + 1):
    a = itr.next()
    m = update_step(a, targets)
    if i % print_loss_after == 0:
        opt.learning_rate.assign(decayed_learning_rate(i))
        print("i", i, "mse:", m.numpy())
        if i and i % save_model_after == 0:
            sys.stdout.flush()
            chck.save(chck_root + chck_name)
print(time.time() - t)

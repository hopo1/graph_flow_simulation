import tensorflow as tf
import os
import json
import functools

from common import NodeType


def _parse(proto, meta):
    """Parses a trajectory from tf.Example."""
    feature_lists = {k: tf.io.VarLenFeature(tf.string)
                     for k in meta['field_names']}
    features = tf.io.parse_single_example(proto, feature_lists)
    out = {}
    for key, field in meta['features'].items():
        data = tf.io.decode_raw(features[key].values, getattr(tf, field['dtype']))
        data = tf.reshape(data, field['shape'])
        if field['type'] == 'static':
            data = tf.tile(data, [meta['trajectory_length'], 1, 1])
        elif field['type'] == 'dynamic_varlen':
            length = tf.io.decode_raw(features['length_' + key].values, tf.int32)
            length = tf.reshape(length, [-1])
            data = tf.RaggedTensor.from_row_lengths(data, row_lengths=length)
        elif field['type'] != 'dynamic':
            raise ValueError('invalid data format')
        out[key] = data
    return out


def load_dataset(path, split):
    """Load dataset."""
    with open(os.path.join(path, 'meta.json'), 'r') as fp:
        meta = json.loads(fp.read())
    ds = tf.data.TFRecordDataset(os.path.join(path, split + '.tfrecord'))
    ds = ds.map(functools.partial(_parse, meta=meta), num_parallel_calls=8)
    ds = ds.prefetch(1)
    return ds


def split_dataset(ds):
    return ds.flat_map(tf.data.Dataset.from_tensor_slices)


def add_noises(ds, noise_field, noise_scale, noise_gamma):
    def add_noise(frame):
        noise = tf.random.normal(tf.shape(frame[noise_field]),
                                 stddev=noise_scale, dtype=tf.float32)
        # don't apply noise to boundary nodes
        mask = tf.equal(frame['node_type'], NodeType.NORMAL)
        noise = tf.where(mask, noise, tf.zeros_like(noise))
        frame[noise_field] += noise
        frame['target|' + noise_field] += (1.0 - noise_gamma) * noise
        return frame

    return ds.map(add_noise, num_parallel_calls=8)


def prepare(ds, shuffle_size):
    ds = ds.shuffle(shuffle_size)
    ds = ds.repeat(None)
    return ds.prefetch(10)


def add_targets(ds, fields, add_history):
    """Adds target and optionally history fields to dataframe."""

    def fn(trajectory):
        out = {}
        for key, val in trajectory.items():
            out[key] = val[1:-1]
            if key in fields:
                if add_history:
                    out['prev|' + key] = val[0:-2]
                out['target|' + key] = val[2:]
        return out

    return ds.map(fn, num_parallel_calls=8)


def triangles_to_edges(faces):
    """Computes mesh edges from triangles."""
    # collect edges from triangles
    edges = tf.concat([faces[:, 0:2],
                       faces[:, 1:3],
                       tf.stack([faces[:, 2], faces[:, 0]], axis=1)], axis=0)
    # those edges are sometimes duplicated (within the mesh) and sometimes
    # single (at the mesh boundary).
    # sort & pack edges as single tf.int64
    receivers = tf.reduce_min(edges, axis=1)
    senders = tf.reduce_max(edges, axis=1)
    packed_edges = tf.bitcast(tf.stack([senders, receivers], axis=1), tf.int64)
    # remove duplicates and unpack
    unique_edges = tf.bitcast(tf.unique(packed_edges)[0], tf.int32)
    senders, receivers = tf.unstack(unique_edges, axis=1)
    # create two-way connectivity
    return (tf.concat([senders, receivers], axis=0),
            tf.concat([receivers, senders], axis=0))


def add_batch(ds, b_size):
    nodes = ds.element_spec["node_type"].shape[0]

    def batch(item):
        out = {}
        for key, val in item.items():
            if key == 'cells':
                out[key] = tf.concat([val[x] + x * nodes for x in range(val.shape[0])], axis=0)
            else:
                out[key] = tf.reshape(val, [-1, val.shape[2]])
        return out

    ds = ds.batch(b_size, True)
    return ds.map(batch, num_parallel_calls=8)

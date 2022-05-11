import enum
import tensorflow as tf
from graph_nets import utils_tf


class NodeType(enum.IntEnum):
    NORMAL = 0
    AIRFOIL = 2
    OUTFLOW = 5

def triangles_to_edges(faces):
    #taken from https://github.com/deepmind/deepmind-research/blob/master/meshgraphnets/common.py
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


@tf.function
def toGraphsTuple(d, targets=('velocity',)):
    send, recive = triangles_to_edges(d['cells'])
    rel_pos = (tf.gather(d['mesh_pos'], send) - tf.gather(d['mesh_pos'], recive))
    nodes_unique = tf.unique_with_counts(tf.reshape(d["node_type"], [-1]))
    one_hot = tf.one_hot(nodes_unique.idx, len(NodeType), dtype=tf.float32)
    dd = {
        "nodes": tf.concat([*[d[x] for x in targets], one_hot], 1),
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


def rollout(model, grp_, length, loss_mask):
    res = [grp_, ]
    for i in range(length):
        grp2_ = model(grp_, False)
        grp_ = grp_.replace(nodes=tf.where(loss_mask, grp2_.nodes, grp_.nodes))
        res.append(grp_)
    return res

import sonnet as snt
import tensorflow as tf
from graph_nets import blocks

from Normalizer import Normalizer


class EncodeProcessDecode(snt.Module):

    def __init__(self, steps, learn_features, node_net_sizes, edge_net_sizes, edge_feat_cnt, node_feat_cnt,
                 name="EncodeProcessDecode"):
        super(EncodeProcessDecode, self).__init__(name=name)
        if node_net_sizes[-1] != learn_features:
            print("adding output layer of size:", learn_features, "to node net.")
            node_net_sizes += [learn_features]
        self._to_nodes = blocks.NodeBlock(
            node_model_fn=lambda: snt.nets.MLP(node_net_sizes),
            use_sent_edges=True,
            use_globals=False)
        self._to_edges = blocks.EdgeBlock(
            edge_model_fn=lambda: snt.nets.MLP(edge_net_sizes),
            use_globals=False)
        self.steps = steps
        self.learn_features = learn_features
        self._edge_norm = Normalizer(edge_feat_cnt)
        self._node_norm = Normalizer(node_feat_cnt)

    def predict_next(self, grp):
        d2 = self._to_edges(grp)
        d3 = self._to_nodes(d2)
        new_speed = grp.nodes[..., :self.learn_features] + d3.nodes[..., :self.learn_features]
        return grp.replace(nodes=tf.concat([new_speed, grp.nodes[..., self.learn_features:]], 1))

    def __call__(self, grp, is_learning=False):
        grp = grp.replace(nodes=self._node_norm(grp.nodes, is_learning), edges=self._edge_norm(grp.edges, is_learning))
        for _ in range(self.steps):
            grp = self.predict_next(grp)
        return grp.replace()

    def acumulate(self, grp):
        self._edge_norm.accumulate(grp.edges)
        self._node_norm.accumulate(grp.edges)

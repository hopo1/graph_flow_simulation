import sonnet as snt
import tensorflow as tf
from graph_nets import blocks


class EncodeProcessDecode(snt.Module):
    def __init__(self, steps, learn_features, node_net_sizes, edge_net_sizes, name="EncodeProcessDecode"):
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

    def predict_next(self, grp):
        d2 = self._to_edges(grp)
        d3 = self._to_nodes(d2)
        new_speed = grp.nodes[..., :self.learn_features] + d3.nodes[..., :self.learn_features]
        return grp.replace(nodes=tf.concat([new_speed, grp.nodes[..., self.learn_features:]], 1))

    def __call__(self, grp, is_learning=False):
        for _ in range(self.steps):
            grp = self.predict_next(grp)
        return grp


if __name__ == '__main__':
    EncodeProcessDecode(15, 2, [128] * 2 + [2], [128] * 2 + [4])

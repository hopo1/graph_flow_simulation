import sonnet as snt
import tensorflow as tf
from graph_nets import blocks

from Normalizer import Normalizer
from common import NodeType


class EncodeProcessDecode(snt.Module):

    def __init__(self, steps, learn_features, n_layers, lat_size, edge_feat_cnt, node_feat_cnt,
                 name="EncodeProcessDecode"):
        super(EncodeProcessDecode, self).__init__(name=name)
        self.n_layers = n_layers
        self.lat_size = lat_size
        self._to_nodes = blocks.NodeBlock(
            node_model_fn=lambda: self._make_mlp(self.lat_size),
            use_globals=False)
        self._to_edges = blocks.EdgeBlock(
            edge_model_fn=lambda: self._make_mlp(self.lat_size),
            use_globals=False)
        self.steps = steps
        self.learn_features = learn_features
        self._edge_norm = Normalizer(edge_feat_cnt)
        self._node_norm = Normalizer(node_feat_cnt)
        self._out_norm = Normalizer(self.learn_features)
        self._encode_nodes = self._make_mlp(self.lat_size)
        self._encode_features = self._make_mlp(self.lat_size)
        self._decode_nodes = self._make_mlp(self.learn_features, False)

    def _encode(self, grp):
        return grp.replace(nodes=self._encode_nodes(grp.nodes), edges=self._encode_features(grp.edges))

    def _decode(self, grp):
        return grp.replace(nodes=self._decode_nodes(grp.nodes))

    def _make_mlp(self, output_size, layer_norm=True):
        # todo add citation
        """Builds an MLP."""
        widths = [self.lat_size] * self.n_layers + [output_size]
        network = snt.nets.MLP(widths, activate_final=False)
        if layer_norm:
            network = snt.Sequential([network, snt.LayerNorm(-1, True, True)])
        return network

    def predict_next(self, grp):
        d2 = self._to_edges(grp)
        d3 = self._to_nodes(d2)
        new_speed = grp.nodes[..., :self.learn_features] + d3.nodes[..., :self.learn_features]
        return grp.replace(nodes=tf.concat([new_speed, grp.nodes[..., self.learn_features:]], 1))

    def __call__(self, grp, is_learning=False):
        st = grp
        grp = grp.replace(nodes=self._node_norm(grp.nodes, is_learning), edges=self._edge_norm(grp.edges, is_learning))
        grp = self._encode(grp)
        for _ in range(self.steps):
            grp = self._to_edges(grp)
            grp = self._to_nodes(grp)
        if is_learning:
            return self._decode(grp)
        else:
            new_speed = st.nodes[..., :self.learn_features] + self._out_norm.inverse(
                self._decode(grp).nodes[..., :self.learn_features])
            return st.replace(nodes=tf.concat([new_speed, st.nodes[..., self.learn_features:]], 1))

    def loss(self, grp, inputs):
        res = self(grp, is_learning=True)

        # build target velocity change
        cur_velocity = inputs['velocity']
        target_velocity = inputs['target|velocity']
        target_velocity_change = target_velocity - cur_velocity
        target_normalized = self._out_norm(target_velocity_change, True)

        # build loss
        node_type = inputs['node_type'][:, 0]
        loss_mask = tf.logical_or(tf.equal(node_type, NodeType.NORMAL),
                                  tf.equal(node_type, NodeType.OUTFLOW))
        error = tf.reduce_sum((target_normalized - res.nodes) ** 2, axis=1)
        loss = tf.reduce_mean(error[loss_mask])
        return loss


if __name__ == '__main__':
    EncodeProcessDecode(15, 2, 2, 128, 3, 5)

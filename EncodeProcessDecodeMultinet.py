import sonnet as snt
import tensorflow as tf

from Normalizer import Normalizer
from common import NodeType


class EncodeProcessDecode(snt.Module):

    def __init__(self, steps, learn_features, n_layers, lat_size, edge_feat_cnt, node_feat_cnt, la_norm=True,
                 name="EncodeProcessDecode"):
        super(EncodeProcessDecode, self).__init__(name=name)
        self.n_layers = n_layers
        self.lat_size = lat_size
        self.la_norm = la_norm
        self.steps = steps
        self._to_nodes = [self._make_mlp(self.lat_size) for _ in range(self.steps)]
        self._to_edges = [self._make_mlp(self.lat_size) for _ in range(self.steps)]
        self.learn_features = learn_features
        self._edge_norm = Normalizer(edge_feat_cnt)
        self._node_norm = Normalizer(node_feat_cnt)
        self._out_norm = Normalizer(self.learn_features)
        self._encode_nodes = self._make_mlp(self.lat_size)
        self._encode_edges = self._make_mlp(self.lat_size)
        self._decode_nodes = self._make_mlp(self.learn_features, False)

    def _encode(self, grp):
        return grp.replace(nodes=self._encode_nodes(grp.nodes), edges=self._encode_edges(grp.edges))

    def _decode(self, grp):
        return grp.replace(nodes=self._decode_nodes(grp.nodes))

    def _make_mlp(self, output_size, layer_norm=True, name=None):
        # todo add citation
        """Builds an MLP."""
        widths = [self.lat_size] * self.n_layers + [output_size]
        network = snt.nets.MLP(widths, activate_final=False, name=name)
        layer_norm = self.la_norm and layer_norm
        if layer_norm:
            network = snt.Sequential([network, snt.LayerNorm(-1, True, True, name=name)], name=name)
        return network

    def _update_edge_features(self, grp, pass_num):
        """Aggregrates node features, and applies edge function."""
        sender_features = tf.gather(grp.nodes, grp.senders)
        receiver_features = tf.gather(grp.nodes, grp.receivers)
        features = [sender_features, receiver_features, grp.edges]
        return self._to_edges[pass_num](tf.concat(features, axis=-1))

    def _update_node_features(self, grp, new_edges, pass_num):
        """Aggregrates edge features, and applies node function."""
        num_nodes = grp.nodes.shape[0]
        features = [grp.nodes, tf.math.unsorted_segment_sum(new_edges, grp.receivers, num_nodes)]
        return self._to_nodes[pass_num](tf.concat(features, axis=-1))

    def _pass_message(self, graph, pass_num):
        # apply edge functions
        new_edges = self._update_edge_features(graph, pass_num)
        # apply node function
        new_nodes = self._update_node_features(graph, new_edges, pass_num)

        return graph.replace(nodes=graph.nodes + new_nodes, edges=graph.edges + new_edges)

    def __call__(self, grp, is_learning=False):
        st = grp
        grp = grp.replace(nodes=self._node_norm(grp.nodes, is_learning), edges=self._edge_norm(grp.edges, is_learning))
        grp = self._encode(grp)
        for i in range(self.steps):
            grp = self._pass_message(grp, i)
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

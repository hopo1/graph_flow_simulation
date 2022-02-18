import sonnet as snt
import tensorflow as tf


class Normalizer(snt.Module):
    """Feature normalizer that accumulates statistics online."""

    def __init__(self, size, std_epsilon=1e-8,
                 name='Normalizer'):
        super(Normalizer, self).__init__(name=name)
        self._std_epsilon = std_epsilon
        self._cnt = tf.Variable(0, dtype=tf.float32, trainable=False)
        self._sum = tf.Variable(tf.zeros(size, tf.float32), trainable=False)
        self._sum_squared = tf.Variable(tf.zeros(size, tf.float32),
                                        trainable=False)

    def __call__(self, data, is_training=False):
        """Normalizes input data and accumulates statistics."""
        if is_training:
            self.accumulate(data)
        return (data - self._mean()) / self._std_with_epsilon()

    def accumulate(self, data):
        self._cnt.assign_add(tf.cast(tf.shape(data)[0], tf.float32))
        self._sum.assign_add(tf.reduce_sum(data, axis=0))
        self._sum_squared.assign_add(tf.reduce_sum(data ** 2, axis=0))

    def inverse(self, normalized_batch_data):
        """Inverse transformation of the normalizer."""
        return normalized_batch_data * self._std_with_epsilon() + self._mean()

    def _mean(self):
        return self._sum / self._cnt

    def _std_with_epsilon(self):
        std = tf.sqrt(self._sum_squared / self._cnt - self._mean() ** 2)
        return tf.math.maximum(std, self._std_epsilon)


if __name__ == '__main__':
    Normalizer(5)

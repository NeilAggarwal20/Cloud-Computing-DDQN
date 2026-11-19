import unittest
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from dqn_targets import bellman_targets, next_state_values, synchronize_target


class BellmanTargetTests(unittest.TestCase):
    def setUp(self):
        self.online = tf.constant([[9.0, 1.0, 50.0]])
        self.target = tf.constant([[2.0, 7.0, 100.0]])
        self.valid = tf.constant([[True, True, False]])

    def test_standard_dqn_selects_maximum_target_q(self):
        value = next_state_values('dqn', self.online, self.target, self.valid)
        self.assertEqual(float(value[0]), 7.0)

    def test_double_dqn_selects_online_and_evaluates_target(self):
        value = next_state_values('ddqn', self.online, self.target, self.valid)
        self.assertEqual(float(value[0]), 2.0)

    def test_terminal_transition_does_not_bootstrap(self):
        result = bellman_targets(
            'dqn', tf.constant([3.5]), tf.constant([1.0]), self.online,
            self.target, self.valid, .95
        )
        self.assertEqual(float(result[0]), 3.5)

    def test_nonterminal_target_uses_discounted_value(self):
        result = bellman_targets(
            'dqn', tf.constant([1.0]), tf.constant([0.0]), self.online,
            self.target, self.valid, .5
        )
        self.assertEqual(float(result[0]), 4.5)

    def test_hard_target_sync_copies_online_weights(self):
        online = tf.keras.Sequential([tf.keras.layers.Input((1,)), tf.keras.layers.Dense(1)])
        target = tf.keras.Sequential([tf.keras.layers.Input((1,)), tf.keras.layers.Dense(1)])
        online(np.zeros((1, 1), np.float32)); target(np.zeros((1, 1), np.float32))
        online.set_weights([np.array([[3.0]], np.float32), np.array([2.0], np.float32)])
        target.set_weights([np.array([[-1.0]], np.float32), np.array([0.0], np.float32)])
        synchronize_target(online, target)
        for expected, actual in zip(online.get_weights(), target.get_weights()):
            np.testing.assert_array_equal(expected, actual)


if __name__ == '__main__':
    unittest.main()

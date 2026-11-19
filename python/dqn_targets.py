"""Bellman target helpers shared by the standard DQN and Double DQN agents."""

import tensorflow as tf


def next_state_values(algorithm, online_q, target_q, valid_actions):
    """Return next-state Q values using the selected algorithm's target rule.

    `valid_actions` is a boolean mask with the same [batch, actions] shape as
    the Q tensors. The project constrains decisions to estimated-finish
    candidates, so both algorithms bootstrap over that same action set.
    """
    valid_actions = tf.cast(valid_actions, tf.bool)
    negative = tf.cast(-1e9, target_q.dtype)
    target_valid = tf.where(valid_actions, target_q, negative)
    if algorithm == 'dqn':
        # Standard DQN uses the TARGET network for both selection and value.
        return tf.reduce_max(target_valid, axis=1)
    if algorithm == 'ddqn':
        # Double DQN selects with online Q and evaluates with target Q.
        online_valid = tf.where(valid_actions, online_q, negative)
        actions = tf.argmax(online_valid, axis=1, output_type=tf.int32)
        return tf.reduce_sum(target_q * tf.one_hot(actions, tf.shape(target_q)[1]), axis=1)
    raise ValueError(f'Unknown value-learning algorithm: {algorithm}')


def bellman_targets(algorithm, rewards, done, online_q, target_q,
                    valid_actions, gamma):
    """Compute one-step targets; terminal rows never bootstrap."""
    next_values = next_state_values(algorithm, online_q, target_q, valid_actions)
    return tf.stop_gradient(rewards + gamma * (1.0 - done) * next_values)


def synchronize_target(online, target):
    """Apply the project's periodic hard target-network update."""
    target.set_weights(online.get_weights())

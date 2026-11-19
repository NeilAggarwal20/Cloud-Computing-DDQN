import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse, json, random, threading
from collections import deque
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
import numpy as np
import tensorflow as tf
from dqn_targets import bellman_targets, synchronize_target
p = argparse.ArgumentParser()
p.add_argument('--seed', type=int, default=42)
p.add_argument('--model')
p.add_argument('--algorithm', choices=['dqn','ddqn'], default='ddqn')
p.add_argument('--eval', action='store_true')
p.add_argument('--resume', action='store_true')
p.add_argument('--run-id', default='')
p.add_argument('--gamma', type=float, default=.95)
p.add_argument('--learning-rate', type=float, default=.001)
p.add_argument('--batch-size', type=int, default=64)
p.add_argument('--target-sync', type=int, default=100)
p.add_argument('--replay-capacity', type=int, default=10000)
p.add_argument('--epsilon-start', type=float, default=.5)
p.add_argument('--epsilon-min', type=float, default=.05)
p.add_argument('--epsilon-decay', type=float, default=.995)
p.add_argument('--candidate-count', type=int, default=16)
p.add_argument('--dataset-sha256', default='')
p.add_argument('--experiment-id', default='')
a = p.parse_args()
if not (0 < a.gamma <= 1 and a.learning_rate > 0 and a.batch_size > 0 and
        a.target_sync > 0 and a.replay_capacity >= a.batch_size and
        0 <= a.epsilon_min <= a.epsilon_start <= 1 and 0 < a.epsilon_decay <= 1 and
        1 <= a.candidate_count <= 200):
    p.error('invalid DQN hyperparameter range')
random.seed(a.seed); np.random.seed(a.seed); tf.random.set_seed(a.seed)

def network():
    return tf.keras.Sequential([tf.keras.layers.Input((1800,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'), tf.keras.layers.Dense(200)])

online, target = network(), network()
optimizer = tf.keras.optimizers.Adam(learning_rate=a.learning_rate)
replay = deque(maxlen=a.replay_capacity)
updates, episodes, epsilon = 0, 0, a.epsilon_start
episode_return = 0.0
model = Path(a.model or f'models/{a.algorithm}.weights.h5')
metadata = Path(str(model) + '.metadata.json')
state_schema = 'dataset-workload-state-v1-9f'
reward_schema = 'dataset-workload-objective-v1'
candidate_count = a.candidate_count
policy_schema = f'earliest-finish-candidates-{candidate_count}'
if a.eval or a.resume:
    if not model.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {model.resolve()}")
    if not metadata.is_file():
        raise ValueError(f"Checkpoint metadata is missing: {metadata.resolve()}; train with this state schema first")
    saved_metadata = json.loads(metadata.read_text())
    if saved_metadata.get('state_schema') != state_schema:
        raise ValueError(f"Checkpoint state schema mismatch: expected {state_schema}")
    if saved_metadata.get('reward_schema') != reward_schema:
        raise ValueError(f"Checkpoint reward schema mismatch: expected {reward_schema}")
    if saved_metadata.get('policy_schema') != policy_schema:
        raise ValueError(f"Checkpoint policy schema mismatch: expected {policy_schema}")
    if saved_metadata.get('algorithm') != a.algorithm:
        raise ValueError(f"Checkpoint algorithm mismatch: expected {a.algorithm}")
    online.load_weights(str(model))
synchronize_target(online, target)

@tf.function
def learn(s, action, r, ns, done):
    waits = tf.math.expm1(ns[:, 0::9] * tf.math.log(20001.0))
    lengths = ns[:, 3:4] * 20000.0
    speeds = ns[:, 2::9] * 2000.0
    cpu = ns[:, 5::9]
    finish = waits + lengths / (speeds * cpu)
    candidate_indices = tf.math.top_k(-finish, k=candidate_count).indices
    valid = tf.reduce_sum(tf.one_hot(candidate_indices, 200), axis=1) > 0
    online_next_q = online(ns, training=False)
    target_next_q = target(ns, training=False)
    y = bellman_targets(a.algorithm, r, done, online_next_q, target_next_q,
                        valid, a.gamma)
    with tf.GradientTape() as tape:
        q = online(s, training=True)
        predicted = tf.reduce_sum(q * tf.one_hot(action, 200), axis=1)
        loss = tf.reduce_mean(tf.square(y - predicted))
    gradients = tape.gradient(loss, online.trainable_variables)
    tf.debugging.assert_all_finite(loss, 'Non-finite DDQN loss')
    for gradient in gradients:
        if gradient is not None:
            tf.debugging.assert_all_finite(gradient, 'Non-finite DDQN gradient')
    optimizer.apply_gradients(zip(gradients, online.trainable_variables))
    return loss

def handle(path, data):
    global updates, episodes, epsilon, episode_return
    if path == '/act':
        s = np.asarray(data['state'], dtype=np.float32)
        if s.shape != (1800,) or not np.isfinite(s).all():
            raise ValueError('Expected 1800 finite state values')
        waits = np.expm1(s[0::9] * np.log(20001.0))
        lengths = s[3] * 20000.0
        speeds = s[2::9] * 2000.0
        cpu = s[5::9]
        finish = waits + lengths / (speeds * cpu)
        candidates = np.argpartition(finish, candidate_count - 1)[:candidate_count]
        if not a.eval and random.random() < epsilon:
            action = int(random.choice(candidates.tolist()))
        else:
            q_values = online(s[None], training=False).numpy()[0]
            if not np.isfinite(q_values).all():
                raise ValueError('The online network returned non-finite Q values')
            action = int(candidates[np.argmax(q_values[candidates])])
        if not 0 <= action < 200:
            raise ValueError(f'Agent selected invalid VM index: {action}')
        return {'action': action}
    if path == '/observe':
        if a.eval: return {'evaluation': True}
        s = np.asarray(data['state'], dtype=np.float32)
        ns = np.asarray(data['next_state'], dtype=np.float32)
        action, reward = int(data['action']), float(data['reward'])
        if s.shape != (1800,) or ns.shape != (1800,) or not np.isfinite(s).all() or not np.isfinite(ns).all():
            raise ValueError('Transitions require two 1800-value finite states')
        if not 0 <= action < 200 or not np.isfinite(reward):
            raise ValueError('Transition action or reward is invalid')
        replay.append((s, action, reward, ns, float(bool(data['done']))))
        episode_return += reward
        loss = None
        if len(replay) >= a.batch_size:
            batch = random.sample(replay, a.batch_size)
            s, action, r, ns, done = zip(*batch)
            loss = float(learn(np.asarray(s,np.float32), np.asarray(action,np.int32),
                np.asarray(r,np.float32), np.asarray(ns,np.float32), np.asarray(done,np.float32)))
            updates += 1
            if updates % a.target_sync == 0: synchronize_target(online, target)
        if data['done']:
            episodes += 1; epsilon = max(a.epsilon_min, a.epsilon_start * a.epsilon_decay ** episodes)
            model.parent.mkdir(parents=True, exist_ok=True)
            online.save_weights(str(model))
            metadata.write_text(json.dumps({
                'state_schema': state_schema,
                'reward_schema': reward_schema,
                'policy_schema': policy_schema,
                'algorithm': a.algorithm,
                'dataset_sha256': a.dataset_sha256,
                'experiment_id': a.experiment_id,
                'episodes': episodes,
                'training': {
                    'gamma': a.gamma, 'learning_rate': a.learning_rate,
                    'batch_size': a.batch_size, 'target_sync': a.target_sync,
                    'replay_capacity': a.replay_capacity,
                    'epsilon_start': a.epsilon_start, 'epsilon_min': a.epsilon_min,
                    'epsilon_decay': a.epsilon_decay, 'candidate_count': candidate_count
                },
                'weights_only': True
            }, indent=2))
            print(f"Episode {episodes}: updates={updates}, epsilon={epsilon:.4f}, last_loss={loss}, shaped_return={episode_return:.6f}", flush=True)
            episode_return = 0.0
        return {'loss': loss, 'updates': updates, 'epsilon': epsilon}
    raise ValueError('Unknown endpoint')

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            length = int(self.headers.get('Content-Length', '0'))
            data = json.loads(self.rfile.read(length))
            result = handle(self.path, data); status = 200
        except Exception as exc:
            result = {'error': str(exc)}; status = 500
        body = json.dumps(result).encode()
        self.send_response(status); self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(body))); self.send_header('Connection','close')
        self.close_connection = True; self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        body = b'{"ready":true}'
        self.send_response(200); self.send_header('Content-Length',str(len(body)))
        self.send_header('Connection','close'); self.close_connection = True
        self.end_headers(); self.wfile.write(body)
    def log_message(self, *args): pass

print(f"{a.algorithm.upper()} ready at http://127.0.0.1:8765", flush=True)
HTTPServer(('127.0.0.1',8765), Handler).serve_forever()

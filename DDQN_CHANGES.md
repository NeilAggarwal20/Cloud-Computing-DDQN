# DQN and DDQN baselines

`python/agent_server.py` supports both algorithms through the same online and target networks, state, action candidates, rewards, replay memory, optimizer, and exploration settings.

- Standard DQN uses `r + gamma * (1-done) * max_a Q_target(s_next,a)`.
- Double DQN selects `argmax_a Q_online(s_next,a)` and evaluates it with `Q_target`.
- Both stop bootstrapping on terminal transitions and synchronize target weights on the configured update interval.

Fresh models and results for an experiment are isolated in its `results/<experiment>/` directory. Existing `models/` checkpoints are historical and are not loaded by the current dataset workflow. Use the `experiment` command in README.md to train both from scratch and compare RR, EFT, DQN, and DDQN on paired dataset workloads.

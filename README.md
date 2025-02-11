# value-of-intent
restructured version of the oracle play framework

given a full set of prior distributions of the opponent's policy, what is the minimal information needed to know to determine the optimal best response strategy?

value-of-intent/
├── src/
│   ├── core/                             # All core logic
│   │   ├── __init__.py                   # Central import hub for networks
│   │   ├── networks.py                   # Actor-Critic network and observation processing
│   │   ├── base_agent.py                 # Base agent class (abstract methods for policy and value computation, log probability computation, action selection)
│   │   └── oracle_agent.py               # Oracle agent (partner-action-augmented inputs)
│
│   ├── training/                         # Core training logic
│   │   ├── __init__.py                   # Central import hub for trainers
│   │   ├── base_trainer.py               # Trainer for baseline PPO (IPPO) somewhat equivalent to env_step in original code
│   │   └── oracle_trainer.py             # Trainer for oracle PPO (augmented observations)
│           
│   ├── evaluation/                       # Metrics and visualization utilities
│   │   ├── __init__.py
│   │   ├── metrics.py                    # Tracking and reporting key performance metrics
│   │   └── visualization.py              # Advanced visualization utilities
│
│   ├── utils/                            # General utilities
│   │   ├── __init__.py
│   │   ├── env_utils.py                  # Environment-specific utilities
│   │   ├── io.py                         # For file I/O (loading/saving checkpoints, metrics)
│   │   ├── trajectory_utils.py           # Trajectory-specific utilities
│   │   ├── network_utils.py              # Network-specific utilities
│   │   ├── policy_objectives.py          # Policy objective utilities
│   │   ├── lr_schedules.py               # Learning rate schedule utilities
│
│   └── variants/                         # PPO variants for experiments
│       ├── baseline.py                   # Baseline IPPO implementation
│       └── oracle.py                     # Oracle agent implementation
│
├── config/                               # Configuration files for experiments
│   ├── base_config.yaml                  # Base configuration for all experiments
│   └── sweep.yaml                        # Configuration for hyperparameter sweeps
│
└── experiments/                          # Experiment scripts
    ├── train.py                          # Main entry point for running experiments (equivalent to make_train in original code)
    ├── adaptability.py                   # Experiment comparing pretrained fixed-policy agents with a learning agent

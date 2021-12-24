# Emergence of Theory of Mind Collaboration in Multiagent Systems (NeurIPS 2019 Workshop) https://arxiv.org/pdf/2110.00121 #

Detailed the implementations for our algorithm and benchmarks can be found in this folder.

ToM:
  Kitchen:
    ./Agent/*
    ./Experiments/Example/config.py (finely adjusted network structure and hyper-parameters)
    ./Game/*
    ./train.py
    ./test.py
    ./interact.py
  Appointment:
    ./AgentC/*
    ./Experiments/ExampleC/config.py (finely adjusted network structure and hyper-parameters)
    ./Game/*
    ./trainC.py
    ./testC.py
    ./interactC.py

BAD:
  Kitchen:
    ./Benchmark/BAD/Kitchen/*
    ./Benchmark/BAD/Kitchen/config.py (finely adjusted network structure and hyper-parameters)
  Apppointment:
    ./Benchmark/BAD/Calendar/*
    ./Benchmark/BAD/Calendar/config.py (finely adjusted network structure and hyper-parameters) 

CIRL:
  Kitchen:
    ./Benchmark/ValueAlignment/*
    ./Benchmark/ValueAlignment/[config.py|train.py] (finely adjusted network structure and hyper-parameters)

COMA:
  Kitchen:
    ./Benchmark/Counterfactual/Kitchen/*
    ./Benchmark/Counterfactual/Kitchen/config.py (finely adjusted network structure and hyper-parameters)
  Appointment:
    ./Benchmark/Counterfactual/Calendar_MLP/*
    ./Benchmark/Counterfactual/Calendar_MLP/config.py (finely adjusted network structure and hyper-parameters)

DDRQN:
  Kitchen:
    ./Benchmark/Riddle/Kitchen/*
    ./Benchmark/Riddle/Kitchen/config.py (finely adjusted network structure and hyper-parameters)
  Appointment:
    ./Benchmark/Riddle/Calendar/*
    ./Benchmark/Riddle/Calendar/configC2.py (finely adjusted network structure and hyper-parameters)

DIAL:
  Kitchen:
    ./Benchmark/DIAL/*
    ./Benchmark/DIAL/config.py (finely adjusted network structure and hyper-parameters)
    
MADDPG:
  Kitchen:
    ./Benchmark/MADDPG/rnn_gumbel/*
    ./Benchmark/MADDPG/rnn_gumbel/[config.py|train.py] (finely adjusted network structure and hyper-parameters)
    (other folders in ./Benchmark/MADDPG/ are ours attempts to increase MADDPG's performance in Kitchen Collaboration task)
  Appointment:
    ./Benchmark/MADDPGC/rnn_gumbel/*
    ./Benchmark/MADDPGC/rnn_gumbel/[config.py|train.py] (finely adjusted network structure and hyper-parameters)
    (other folders in ./Benchmark/MADDPGC/ are ours attempts to increase MADDPG's performance in Appointment Scheduling task)

Random (baseline):
  Kitchen:
    ./Benchmark/Random/*
  Appointment:
    (calculated analytically)

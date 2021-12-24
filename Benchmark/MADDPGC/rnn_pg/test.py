from Game.calendar import Calendar
from Benchmark.MADDPGC.rnn_pg.config import *
from Benchmark.MADDPGC.rnn_pg.agents import *

from shutil import copyfile
from copy import deepcopy
import tensorflow as tf
import os
import sys
import logging
import datetime
import argparse

## game
NUM_SLOTS = 8

## agents
EMBED_DIM = 8
NUM_ENSEMBLE = 1
DENSE = 128
STEP_REWARD = 0
REWARD = 1
DECAY_STEP = 5e4



A_LR = 1e-5
C_LR = 1e-4
DISCOUNT = 0.6
LAMBDA = 0.5
BETA_H = 1e-3
BETA_R = 1e-3
BREAK_CORR = True

BUFFER_ROOM = 1000
BS = 200
TRAIN_STEP = BUFFER_ROOM // BS

MAX_EP = 10000000
MAX_TIME = 20
UPDATE_STEP = 100

QP_MATCH_STEP = 200

CKPT_STEP = 5000
SUM_SEC = 600

LOG_STEP = 1000

DEVICE_COUNT=1

def setup_logger(log_dir):
    logger = logging.getLogger()
    logger.handlers = []
    file_handler = logging.FileHandler(os.path.join(log_dir, 'log.log'))
    console_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)

def get_trainers(num_actions, calendar_tensor, action_tensor):
    trainer = []
    for i in range(2):

        if i == 0:
            name = 'human'
            beta = BETA_H
        else:
            name = 'robot'
            beta = BETA_R
        if DEVICE_COUNT == 0:
            use_gpu = False
        else:
            use_gpu = True
        config = Config(NUM_SLOTS, NUM_ENSEMBLE, num_actions, DENSE,
                        EMBED_DIM, A_LR, C_LR, DISCOUNT, LAMBDA, beta, BS, BUFFER_ROOM, MAX_TIME, UPDATE_STEP,
                        BREAK_CORR, use_gpu)
        agent = MADDPGAgentTrainer(name, 2, i, config, p_func, q_func, calendar_tensor, action_tensor)
        trainer.append(agent)

    return trainer



def test_model(sess, env, agents):

    argmax_q = [[], []]
    argmax_policy = [[], []]
    mistake = []
    test = []
    for ep in range(100):
        env.reset()
        obs = env.start()
        private = [deepcopy(obs[o].private[np.newaxis,:]) for o in ['human', 'robot']]

        # init
        for agent in agents:
            agent.set_policy_index()
        last_action_n = [- np.ones((1,)) for _ in range(2)]
        in_state = [None, None]

        ## store stuff
        private_traj = [[] for _ in range(2)]
        last_act_traj = [[] for _ in range(2)]
        act_traj = [[] for _ in range(2)]
        reward_traj = [[] for _ in range(2)]
        terminal_traj = [[] for _ in range(2)]

        ## for logging
        if ep % LOG_STEP == 0:
            logging.info(
                'Episode: {}\n'.format(ep))


        for t in range(MAX_TIME):

            for i, name in enumerate(['human', 'robot']):
                a, in_state[i], ent, logit = agents[i].action(sess, private[i][:,None,:],
                                                                      [lact[:,None] for lact in last_action_n],
                                                    in_state=in_state[i], train=False)
                _, in_state[(i+1)%2], _, _ = agents[(i+1)%2].action(sess, private[(i+1)%2][:, None, :],
                                                                       [lact[:,None] for lact in last_action_n],
                                                                 in_state=in_state[(i+1)%2], train=False)
                a = np.squeeze(a, axis=1)
                next_obs = deepcopy(env.proceed({name: a[0]}))

                for j in range(2):
                    private_traj[j].append(private[j])
                    last_act_traj[j].append(last_action_n[j])
                    act_traj[j].append(a if j == i else np.array([-1]))
                    reward_traj[j].append(np.array([next_obs[agents[j].name].reward]))
                    terminal_traj[j].append(np.array([obs[agents[j].name].terminate]))


                if next_obs[name].terminate:
                    if next_obs[name].get('msg_success') is not None:
                        if name == 'human':
                            mistake.append(0)
                        else:
                            mistake.append(1)
                        test.append(0)

                        break

                    if next_obs[name].get('action_success') is not None and not next_obs[name].get('action_success'):
                        if name == 'human':
                            mistake.append(0)
                        else:
                            mistake.append(1)
                        test.append(0)
                    else:
                        test.append(1)



                    break

                last_action_n[i] = a
                last_action_n[(i+1)%2] = np.array([-1])

            if next_obs[name].terminate:
                break

        if not next_obs[name].terminate:
            mistake.append(2)
            test.append(0)


    acc = sum(test) / len(test)
    if not mistake:
        herr = rerr = 0
    else:
        herr = (np.array(mistake) == 0).mean()
        rerr = (np.array(mistake) == 1).mean()

    logging.info(
        'Episode: {} Overall Accuracy: {} Human Error Percentage: {} Robot Error Percentage: {}\n'.format(
            ep, acc, herr, rerr))

    return acc

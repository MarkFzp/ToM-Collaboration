from Game.calendar import Calendar
from Benchmark.MADDPGC.rnn_gumbel.config import *
from Benchmark.MADDPGC.rnn_gumbel.agents import *
from Benchmark.MADDPGC.rnn_gumbel.policy import *
import numpy as np
from shutil import copyfile
from copy import deepcopy
import tensorflow as tf
import os
import argparse
import sys
import logging
import datetime
import argparse

## game
NUM_SLOTS = 8

## agents
EMBED_DIM = 8
NUM_ENSEMBLE = 1
DENSE = 100
STEP_REWARD = -0.1
FAIL_REWARD = -2
REWARD = 1
DECAY_STEP = 5e4


TEMP_MIN = 0.1
TEMP_START = 100
TEMP_DECAY = 5e4


A_LR = 1e-4
C_LR = 1e-3
DISCOUNT = 0.6
LAMBDA = 0.5
BETA_H = 3e-2
BETA_R = 8e-2
BREAK_CORR = True

BUFFER_ROOM = 1000
BS = 200
TRAIN_STEP = BUFFER_ROOM // BS

MAX_EP = 1000000
MAX_TIME = (NUM_SLOTS+1) // 2
UPDATE_STEP = 100

QP_MATCH_STEP = 200

NUM_CKPT = 50
CKPT_STEP = MAX_EP // NUM_CKPT
SUM_SEC = 600

LOG_STEP = 1000

DEVICE_COUNT=1


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
                        EMBED_DIM, A_LR, C_LR, DISCOUNT, LAMBDA, beta, BS, BUFFER_ROOM, MAX_TIME * 2, UPDATE_STEP,
                        BREAK_CORR, use_gpu)
        agent = MADDPGAgentTrainer(name, 2, i, config, p_func, q_func, calendar_tensor, action_tensor)
        trainer.append(agent)

    return trainer


def test_model(sess, env, agents, itr=10000):

    mistake = []
    test = []
    for ep in range(itr):
        env.reset(train=False)
        obs = env.start()
        private = [deepcopy(obs[o].private[np.newaxis,:]) for o in ['human', 'robot']]

        # init
        for agent in agents:
            agent.set_policy_index()
        last_action_n = [- np.ones((1,)) for _ in range(2)]
        in_state = [None, None]


        for t in range(MAX_TIME):

            for i, name in enumerate(['human', 'robot']):
                a, in_state[i], ent, logit, a_cont = agents[i].action(sess, private[i][:, None, :],
                                                                      [lact[:, None] for lact in last_action_n],
                                                                      0.01,
                                                                      in_state=in_state[i], train=False)
                _, in_state[(i + 1) % 2], _, _, _ = agents[(i + 1) % 2].action(sess, private[(i + 1) % 2][:, None, :],
                                                                               [lact[:, None] for lact in
                                                                                last_action_n],
                                                                               0.01,
                                                                               in_state=in_state[(i + 1) % 2],
                                                                               train=False)
                a = np.squeeze(a, axis=1)
                next_obs = deepcopy(env.proceed({name: a[0]}))


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


def test(ckpt_dir, ds_order):
    env = Calendar('human', 'robot', -0.1, -2, 1, 8,
                   dataset_path_prefix='/home/Datasets/Calendar/calendar_8_{}'.format(
                       ds_order))  # /home/Datasets/Calendar # utils

    agents = get_trainers(env.num_actions_, env.tensor_, env.action_tensor_)
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))
    mistake = []
    test = []


    for ep in range(10000):
        env.reset(train=False)
        obs = env.start()
        private = [deepcopy(obs[o].private[np.newaxis, :]) for o in ['human', 'robot']]

        # init
        for agent in agents:
            agent.set_policy_index()
        last_action_n = [- np.ones((1,)) for _ in range(2)]
        in_state = [None, None]

        for t in range(MAX_TIME):

            for i, name in enumerate(['human', 'robot']):
                a, in_state[i], ent, logit, a_cont = agents[i].action(sess, private[i][:, None, :],
                                                                       [lact[:, None] for lact in last_action_n],
                                                                       0.01,
                                                                       in_state=in_state[i], train=False)
                _, in_state[(i + 1) % 2], _, _, _ = agents[(i + 1) % 2].action(sess, private[(i + 1) % 2][:, None, :],
                                                                                [lact[:, None] for lact in
                                                                                 last_action_n],
                                                                                0.01,
                                                                                in_state=in_state[(i + 1) % 2],
                                                                                train=False)
                a = np.squeeze(a, axis=1)
                next_obs = deepcopy(env.proceed({name: a[0]}))

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
                last_action_n[(i + 1) % 2] = np.array([-1])

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

    print(
        'Episode: {} Overall Accuracy: {} Human Error Percentage: {} Robot Error Percentage: {}\n'.format(
            ep, acc, herr, rerr))


def swap_test(ckpt_dir1, ckpt_dir2, ds_order):
    env = Calendar('human', 'robot', -0.1, -2, 1, 8,
                   dataset_path_prefix='/home/Datasets/Calendar/calendar_8_{}'.format(
                       ds_order))  # /home/Datasets/Calendar # utils

    with tf.Graph().as_default() as group1:
        agents1 = get_trainers(env.num_actions_, env.tensor_, env.action_tensor_)
        saver1 = tf.train.Saver()
        sess1 = tf.Session(graph=group1)

        sess1.run(tf.initializers.global_variables())


    with tf.Graph().as_default() as group2:
        agents2 = get_trainers(env.num_actions_, env.tensor_, env.action_tensor_)
        saver2 = tf.train.Saver()
        sess2 = tf.Session(graph=group2)

        sess2.run(tf.initializers.global_variables())

    saver1.restore(sess1, tf.train.latest_checkpoint(ckpt_dir1))
    saver2.restore(sess2, tf.train.latest_checkpoint(ckpt_dir2))

    for agent in agents1:
        agent.p_update(sess1)
        agent.q_update(sess1)
    for agent in agents2:
        agent.p_update(sess2)
        agent.q_update(sess2)

    agents_to_act = [agents1[0], agents2[1]]
    sesses = [sess1, sess2]

    mistake = []
    test = []
    for ep in range(10000):
        env.reset(train=False)
        obs = env.start()
        private = [deepcopy(obs[o].private[np.newaxis,:]) for o in ['human', 'robot']]

        # init
        for agent in agents1:
            agent.set_policy_index()
        for agent in agents2:
            agent.set_policy_index()
        last_action_n = [- np.ones((1,)) for _ in range(2)]
        in_state = [None, None]


        for t in range(MAX_TIME):

            for i, name in enumerate(['human', 'robot']):
                a, in_state[i], ent, logit, a_cont = agents_to_act[i].action(sesses[i], private[i][:, None, :],
                                                                      [lact[:, None] for lact in last_action_n],
                                                                      0.01,
                                                                      in_state=in_state[i], train=False)
                _, in_state[(i + 1) % 2], _, _, _ = agents_to_act[(i + 1) % 2].action(sesses[(i + 1) % 2], private[(i + 1) % 2][:, None, :],
                                                                               [lact[:, None] for lact in
                                                                                last_action_n],
                                                                               0.01,
                                                                               in_state=in_state[(i + 1) % 2],
                                                                               train=False)
                a = np.squeeze(a, axis=1)
                next_obs = deepcopy(env.proceed({name: a[0]}))


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

    print(
        'Overall Accuracy: {} Human Error Percentage: {} Robot Error Percentage: {}\n'.format(
            acc, herr, rerr))

    return acc


def plot_all_test_acc(ckpt_dir, ds_order):
    env = Calendar('human', 'robot', -0.1, -2, 1, 8,
                   dataset_path_prefix='/home/Datasets/Calendar/calendar_8_{}'.format(
                       ds_order))  # /home/Datasets/Calendar # utils

    agents = get_trainers(env.num_actions_, env.tensor_, env.action_tensor_)
    saver = tf.train.Saver()
    sess = tf.Session()

    ckpt_files = tf.train.get_checkpoint_state(ckpt_dir).all_model_checkpoint_paths[-1:]

    for i, ckpt in enumerate(ckpt_files):
        saver.restore(sess, ckpt)
        acc= test_model(sess, env, agents, 10000)
        print('CKPT name: {}, Acc: {}'.format(ckpt.split('/')[-1], acc))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir',
                        default='/home/alexzhou907/My_Items/Communication_Game/Benchmark/MADDPGC/rnn_gumbel/output/screen_dataset3_run2/ckpt/best/')

    parser.add_argument('--ckpt-dir2',
                        default='/home/alexzhou907/My_Items/Communication_Game/Benchmark/MADDPGC/rnn_gumbel/output/screen_dataset3_run2/ckpt/target_best/')
    parser.add_argument('--swap', default=True, action='store_true')
    parser.add_argument('--get_test_acc_over_time', default=False, action='store_true')
    parser.add_argument('--ds', default=3)

    opt = parser.parse_args()

    if opt.swap:
        swap_test(opt.ckpt_dir, opt.ckpt_dir2, opt.ds)
        swap_test(opt.ckpt_dir2, opt.ckpt_dir, opt.ds)
    else:
        if opt.get_test_acc_over_time:
            plot_all_test_acc(opt.ckpt_dir, opt.ds)
        else:
            test(opt.ckpt_dir, opt.ds)


if __name__ == '__main__':
    main()

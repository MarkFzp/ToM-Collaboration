import numpy as np

from shutil import copyfile
from copy import deepcopy
import tensorflow as tf
import os
import sys
import logging
import datetime
import argparse
from Game.kitchen import Kitchen
from Benchmark.MADDPG.rnn_gumbel.config import *
from Benchmark.MADDPG.rnn_gumbel.agents import *
from Benchmark.MADDPG.rnn_gumbel.policy import *

## game
MENU = None #[(0, 1, 2, 2, 9), (7,8, 1, 4, 5), (6,1, 0, 6, 4), (4,0, 5, 7,3)]
INGREDIENTS = list(range(12))
NUM_DISHES = 7
MAX_DISH_INGREDIENTS = 5

## agents
NUM_ACTIONS = len(INGREDIENTS)
NUM_GOALS = NUM_DISHES
EMBED_DIM = 8
NUM_ENSEMBLE = 1
DENSE = 100
STEP_REWARD = 0
REWARD = 1
DECAY_STEP = 1e6
DECAY_RATE = 0.5

TEMP_MIN = 0.1
TEMP_START = 100
TEMP_DECAY = 5e4


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

MAX_EP = 5000000
MAX_TIME = 5
UPDATE_STEP = 100

QP_MATCH_STEP = 200


NUM_CKPT = 50
CKPT_STEP = MAX_EP // NUM_CKPT
SUM_SEC = 600

LOG_STEP = 1000

def get_UI(UI_stats, name, menu, target_goal, a):
    t = menu[0,target_goal]
    t2 = (menu[0].sum(axis=0) == t) & (t > 0)
    has_UI = t2.any()
    use_UI = t2[a]
    UI_stats[name][0].append(use_UI)
    UI_stats[name][1].append(has_UI)

    return UI_stats

def get_trainers(init_global_step, device_count):
    trainer = []
    for i in range(2):

        if i == 0:
            name = 'human'
            beta = BETA_H
        else:
            name = 'robot'
            beta = BETA_R
        if device_count == 0:
            use_gpu = False
        else:
            use_gpu = True
        config = Config(NUM_DISHES, len(INGREDIENTS), NUM_ENSEMBLE, NUM_ACTIONS, DENSE,
                        EMBED_DIM, A_LR, C_LR, init_global_step ,DECAY_STEP, DECAY_RATE, DISCOUNT, LAMBDA, beta, BS, BUFFER_ROOM, MAX_TIME, UPDATE_STEP,
                        BREAK_CORR, use_gpu)
        agent = MADDPGAgentTrainer(name, 2, i, config, p_func, q_func)
        trainer.append(agent)

    return trainer


def test_model(sess, env, agents, itr=10000):

    mistake = []
    test = []
    step = 0
    for ep in range(itr):
        ep_r = 0
        temp = 0.01#TEMP_MIN + (TEMP_START - TEMP_MIN) * np.exp(-1 * ep / DECAY_STEP)

        env.reset(MENU, train=False)
        goal = np.array([env.state_.order])
        obs = env.start()
        menu = deepcopy(obs['robot'].observation.menu_embed[np.newaxis, :])
        wpeb = deepcopy(obs['robot'].observation.workplace_embed[np.newaxis,:])

        in_state = [None, None]


        for agent in agents:
            agent.set_policy_index()

        for t in range(MAX_TIME):

            for i, name in enumerate(['human', 'robot']):
                a, in_state[i], ent, _, _ = agents[i].action(sess, menu[:,None,:,:], wpeb[:,None,:], goal, temp,
                                                    np.array([[True]]),in_state[i], train=False)
                _, in_state[(i+1)%2], _, _, _ = agents[(i+1)%2].action(sess, menu[:, None, :, :], wpeb[:, None, :], goal, temp,
                                                       np.array([[True]]), in_state[(i+1)%2], train=False)
                a = np.squeeze(a, axis=1)
                next_obs = deepcopy(env.proceed({name: a[0]}))
                next_wpeb = next_obs[name].observation.workplace_embed[np.newaxis, :]

                ep_r += next_obs[name].reward
                step += 1


                if next_obs[name].terminate:
                    step += 1
                    t += 1
                    if next_obs[name].success:
                        test.append(1)
                    else:
                        if name == 'human':
                            mistake.append(0)
                        else:
                            mistake.append(1)
                        test.append(0)



                    break

                wpeb = next_wpeb

            if next_obs[name].terminate:
                break

        if not next_obs[name].terminate:
            mistake.append(2)
            test.append(0)


    acc = sum(test) / len(test)
    herr = (np.array(mistake) == 0).mean()
    rerr = (np.array(mistake) == 1).mean()

    print('Test: Overall Accuracy: {} Human Error Percentage: {} Robot Error Percentage: {}\n'.format(
        acc, herr, rerr))

    return acc

def test(ckpt_dir, env, gpu):
    if gpu == -1:
        device_count = 0
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        device_count = 1
    agents = get_trainers(0, device_count)
    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
    config = tf.ConfigProto(device_count={'GPU': device_count})
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.initializers.global_variables())
    saver.restore(sess, os.path.join(os.path.dirname(__file__), 'output',
                                       tf.train.latest_checkpoint(ckpt_dir).split('output/')[-1]))

    for agent in agents:
        agent.p_update(sess)
        agent.q_update(sess)

    mistake = []
    test = []
    step = 0

    UI_stats = {'human': [[], []], 'robot': [[], []]}
    for ep in range(10000):
        ep_r = 0
        temp = 0.01#TEMP_MIN + (TEMP_START - TEMP_MIN) * np.exp(-1 * ep / DECAY_STEP)

        env.reset(MENU, train=False)
        goal = np.array([env.state_.order])
        obs = env.start()
        menu = deepcopy(obs['robot'].observation.menu_embed[np.newaxis, :])
        wpeb = deepcopy(obs['robot'].observation.workplace_embed[np.newaxis,:])

        in_state = [None, None]


        for agent in agents:
            agent.set_policy_index()

        for t in range(MAX_TIME):

            for i, name in enumerate(['human', 'robot']):
                a, in_state[i], ent, _, _ = agents[i].action(sess, menu[:,None,:,:], wpeb[:,None,:], goal, temp,
                                                    np.array([[True]]),in_state[i], train=False)
                _, in_state[(i+1)%2], _, _, _ = agents[(i+1)%2].action(sess, menu[:, None, :, :], wpeb[:, None, :], goal, temp,
                                                       np.array([[True]]), in_state[(i+1)%2], train=False)
                a = np.squeeze(a, axis=1)
                next_obs = deepcopy(env.proceed({name: a[0]}))
                next_wpeb = next_obs[name].observation.workplace_embed[np.newaxis, :]

                ep_r += next_obs[name].reward
                step += 1

                UI_stats = get_UI(UI_stats, name, menu, env.state_.order, a[0])


                if next_obs[name].terminate:
                    step += 1
                    t += 1
                    if next_obs[name].success:
                        test.append(1)
                    else:
                        if name == 'human':
                            mistake.append(0)
                        else:
                            mistake.append(1)
                        test.append(0)



                    break

                wpeb = next_wpeb

            if next_obs[name].terminate:
                break

        if not next_obs[name].terminate:
            mistake.append(2)
            test.append(0)


    acc = sum(test) / len(test)
    herr = (np.array(mistake) == 0).mean()
    rerr = (np.array(mistake) == 1).mean()

    UI_stats = {n: [np.array(s) for s in UI_stats[n]] for n in UI_stats}
    UI_ratio = {n: UI_stats[n][0][UI_stats[n][1]].mean() for n in UI_stats}

    print('Test: Overall Accuracy: {} Human Error Percentage: {} Robot Error Percentage: {} UI(Human): {}, UI(robot):{}\n'.format(
        acc, herr, rerr, UI_ratio['human'], UI_ratio['robot']))

    return acc


def swap_test(ckpt_dir1, ckpt_dir2, env, gpu):
    if gpu == -1:
        device_count = 0
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        device_count = 1

    config = tf.ConfigProto(device_count={'GPU': device_count})
    config.gpu_options.allow_growth = True
    with tf.Graph().as_default() as group1:
        agents1 = get_trainers(0, device_count)
        saver1 = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
        sess1 = tf.Session(graph=group1, config=config)
        sess1.run(tf.initializers.global_variables())
        saver1.restore(sess1, os.path.join(os.path.dirname(__file__), 'output', tf.train.latest_checkpoint(ckpt_dir1).split('output/')[-1]))


    with tf.Graph().as_default() as group2:
        agents2 = get_trainers(0, device_count)
        saver2 = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

        sess2 = tf.Session(graph=group2, config=config)
        sess2.run(tf.initializers.global_variables())
        saver2.restore(sess2, os.path.join(os.path.dirname(__file__), 'output',
                                           tf.train.latest_checkpoint(ckpt_dir2).split('output/')[-1]))

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
        env.reset(MENU, train=False)
        goal = np.array([env.state_.order])
        obs = env.start()
        menu = deepcopy(obs['robot'].observation.menu_embed[np.newaxis, :])
        wpeb = deepcopy(obs['robot'].observation.workplace_embed[np.newaxis, :])

        in_state = [None, None]

        # init
        for agent in agents1:
            agent.set_policy_index()
        for agent in agents2:
            agent.set_policy_index()



        for t in range(MAX_TIME):

            for i, name in enumerate(['human', 'robot']):
                a, in_state[i], ent, _, _ = agents_to_act[i].action(sesses[i], menu[:,None,:,:], wpeb[:,None,:], goal, 0.01,
                                                    np.array([[True]]),in_state[i], train=False)
                _, in_state[(i+1)%2], _, _, _ = agents_to_act[(i+1)%2].action(sesses[(i+1)%2], menu[:, None, :, :], wpeb[:, None, :], goal, 0.01,
                                                       np.array([[True]]), in_state[(i+1)%2], train=False)
                a = np.squeeze(a, axis=1)
                next_obs = deepcopy(env.proceed({name: a[0]}))
                next_wpeb = next_obs[name].observation.workplace_embed[np.newaxis, :]


                if next_obs[name].terminate:
                    if next_obs[name].success:
                        test.append(1)
                    else:
                        if name == 'human':
                            mistake.append(0)
                        else:
                            mistake.append(1)
                        test.append(0)



                    break

                wpeb = next_wpeb

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


def plot_all_test_acc(ckpt_dir, env, gpu):

    if gpu == -1:
        device_count = 0
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        device_count = 1

    agents = get_trainers(0, device_count)
    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=NUM_CKPT)
    config = tf.ConfigProto(device_count={'GPU': device_count})
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    ckpt_files = [os.path.join(ckpt_dir, f.split('.')[0]) for f in os.listdir(ckpt_dir) if f.startswith('model-') and f.endswith('.meta')]
    ckpt_files.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('-')[-1]))

    ckpt_files = list(filter(lambda x: (int(x.split('/')[-1].split('.')[0].split('-')[-1]) - 200000) % 300000 != 0 and int(x.split('/')[-1].split('.')[0].split('-')[-1]) > 500000, ckpt_files))

    for i, ckpt in enumerate(ckpt_files):
        sess.run(tf.initializers.global_variables())
        saver.restore(sess, ckpt)
        for agent in agents:
            agent.p_update(sess)
            agent.q_update(sess)
        acc= test_model(sess, env, agents, 10000)
        print('CKPT name: {}, Acc: {}'.format(ckpt.split('/')[-1], acc))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir',
                        default='/home/alexzhou907/My_Items/Communication_Game/Benchmark/MADDPG/rnn_gumbel/output/cara_maddpg_kitchen/screen_dataset3_run2/2019-08-29-19-02-14/ckpt/best/')

    parser.add_argument('--ckpt-dir2',
                        default='/home/alexzhou907/My_Items/Communication_Game/Benchmark/MADDPG/rnn_gumbel/output/cara_maddpg_kitchen/screen_dataset3_run2/2019-08-29-19-02-14/ckpt/best/')
    parser.add_argument('--swap', default=False, action='store_true')
    parser.add_argument('--get_test_acc_over_time', default=False, action='store_true')
    parser.add_argument('--ds', default=3)
    parser.add_argument('--gpu', default=0)

    opt = parser.parse_args()
    print(opt)

    env = Kitchen('human', 'robot', 0, -1, 1, INGREDIENTS, NUM_DISHES, MAX_DISH_INGREDIENTS,
                  dataset_path='/home/Datasets/Kitchen/dataset{}_with_10_ingred_4_dish_5_maxingred_1000000_size_0.7_ratio.npz'.format(
                      opt.ds))

    if opt.swap:
        swap_test(opt.ckpt_dir, opt.ckpt_dir2, env, opt.gpu)
        swap_test(opt.ckpt_dir2, opt.ckpt_dir, env, opt.gpu)
    else:
        if opt.get_test_acc_over_time:
            plot_all_test_acc(opt.ckpt_dir, env, opt.gpu)
        else:
            test(opt.ckpt_dir, env, opt.gpu)


if __name__ == '__main__':
    main()


import sys
sys.path.append("../..")
from Game.kitchen import Kitchen
from Benchmark.ValueAlignment.config import *
from Benchmark.ValueAlignment.agents import *
from Benchmark.ValueAlignment.buffer import *
from Benchmark.ValueAlignment.qnetwork import *

from shutil import copyfile
from copy import deepcopy
import tensorflow as tf
import os
import sys
import logging
import datetime
import argparse
from pprint import pprint

## game
MENU = None #[(0, 1, 2, 2, 9), (7,8, 1, 4, 5), (6,1, 0, 6, 4), (4,0, 5, 7,3)]
INGREDIENTS = list(range(12))
NUM_DISHES = 7
MAX_DISH_INGREDIENTS = 5

## agents
NUM_HACTIONS = len(INGREDIENTS)
NUM_RACTIONS = len(INGREDIENTS)
NUM_GOALS = NUM_DISHES
EMBED_DIM = 10
DENSE = 100
STEP_REWARD = 0
REWARD = 1


EPSILON_START = 0.95
EPSILON_MIN = 0.05
SWITCH_ITER = 5e4
EPSILON_DECAY = SWITCH_ITER / 4

TEMP_MIN = 0.1
TEMP_START = 100
TEMP_DECAY = SWITCH_ITER


LR = 1e-4
LR_BETA1 = 0.9
LR_EPSILON = 1e-8
DISCOUNT = 0.9

BUFFER_ROOM = 2000
BS = 200
TRAIN_STEP = BUFFER_ROOM // 20

MAX_EP = 40000
MAX_TIME = 10
UPDATE_STEP = 100

NUM_CKPT = 50
CKPT_STEP = MAX_EP // NUM_CKPT
SUM_SEC = 600

LOG_STEP = 100

DEVICE_COUNT=0


def get_UI(UI_stats, name, menu, target_goal, a):
    t = menu[0,target_goal]
    t2 = (menu[0].sum(axis=0) == t) & (t > 0)
    has_UI = t2.any()
    use_UI = t2[a]
    UI_stats[name][0].append(use_UI)
    UI_stats[name][1].append(has_UI)

    return UI_stats

def test_model(sess, env, human, robot, qnet):
    mistake = []
    test = []

    belief_acc = []

    UI_stats = {'human':[[], []], 'robot':[[],[]]}
    for ep in range(int(1e4)):
        rbelief = np.zeros((1, NUM_DISHES))
        rbelief.fill(0.25)

        env.reset(MENU, train=False)
        target_goal = env.state_.order
        obs = env.start()
        menu = deepcopy(obs['robot'].observation.menu_embed[np.newaxis, :])
        wpeb = deepcopy(obs['robot'].observation.workplace_embed[np.newaxis, :])



        for t in range(MAX_TIME):

            # robot's turn
            if t == 0:
                raction = np.zeros((1, NUM_RACTIONS))

            else:

                qvalues_all, hprob_all = qnet.get_qvalues_and_hprob_matrix(sess, menu, wpeb, rbelief, 0.1, train=False)

                raction = robot.act(sess, qvalues_all, hprob_all, rbelief, train=False)
                ra = np.argmax(raction, axis=-1)[0]

                next_obs = deepcopy(env.proceed({'robot': ra}))
                #####

                UI_stats = get_UI(UI_stats, 'robot', menu, target_goal, ra)


                if next_obs['robot'].terminate:
                    if next_obs['robot'].success:
                        test.append(1)
                    else:
                        mistake.append(0)
                        test.append(0)
                    t += 1

                    belief_acc.append(int(np.argmax(rbelief) == target_goal))

                    break



            # human's turn
            qvalues_per_ra, hprob_per_ra = qnet.get_q_all_hactions(sess, menu, wpeb, rbelief, raction, 0.1, train=False)

            haction = human.act(sess, qvalues_per_ra, np.array([target_goal]), 0.1, train=False)
            ha = np.argmax(haction, axis=-1)[0]

            ## step
            next_obs = deepcopy(env.proceed({'human': ha}))
            next_wpeb = next_obs['human'].observation.workplace_embed[np.newaxis, :]

            #########

            UI_stats = get_UI(UI_stats, 'human', menu, target_goal, ha)

            obs = next_obs
            wpeb = next_wpeb
            hprob_per_ha_ra = hprob_per_ra[np.arange(haction.shape[0]), np.argmax(haction, axis=-1),
                              :]  # get per haction
            rbelief = robot.belief.fixed_point_belief(rbelief, hprob_per_ha_ra)

            if obs['human'].terminate:
                if obs['human'].success:
                    test.append(1)
                else:
                    mistake.append(1)
                    test.append(0)
                t += 1

                belief_acc.append(int(np.argmax(rbelief) == target_goal))

                break



        if not next_obs['human'].terminate:
            mistake.append(2)
            test.append(0)
        ## eval


    acc = sum(test) / len(test)
    belief_acc = sum(belief_acc) / len(belief_acc)
    herr = (np.array(mistake) == 1).mean()
    rerr = (np.array(mistake) == 0).mean()

    UI_stats = {n: [np.array(s) for s in UI_stats[n]] for n in UI_stats}
    UI_ratio = {n: UI_stats[n][0][UI_stats[n][1]].mean() for n in UI_stats}

    print(
            'Episode: {} Overall Accuracy: {} Human Error Percentage: {} Robot Error Percentage: {}\n'
            'UI(human): {} UI(robot): {} Belief Acc: {}\n'.format(
                ep, acc, herr, rerr, UI_ratio['human'], UI_ratio['robot'], belief_acc))

    return acc

def test(ckpt_dir, ds):
    env = Kitchen('human', 'robot', STEP_REWARD, -REWARD, REWARD, INGREDIENTS, NUM_DISHES, MAX_DISH_INGREDIENTS,
                  dataset_path='/home/luyao/Datasets/Kitchen/dataset%d_with_%d_ingred_%d_dish_5_maxingred_1000000_size_0.7_ratio.npz'\
                               % (int(ds), NUM_RACTIONS, NUM_DISHES))

    rconfig = RobotConfig(NUM_HACTIONS, NUM_RACTIONS, NUM_GOALS)
    hconfig = HumanConfig(NUM_HACTIONS, NUM_RACTIONS, NUM_GOALS)
    qconfig = QNetConfig(NUM_DISHES, len(INGREDIENTS), EMBED_DIM, NUM_HACTIONS, NUM_RACTIONS, NUM_GOALS, DENSE, LR,
                         LR_BETA1, LR_EPSILON, DISCOUNT)

    human = Human(hconfig)
    robot = Robot(rconfig)
    qnet = QNetwork(qconfig)

    saver = tf.train.Saver()
    config = tf.ConfigProto(device_count={'GPU': DEVICE_COUNT})
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))

        mistake = []
        test = []

        belief_acc = []

        UI_stats = {'human':[[], []], 'robot':[[],[]]}
        for ep in range(10000):
            rbelief = np.zeros((1, NUM_DISHES))
            rbelief.fill(0.25)

            env.reset(MENU, train=False)
            target_goal = env.state_.order
            obs = env.start()
            menu = deepcopy(obs['robot'].observation.menu_embed[np.newaxis, :])
            wpeb = deepcopy(obs['robot'].observation.workplace_embed[np.newaxis, :])



            for t in range(MAX_TIME):

                # robot's turn
                if t == 0:
                    raction = np.zeros((1, NUM_RACTIONS))

                else:

                    qvalues_all, hprob_all = qnet.get_qvalues_and_hprob_matrix(sess, menu, wpeb, rbelief, 0.1, train=False)

                    raction = robot.act(sess, qvalues_all, hprob_all, rbelief, train=False)
                    ra = np.argmax(raction, axis=-1)[0]

                    next_obs = deepcopy(env.proceed({'robot': ra}))
                    #####

                    UI_stats = get_UI(UI_stats, 'robot', menu, target_goal, ra)


                    if next_obs['robot'].terminate:
                        if next_obs['robot'].success:
                            test.append(1)
                        else:
                            mistake.append(0)
                            test.append(0)
                        t += 1

                        belief_acc.append(int(np.argmax(rbelief) == target_goal))

                        break



                # human's turn
                qvalues_per_ra, hprob_per_ra = qnet.get_q_all_hactions(sess, menu, wpeb, rbelief, raction, 0.1, train=False)

                haction = human.act(sess, qvalues_per_ra, np.array([target_goal]), 0.1, train=False)
                ha = np.argmax(haction, axis=-1)[0]

                ## step
                next_obs = deepcopy(env.proceed({'human': ha}))
                next_wpeb = next_obs['human'].observation.workplace_embed[np.newaxis, :]

                #########

                UI_stats = get_UI(UI_stats, 'human', menu, target_goal, ha)

                obs = next_obs
                wpeb = next_wpeb
                hprob_per_ha_ra = hprob_per_ra[np.arange(haction.shape[0]), np.argmax(haction, axis=-1),
                                  :]  # get per haction
                rbelief = robot.belief.fixed_point_belief(rbelief, hprob_per_ha_ra)

                if obs['human'].terminate:
                    if obs['human'].success:
                        test.append(1)
                    else:
                        mistake.append(1)
                        test.append(0)
                    t += 1

                    belief_acc.append(int(np.argmax(rbelief) == target_goal))

                    break



            if not next_obs['human'].terminate:
                mistake.append(2)
                test.append(0)
            ## eval


        acc = sum(test) / len(test)
        belief_acc = sum(belief_acc) / len(belief_acc)
        herr = (np.array(mistake) == 1).mean()
        rerr = (np.array(mistake) == 0).mean()

        UI_stats = {n: [np.array(s) for s in UI_stats[n]] for n in UI_stats}
        UI_ratio = {n: UI_stats[n][0][UI_stats[n][1]].mean() for n in UI_stats}

        print(
                'Episode: {} Overall Accuracy: {} Human Error Percentage: {} Robot Error Percentage: {}\n'
                'UI(human): {} UI(robot): {} Belief Acc: {}'.format(
                    ep, acc, herr, rerr, UI_ratio['human'], UI_ratio['robot'], belief_acc))

        return acc
        #
        # def numpy_fillna(data):
        #     # Get lengths of each row of data
        #     lens = np.array([len(i) for i in data])
        #
        #     # Mask of valid places in each row
        #     mask = np.arange(lens.max()) < lens[:, None]
        #
        #     # Setup output array and put elements from data into masked positions
        #     out = np.zeros(mask.shape, dtype=data.dtype)
        #     out[mask] = np.concatenate(data)
        #     return out
        #
        # first = []
        # second = []
        # third = []
        # for ep in success_stats['success']:
        #     for i, k in enumerate(ep):
        #         if i == 0:
        #             first.append(k)
        #         elif i == 1:
        #             second.append(k)
        #         else:
        #             third.append(k)
        #
        # print(' Step: 1, Success Rate: {}'.format(np.array(first).mean()))
        # print(' Step: 2, Success Rate: {}'.format(np.array(second).mean()))
        # print(' Step: 3, Success Rate: {}'.format(np.array(third).mean()))


        # success_stats_np = numpy_fillna()
        # stats = success_stats_np.mean(axis=0)
        #
        # print('Max length: ',len(stats))
        # for i, k in enumerate(stats):
        #     print(' Step: {}, Success Rate: {}'.format(i, k))
        #
        # belief_success_np = numpy_fillna(np.array(success_stats['belief']))
        # for i, k in enumerate(belief_success_np.mean(axis=0)):
        #     print(' Step: {}, Belief Accuracy: {}'.format(i, k))


def swap_test(ckpt_dir1, ckpt_dir2, ds):
    env = Kitchen('human', 'robot', STEP_REWARD, -REWARD, REWARD, INGREDIENTS, NUM_DISHES, MAX_DISH_INGREDIENTS,
                  dataset_path='/home/Datasets/Kitchen/dataset{}_with_10_ingred_4_dish_5_maxingred_1000000_size_0.7_ratio.npz'.format(ds))


    rconfig = RobotConfig(NUM_HACTIONS, NUM_RACTIONS, NUM_GOALS)
    hconfig = HumanConfig(NUM_HACTIONS, NUM_RACTIONS, NUM_GOALS)
    qconfig = QNetConfig(NUM_DISHES, len(INGREDIENTS), EMBED_DIM, NUM_HACTIONS, NUM_RACTIONS, NUM_GOALS, DENSE, LR,
                         LR_BETA1, LR_EPSILON, DISCOUNT)

    with tf.Graph().as_default() as group1:

        human1 = Human(hconfig)
        robot1 = Robot(rconfig)
        qnet1 = QNetwork(qconfig)
        saver1 = tf.train.Saver()

    with tf.Graph().as_default() as group2:
        human2 = Human(hconfig)
        robot2 = Robot(rconfig)
        qnet2 = QNetwork(qconfig)
        saver2 = tf.train.Saver()

    config = tf.ConfigProto(device_count={'GPU': DEVICE_COUNT})
    config.gpu_options.allow_growth = True
    sess1 = tf.Session(graph=group1, config=config)
    sess2 = tf.Session(graph=group2, config=config)

    saver1.restore(sess1, tf.train.latest_checkpoint(ckpt_dir1))
    saver2.restore(sess2, tf.train.latest_checkpoint(ckpt_dir2))

    mistake = []
    test = []

    belief_acc = []
    UI_stats = {'human': [[], []], 'robot': [[], []]}
    for ep in range(10000):
        rbelief = np.zeros((1, NUM_DISHES))
        rbelief.fill(0.25)

        env.reset(MENU, train=False)
        target_goal = env.state_.order
        obs = env.start()
        menu = deepcopy(obs['robot'].observation.menu_embed[np.newaxis, :])
        wpeb = deepcopy(obs['robot'].observation.workplace_embed[np.newaxis, :])



        for t in range(MAX_TIME):

            # robot's turn
            if t == 0:
                raction = np.zeros((1, NUM_RACTIONS))

            else:

                qvalues_all, hprob_all = qnet2.get_qvalues_and_hprob_matrix(sess2, menu, wpeb, rbelief, 0.1, train=False)

                raction = robot2.act(sess2, qvalues_all, hprob_all, rbelief, train=False)
                ra = np.argmax(raction, axis=-1)[0]
                next_obs = deepcopy(env.proceed({'robot': ra}))
                #####

                UI_stats = get_UI(UI_stats, 'robot', menu, target_goal, ra)

                if next_obs['robot'].terminate:
                    if next_obs['robot'].success:
                        test.append(1)
                    else:
                        mistake.append(0)
                        test.append(0)
                    t += 1

                    belief_acc.append(int(np.argmax(rbelief) == target_goal))


                    break

            # human's turn
            # if t == 0:
            qvalues_per_ra, hprob_per_ra = qnet1.get_q_all_hactions(sess1, menu, wpeb, rbelief, raction, 0.1, train=False)
            #
            # else:
            #     qvalues_per_ra, hprob_per_ra = qvalues_all[np.arange(menu.shape[0]), :, np.argmax(raction, axis=-1),:], hprob_all[np.arange(menu.shape[0]), :, np.argmax(raction, axis=-1),:]

            haction = human1.act(sess1, qvalues_per_ra, np.array([target_goal]), 0.1, train=False)
            ha = np.argmax(haction, axis=-1)[0]

            ## step
            next_obs = deepcopy(env.proceed({'human': ha}))
            next_wpeb = next_obs['human'].observation.workplace_embed[np.newaxis, :]

            #########

            UI_stats = get_UI(UI_stats, 'human', menu, target_goal, ha)

            obs = next_obs
            wpeb = next_wpeb
            qvalues_per_ra, hprob_per_ra = qnet2.get_q_all_hactions(sess2, menu, wpeb, rbelief, raction, 0.1,
                                                                    train=False)
            hprob_per_ha_ra = hprob_per_ra[np.arange(haction.shape[0]), np.argmax(haction, axis=-1),
                              :]  # get per haction
            rbelief = robot2.belief.fixed_point_belief(rbelief, hprob_per_ha_ra)

            if obs['human'].terminate:
                if obs['human'].success:
                    test.append(1)
                else:
                    mistake.append(1)
                    test.append(0)
                t += 1
                belief_acc.append(int(np.argmax(rbelief) == target_goal))
                break



        if not next_obs['human'].terminate:
            mistake.append(2)
            test.append(0)
        ## eval


    acc = sum(test) / len(test)

    belief_acc = sum(belief_acc) / len(belief_acc)
    herr = (np.array(mistake) == 1).mean()
    rerr = (np.array(mistake) == 0).mean()

    UI_stats = {n: [np.array(s) for s in UI_stats[n]] for n in UI_stats}
    UI_ratio = {n: UI_stats[n][0][UI_stats[n][1]].mean() for n in UI_stats}

    print(
            'Episode: {} Overall Accuracy: {} Human Error Percentage: {} Robot Error Percentage: {}\n'
            'UI(human): {} UI(robot): {} Belief Acc: {}'.format(
                ep, acc, herr, rerr, UI_ratio['human'], UI_ratio['robot'], belief_acc))


    #
    # def numpy_fillna(data):
    #     # Get lengths of each row of data
    #     lens = np.array([len(i) for i in data])
    #
    #     # Mask of valid places in each row
    #     mask = np.arange(lens.max()) < lens[:, None]
    #
    #     # Setup output array and put elements from data into masked positions
    #     out = np.zeros(mask.shape, dtype=data.dtype)
    #     out[mask] = np.concatenate(data)
    #     return out
    #
    # first = []
    # second = []
    # third = []
    # for ep in success_stats['success']:
    #     for i, k in enumerate(ep):
    #         if i == 0:
    #             first.append(k)
    #         elif i == 1:
    #             second.append(k)
    #         else:
    #             third.append(k)
    #
    # print(' Step: 1, Success Rate: {}'.format(np.array(first).mean()))
    # print(' Step: 2, Success Rate: {}'.format(np.array(second).mean()))
    # print(' Step: 3, Success Rate: {}'.format(np.array(third).mean()))


    # success_stats_np = numpy_fillna()
    # stats = success_stats_np.mean(axis=0)
    #
    # print('Max length: ',len(stats))
    # for i, k in enumerate(stats):
    #     print(' Step: {}, Success Rate: {}'.format(i, k))
    #
    # belief_success_np = numpy_fillna(np.array(success_stats['belief']))
    # for i, k in enumerate(belief_success_np.mean(axis=0)):
    #     print(' Step: {}, Belief Accuracy: {}'.format(i, k))


def plot_all_test_acc(ckpt_dir, ds):
    env = Kitchen('human', 'robot', STEP_REWARD, -REWARD, REWARD, INGREDIENTS, NUM_DISHES, MAX_DISH_INGREDIENTS,
                  dataset_path='/home/Datasets/Kitchen/dataset{}_with_10_ingred_4_dish_5_maxingred_1000000_size_0.7_ratio.npz'.format(ds))

    rconfig = RobotConfig(NUM_HACTIONS, NUM_RACTIONS, NUM_GOALS)
    hconfig = HumanConfig(NUM_HACTIONS, NUM_RACTIONS, NUM_GOALS)
    qconfig = QNetConfig(NUM_DISHES, len(INGREDIENTS), EMBED_DIM, NUM_HACTIONS, NUM_RACTIONS, NUM_GOALS, DENSE, LR,
                         LR_BETA1, LR_EPSILON, DISCOUNT)

    human = Human(hconfig)
    robot = Robot(rconfig)
    qnet = QNetwork(qconfig)

    saver = tf.train.Saver()
    config = tf.ConfigProto(device_count={'GPU': DEVICE_COUNT})
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)

    ckpt_files = tf.train.get_checkpoint_state(ckpt_dir).all_model_checkpoint_paths[-1:]

    for i, ckpt in enumerate(ckpt_files):
        saver.restore(sess, ckpt)
        acc= test_model(sess, env, human, robot, qnet)
        print('CKPT name: {}, Acc: {}'.format(ckpt.split('/')[-1], acc))



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--ckpt_dir',
                        default='/home/alexzhou907/My_Items/Communication_Game/Benchmark/ValueAlignment/output/thor_cirl_results/screen_dataset1_run1/2019-09-03-15-08-46/ckpt/')

    parser.add_argument('--ckpt-dir2', default='/home/alexzhou907/My_Items/Communication_Game/Benchmark/ValueAlignment/output/screen_dataset2/2019-08-27-12-52-59/ckpt/best/')
    parser.add_argument('--swap', default=False, action='store_true')
    parser.add_argument('--plot_all_acc_over_time', default=True, action='store_true')
    parser.add_argument('--ds', default=1)


    opt = parser.parse_args()

    if opt.swap:
        swap_test(opt.ckpt_dir, opt.ckpt_dir2, opt.ds)
        swap_test(opt.ckpt_dir2, opt.ckpt_dir, opt.ds)
    else:
        if opt.plot_all_acc_over_time:
            plot_all_test_acc(opt.ckpt_dir, opt.ds)
        else:
            test(opt.ckpt_dir, opt.ds)


if __name__ == '__main__':
    main()


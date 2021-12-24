from Game.kitchen import Kitchen
from Benchmark.MADDPG.rnn_gumbel.config import *
from Benchmark.MADDPG.test_agents import *

from shutil import copyfile
from copy import deepcopy
import tensorflow as tf
import os
import sys
import logging
import datetime
import argparse

## game
MENU = None #[(0, 1, 2, 2, 9), (7,8, 1, 4, 5), (6,1, 0, 6, 4), (4,0, 5, 7,3)]
INGREDIENTS = list(range(10))
NUM_DISHES = 4
MAX_DISH_INGREDIENTS = 5

## agents
NUM_ACTIONS = len(INGREDIENTS)
NUM_GOALS = NUM_DISHES
EMBED_DIM = 8
NUM_ENSEMBLE = 1
DENSE = 100
STEP_REWARD = 0
REWARD = 1
TEMP_START = 0.95
TEMP_MIN = 0.1
DECAY_STEP = 5e4


A_LR = 5e-4
C_LR = 5e-3
DISCOUNT = 0.8
LAMBDA = 0.
BETA_H = 1e-2
BETA_R = 1e-2
BREAK_CORR = False

BUFFER_ROOM = 1000
BS = 200
TRAIN_STEP = BUFFER_ROOM // BS

MAX_EP = 10000000
MAX_TIME = 5
UPDATE_STEP = 100

CKPT_STEP = 5000
SUM_SEC = 600

LOG_STEP = 10

DEVICE_COUNT=0


def setup_logger(log_dir):
    logger = logging.getLogger()
    logger.handlers = []
    file_handler = logging.FileHandler(os.path.join(log_dir, 'log.log'))
    console_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)

def get_trainers():
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
        config = Config(NUM_DISHES, len(INGREDIENTS), NUM_ENSEMBLE, NUM_ACTIONS, DENSE,
                        EMBED_DIM, A_LR, C_LR, DISCOUNT, LAMBDA, beta, BS, BUFFER_ROOM, MAX_TIME, UPDATE_STEP,
                        BREAK_CORR, use_gpu)
        agent = MADDPGAgentTrainer(name, 2, i, config, p_func, q_func)
        trainer.append(agent)

    return trainer




def test(ckpt_dir):

    env = Kitchen('human', 'robot', STEP_REWARD, -REWARD, REWARD, INGREDIENTS, NUM_DISHES, MAX_DISH_INGREDIENTS)

    agents = get_trainers()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))

        mistake = []
        test = []
        step = 0
        for ep in range(500):
            ep_r = 0
            temp = 0.01#TEMP_MIN + (TEMP_START - TEMP_MIN) * np.exp(-1 * ep / DECAY_STEP)

            env.reset(MENU)
            goal = np.array([env.state_.order])
            obs = env.start()
            menu = deepcopy(obs['robot'].observation.menu_embed[np.newaxis, :])
            wpeb = deepcopy(obs['robot'].observation.workplace_embed[np.newaxis,:])

            in_state = [None, None]

            if ep % LOG_STEP == 0:
                print(
                    'Episode: {}\n'.format(ep))

            for agent in agents:
                agent.set_policy_index()

            for t in range(1000):

                for i, name in enumerate(['human', 'robot']):
                    a, in_state[i], ent, _, _ = agents[i].action(sess, menu[:,None,:,:], wpeb[:,None,:], goal, temp,
                                                        np.array([[True]]),in_state[i], train=False)
                    _, in_state[(i+1)%2], _, _, _ = agents[(i+1)%2].action(sess, menu[:, None, :, :], wpeb[:, None, :], goal, temp,
                                                           np.array([[True]]), in_state[(i+1)%2], train=False)
                    a = np.squeeze(a, axis=1)
                    next_obs = deepcopy(env.proceed({name: a[0]}))
                    next_wpeb = next_obs[name].observation.workplace_embed[np.newaxis, :]

                    if ep % LOG_STEP == 0:
                        q = agents[i].get_all_action_q(sess, menu[:,None,:,:], wpeb[:,None,:], goal, True)
                        logit = agents[i].get_pi_logit(sess,  menu[:,None,:,:], wpeb[:,None,:], goal, True)

                        targets = menu[0, env.state_.order, :]
                        q_at_goal = q[0, :, env.state_.order]  # [1:]
                        correct_ingredient_avg_q_per_ra = (q_at_goal * (targets != 0.)).mean()
                        wrong_ingredient_avg_q_per_ra = (q_at_goal * (targets == 0)).mean()

                        debug_msg = {
                            'goal': targets,
                            'correct_ingredient_avg_q_per_ra': correct_ingredient_avg_q_per_ra,
                            'wrong_ingredient_avg_q_per_ra': wrong_ingredient_avg_q_per_ra,
                            'average_q_diff': correct_ingredient_avg_q_per_ra - wrong_ingredient_avg_q_per_ra
                        }

                        print(
                            '   Name: {} Action: {}\n   Menu:{} Goal: {}\n   Workplace:{} Reward: {}\n'
                            '     Q: {} \n    Policy logits: {}\n  Entropy: {} average_q_diff: {}\n'.format(
                                name, a, menu, goal, wpeb, next_obs[name].reward, q, logit, ent,
                                debug_msg['average_q_diff']))

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

            # for agent in agents:
            #     agent.preupdate()
            # for l, agent in enumerate(agents):
            #     result = agent.update(sess, agents, temp, ep)
            #
            #     if result is not None and ep % LOG_STEP == 0:
            #         q_loss, p_loss, target_q_mean, reward_mean, target_q_std_mean = result
            #
            #         print(
            #             '       Name: {}\n      Q_Loss: {} P_Loss: {}\n'
            #             '       Mean Target Q: {} Mean Target Q std: {}\n'
            #             '       Mean Reward: {} \n'.format(
            #                 agent.name, q_loss, p_loss, target_q_mean, target_q_std_mean,
            #                 reward_mean))

        acc = sum(test) / len(test)
        herr = (np.array(mistake) == 0).mean()
        rerr = (np.array(mistake) == 1).mean()

        print('Overall Accuracy: {} Human Error Percentage: {} Robot Error Percentage: {}\n'.format(
            acc, herr, rerr))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--screen', default='0')
    parser.add_argument('--ckpt_dir', default='/home/alexzhou907/My_Items/Communication_Game/Benchmark/MADDPG/rnn_gumbel/output/screen_0.6G0.8L/2019-08-18-12-02-24/ckpt/best/')
    parser.add_argument('--sum_dir', default='')
    parser.add_argument('--log_dir', default='')
    opt = parser.parse_args()

    test(opt.ckpt_dir)

if __name__ == '__main__':
    main()

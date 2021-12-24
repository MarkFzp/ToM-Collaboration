from Game.kitchen import Kitchen
from Benchmark.MADDPG.mlp.config import *
from Benchmark.MADDPG.mlp.agents import *

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
EMBED_DIM = 5
NUM_ENSEMBLE = 1
DENSE = 100
STEP_REWARD = 0
REWARD = 1
TEMP_START = 0.95
TEMP_MIN = 0.01
DECAY_STEP = 5e4

A_LR = 1e-4
C_LR = 5e-4
DISCOUNT = 0.9

BUFFER_ROOM = 1000
BS = 200
TRAIN_STEP = BUFFER_ROOM // 20

MAX_EP = 1000000
MAX_TIME = 5
UPDATE_STEP = 500

CKPT_STEP = 5000
SUM_SEC = 600

LOG_STEP = 1000


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
        config = Config(NUM_DISHES, len(INGREDIENTS), NUM_ENSEMBLE, NUM_ACTIONS, DENSE, EMBED_DIM, A_LR, C_LR, DISCOUNT, BS, BUFFER_ROOM, UPDATE_STEP)
        if i == 0:
            name = 'human'
        else:
            name = 'robot'
        agent = MADDPGAgentTrainer(name, 2, i, config, p_func, q_func)
        trainer.append(agent)

    return trainer

def train(ckpt_dir, sum_dir):

    env = Kitchen('human', 'robot', STEP_REWARD, -REWARD, REWARD, INGREDIENTS, NUM_DISHES, MAX_DISH_INGREDIENTS)

    agents = get_trainers()

    writer = tf.summary.FileWriter(sum_dir, flush_secs=SUM_SEC)

    saver = tf.train.Saver()
    config = tf.ConfigProto(device_count = {'GPU': 1})
    config.gpu_options.allow_growth = True
    with tf.Session( config=config) as sess:
        sess.run(tf.initializers.global_variables())

        mistake = []
        test = []
        step = 0
        for ep in range(MAX_EP):
            ep_r = 0
            ep_q_loss = [0,0]
            ep_p_loss = [0,0]

            temp = 0.1#TEMP_MIN + (TEMP_START - TEMP_MIN) * np.exp(-1 * ep / DECAY_STEP)

            env.reset(MENU)
            goal = np.array([env.state_.order])
            obs = env.start()
            menu = deepcopy(obs['robot'].observation.menu_embed[np.newaxis, :])
            wpeb = deepcopy(obs['robot'].observation.workplace_embed[np.newaxis,:])

            if ep % LOG_STEP == 0:
                logging.info(
                    'Episode: {}\n'.format(ep))

            for i in range(2):
                agents[i].set_policy_index()


            for t in range(MAX_TIME):

                for i, name in enumerate(['human', 'robot']):
                    a, ent = agents[i].action(sess, menu, wpeb, goal, temp, train=True)
                    next_obs = deepcopy(env.proceed({name: a[0]}))
                    next_wpeb = next_obs[name].observation.workplace_embed[np.newaxis, :]

                    ## store
                    menu_step = [menu, menu]
                    wpeb_step = [wpeb, wpeb]
                    action_step = [a if k == i else np.array([-1]) for k in range(2)]
                    reward_step = [np.array([next_obs[name].reward]), np.array([next_obs[name].reward])]
                    next_wpeb_step = [next_wpeb, next_wpeb]
                    goal_step = [goal, goal]
                    terminal_step = [np.array([next_obs[name].terminate]), np.array([next_obs[name].terminate])]
                    for agent in agents:
                        agent.experience(menu_step, wpeb_step, action_step, reward_step, next_wpeb_step, goal_step, terminal_step)

                    if ep % LOG_STEP == 0:
                        q = agents[i].get_all_action_q(sess, menu, wpeb, goal, True)
                        logit = agents[i].get_pi_logit(sess, menu, wpeb, goal, True)
                        logging.info(
                            '   Name: {} Action: {}\n   Menu:{} Goal: {}\n   Workplace:{} Reward: {}\n'
                            '     Q: {} \n    Policy logits: {}\n  Entropy: {}\n'.format(
                                name, a, menu, goal, wpeb, next_obs[name].reward, q, logit,ent))

                    ep_r += next_obs[name].reward
                    step += 1

                    for agent in agents:
                        agent.preupdate()
                    for l, agent in enumerate(agents):
                        result = agent.update(sess, agents, temp, step)

                        if result is not None and ep % LOG_STEP == 0:
                            q_loss, p_loss, target_q_mean, reward_mean, target_q_next_mean, target_q_std_mean = result

                            logging.info(
                                '       Name: {}\n      Q_Loss: {} P_Loss: {}\n'
                                '       Mean Target Q: {} Mean Target Q std: {} Mean Target Next Q: {}\n'
                                '       Mean Reward: {} \n'.format(
                                    agent.name, q_loss, p_loss, target_q_mean, target_q_std_mean, target_q_next_mean,
                                    reward_mean))

                            ep_q_loss[l] += q_loss
                            ep_p_loss[l] += p_loss


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
            if not mistake:
                herr = rerr = 0
            else:
                herr = (np.array(mistake) == 0).mean()
                rerr = (np.array(mistake) == 1).mean()

            if ep % LOG_STEP == 0:
                logging.info('Episode: {} Overall Accuracy: {} Human Error Percentage: {} Robot Error Percentage: {}\n'.format(
                    ep, acc, herr, rerr))

            if len(test) >= 500:
                test = test[1:]
            if len(mistake) >= 500:
                mistake = mistake[1:]


            ep_r_mean = ep_r / t
            ep_q_loss_mean = [ql / t for ql in ep_q_loss]
            ep_p_loss_mean = [pl / t for pl in ep_p_loss]

            if ep % CKPT_STEP == 0:
                saver.save(sess, os.path.join(ckpt_dir, 'model'), step)

            summary = tf.Summary(value=[
                tf.Summary.Value(tag="episode_reward", simple_value=ep_r_mean),
                tf.Summary.Value(tag="overall_accuracy", simple_value=acc),
                *[tf.Summary.Value(tag="episode_q_loss_agent_%d"%d, simple_value=ep_ql) for d, ep_ql in enumerate(ep_q_loss_mean)],
                *[tf.Summary.Value(tag="episode_p_loss_agent_%d"%d, simple_value=ep_pl) for d, ep_pl in enumerate(ep_p_loss_mean)],

            ])


            writer.add_summary(summary, ep)
            writer.flush()


def get_output_dir(exp_id):
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_dir = os.path.join(os.path.dirname(__file__), exp_id, t)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--screen', default='0')
    parser.add_argument('--ckpt_dir', default='')
    parser.add_argument('--sum_dir', default='')
    parser.add_argument('--log_dir', default='')
    parser.add_argument('--with_null', type=bool, default=False)
    opt = parser.parse_args()

    output_dir = get_output_dir('screen_'+str(opt.screen))

    if opt.ckpt_dir == '':
        ckpt_dir = os.path.join(output_dir, 'ckpt')
    else:
        ckpt_dir = opt.ckpt_dir

    if opt.sum_dir == '':
        sum_dir = os.path.join(output_dir, 'summary')
    else:
        sum_dir = opt.sum_dir

    if opt.log_dir == '':
        log_dir = os.path.join(output_dir, 'log')
    else:
        log_dir = opt.log_dir

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    if not os.path.exists(sum_dir):
        os.makedirs(sum_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    setup_logger(log_dir)

    copyfile(__file__, os.path.join(ckpt_dir, os.path.basename(__file__)))
    copyfile(os.path.join(os.path.dirname(__file__),'agents.py'), os.path.join(ckpt_dir, os.path.basename('agents.py')))

    train(ckpt_dir, sum_dir)

if __name__ == '__main__':
    main()

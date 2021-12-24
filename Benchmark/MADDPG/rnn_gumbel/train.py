from Game.kitchen import Kitchen
from Benchmark.MADDPG.rnn_gumbel.config import *
from Benchmark.MADDPG.rnn_gumbel.agents import *
from Benchmark.MADDPG.rnn_gumbel.test import test_model

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

MAX_EP = 2000000
MAX_TIME = 5
UPDATE_STEP = 100

QP_MATCH_STEP = 200


NUM_CKPT = 50
CKPT_STEP = MAX_EP // NUM_CKPT
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



def train(ckpt_dir, sum_dir, dataset_order, gpu):
    ckpt = tf.train.latest_checkpoint(ckpt_dir)

    if ckpt is None:
        init_ep = 0
    else:

        init_ep = int(ckpt.split('-')[-1])

    env = Kitchen('human', 'robot', STEP_REWARD, -REWARD, REWARD, INGREDIENTS, NUM_DISHES, MAX_DISH_INGREDIENTS,
                  dataset_path='/home/luyao/Datasets/Kitchen/dataset%d_with_%d_ingred_%d_dish_5_maxingred_1000000_size_0.7_ratio.npz'\
                               % (int(dataset_order), NUM_ACTIONS, NUM_DISHES))

    if gpu == -1:
        device_count = 0
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        device_count = 1

    agents = get_trainers(init_ep, device_count)

    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=NUM_CKPT)
    best_saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
    config = tf.ConfigProto(device_count = {'GPU':device_count})
    config.gpu_options.allow_growth = True
    with tf.Session( config=config) as sess:
        sess.run(tf.initializers.global_variables())
        if ckpt is not None:
            saver.restore(sess, ckpt)

        for agent in agents:
            agent.p_update(sess)
            agent.q_update(sess)


        argmax_q = [[], []]
        argmax_policy = [[], []]
        mistake = []
        test = []
        step = 0
        max_acc = 0
        test_acc=0
        for ep in range(init_ep, MAX_EP):
            ep_r = 0
            ep_q_loss = [0,0]
            ep_p_loss = [0,0]

            temp = TEMP_MIN + (TEMP_START - TEMP_MIN) * np.exp(-1 * ep / TEMP_DECAY)

            env.reset(MENU)
            goal = np.array([env.state_.order])
            obs = env.start()
            menu = deepcopy(obs['robot'].observation.menu_embed[np.newaxis, :])
            wpeb = deepcopy(obs['robot'].observation.workplace_embed[np.newaxis,:])

            in_state = [None, None]

            ## store stuff
            menu_traj = [[] for _ in range(2)]
            wpeb_traj = [[] for _ in range(2)]
            act_traj = [[] for _ in range(2)]
            reward_traj = [[] for _ in range(2)]
            terminal_traj = [[] for _ in range(2)]

            if ep % LOG_STEP == 0:
                logging.info(
                    'Episode: {}\n'.format(ep))

            for agent in agents:
                agent.set_policy_index()

            for t in range(MAX_TIME):

                for i, name in enumerate(['human', 'robot']):
                    a, in_state[i], ent, logit, a_cont = agents[i].action(sess, menu[:,None,:,:], wpeb[:,None,:], goal, temp,
                                                        in_state=in_state[i], train=True)
                    _, in_state[(i+1)%2], _, _, _ = agents[(i+1)%2].action(sess, menu[:, None, :, :], wpeb[:, None, :], goal, temp,
                                                                     in_state=in_state[(i+1)%2], train=True)
                    a = np.squeeze(a, axis=1)
                    next_obs = deepcopy(env.proceed({name: a[0]}))
                    next_wpeb = next_obs[name].observation.workplace_embed[np.newaxis, :]

                    for j in range(2):
                        menu_traj[j].append(menu)
                        wpeb_traj[j].append(wpeb)
                        act_traj[j].append(a if j == i else np.array([-1]))
                        reward_traj[j].append(np.array([next_obs[agents[j].name].reward]))
                        terminal_traj[j].append(np.array([obs[agents[j].name].terminate]))

                    # if ep % LOG_STEP == 0:
                    #     q = agents[i].get_all_action_q(sess, menu[:, None, :, :], wpeb[:, None, :], goal, True)
                    #     top_q = np.argsort(q[0,0])[-3:][::-1]
                    #     top_pi = np.argsort(logit[0,0])[-3:][::-1]
                    #     argmax_q[i].append(np.sort(top_q))
                    #     argmax_policy[i].append(np.sort(top_pi))
                    #
                    #     # logging.info(
                    #     #     '   Name: {} Action: {}\n   Menu:{} Goal: {}\n   Workplace:{} Reward: {}\n'
                    #     #     '     Q: {} \n    Policy logits: {}\n  Entropy: {} \n'.format(
                    #     #         name, a, menu, goal, wpeb, next_obs[name].reward, q, logit, ent))
                    #     targets = menu[0, env.state_.order, :]  # [1:]
                    #     correct_ingredient_avg_q_per_ra = (q[0][0] * (targets != 0.)).mean()
                    #     wrong_ingredient_avg_q_per_ra = (q[0][0] * (targets == 0)).mean()
                    #
                    #     debug_msg = {
                    #         'goal': targets,
                    #         'correct_ingredient_avg_q_per_ra': correct_ingredient_avg_q_per_ra,
                    #         'wrong_ingredient_avg_q_per_ra': wrong_ingredient_avg_q_per_ra,
                    #         'average_q_diff': correct_ingredient_avg_q_per_ra - wrong_ingredient_avg_q_per_ra,
                    #         'first_second_q_diff': np.sort(q[0,0])[-1] - np.sort(q[0,0])[-2]
                    #     }
                    #
                    #     logging.info(
                    #         '   Name: {} Action: {}\n   Gumbel Output: {}\n   '
                    #         'Menu:{} Goal: {}\n   Workplace:{} Reward: {}\n'
                    #         '     Q: {} \n    Policy logits: {}\n  Entropy: {} Temp: {}\n '
                    #         '   Average_q_diff: {}, First_second_q_diff: {}\n'
                    #         '   Top 3 Q: {}, Top 3 Policy: {}\n'.format(
                    #             name, a, a_cont, menu, goal, wpeb, next_obs[name].reward, q, logit, ent, temp,
                    #             debug_msg['average_q_diff'], debug_msg['first_second_q_diff'],
                    #             top_q, top_pi))

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

            ## store
            for i, agent in enumerate(agents):
                agent.experience(menu_traj, wpeb_traj, act_traj, reward_traj, [goal, goal], terminal_traj)

            ## train

            for agent in agents:
                agent.preupdate()
            for l, agent in enumerate(agents):
                result = agent.update(sess, agents, temp, ep)

                if result is not None and ep % LOG_STEP == 0:
                    q_loss, p_loss, target_q_mean, reward_mean, target_q_std_mean = result

                    logging.info(
                        '       Name: {}\n      Q_Loss: {} P_Loss: {}\n'
                        '       Mean Target Q: {} Mean Target Q std: {}\n'
                        '       Mean Reward: {} \n'.format(
                            agent.name, q_loss, p_loss, target_q_mean, target_q_std_mean,
                            reward_mean))

                    ep_q_loss[l] = q_loss
                    ep_p_loss[l] = p_loss
            # if argmax_q[1]:
            #     qp_match_prob = [np.mean(np.any(np.stack(argmax_q[k], axis=0) == np.stack(argmax_policy[k], axis=0), axis=-1)) for k in range(2)]
            # else:
            #     qp_match_prob = None
            for k in range(2):
                if len(argmax_q[k]) >= 500:
                    argmax_q[k] = argmax_q[k][1:]
                if len(argmax_policy[k]) >= 500:
                    argmax_policy[k] = argmax_policy[k][1:]

            acc = sum(test) / len(test)
            # if not mistake:
            #     herr = rerr = 0
            # else:
            #     herr = (np.array(mistake) == 0).mean()
            #     rerr = (np.array(mistake) == 1).mean()

            if len(test) >= 500:
                test = test[1:]
            if len(mistake) >= 500:
                mistake = mistake[1:]

            if len(test) == 499 and acc > max_acc:
                max_acc = acc
                if not os.path.exists(os.path.join(ckpt_dir,'best')):
                    os.makedirs(os.path.join(ckpt_dir,'best'))

                best_saver.save(sess, os.path.join(ckpt_dir,'best', 'model'), ep)
                logging.info('Best Acc Found')
                test_acc = test_model(sess, env, agents)

            if ep % CKPT_STEP == 0 and ep > 0:
                saver.save(sess, os.path.join(ckpt_dir, 'model'), ep)

            if ep % LOG_STEP == 0:
                # logging.info('Episode: {} Overall Accuracy: {} Max Accuracy: {} Human Error Percentage: {} Robot Error Percentage: {} '
                #              'Q_Logits_match_prob: {} Test Acc: {}\n'.format(
                #     ep, acc, max_acc, herr, rerr, qp_match_prob, test_acc))
                logging.info(
                    'Episode: {} Overall Accuracy: {} Max Accuracy: {} Test Acc: {}\n'.format(
                        ep, acc, max_acc, test_acc))


        test_acc = test_model(sess, env, agents)
        print('final test acc:', test_acc)

def get_output_dir(exp_id):
    output_dir = os.path.join(os.path.dirname(__file__),'output', exp_id)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--screen', default='0')
    parser.add_argument('--ckpt_dir', default='')
    parser.add_argument('--sum_dir', default='')
    parser.add_argument('--log_dir', default='')
    parser.add_argument('--dataset_order', default=1)
    parser.add_argument('--gpu', default=0)
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
    copyfile(os.path.join(os.path.dirname(__file__), 'policy.py'),
             os.path.join(ckpt_dir, os.path.basename('policy.py')))

    train(ckpt_dir, sum_dir, opt.dataset_order, opt.gpu)

if __name__ == '__main__':
    main()

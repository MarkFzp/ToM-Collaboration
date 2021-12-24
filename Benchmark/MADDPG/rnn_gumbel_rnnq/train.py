from Game.kitchen import Kitchen
from Benchmark.MADDPG.rnn_gumbel_rnnq.config import *
from Benchmark.MADDPG.rnn_gumbel_rnnq.agents import *

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
DECAY_STEP = 5e4

SWITCH_ITER = 5e4
TEMP_MIN = 0.1
TEMP_START = 100
TEMP_DECAY = SWITCH_ITER


A_LR = 1e-5
C_LR = 1e-4
DISCOUNT = 0.9
LAMBDA = 0.5
BETA_H = 1e-3
BETA_R = 1e-3
BREAK_CORR = False

BUFFER_ROOM = 1000
BS = 200
TRAIN_STEP = BUFFER_ROOM // BS

MAX_EP = 10000000
MAX_TIME = 5
UPDATE_STEP = 100

QP_MATCH_STEP = 200

CKPT_STEP = 5000
SUM_SEC = 600

LOG_STEP = 1000

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



def train(ckpt_dir, restore_ckpt, sum_dir):

    env = Kitchen('human', 'robot', STEP_REWARD, -REWARD, REWARD, INGREDIENTS, NUM_DISHES, MAX_DISH_INGREDIENTS)

    agents = get_trainers()

    writer = tf.summary.FileWriter(sum_dir, flush_secs=SUM_SEC)

    saver = tf.train.Saver()
    best_saver = tf.train.Saver()
    config = tf.ConfigProto(device_count = {'GPU':DEVICE_COUNT})
    config.gpu_options.allow_growth = True
    with tf.Session( config=config) as sess:
        if restore_ckpt != '':
            saver.restore(sess, tf.train.latest_checkpoint(restore_ckpt))
        else:
            sess.run(tf.initializers.global_variables())

        argmax_q = [None, None]
        argmax_policy = [[], []]
        mistake = []
        test = []
        step = 0
        max_acc = 0
        for ep in range(MAX_EP):
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

                    if ep % LOG_STEP == 0:

                        top_pi = np.argsort(logit[0,0])[-3:][::-1]
                        argmax_policy[i].append(np.sort(top_pi))

                        logging.info(
                            '   Name: {} Action: {}\n   Gumbel Output: {}\n   '
                            'Menu:{} Goal: {}\n   Workplace:{} Reward: {}\n'
                            '     Policy logits: {}\n  Entropy: {} Temp: {}\n '
                            '    Top 3 Policy: {}\n'.format(
                                name, a, a_cont, menu, goal, wpeb, next_obs[name].reward, logit, ent, temp,
                                top_pi))




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
            ## print q
            # if ep % LOG_STEP == 0:
            #     for i in range(2):
            #         menu = np.stack(menu_traj[i], axis=1)
            #         wpeb = np.stack(wpeb_traj[i], axis=1)
            #         q = agents[i].get_all_action_q(sess, menu, wpeb, goal, True)
            #         top_q = np.argsort(q[0])[:,-3:][:,::-1]
            #         argmax_q[i] = np.sort(top_q, axis=-1)
            #
            #         # logging.info(
            #         #     '   Name: {} Action: {}\n   Menu:{} Goal: {}\n   Workplace:{} Reward: {}\n'
            #         #     '     Q: {} \n    Policy logits: {}\n  Entropy: {} \n'.format(
            #         #         name, a, menu, goal, wpeb, next_obs[name].reward, q, logit, ent))
            #         targets = menu[0,:, env.state_.order, :]  # [1:]
            #         correct_ingredient_avg_q_per_ra = (q[0] * (targets != 0.)).mean(axis=-1)
            #         wrong_ingredient_avg_q_per_ra = (q[0] * (targets == 0)).mean(axis=-1)
            #
            #         debug_msg = {
            #             'goal': targets,
            #             'correct_ingredient_avg_q_per_ra': correct_ingredient_avg_q_per_ra,
            #             'wrong_ingredient_avg_q_per_ra': wrong_ingredient_avg_q_per_ra,
            #             'average_q_diff': correct_ingredient_avg_q_per_ra - wrong_ingredient_avg_q_per_ra,
            #             'first_second_q_diff': np.sort(q[0])[:,-1] - np.sort(q[0])[:,-2]
            #         }
            #
            #         logging.info(
            #             '   Q_stats: Name: {} \n'
            #             '     Q: {} \n '
            #             '   Average_q_diff: {}, First_second_q_diff: {}\n'
            #             '   Top 3 Q: {}\n'.format(
            #                 agents[i].name, q,
            #                 debug_msg['average_q_diff'], debug_msg['first_second_q_diff'],
            #                 top_q))


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
            if argmax_policy[1]:
                qp_match_prob = [np.mean(np.any(argmax_q[k] == np.stack(argmax_policy[k], axis=0), axis=-1)) for k in range(2)]
            else:
                qp_match_prob = None
            # for k in range(2):
            #     if len(argmax_q[k]) >= 500:
            #         argmax_q[k] = argmax_q[k][1:]
            #     if len(argmax_policy[k]) >= 500:
            #         argmax_policy[k] = argmax_policy[k][1:]

            acc = sum(test) / len(test)
            if not mistake:
                herr = rerr = 0
            else:
                herr = (np.array(mistake) == 0).mean()
                rerr = (np.array(mistake) == 1).mean()

            if ep % LOG_STEP == 0:
                logging.info('Episode: {} Overall Accuracy: {} Max Accuracy: {} Human Error Percentage: {} Robot Error Percentage: {} '
                             'Q_Logits_match_prob: {}\n'.format(
                    ep, acc, max_acc, herr, rerr, qp_match_prob))

            if len(test) >= 500:
                test = test[1:]
            if len(mistake) >= 500:
                mistake = mistake[1:]

            if acc > max_acc:
                max_acc = acc
                if not os.path.exists(os.path.join(ckpt_dir,'best')):
                    os.makedirs(os.path.join(ckpt_dir,'best'))

                best_saver.save(sess, os.path.join(ckpt_dir,'best', 'model'), ep)

            ep_r_mean = ep_r / t
            ep_q_loss_mean = [ql / t for ql in ep_q_loss]
            ep_p_loss_mean = [pl / t for pl in ep_p_loss]

            if ep % CKPT_STEP == 0:
                saver.save(sess, os.path.join(ckpt_dir, 'model'), ep)

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
    output_dir = os.path.join(os.path.dirname(__file__),'output', exp_id, t)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--screen', default='0')
    parser.add_argument('--ckpt_dir', default='')
    parser.add_argument('--restore_ckpt', default='')
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
    copyfile(os.path.join(os.path.dirname(__file__), 'policy.py'),
             os.path.join(ckpt_dir, os.path.basename('policy.py')))

    train(ckpt_dir, opt.restore_ckpt, sum_dir)

if __name__ == '__main__':
    main()

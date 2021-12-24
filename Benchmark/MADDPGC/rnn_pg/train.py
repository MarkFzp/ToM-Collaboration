from Game.calendar import Calendar
from Benchmark.MADDPGC.rnn_pg.config import *
from Benchmark.MADDPGC.rnn_pg.agents import *
from Benchmark.MADDPGC.rnn_pg.test import *

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
DENSE = 100
STEP_REWARD = -0.1
REWARD = 1
FAIL_REWARD = -2
DECAY_STEP = 1e6
DECAY_RATE = 0.1



TEMP_MIN = 0.1
TEMP_START = 100
TEMP_DECAY = 5e4


A_LR = 1e-4
C_LR = 5e-4
DISCOUNT = 0.6
LAMBDA = 0.5
BETA_H = 5e-1
BETA_R = 5e-1
BREAK_CORR = True

BUFFER_ROOM = 1000
BS = 200
TRAIN_STEP = BUFFER_ROOM // BS

MAX_EP = 5000000
MAX_TIME = (NUM_SLOTS + 1) // 2
UPDATE_STEP = 100

QP_MATCH_STEP = 200

CKPT_STEP = 50000
SUM_SEC = 600

LOG_STEP = 100

DEVICE_COUNT=1


def setup_logger(log_dir):
    logger = logging.getLogger()
    logger.handlers = []
    file_handler = logging.FileHandler(os.path.join(log_dir, 'log.log'))
    console_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)

def get_trainers(num_actions, init_global_step, calendar_tensor, action_tensor):
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
                        EMBED_DIM, A_LR, C_LR, init_global_step, DECAY_STEP, DECAY_RATE,
                        DISCOUNT, LAMBDA, beta, BS, BUFFER_ROOM, MAX_TIME * 2, UPDATE_STEP,
                        BREAK_CORR, use_gpu)
        agent = MADDPGAgentTrainer(name, 2, i, config, p_func, q_func, calendar_tensor, action_tensor)
        trainer.append(agent)

    return trainer



def train(ckpt_dir, restore_ckpt, sum_dir):
    if restore_ckpt == '':
        init_ep = 0
    else:
        ckpt = tf.train.latest_checkpoint(restore_ckpt)
        init_ep = int(ckpt.split('-')[-1])

    env = Calendar('human', 'robot', STEP_REWARD, FAIL_REWARD, REWARD, NUM_SLOTS,
                   dataset_path_prefix='utils/calendar_8_1') # /home/Datasets/Calendar # utils

    agents = get_trainers(env.num_actions_, init_ep, env.tensor_, env.action_tensor_)

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

        argmax_q = [[], []]
        argmax_policy = [[], []]
        mistake = []
        test = []
        step = 0
        max_acc = 0
        test_acc = 0
        for ep in range(init_ep, MAX_EP):
            ep_r = 0
            ep_q_loss = [0,0]
            ep_p_loss = [0,0]

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
            calendar_id = [int(''.join(str(p[0])[1: -1].split()), 2) for p in private]
            valid_msgs = [env.get_msg(cid)[0] for cid in calendar_id]
            valid_msg_idx = [[env.all_msgs_.index(vm) for vm in valid_msgs_per_cid] for valid_msgs_per_cid in valid_msgs]
            valid_mask = [np.zeros(env.num_msgs_) for _ in valid_msg_idx]
            slot_mask = [np.squeeze(1 - p) for p in private]
            [np.put(vm, valid_msg_idx_per_cid, 1) for vm, valid_msg_idx_per_cid in zip(valid_mask, valid_msg_idx)]





            for t in range(MAX_TIME):

                for i, name in enumerate(['human', 'robot']):
                    a, in_state[i], ent, logit = agents[i].action(sess, private[i][:,None,:],
                                                                          [lact[:,None] for lact in last_action_n],
                                                        in_state=in_state[i], train=True)
                    _, in_state[(i+1)%2], ent2, logit2 = agents[(i+1)%2].action(sess, private[(i+1)%2][:, None, :],
                                                                           [lact[:,None] for lact in last_action_n],
                                                                     in_state=in_state[(i+1)%2], train=True)
                    a = np.squeeze(a, axis=1)
                    next_obs = deepcopy(env.proceed({name: a[0]}))

                    for j in range(2):
                        private_traj[j].append(private[j])
                        last_act_traj[j].append(last_action_n[j])
                        act_traj[j].append(a if j == i else np.array([-1]))
                        reward_traj[j].append(np.array([next_obs[agents[j].name].reward]))
                        terminal_traj[j].append(np.array([obs[agents[j].name].terminate]))

                    if ep % LOG_STEP == 0:
                        q = agents[i].get_all_action_q(sess, [p[:, None, :] for p in private],
                                                       [act[:, None] for act in last_action_n], True)
                        top_q = np.argsort(q[0, 0])[-3:][::-1]
                        top_pi = np.argsort(logit[0, 0])[-3:][::-1]
                        argmax_q[i].append(np.sort(top_q))
                        argmax_policy[i].append(np.sort(top_pi))

                        valid_msg_avg_q = (q[0, 0, 0:env.num_msgs_] * valid_mask[i]) / np.sum(valid_mask[i])
                        invalid_msg_avg_q = (q[0, 0, 0:env.num_msgs_] * (1 - valid_mask[i])) / np.sum(1 - valid_mask[i])
                        valid_propose_avg_q = (q[0, 0, env.num_msgs_:-1] * slot_mask[i]) / np.sum(slot_mask[i])
                        invalid_propose_avg_q = (q[0, 0, env.num_msgs_:-1] * (1 - slot_mask[i])) / np.sum(
                            (1 - slot_mask[i]))

                        debug_msg = {
                            'average_msg_q_diff': valid_msg_avg_q - invalid_msg_avg_q,
                            'aveage_propose_q_diff': valid_propose_avg_q - invalid_propose_avg_q
                        }

                        logging.info(
                            '   Name: {} Action: {}, {}\n   '
                            '   Private:{} \n '
                            '   Reward: {}\n'
                            '     Q: {} \n    Policy logits: {}\n  Entropy: {}\n '
                            '   Average_msg_q_diff: {}\n   Average_propose_q_diff: {}\n'
                            '   Top 3 Q: {}, {} \n   Top 3 Policy: {}, {}\n'.format(
                                name, env.action_tensor_[a], a, private, next_obs[name].reward, q,
                                logit, ent,
                                debug_msg['average_msg_q_diff'], debug_msg['aveage_propose_q_diff'],
                                env.action_tensor_[top_q], top_q, env.action_tensor_[top_pi], top_pi))

                    ep_r += next_obs[name].reward
                    step += 1


                    if next_obs[name].terminate:
                        step += 1
                        t += 1
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

            ## store
            for i, agent in enumerate(agents):
                agent.experience(private_traj, last_act_traj, act_traj, reward_traj, terminal_traj)


            ## train

            for agent in agents:
                agent.preupdate()
            for l, agent in enumerate(agents):
                result = agent.update(sess, agents, ep)

                if result is not None and ep % LOG_STEP == 0:
                    q_loss, p_loss, target_q_mean, reward_mean, target_q_std_mean, mean_traj_len = result

                    logging.info(
                        '       Name: {}\n      Q_Loss: {} P_Loss: {}\n'
                        '       Mean Target Q: {} Mean Target Q std: {}\n'
                        '       Mean Reward: {} Mean Trajectory Len: {}\n'.format(
                            agent.name, q_loss, p_loss, target_q_mean, target_q_std_mean,
                            reward_mean, mean_traj_len))

                    ep_q_loss[l] = q_loss
                    ep_p_loss[l] = p_loss
            if argmax_q[1]:
                qp_match_prob = [np.mean(np.any(np.stack(argmax_q[k], axis=0) == np.stack(argmax_policy[k], axis=0), axis=-1)) for k in range(2)]
            else:
                qp_match_prob = None
            for k in range(2):
                if len(argmax_q[k]) >= 500:
                    argmax_q[k] = argmax_q[k][1:]
                if len(argmax_policy[k]) >= 500:
                    argmax_policy[k] = argmax_policy[k][1:]

            acc = sum(test) / len(test)
            if not mistake:
                herr = rerr = 0
            else:
                herr = (np.array(mistake) == 0).mean()
                rerr = (np.array(mistake) == 1).mean()

            if len(test) > 500:
                test = test[1:]
            if len(mistake) > 500:
                mistake = mistake[1:]

            if len(test) == 500 and acc > max_acc:
                max_acc = acc
                if not os.path.exists(os.path.join(ckpt_dir,'best')):
                    os.makedirs(os.path.join(ckpt_dir,'best'))

                best_saver.save(sess, os.path.join(ckpt_dir,'best', 'model'), ep)


            if ep % CKPT_STEP == 0:
                saver.save(sess, os.path.join(ckpt_dir,'model'), ep)
                test_acc = test_model(sess, env, agents)


            if ep % LOG_STEP == 0:
                logging.info('Episode: {} Overall Accuracy: {} Max Accuracy: {} Human Error Percentage: {} Robot Error Percentage: {} '
                             'Q_Logits_match_prob: {}\n'
                             'Test Acc: {}\n'.format(
                    ep, acc, max_acc, herr, rerr, qp_match_prob, test_acc))


            ep_r_mean = ep_r / t
            ep_q_loss_mean = [ql / t for ql in ep_q_loss]
            ep_p_loss_mean = [pl / t for pl in ep_p_loss]

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

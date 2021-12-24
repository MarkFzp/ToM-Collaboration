import sys
sys.path.append("../..")
from Game.kitchen import Kitchen
from Benchmark.ValueAlignment.config import *
from Benchmark.ValueAlignment.agents import *
from Benchmark.ValueAlignment.buffer import *
from Benchmark.ValueAlignment.qnetwork import *
from Benchmark.ValueAlignment.test import test_model

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
DISCOUNT = 0.8

BUFFER_ROOM = 2000
BS = 200
TRAIN_STEP = BUFFER_ROOM // 20

MAX_EP = 2000000
MAX_TIME = 10
UPDATE_STEP = 100

NUM_CKPT = 20
CKPT_STEP = MAX_EP // NUM_CKPT
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

def log_and_print(human:bool, menu, wpeb, next_wpeb, ha, ra, r, target_goal, q, hp, rbelief, t, temp):
    ## debug
    if human:
        targets = menu[0, target_goal, :]
        q_at_goal = q[0, :, target_goal]#[1:]
        correct_ingredient_avg_q_per_ra = (q_at_goal * (targets != 0.)).mean()
        wrong_ingredient_avg_q_per_ra = (q_at_goal * (targets == 0)).mean()

        pi_at_goal = hp[0, :, target_goal]
        top_q = np.argsort(q_at_goal)[-3:][::-1]
        entropy = -(pi_at_goal * np.log(pi_at_goal+1e-8)).sum()

        debug_msg = {
            'goal': targets,
            'correct_ingredient_avg_q_per_ra': correct_ingredient_avg_q_per_ra,
            'wrong_ingredient_avg_q_per_ra': wrong_ingredient_avg_q_per_ra,
            'average_q_diff': correct_ingredient_avg_q_per_ra - wrong_ingredient_avg_q_per_ra
        }

        logging.info('    T: {} Human:\n    Menu: {}\n    Goal: {}\n    Workplace: {} Next Workplace {}\n'
                     '    Q: {}\n    PI: {} \n    Robot Action: {}, Human Action: {}\n    Reward: {}\n'
                     '    Average Q Diff: {}\n'
                     '    Top 3 Q: {} Entropy: {} Temp: {}\n'.format(
            t, menu, target_goal, wpeb, next_wpeb,
            q_at_goal,
            hp[:, :, target_goal],
            ra, ha, r, debug_msg['average_q_diff'], top_q, entropy, temp))

    else:
        targets = menu[0, target_goal, :]
        exp_q = (q * hp * rbelief[:, None, None, :]).sum(axis=(0, 1, -1))#[1:]
        correct_ingredient_exp_q = (exp_q * (targets != 0.)).mean()
        wrong_ingredient_exp_q = (exp_q * (targets == 0)).mean()

        top_q = np.argsort(exp_q)[-3:][::-1]

        debug_msg = {
            'goal': targets,
            'correct_ingredient_exp_q': correct_ingredient_exp_q,
            'wrong_ingredient_exp_q': wrong_ingredient_exp_q,
            'average_q_diff': correct_ingredient_exp_q - wrong_ingredient_exp_q
        }

        logging.info(
            '    T: {} Robot:\n    Menu: {}\n    Goal: {}\n    Belief: {}\n    Workplace: {} Next Workplace {}\n'
            '    E(Q): {}\n  Human Action: {}, Robot Action: {}\n    Reward: {}\n'
            '    Average Q Diff: {}\n'
            '    Top 3 Q: {}\n'.format(
                t, menu, target_goal, rbelief, wpeb, next_wpeb,
                exp_q,
                ha, ra, r, debug_msg['average_q_diff'], top_q))

def train(ckpt_dir, restore_ckpt, sum_dir, dataset_order):

    env = Kitchen('human', 'robot', STEP_REWARD, -REWARD, REWARD, INGREDIENTS, NUM_DISHES, MAX_DISH_INGREDIENTS,
                  dataset_path='/home/luyao/Datasets/Kitchen/dataset%d_with_%d_ingred_%d_dish_5_maxingred_1000000_size_0.7_ratio.npz'\
                               % (int(dataset_order), NUM_RACTIONS, NUM_DISHES))

    rconfig = RobotConfig(NUM_HACTIONS, NUM_RACTIONS, NUM_GOALS)
    hconfig = HumanConfig(NUM_HACTIONS, NUM_RACTIONS, NUM_GOALS)
    qconfig = QNetConfig(NUM_DISHES, len(INGREDIENTS), EMBED_DIM, NUM_HACTIONS, NUM_RACTIONS, NUM_GOALS, DENSE,LR, LR_BETA1, LR_EPSILON, DISCOUNT)

    human = Human(hconfig)
    robot = Robot(rconfig)
    qnet = QNetwork(qconfig)

    # buffer = Buffer(NUM_DISHES, len(INGREDIENTS), NUM_HACTIONS, NUM_RACTIONS, BUFFER_ROOM, BS)
    buffer = Bufferv2(BUFFER_ROOM, BS)

    writer = tf.summary.FileWriter(sum_dir, flush_secs=SUM_SEC)

    saver = tf.train.Saver(max_to_keep=NUM_CKPT)
    best_saver = tf.train.Saver()
    config = tf.ConfigProto(device_count={'GPU': DEVICE_COUNT})
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        if restore_ckpt != '':
            saver.restore(sess, tf.train.latest_checkpoint(restore_ckpt))
        else:
            sess.run(tf.initializers.global_variables())

        mistake = []
        test = []
        max_acc = 0
        test_acc = 0
        for ep in range(MAX_EP):
            if ep % UPDATE_STEP == 0:
                qnet.update_target_qnet(sess)

            ep_r = 0
            ep_loss = 0
            epsilon = EPSILON_MIN + (EPSILON_START - EPSILON_MIN) * np.exp(-1 * SWITCH_ITER / EPSILON_DECAY)
            temp = 0.1 #TEMP_MIN + (TEMP_START - TEMP_MIN) * np.exp(-1 * ep / TEMP_DECAY)
            rbelief = np.zeros((1, NUM_DISHES))
            rbelief.fill(0.25)

            env.reset(MENU)
            target_goal = env.state_.order
            obs = env.start()
            menu = deepcopy(obs['robot'].observation.menu_embed[np.newaxis, :])
            wpeb = deepcopy(obs['robot'].observation.workplace_embed[np.newaxis, :])

            ha = 'null'

            if ep % LOG_STEP == 0:
                logging.info('Episode {}'.format(ep))

            for t in range(MAX_TIME):

                # robot's turn
                if t == 0:
                    raction = np.zeros((1, NUM_RACTIONS))
                    ra = 'null'
                    r = 0

                else:

                    qvalues_all, hprob_all = qnet.get_qvalues_and_hprob_matrix(sess, menu, wpeb, rbelief, temp)

                    raction = robot.act(sess, qvalues_all, hprob_all, rbelief, epsilon=epsilon)
                    ra = np.argmax(raction, axis=-1)[0]

                    next_obs = deepcopy(env.proceed({'robot': ra}))
                    next_wpeb = next_obs['robot'].observation.workplace_embed[np.newaxis, :]

                    r = next_obs['robot'].reward

                    ep_r += next_obs['robot'].reward
                    #####

                    if ep % LOG_STEP == 0:
                        pass
                        # log_and_print(False, menu, wpeb, next_wpeb, ha, ra, r, target_goal, qvalues_all,
                        #               hprob_all, rbelief, t, temp)

                    if next_obs['robot'].terminate:
                        if next_obs['robot'].success:
                            test.append(1)
                            # j = target_goal
                        else:
                            mistake.append(0)
                            test.append(0)
                            # j = np.argmax(rbelief)
                        t += 1

                        ## store stuff

                        # for i in range(NUM_HACTIONS):
                        i = np.random.choice(NUM_HACTIONS)
                        haction = np.eye(NUM_HACTIONS)[i:i + 1]
                        buffer.store(menu, wpeb, rbelief, haction, raction, r,
                                     menu, next_wpeb,
                                     target_goal, next_obs['robot'].terminate)

                        ## train
                        if buffer.is_available():
                            # for data in buffer.get_data():
                            loss, mean_r = qnet.train(sess, robot, human, temp, *buffer.get_data())
                            if ep % LOG_STEP == 0:
                                logging.info('      Episode: {}  Loss: {}  Average Reward: {}\n'.format(ep, loss, mean_r))
                            ep_loss += loss

                        break

                # obs = next_obs
                # wpeb = next_wpeb

                # human's turn
                qvalues_per_ra, hprob_per_ra = qnet.get_q_all_hactions(sess, menu, wpeb, rbelief, raction,  temp)

                haction = human.act(sess, qvalues_per_ra, np.array([target_goal]), temp)
                ha = np.argmax(haction, axis=-1)[0]

                ## step
                next_obs = deepcopy(env.proceed({'human': ha}))
                next_wpeb = next_obs['human'].observation.workplace_embed[np.newaxis, :]

                if next_obs['human'].get('success') is not None and next_obs['human'].success:
                    human_step_reward = next_obs['human'].reward
                else:
                    human_step_reward = DISCOUNT * next_obs['human'].reward
                r += human_step_reward

                ep_r += human_step_reward
                #########

                buffer.store(menu, wpeb, rbelief, haction, raction, r,
                             menu, next_wpeb, target_goal, next_obs['human'].terminate)

                if ep % LOG_STEP == 0:
                    pass
                    # log_and_print(True, menu, wpeb, next_wpeb, ha, ra, r, target_goal, qvalues_per_ra, hprob_per_ra,
                    #               rbelief, t, temp)

                ## train
                if buffer.is_available():
                    # for data in buffer.get_data():
                    loss, mean_r = qnet.train(sess, robot, human, temp, *buffer.get_data())
                    if ep % LOG_STEP == 0:
                        logging.info('      Episode: {}  Loss: {}  Average Reward: {}\n'.format(ep, loss, mean_r))
                    ep_loss += loss


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

                    break




            if not next_obs['robot'].terminate:
                mistake.append(2)
                test.append(0)
            ## eval

            if ep % UPDATE_STEP == 0:
                qnet.update_target_qnet(sess)

            acc = sum(test) / len(test)
            if acc > max_acc:
                max_acc = acc
                if not os.path.exists(os.path.join(ckpt_dir,'best')):
                    os.makedirs(os.path.join(ckpt_dir,'best'))

                best_saver.save(sess, os.path.join(ckpt_dir,'best', 'model'), ep)
                # test_acc = test_model(sess, env, human, robot, qnet)


            if ep % CKPT_STEP == 0 or ep == MAX_EP-1:
                saver.save(sess, os.path.join(ckpt_dir, 'model'), ep)
                test_acc = test_model(sess, env, human, robot, qnet)


            if not mistake:
                herr = rerr = 0
            else:
                herr = (np.array(mistake) == 1).mean()
                rerr = (np.array(mistake) == 0).mean()

            if ep % LOG_STEP == 0:
                logging.info('Episode: {} Overall Accuracy: {} Max Accuracy: {} Human Error Percentage: {} Robot Error Percentage: {}\n'
                             'Test Acc: {}\n'.format(
                    ep, acc, max_acc, herr, rerr, test_acc))

            if len(test) >= 500:
                test = test[1:]
            if len(mistake) >= 500:
                mistake = mistake[1:]


            ep_r_mean = ep_r / t
            ep_loss_mean = ep_loss / t

            summary = tf.Summary(value=[
                tf.Summary.Value(tag="episode_loss", simple_value=ep_loss_mean),
                tf.Summary.Value(tag="episode_reward", simple_value=ep_r_mean),
                tf.Summary.Value(tag="overall_accuracy", simple_value=acc),
            ])


            writer.add_summary(summary, ep)
            writer.flush()

        test_acc = test_model(sess, env, human, robot, qnet)
        print('final test acc:', test_acc)

def get_output_dir(exp_id):
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_dir = os.path.join(os.getcwd(),'Benchmark/ValueAlignment/output/' + exp_id, t)
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
    parser.add_argument('--dataset_order', default=1)
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
    copyfile(os.path.join(os.getcwd(), 'Benchmark/ValueAlignment/qnetwork.py'), os.path.join(ckpt_dir, os.path.basename('qnet.py')))
    copyfile(os.path.join(os.getcwd(), 'Benchmark/ValueAlignment/agents.py'),
             os.path.join(ckpt_dir, os.path.basename('agents.py')))

    train(ckpt_dir, opt.restore_ckpt, sum_dir, opt.dataset_order)

if __name__ == '__main__':
    main()

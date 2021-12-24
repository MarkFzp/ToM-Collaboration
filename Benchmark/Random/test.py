
from Game.calendar import Calendar
from copy import deepcopy
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds', default=3)
    opt = parser.parse_args()

    env = Calendar('human', 'robot', -0.1, -2, 1, 8,
                   dataset_path_prefix='/home/Datasets/Calendar/calendar_8_{}'.format(
                       opt.ds))  # /home/Datasets/Calendar # utils

    mistake = []
    test = []


    for ep in range(10000):
        env.reset(train=False)

        for t in range(8):

            for i, name in enumerate(['human', 'robot']):
                a = np.random.randint(env.num_actions_)
                next_obs = deepcopy(env.proceed({name: a}))

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

if __name__ == '__main__':
    main()
def main():
    f = open('CG_results.csv', 'w+')
    #methods = ['ToM', 'DIAL no msg', 'DIAL msg', 'BAD', 'MADDPG', 'DDRQN', 'Counterfactual', 'Value Alignment']
    methods = ['ToM', 'BAD', 'MADDPG', 'DDRQN', 'Counterfactual']
    num_points = 20
    training_progress = [i / num_points for i in range(1, 21)]
    num_splits = 3
    num_exp_split = 2
    experiment_name = 'Calendar8'
    f.write('Method,Dataset,Training_Progress,Accuracy\n')
    for m in methods:
        for ns in range(1, num_splits + 1):
            for nes in range(1, num_exp_split + 1):
                for tp in training_progress:
                    f.write('%s,%s,%.2f,\n' % (m, experiment_name + '-' + str(ns) + '-' + str(nes), tp))

if __name__ == '__main__':
    main()

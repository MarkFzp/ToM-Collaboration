import numpy as np 
import os


def main(num_slot, train_ratio = 0.7, num_split = 3):
    all_cid = set(range(2 ** num_slot))
    all_cid_count = len(all_cid)

    train_file_base = 'calendar_{}_ID_train.txt'.format(num_slot)
    test_file_base = 'calendar_{}_ID_test.txt'.format(num_slot)
    
    for i in range(1, num_split + 1):
        train_cid = list(np.random.choice(all_cid_count, size = round(all_cid_count * train_ratio), replace = False))
        test_cid = list(all_cid - set(train_cid))

        train_file = train_file_base.replace('ID', str(i))
        if not os.path.exists(train_file):
            with open(train_file, 'w+') as train_f:
                for train_cidd in train_cid:
                    train_f.write('{}\n'.format(train_cidd))
        else:
            print('train file {} exists'.format(train_file))
        
        test_file = test_file_base.replace('ID', str(i))
        if not os.path.exists(test_file):
            with open(test_file, 'w+') as test_f:
                for test_cidd in test_cid:
                    test_f.write('{}\n'.format(test_cidd))
        else:
            print('test file {} exists'.format(test_file))



if __name__ == '__main__':
    main(8)
    main(10)

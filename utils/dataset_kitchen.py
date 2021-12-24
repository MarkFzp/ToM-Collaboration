import numpy as np
from itertools import permutations, product
import argparse
import os

def generate(num_ingred=10, num_dishes=4, max_ingred_num=5, train_ratio=0.7, dataset_size=10000):

    '''generate all combinations'''
    assert max_ingred_num <= num_ingred
    ingred = list(range(num_ingred))

    dish_embed_partition = {'train':[], 'test':[]}
    comb_list = []


    for n in range(1,max_ingred_num+1):
        dishes_set = set([tuple(prod) for prod in product(*(range(i) for i in range(1,n+1)))])
        dish_set_embed = np.zeros((len(dishes_set), num_ingred))
        for idx, dish in enumerate(dishes_set):
            for ing in dish:
                dish_set_embed[idx, ing] += 1
            dish_set_embed[idx] = np.sort(dish_set_embed[idx])

        dish_set_embed = np.unique(dish_set_embed, axis=0)
        dishes_embed = []
        for unique_dish in dish_set_embed:
            perm = np.array(list(set(permutations(unique_dish!=0))), dtype=np.float32, copy=False) ## positional permutation
            perm[perm!=0] = np.tile(unique_dish[unique_dish!=0][None,:], (perm.shape[0],1)).reshape(-1) ## fill in values
            all_perm_per_unique_dish = []
            for p in perm:
                non_zero_perm = np.array(list(permutations(p[p!=0])), copy=False)
                all_perm_per_p = np.zeros((non_zero_perm.shape[0], p.shape[0]))
                all_perm_per_p[:,p!=0] = non_zero_perm
                all_perm_per_unique_dish.append(all_perm_per_p)
            dishes_embed.append(np.concatenate(all_perm_per_unique_dish, axis=0))

        dishes_embed = np.concatenate(dishes_embed, axis=0)
        embed_size = dishes_embed.shape[0]
        perm_dishes_embed = np.random.permutation(dishes_embed)
        train_embed = perm_dishes_embed[:int(embed_size*train_ratio)]
        test_embed = perm_dishes_embed[int(embed_size * train_ratio):]

        dish_embed_partition['train'].append(train_embed)
        dish_embed_partition['test'].append(test_embed)

        ## calculate prob
        comb_list.append(np.unique(dishes_embed,axis=0).shape[0])

    max_comb = sum(comb_list)
    prob_list = [c/max_comb for c in comb_list]

    train_size = int(train_ratio * dataset_size)
    test_size = dataset_size - train_size
    dsize = {'train':train_size, 'test':test_size}
    dataset = {'train':{'menu':[], 'order':[]}, 'test':{'menu':[], 'order':[]}}
    for key in dataset:
        for idx in range(dsize[key]):
            partition_ind = np.random.choice(max_ingred_num, num_dishes, p=prob_list)
            order = np.random.choice(num_dishes)
            menu = np.zeros((num_dishes, num_ingred))
            for i, ind in enumerate(partition_ind):
                rand_row = np.random.choice(dish_embed_partition[key][ind].shape[0])
                menu[i] = dish_embed_partition[key][ind][rand_row]

            dataset[key]['menu'].append(menu)
            dataset[key]['order'].append(order)


        dataset[key]['menu'] = np.stack(dataset[key]['menu'], axis=0)
        dataset[key]['order'] = np.array(dataset[key]['order'], copy=False)



    return dataset


def generate_v2(num_ingred=10, num_dishes=4, max_ingred_num=5, train_ratio=0.7, dataset_size=10000):

    '''generate all combinations'''

    ingred = list(range(num_ingred))
    aug_ingred = list(range(num_ingred+1))

    partition = [set() for _ in range(max_ingred_num)]

    all_dishes = list(product(*[aug_ingred for i in range(max_ingred_num)])) ## avoids 0 len
    ## there are duplicates in the above set e.g. (0,10,0,0,0) and (0,0,10,0,0)

    for i, dish in enumerate(all_dishes):
        actual_dish = tuple([ingred[ci] for ci in dish if ci < num_ingred])
        if i % 1000 == 0:
            print('Processing dish [%7d/%7d]: '%(i, len(all_dishes)), actual_dish)
        if len(actual_dish) == 0:
            continue
        partition[len(actual_dish)-1].add(actual_dish)

    dish_embed_partition = {'train':[], 'test':[]}
    for p in partition:
        embed = np.zeros((len(p), num_ingred))
        for idx, dish in enumerate(p):
            for ing in dish:
                embed[idx, ing] += 1

        embed_size = embed.shape[0]
        perm_dishes_embed = np.random.permutation(embed)
        train_embed = perm_dishes_embed[:int(embed_size * train_ratio)]
        test_embed = perm_dishes_embed[int(embed_size * train_ratio):]

        dish_embed_partition['train'].append(train_embed)
        dish_embed_partition['test'].append(test_embed)

    all_dishes = dict()
    for key in dish_embed_partition:
        all_dishes[key] = np.concatenate(dish_embed_partition[key], axis=0)

    train_size = int(train_ratio * dataset_size)
    test_size = dataset_size - train_size
    dsize = {'train': train_size, 'test': test_size}
    dataset = {'train': {'menu': [], 'order': []}, 'test': {'menu': [], 'order': []}}
    for key in dataset:
        for idx in range(dsize[key]):
            menu_ind = np.random.choice(all_dishes[key].shape[0], (num_dishes,), replace=False)
            order = np.random.choice(num_dishes)
            menu = all_dishes[key][menu_ind]

            dataset[key]['menu'].append(menu)
            dataset[key]['order'].append(order)
            if idx % 1000 == 0:
                print('{} [%7d/%7d]:\n  Menu:{} Order:{}'.format(key, menu, order)%(idx, dsize[key]))

        dataset[key]['menu'] = np.stack(dataset[key]['menu'], axis=0)
        dataset[key]['order'] = np.array(dataset[key]['order'], copy=False)

    print('Unique dish - train:', np.unique(all_dishes['train'], axis=0).shape[0])
    print('Unique dish - test:', np.unique(all_dishes['test'], axis=0).shape[0])
    print('Unique menu - train', count_unique_menu(dataset['train']['menu']))
    print('Unique menu - test', count_unique_menu(dataset['test']['menu']))

    return dataset

def load_data(path):
    ''' return nested dictionary

    return: {'train':{'menu':data, 'order':data}, 'test':{'menu':data, 'order':data}}
    '''
    data = np.load(path, allow_pickle=True)
    res = dict()
    for key in data:
        res[key] = data[key].item()
    return res

def count_unique_menu(all_menu):
    from itertools import permutations
    ind = np.array(list(set(permutations(range(all_menu.shape[1])))))

    s = [all_menu[0]]
    for i, menu in enumerate(all_menu[1:]):
        k = 0
        is_perm = False
        while k < len(s):
            is_perm = np.all(s[k][ind] == menu, axis=(-2, -1)).any()
            if is_perm:
                break
            k += 1
        if not is_perm:
            s.append(menu)

        if i % 1000 == 0:
            print('Current num of unique menu:', len(s))
    return len(s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-dir', default='/home/Datasets/Kitchen/', help='relative to your working dir', type=str)
    parser.add_argument('--num-ingredients', default=10, type=int)
    parser.add_argument('--dataset-order', default=1, type=int)
    parser.add_argument('--num-dishes', default=4, type=int)
    parser.add_argument('--max-ingredient-num', default=5, type=int)
    parser.add_argument('--dataset-size', default=1000000, type=int)
    parser.add_argument('--train-ratio', default=0.7, type=float)
    opt = parser.parse_args()

    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)

    # dataset = generate_v2(opt.num_ingredients, opt.num_dishes, opt.max_ingredient_num, opt.train_ratio, opt.dataset_size)
    #
    # np.savez(os.path.join(opt.save_dir,
    #                       'dataset{}_with_{}_ingred_{}_dish_{}_maxingred_{}_size_{}_ratio.npz'.format(
    #                           opt.dataset_order, opt.num_ingredients, opt.num_dishes, opt.max_ingredient_num, opt.dataset_size,
    #                           opt.train_ratio,
    #
    #                       )), **dataset)
    for i in range(3):
        data = load_data(os.path.join(opt.save_dir, 'dataset{}_with_{}_ingred_{}_dish_{}_maxingred_{}_size_{}_ratio.npz'.format(
                              opt.dataset_order, opt.num_ingredients, opt.num_dishes, opt.max_ingredient_num, opt.dataset_size,
                              opt.train_ratio,

                          )))

        # print('Unique dish - train:', np.unique(data['train']['menu'].reshape(data['train']['menu'].shape[0] *  data['train']['menu'].shape[1],data['train']['menu'].shape[2]), axis=0).shape[0])
        # print('Unique dish - test:', np.unique(data['test']['menu'].reshape(data['test']['menu'].shape[0] * data['test']['menu'].shape[1],  data['test']['menu'].shape[2]), axis=0).shape[0])
        print('Unique menu - train', count_unique_menu(data['train']['menu']))
        print('Unique menu - test', count_unique_menu(data['test']['menu']))

    print('\ndone.')

if __name__ == '__main__':

    main()
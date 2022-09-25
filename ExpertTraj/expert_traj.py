import numpy as np
from env_32_none import abs_dis, ele_list_gen, cor_func_all, ideal_cor_func, swap_step, ideal_cor_func
import random
import math
import multiprocessing as mp
import time
from itertools import combinations
path = '/media/wz/a7ee6d50-691d-431a-8efb-b93adc04896d/Github/SQSdrl/'
ind_1nn = np.load(path+'struc_info/ind_1nn.npy')
action_list = np.array([[i,j] for i, j in combinations(np.arange(32), 2)])

def remove_none(a):
    return a[a != None]

def second_to_hour(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    print("calc cost: %d:%02d:%02d" % (h, m, s))
    return "calc cost: %d:%02d:%02d" % (h, m, s)

def main(iter):
    np.random.seed()
    random.seed()
    while True:
        ac_1 = 1/3*np.random.rand() + 1/6
        ac_2 = 1/3*np.random.rand() + 1/6
        ac_3 = 1 - ac_1 - ac_2
        ideal_cor = ideal_cor_func(ac_1, ac_2, ind_1nn)
        if 1/6 <= ac_3 < 1/2:
            #*Atomic contents
            ele_list = ele_list_gen(ac_1, ac_2, ac_3)
            cor_func_raw = cor_func_all(ele_list, ideal_cor)
            if cor_func_raw >= 20:
                break
    
    step_count = 0
    buffer = []
    cor_func_raw = cor_func_raw
    step_max = 2
    # ele_traj = np.zeros(96)
    # ele_traj[:32] = ele_list
    action_dim = 108

    while step_count <= step_max:
        a_ind = np.random.choice(range(action_dim))
        action = ind_1nn[a_ind]
        ele_list_n, r, cor_func_n, _ = swap_step(action, ele_list, 0, 0, ideal_cor)
        r_ = np.exp(r)
        if np.random.rand() <= np.min([r_, 1]) and cor_func_n != cor_func_raw:
            # a_ind = np.where(np.linalg.norm(action - action_list, axis=1) == 0)[0][0]
            cor_func_raw = cor_func_n
            s_a_pair = np.concatenate([ele_list, action/31])
            ele_list = ele_list_n
            # ele_traj[32+step_count] = a_ind/action_dim
            buffer.append(s_a_pair.tolist())
            step_count += 1

        if cor_func_n < 8:
            print(iter)
            step_count = 21
            return buffer
            
def multicore(iter_time, process_num):
    # pool = mp.Pool(processes=2)
    pool = mp.Pool(processes=process_num)
    output_list = [pool.map(main, range(iter_time))]
    # map the equation to the value
    return output_list

if __name__ == '__main__':
    iter_time = 100000
    trial = 18
    step = 2
    tar = 8
    comment = 'nonenormal_108'
    start_ = time.time()
    output_list = [multicore(iter_time, process_num=20)][0][0]
    output_list = [i for i in output_list if i]
    np.save(path+f'ExpertTraj/expert_traj_32_tar{tar}_try{trial}_step{step}_{comment}.npy', output_list)
    np.save(path+f'ExpertTraj/tar{tar}_try{trial}_step{step}_{comment}.npy', output_list)
    second_to_hour(time.time() - start_)
    try:
        print(len(output_list))
    except:
        print('Sadly nothing reached the goal')
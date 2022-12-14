import math
import numpy as np
from random import randrange

#*Index of 1~4NN pairs in 32-atom cells.
path = '/media/wz/a7ee6d50-691d-431a-8efb-b93adc04896d/Github/SQSdrl/'
ind_1nn = np.load(path+'struc_info/ind_1nn.npy')
ind_2nn = np.load(path+'struc_info/ind_2nn.npy')
ind_3nn = np.load(path+'struc_info/ind_3nn.npy')
ind_4nn = np.load(path+'struc_info/ind_4nn.npy')

def abs_dis(a, b, target):
    return abs(np.linalg.norm(np.array(a) - np.array(b)) - target)

def phi1(x):
    return math.sqrt(3/2)*x

def phi2(x):
    return math.sqrt(2)*(3/2*(x**2) - 1)

def cpr(val1, val2):
    cor_func = (phi1(val1)*phi1(val2)
              +(phi1(val1)*phi2(val2)
              +phi2(val1)*phi1(val2))/2
              +phi2(val1)*phi2(val2))

    return cor_func

def cor_func(ind_nNN, ele_list):
    cor_func_n = 0
    for i in ind_nNN:
        phi1_1 = phi1(ele_list[i[0]])
        phi1_2 = phi2(ele_list[i[0]])
        phi2_1 = phi1(ele_list[i[1]])
        phi2_2 = phi2(ele_list[i[1]])
        cor_func_n += phi1_1*phi2_1+(phi1_1*phi2_2+phi1_2*phi2_1)/2+phi1_2*phi2_2
    return cor_func_n

def ele_list_gen(cr_content, co_content, ni_content, mode='randchoice'):
    np.random.seed()
    # if iter <= len_demo:
    #     return ele_demo[iter]
    assert abs(cr_content+co_content+ni_content-1)<0.001, 'Make sure atomic ratio sum to 1'

    while True:
        if mode == 'randchoice':
            len_cr = randrange(int(cr_content*32),int(cr_content*32)+2)
            len_co = randrange(int(co_content*32),int(co_content*32)+2)
        elif mode == 'int':
            len_cr = int(cr_content*32)
            len_co = int(co_content*32)
        
        len_ni = 32-len_cr-len_co
        if abs(len_ni-32*ni_content) <= 1:
            break

    ele_list_raw = np.concatenate([np.zeros(len_cr),np.ones(len_co),0-np.ones(len_ni)],axis=0)
    np.random.shuffle(ele_list_raw)
    
    return ele_list_raw

def ideal_cor_func(cr_content, co_content, ind_nNN):
    bond_num = len(ind_nNN)
    ni_content = 1-cr_content-co_content
    
    num_crcr = cr_content**2*bond_num
    num_coco = co_content**2*bond_num
    num_nini = ni_content**2*bond_num
    num_crco = 2*cr_content*co_content*bond_num
    num_coni = 2*co_content*ni_content*bond_num
    num_crni = 2*cr_content*ni_content*bond_num
    
    cor_func = (num_crcr*cpr(0,0)
               +num_coco*cpr(1,1)
               +num_nini*cpr(-1,-1)
               +num_crco*cpr(0,1)
               +num_crni*cpr(0,-1)
               +num_coni*cpr(1,-1))

    return cor_func

def cor_func_all(state, ideal):
    return abs(cor_func(ind_1nn, state)-ideal)

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def swap_step(action, state, target_val, step, ideal, reward_type = 'base'):

    cor_func_raw = abs(cor_func(ind_1nn, state[:32])-ideal)

    a1 = action[0]
    a2 = action[1]

    state[a2], state[a1] = state[a1], state[a2]

    cor_func_new = abs(cor_func(ind_1nn, state[:32])-ideal)
    
    if cor_func_new < target_val:
        done = True
    else:
        done = False
    
    diff = cor_func_new/cor_func_raw

    #*Artificially defined reward functions.
    if reward_type == 'base':
        reward = 1 - cor_func_new/cor_func_raw

    # elif reward_type == 'base2': #*bad
    #     reward = np.tanh(cor_func_raw - cor_func_new)

    # elif reward_type == 'sigmoid': #*a bit
    #     if diff < 1:
    #         reward = np.exp(-step)*(1-diff)
    #     elif diff >= 1:
    #         reward = 1/(1+np.exp(-step))*(1-diff)

    elif reward_type == 'determine': #*work
        if done:
            reward = 1
        elif diff < 1:
            reward = 0
        else:
            reward = -0.05

    elif reward_type == 'determine2': #*work
        if done:
            reward = 1
        elif diff < 1:
            reward = 1 - diff
        elif diff >= 1:
            reward = - diff
    
    # reward = np.tanh(reward5)
        
    return state, reward, cor_func_new, done

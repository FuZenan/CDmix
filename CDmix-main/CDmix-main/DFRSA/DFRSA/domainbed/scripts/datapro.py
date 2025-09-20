import os
import numpy as np
import pandas as pd
import sys
os.chdir(sys.path[0])
sys.path.append('../')
from utils import *
import pickle as cp

def DASA(dataset_dir='./DASA/data', WINDOW_SIZE=125, OVERLAP_RATE=0.4, DOMAIN_GROUPS=[1,], Z_SCORE=True, SAVE_PATH=os.path.abspath('../../DSADS')):
    '''
        dataset_dir: 源数据目录 : str
        WINDOW_SIZE: 滑窗大小 : int
        OVERLAP_RATE: 滑窗重叠率 : float in [0，1）
        DOMAIN_GROUPS: 受试者分组，每组受试者的编号组成一个域 : list of lists
        Z_SCORE: 标准化 : bool
        SAVE_PATH: 预处理后npy数据保存目录 : str
    '''

    print('\n原数据分析：原始文件共19个活动，每个活动都由8个受试者进行信号采集，每个受试者在每一类上采集5min的信号数据，采样频率25hz（每个txt是 125*45 的数据，包含5s时间长度，共60个txt）\n')
    print('预处理思路：数据集网站的介绍中说到60个txt是有5min连续数据分割而来,因此某一类别a下同一个受试者p的60个txt数据是时序连续的。\n\
            所以可以将a()p()下的所有txt数据进行时序维度拼接，选择窗口大小为125，重叠率为40%进行滑窗。\n')

    all_subjects = set([*range(1, 9)])
    for group in DOMAIN_GROUPS:
        for subject in group:
            assert subject in all_subjects, f"受试者编号 {subject} 不在有效范围内 (1-8)"

    subject_to_domain = {}
    for domain_id, group in enumerate(DOMAIN_GROUPS):
        for subject in group:
            subject_to_domain[subject] = domain_id

    domains_data = {domain_id: {'x': [], 'y': []} for domain_id in range(len(DOMAIN_GROUPS))} 
    adls = sorted(os.listdir(dataset_dir))
    os.chdir(dataset_dir)
    
    for label_id, adl in enumerate(adls):  # each adl
        print('======================================================\ncurrent activity sequence: 【%s】' % (adl))
        
        participants = sorted(os.listdir(adl))
        os.chdir(adl)
        for participant_idx, participant in enumerate(participants):  # each subject

            subject_id = participant_idx + 1
            domain_id = subject_to_domain.get(subject_id)
            print('      current subject: 【%d】   domain: 【%d】' % (subject_id, domain_id))

            files = sorted(os.listdir(participant))
            os.chdir(participant)
            concat_data = np.vstack([pd.read_csv(file, sep=',', header=None).to_numpy() for file in files])  # concat series data (125*60, 45)
            cur_data = sliding_window(array=concat_data, windowsize=WINDOW_SIZE, overlaprate=OVERLAP_RATE)  # sliding window [n, 125, 45]

            domains_data[domain_id]['x'] += cur_data
            domains_data[domain_id]['y'] += [label_id] * len(cur_data)

            os.chdir('../')
        os.chdir('../')
    os.chdir('../')
    
    for domain_id in domains_data:
        domains_data[domain_id]['x'] = np.array(domains_data[domain_id]['x'], dtype=np.float32)
        domains_data[domain_id]['y'] = np.array(domains_data[domain_id]['y'], dtype=np.int64)

    if Z_SCORE:
        for domain_id in domains_data:
            domains_data[domain_id]['x'], _ = z_score_standard(xtrain=domains_data[domain_id]['x'], xtest=None)
    print('\n---------------------------------------------------------------------------------------------------------------------\n')
    for domain_id in domains_data:
        print(f'Domain {domain_id} - x shape: {domains_data[domain_id]["x"].shape}, y shape: {domains_data[domain_id]["y"].shape}')

    if SAVE_PATH:  
        for domain_id in domains_data:
            domain_save_path = os.path.join(SAVE_PATH)
            os.makedirs(domain_save_path, exist_ok=True) 
            X = domains_data[domain_id]['x']
            y = domains_data[domain_id]['y']
            d = np.full(len(X), domain_id)
            
            X = np.transpose(X.reshape((-1, 1, 125, 45)), (0, 1, 2, 3))  # 将其重构为 (n, 45, 1, 125)

            obj = [(X, y, d)]  
            
            saved_filename = 'DSADS' + str(domain_id) + '_wd.data'

            with open(os.path.join(domain_save_path, saved_filename), 'wb') as f:
                cp.dump(obj, f, protocol=cp.HIGHEST_PROTOCOL)  
            
            print(f'Domain {domain_id} data saved to {os.path.join(domain_save_path, saved_filename)}')

    return domains_data

if __name__ == '__main__':
    DASA()
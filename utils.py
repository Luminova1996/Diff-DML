# -*- coding: utf-8 -*-
import os
import pickle
import re
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
import time
import torch
from loguru import logger

ignore_index=-100

def multi_process(func, lst, num_cores=multiprocessing.cpu_count(), backend='multiprocessing'):
    workers = Parallel(n_jobs=num_cores, backend=backend)
    output = workers(delayed(func)(one) for one in tqdm(lst))
    return [x for x in output if x]


def DBC2SBC(ustring):
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 0x3000:
            inside_code = 0x0020
        else:
            inside_code -= 0xfee0
        if not (0x0021 <= inside_code <= 0x7e):
            rstring += uchar
            continue
        rstring += chr(inside_code)
    return rstring


def clean(txt):
    txt = DBC2SBC(txt)
    txt = txt.lower()
    return re.sub('\s*', '', txt)


def pickle_dump_file(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def pickle_load_file(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def get_dir_files(dirname):
    L = []
    for root, dirs, files in os.walk(dirname):
        for file in files:
            L.append(os.path.join(root, file))
    return L


def padding(sequence, pads=0, max_len=None, fill_to_max=False, dtype='int32'):
    v_length = [len(x) for x in sequence]  # every sequence length
    seq_max_len = max(v_length)
    if (not fill_to_max) and ((max_len is None) or (max_len > seq_max_len)):
        max_len = seq_max_len
    x = (np.ones((len(sequence), max_len)) * pads).astype(dtype)
    for idx, s in enumerate(sequence):
        trunc = s[:max_len]
        x[idx, :len(trunc)] = trunc
    return x

def check_chinese_words(check_str):
    x=re.match(r'[\u4e00-\u9fff]+\Z',check_str)
    if x!=None:
        return True
    else:
        return False

def check_english_words(check_str):
    x=re.match(r'[a-zA-Z]+\Z',check_str)
    if x!=None:
        return True
    else:
        return False

def load_state_dict(input,input_type,path):
    if os.path.isfile(path):
        logger.info('%s load from %s'%(input_type, path))
        input.load_state_dict(torch.load(path))
    else:
        logger.info('can\'t find %s !'%(path))

def timmer(func):
    def deco(*args, **kwargs):
        logger.info('\nfunction: {_funcname_} starts'.format(_funcname_=func.__name__))
        start_time = time.time()
        res = func(*args, **kwargs)
        end_time = time.time()
        logger.info('function:{_funcname_} cost {_time_} seconds.'
              .format(_funcname_=func.__name__, _time_=round(end_time - start_time,2)))
        return res

    return deco

def take_index(input,index,fill_with=0): #input:[...,m] index:[...,n] output:[...,n]
    index_=index.reshape(-1,index.size(-1))
    index_t=index_+(torch.arange(index_.size(0),device=index.device)*input.size(-1))[:,None]
    output=torch.take(input,index_t)
    output=output.view_as(index)
    output[index==ignore_index]=fill_with
    return output


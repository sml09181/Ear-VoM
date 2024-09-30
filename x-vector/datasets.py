#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import glob
import argparse

def create_meta(files_list,store_loc,mode='train'):
    if not os.path.exists(store_loc):
        os.makedirs(store_loc)
    
    if mode=='train':
        meta_store = store_loc+'/training.txt'
        fid = open(meta_store,'w')
        for filepath in files_list:
            fid.write(filepath+'\n')
        fid.close()
    elif mode=='test':
        meta_store = store_loc+'/testing.txt'
        fid = open(meta_store,'w')
        for filepath in files_list:
            fid.write(filepath+'\n')
        fid.close()
    elif mode=='validation':
        meta_store = store_loc+'/validation.txt'
        fid = open(meta_store,'w')
        for filepath in files_list:
            fid.write(filepath+'\n')
        fid.close()
    else:
        print('Error in creating meta files')
    
def extract_files(folder_path):
    train_lists=[]
    test_lists = []
    val_lists=[]
    
    sub_folders = sorted(glob.glob(folderpath+'/*/'))
    train_nums = len(sub_folders)-int(len(sub_folders)*0.1)-int(len(sub_folders)*0.05)
    
    for i in range(train_nums):
        sub_folder = sub_folders[i]
        all_files = sorted(glob.glob(sub_folder+'/*.wav'))
        for audio_filepath in all_files:
            to_write = audio_filepath+' '+str(class_ids[language])
            train_lists.append(to_write)
            
    for i in range(train_nums,train_nums+int(len(sub_folders)*0.05)):
        sub_folder = sub_folders[i]
        all_files = sorted(glob.glob(sub_folder+'/*.wav'))
        for audio_filepath in all_files:
            to_write = audio_filepath+' '+str(class_ids[language])
            val_lists.append(to_write)
    
    for i in range(train_nums+int(len(sub_folders)*0.05),len(sub_folders)):
        sub_folder = sub_folders[i]
        all_files = sorted(glob.glob(sub_folder+'/*.wav'))
        for audio_filepath in all_files:
            to_write = audio_filepath+' '+str(class_ids[language])
            test_lists.append(to_write)
    return train_lists,test_lists,val_lists


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Configuration for data preparation")
    parser.add_argument("--processed_data", default="/scratch/ASVspoof2019", type=str, help='Dataset path')
    parser.add_argument("--meta_store_path", default="/result/meta/", type=str, help='Save directory after processing')
    config = parser.parse_args()
    train_list, test_list, val_lists = extract_files(config.processed_data)

    create_meta(train_list, config.meta_store_path, mode='train')
    create_meta(test_list, config.meta_store_path, mode='test')
    create_meta(val_lists, config.meta_store_path, mode='validation')
    
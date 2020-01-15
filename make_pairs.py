'''
  make pairs from 0 to N in a directory

  (c) 2020 Matiur Rahman Minar and Heejune Ahn @ icomlab.seoutech.ac.kr


  Description
  ============

  
'''
import os
import json
import numpy as np


def make_pairs_file(img_path, cloth_path):

    # 1. get the list      
    img_files = os.listdir(img_path)
    cloth_files = os.listdir(cloth_path)

    the_file = open('viton_test_pairs.txt', 'a')

    count = 1
    for each in zip(img_files, cloth_files):
        if count <= 100:
            the_file.write(each[0] + ' ' + each[1] + '\n')
            count += 1


def make_pairs_file_from_clothes(cloth_path):

    # 1. get the list      
    # img_files = os.listdir(img_path)
    cloth_files = os.listdir(cloth_path)

    the_file = open('viton_test_pairs.txt', 'a')

    for each in cloth_files:
        the_file.write(each[:-4].replace('_1', '_0') + ' ' + each[:-4] + '\n')
    

def make_pairs_file_from_warped_clothes(cloth_path):

    # 1. get the list
    cloth_files = os.listdir(cloth_path)

    the_file = open('viton_test_pairs.txt', 'a')

    for each in cloth_files:
        the_file.write(each[:-4].split("_")[0] + '_0 ' + each[:-4].split("_")[0] + '_1\n')


if __name__ =='__main__':

    cloth_2dw_dir = "./results/viton/c2dw/"
    cloth_3dw_dir = "./results/viton/c3dw/"

    # make_pairs_file(img_dir, cloth_dir)
    make_pairs_file_from_clothes(cloth_2dw_dir)
    # make_pairs_file_from_warped_clothes(cloth_3dw_dir)


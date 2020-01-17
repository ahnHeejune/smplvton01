'''
  Reorder/Rename human images and smple params into original viton names

  (c) 2019 Matiur Rahman Minar and Heejune Ahn @ icomlab.seoutech.ac.kr


  Description
  ============

  
'''
import os
import sys
import numpy as np


def reorder_human_images(list_file, image_dir):
    image_names = os.listdir(image_dir)
    # pf = open(list_file)
    filenp = np.load(list_file)

    for each in zip(image_names, filenp):
        src_path = image_dir + each[0]
        dst_path = image_dir + each[1].split(" ")[0]
        os.rename(src_path, dst_path)
        print("Converting " + each[0] + " to " + each[1].split(" ")[0])


def reorder_smpl_params(list_file, smpl_dir):
    # all_files = os.listdir(smpl_dir)
    # pf = open(list_file)
    filenp = np.load(list_file)

    count = 0
    # for each in pf.readlines():
    for each in filenp:
        fname = str(each.decode("utf-8"))
        src_smpl_path = os.path.join(smpl_dir + '%04d.pkl' % count)
        dst_smpl_path = os.path.join(smpl_dir + fname.split(" ")[0].replace(".jpg", ".pkl"))
        os.rename(src_smpl_path, dst_smpl_path)

        src_image_path = os.path.join(smpl_dir + '%04d.png' % count)
        dst_image_path = os.path.join(smpl_dir + fname.split(" ")[0].replace(".jpg", ".png"))
        os.rename(src_image_path, dst_image_path)

        print("Converting " + src_smpl_path + " to " + fname.split(" ")[0].replace(".jpg", ".pkl"))
        count = count + 1
        if count == 1000:
            break


if __name__ =='__main__':

    if len(sys.argv) < 3:
        print('usage: %s listnpyfile smpldir' % sys.argv[0])
        exit()

    # list_file = "test_pairs.txt"  # viton original test pairs file
    list_file = "list.npy"  # viton original test files names' list

    # image_dir = "./images/viton/"  # human images directory path
    smpl_dir = "./results/viton/smpl/"  # saved smpl parameters directory path

    # ============Re-order===========
    # reorder_human_images(list_file, image_dir)
    reorder_smpl_params(list_file, smpl_dir)

'''
  sort file and rename from 0 to N in a directory 
  unsort it back 

  (c) 2019 matiur Rahman and heejune Ahn @ icomlab.seoutech.ac.kr


  Description
  ============

  
'''
import os
import json
import numpy as np
import sys


def rename_by_sort(dir_path):

    # 1. get the list
    files = os.listdir(dir_path)  
    files.sort()  # make the ordering 

    filesnp = np.array(files)

    for count, each in enumerate(files):
	print(count, " converting ", each , 'to', 'viton_' + format(count, '06d') + '.jpg') 
        src_path = dir_path + each   
        dst_path = dir_path + 'viton_' + format(count, '06d') + '.jpg' 
        os.rename(src_path, dst_path)

    # save the ordering
    np.save('viton_list.npy', filesnp) 


def rename_by_list(dir_path):

    print('loading....' + dir_path + 'viton_list.npy')

    #with np.load(dir_path + 'list.npy') as filelist: # zip file loading
    filesnp =  np.load('list.npy') 
    fileslist = filesnp.tolist()

    for count, each in enumerate(fileslist):
	print(count, " converting ", 'viton_' + format(count, '06d') + '.jpg', 'to', each) 
        src_path = dir_path + 'viton_' + format(count, '06d') + '.jpg' 
        dst_path = dir_path + each  
        os.rename(src_path, dst_path)


def rename_images_by_sort():

    img_dir = "./images/"
    rename_by_sort(img_dir)

def restore_images_name():    

    c = input('finished renaming. restore? (1 for yes):') 
    if c == 1:
        rename_by_list(img_dir)


def check_list(file_path):
    filesnp =   np.load(file_path) 
    fileslist = filesnp.tolist()
    for idx, each in enumerate(fileslist):
	print(idx, ":", each) 

def remove_ext_list(file_path):
    filenp =   np.load(file_path) 
    filelist = filenp.tolist()
    for idx, each  in enumerate(filelist):
	#print(idx, ":", filelist[idx]) 
        filelist[idx] = filelist[idx].replace('.jpg','') # remove the .jpg ext

    _ = input('>>')

    for idx, each in enumerate(filelist):
	print(idx, ":", each) 

    filenp = np.array(filelist)
    np.save('viton_list_numbers.npy', filenp) 

def restore_name(sys.argv):

    if len(sys.argv) < 5:
        print('usage: %s listnpyfile srcdir dstdir extension'%sys.argv[0])
        return

    filenp = np.load(file_path) 
    filelist= os.listdir(dir_path)  
    for i in range(filelist):
        src_path  = dir_path + '/' +  '%06d'%i  +  '.' + extension 
        dst_path  = dir_path + '/' +  filenp[i] +  '.' + extension 
        os.rename(src_path, dst_path)

if __name__ =='__main__':
	
    #remove_ext_list(sys.argv[1])
    #check_list(sys.argv[1])
	
    #=======================
    restore_name(sys.argv)



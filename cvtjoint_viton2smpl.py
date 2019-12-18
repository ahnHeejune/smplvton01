'''

 conevrt from VITON Half-body model pose to simple npz file 

  (c) 2019 matiur Rahman and heejune Ahn @ icomlab.seoutech.ac.kr


  Description
  ============

  the ordering of npz file  (is strange and in-efficient.) differ from input format
  number of joints for SMPL is 14, input is 18. (@TODO) 
  the order of joints also different. so that  so remapping is needed as same as LSP format

  each json has 18*3 in 1-D format  
 
  3 x 14 x #ofimages but we used all 18 so 3 x 18 x #ofimages
   

  VITON json format
 -------------------
  {
   'version': 1.0,
   'people': [ {"face_keypoints": [],
               "pose_keypoints": [ x1,y1,p1, x2,y2,p2, ..., x18,y18,p18]}, # 18x3 floats  
               "hand_right_keypoints":[]},   
               "hand_left_keypoints":[]} ]  # originally multi-persons now only one assumped
  }
  
  x, y : in pixel unit
  p    : confidence probability


  SMPL joint format
 --------------------

  npz of list: index (x/y/p, joint, images)  # pretty stupid format

  [ [ [ img 0    ]
      [ img 1    ]
         ...
      [ img last ]
     ]  #  joint-0
     ...
     ...
     ... # joint-18
    ] # x-pixel, (integer?)

    [ 
    
    
    
    ] # y-pixel  
    [ 
    
    
    
    ] # confidence  
  
'''
import sys
import os
import json
import numpy as np
import pprint
import vis_joints

bcheck = True 

# append one person's joints into list
def append_bodyjoints(bodyjoints, imgidx, joints, n_joints):

    for j in range(n_joints):
        # converting the viton joint ordering to lsp joint order       
        if j < 14:  # LSP joints: mapped 
		joints[0,vis_joints.viton2lsp_joint[j],imgidx] = bodyjoints[3*j]
		joints[1,vis_joints.viton2lsp_joint[j],imgidx] = bodyjoints[3*j+1]
		joints[2,vis_joints.viton2lsp_joint[j],imgidx] = bodyjoints[3*j+2]
	else:    # out of LSP, so same index mapping
		joints[0,j,imgidx] = bodyjoints[3*j]
		joints[1,j,imgidx] = bodyjoints[3*j+1]
		joints[2,j,imgidx] = bodyjoints[3*j+2]

	'''
	joints[0,j,imgidx] = bodyjoints[3*j]
	joints[1,j,imgidx] = bodyjoints[3*j+1]
	joints[2,j,imgidx] = bodyjoints[3*j+2]
        '''


# load all the json files into list of x, y, p
# Note: np is faster than list, pre-allocation is much faster than append
def load_alljointjsons(joint_dir, img_dir):

    # 1. get the list
    joint_files = os.listdir(joint_dir)  
    joint_files.sort()  # make the ordering 

    if bcheck:
        joint_files = joint_files[:5] # for test purpose

    # 1. add joint arrays 
    n_joints = 18# the joint # of viton json file,  14 # LSP SMPL joint number  
    smpl_joints = np.zeros((3, n_joints, len(joint_files)), dtype=float)
    imgidx      = np.zeros((len(joint_files)), dtype=int) # for image index 

    # 2. read joints from json files 
    for count, each in enumerate(joint_files):
        print(count, " converting ", each)
	with open(joint_dir + each) as json_file:
		p = json.load(json_file)
                body_joints = p['people'][0]['pose_keypoints'] # needs body joint only  
                append_bodyjoints(body_joints, count, smpl_joints, n_joints) 

                ## file idex 
                idx = each.replace('_0_keypoints.json', '')
                idx = int(idx)
                imgidx[count] = idx

                # visualize 
                if bcheck:
                    img_f_name = each.replace('_keypoints.json', '.jpg')
                    img_f_path = img_dir + img_f_name 
                    bj = np.array(body_joints)
                    bj = np.reshape(bj, (-1,3))
                    print(bj)
                    vis_joints.visualize_joints(img_f_path, bj, True)

    return smpl_joints, imgidx


# 1. check joint estimation file format
def check_smpl_joint_file(fname):

    with np.load(fname) as zipfile: # zip file loading
        est = zipfile['est_joints']
        print("shape:", est.shape, ", type:", est.dtype)
        for imgidx in range(5):
            joints = est[:2, :, imgidx].T  # T for joint-wise
            conf = est[2, :, imgidx]
            print('joints:', joints)
            print('conf:', conf)

'''
def check_idx(fname):

    with np.load(fname) as zipfile: # zip file loading
        idxes = zipfile['idx']
        print("shape:", idxes.shape, ", type:", idxes.dtype)
        for i in range(50):
            print( idxes[i] )
'''

#
#  vito json files to  smpl  npz file 
#

def cvt_viton2smpl():

    #viton_joints_dir = "D:/Datasets/viton_resize/test/pose/"
    viton_joints_dir = "/home/heejune/Work/VTON/VITON/Dataset/viton_resize/test/pose/"
    img_dir = "/home/heejune/Work/VTON/VITON/Dataset/viton_resize/test/image/"
    joints, idxes = load_alljointjsons(viton_joints_dir, img_dir)


    return


    '''
    pp = pprint.PrettyPrinter(depth=6)
    pp.pprint(joints)
    '''
    np.savez("viton_est_joints2.npz", est_joints=joints)
    #np.savez("viton_est_joints.npz", est_joints=joints, idx=idxes)
    print("\nConversion finished!")

    ## validation #####
    print('### reference file ###########################################')
    check_smpl_joint_file('est_joints.npz')
    #check_smpl_joint_file('10k_est_joints.npz')
    print('### my file        ###########################################')
    check_smpl_joint_file("viton_est_joints2.npz")
    ''' no more using
    print('###  idxes         ###########################################')
    check_idx("viton_est_joints.npz")
    '''


# convert numpy to json for a single person joint
def cvt_np2json(joints_np):

    # 1. re-ordering 
    # same as viton2lsp_joint and reamining 
    order = [13,12,8, 7, 6, 9, 10, 11, 2, 1, 0, 3, 4, 5, 14, 15, 16, 17]

    # 2. build dictionary 
    oneperson = { "face_keypoints": [],
                  "pose_keypoints": joints_np[order].flatten().tolist(),
                  "hand_right_keypoints": [],
                  "hand_left_keypoints":[]}

    people   = {"poeple":  [oneperson]}
    joints_json =  { "version": 1.0, "people": people }

    return joints_json

def test_cvt_smpl2viton():

    fname = 'viton_est_joints_18.npz'
    num   = 1
    with np.load(fname) as zipfile: # zip file loading
        est = zipfile['est_joints']
        print("shape:", est.shape, ", type:", est.dtype)
        for imgidx in range(num):
            joints_np = est[:3, :, imgidx].T  # T for joint-wise
            print('joints:', joints_np)

            joints_json = cvt_np2json(joints_np)

            json_file_path = '%06d_0_keypoints.json'%imgidx 
            with  open(json_file_path, 'w') as json_file:
	          json.dump(joints_json, json_file )

if __name__ =='__main__':

    if len(sys.argv) < 2:
        print( 'usage:' + sys.argv[0] + ' tojson (for viton) or tonp (for smpl)')
        exit()

    if sys.argv[1] == 'tonp':
        cvt_viton2smpl()
    elif sys.argv[1] == 'tojson':
        test_cvt_smpl2viton()
    else:
        print('undefined command')



##
##  joint visualize
##
##  (c) 2019 heejune@snut.ac.kr
##  

import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import cv2
import time
import sys



###############################################################
# Joint Visualize
###############################################################

# This display setting is required to display the label image in matplot
  

## joint


# LSP 
viton2lsp_joint = [13,  # Head top
                   12,  # Neck
                   8,   # Right shoulder
                   7,   # Right elbow
                   6,   # Right wrist
                   9,   # Left shoulder
                   10,  # Left elbow
                   11,  # Left wrist
                   2,   # Right hip
                   1,   # Right knee
                   0,   # Right ankle
                   3,   # Left hip
                   4,   # Left knee
                   5]   # Left ankle
                   
lsp_limbs = [ (13,12),  # head to check/neck
          (12, 8), (12, 9), # chick to shoulders
          (8,  7), (9, 10), # shoulders to elbows
          (7,  6), (10,11), # elbows to wrists
          (8,  2), (9,  3), # shoulders to hips
          (2,  1), (3,  4), # hips to knees
          (1,  0), (4,  5)] # knees to ankles 

## VITON ?
## 18 joints = 14 + 4 (eyes, nose, ???)
viton_joint_order = [0, # nose 
                   1,   # Neck
                   2,   # Right shoulder
                   3,   # Right elbow
                   4,   # Right wrist
                   5,   # Left shoulder
                   6,   # Left elbow
                   7,   # Left wrist
                   8,   # Right hip
                   9,   # Right knee
                   10,  # Right ankle
                   11,  # Left hip
                   12,  # Left knee
                   13,  # Left ankle
                   14,  # Right eye   ## 4 more appended, not mixed up with above numebrs ##
                   15,  # left eye     
                   16,  # left ear
                   17]  # right ear 

viton_limbs = [ (0,1),  # head to check/neck
          (1, 2), (1, 5), # chick to shoulders
          (2,  3), (5, 6), # shoulders to elbows
          (3,  4), (6, 7), # elbows to wrists
          (2,  8), (5, 11), # shoulders to hips
          (8,  9), (11, 12), # hips to knees
          (9,  10), (12, 13), # knees to ankles
          (0,  14), (0, 15), # nose to eyes
          (14,  16), (15, 17)] #  
 
                   
# overlay joints 
def drawJoints(img, joints, T = 0.5, jointInPixel = False):
    
    if len(img.shape) == 2:  # gray 
        color = 0        # black 
    else:                    # rgb
        color = (0,0,0)  # black
    
    if jointInPixel != True:
        height_scale = img.shape[0]
        width_scale = img.shape[1]
        #print('h:', height, 'w:', width)
    else:
        height_scale = 1.0
        width_scale  = 1.0
        
    for i in range(len(joints)):
        if joints[i,2] > T:  # when confidence enough 
            x = int((joints[i,0] +0.499)*width_scale)
            y = int((joints[i,1] +0.499)*height_scale)
            cv2.circle(img, (x,y), 3, color)
        
# overlay limbs 
def drawLimbs(img, joints, T = 0.5, jointInPixel = False):
   
   
    if len(img.shape) == 2:  # gray 
        color = 0        # black 
    else:                    # rgb
        color = (0,0,0)  # black
        
    if jointInPixel != True:
        height_scale = img.shape[0]
        width_scale = img.shape[1]
        #print('h:', height, 'w:', width)
    else:
        height_scale = 1.0
        width_scale  = 1.0
        

    # wich format
    limbs = viton_limbs

    for limb in limbs:  
        if (joints[limb[0],2] > T) and (joints[limb[1],2] > T):  # when confidence enough 
            x = int((joints[limb[0],0] +0.5)*width_scale)
            y = int((joints[limb[0],1] +0.5)*height_scale)
            pt1 = (x,y)
            x = int((joints[limb[1],0] +0.5)*width_scale)
            y = int((joints[limb[1],1] +0.5)*height_scale)
            pt2 = (x,y)
        
            cv2.line(img, pt1, pt2, color, 1) # 0 : black
        

def visualize_joints(imgfname, joints, jointInPixel = False):

    """ 
       joint format 14x2 
    
    """

    img = cv2.imread(imgfname)
    if img is None:
        print('cannot load file:', imgfname)
        return

    fig = plt.figure() 

    # display normal image with joints
    drawJoints(img, joints, 0.1, jointInPixel)
    drawLimbs(img, joints, 0.1, jointInPixel)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # becasue OpenCv use BGR not RGB
    plt.axis('off')
    plt.title('Joints')
    plt.show()


def check_est_joints_file(fname, imgfile):

    print('checking ', fname)
    # 1. check joint estimation file format  
    with np.load(fname) as zipfile: # zip file loading
        est = zipfile['est_joints']          
        print("shape:", est.shape, ", type:", est.dtype) 
        for imgidx in range(1):
            joints = est[:2, :, imgidx].T  # T for joint-wise
            conf = est[2, :, imgidx]
            print('joints:', joints)
            print('conf:', conf)
            
            # visualize the pose 
            #joint2d = np.reshape(joints, (-1,2))
            #print('reshaped:', joint2d)    
            visualize_joints(imgfile, joints, jointInPixel = True)


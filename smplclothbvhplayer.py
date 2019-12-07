'''
 Image2Animation using SMPLify human reverse-rendering   
 --------------------------------------------------------

 (c) copyright 2019 heejune@seoultech.ac.kr

 Prerequisite: SMPL model 
 In : Input image 
         - 2d joint estimated
         - SMPL parameters (cam, shape, pose)  
         - segmenation mask of human object
         - cloth vertices from body 
      BVH file 

      background image file 
 Out:  
      animating video file 

'''
from __future__ import print_function 
import sys
from os.path import join, exists, abspath, dirname
from os import makedirs
import logging
import cPickle as pickle
import time
import cv2
import numpy as np
import chumpy as ch
from opendr.camera import ProjectPoints
from smpl_webuser.serialization import load_model
from smpl_webuser.verts import verts_core
from smpl_webuser.verts import verts_decorated
from render_model import render_model
import inspect  # for debugging
import matplotlib.pyplot as plt
from opendr.lighting import SphericalHarmonics
from opendr.geometry import VertNormals, Rodrigues


from bvh import Bvh, BvhNode
import math

_LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from smpl2cloth import smpl2cloth3dcore

# this texture renderer doesnot have lightening effects
#from smpl2cloth import build_texture_renderer 

_clothlabeldict = {"background": 0,
            "hat" :1,
            "hair":2,
            "sunglass":3, #       3
            "upper-clothes":4,  #  4
            "skirt":5 ,  #          5
            "pants":6 ,  #          6
            "dress":7 , #          7
            "belt": 8 , #           8
            "left-shoe": 9, #      9
            "right-shoe": 10, #     10
            "face": 11,  #           11
            "left-leg": 12, #       12
            "right-leg": 13, #      13
            "left-arm": 14,#       14
            "right-arm": 15, #      15
            "bag": 16, #            16
            "scarf": 17 #          17
        }

def _examine_smpl_params(params):

    print(type(params))
    print(params.keys())
    print('camera params')
    print(" - type:", type(params['cam']))
    #print(" - members:", dir(params['cam']))
    #print(" - cam.t:",params['cam'].t)
    print(" - cam.t:", params['cam'].t.r)    # none-zero, likely only nonzero z
    print(" - cam.rt:", params['cam'].rt.r)  # zero (fixed)
    print(" - cam.camera_mtx:", params['cam'].camera_mtx)  # 
    print(" - cam.k:", params['cam'].k.r)  #

    #    print(params['f'].shape)      # 2
    print('pose')
    print(" - type:", type(params['pose']))
    print(' - shape:', params['pose'].shape)   # 72
    #print(' - values:', params['pose'])
    print('betas')
    print(' - type:', type(params['betas']))
    print(' - shape:', params['betas'].shape)  # 10
    # print(' - values:', params['betas'])  # 10


# 
# print pose with annotation 
#  pose: np for all 23  
#
#               - 13 - 16 [lsh] - 18 [lelb] - 20 [lwr] - 22
#     3 - 6 - 9 - 12 - 15
#               - 14 - 17 [rsh] - 19 [relb] - 21 [rwr] - 23
# 0 -
#     1 [lhip] - 4 [lknee] - 7 [lankle] - 10
#     2 [rhip] - 5 [rknee] - 8 [rankle] - 11
#
jointname = {  0: 'root',
                1: 'lhip', 
                2: 'rhip', 
                4: 'lknee',
                5: 'rknee',
                7: 'lankle',
                8: 'rankle',
                10: 'lfoot',
                11: 'rfoot',
                3: 'lowback',
                6: 'midback',
                9: 'chest',
                12: 'neck',
                15: 'head',
                13: 'lcollar',
                14: 'rcollar',
                16: 'lsh',
                17: 'rsh',
                18: 'lelbow',
                19: 'relbow',
                20: 'lwrist',
                21: 'rwrist',
                22: 'lhand',
                23: 'rhand'}

def print_pose(pose):

    if pose.shape[0]  == 24*3:
      pose = pose.reshape([-1,3])

    if pose.shape[0] == 24 and pose.shape[1] == 3:
        for j in range(24):
            print(jointname[j],  pose[j,:])
    else:
        pass

# print 2 poses for comparison 
def print_poses(pose1, pose2):

    # 1. convert into joint x angles
    if pose1.shape[0]  == 24*3:
      pose1 = pose1.reshape([-1,3])
    if pose2.shape[0]  == 24*3:
      pose2 = pose2.reshape([-1,3])

    # 2. print in human readible format
    if (pose1.shape[0] == 24 and pose1.shape[1] == 3) and (pose2.shape[0] == 24 and pose2.shape[1] == 3):
        for j in range(24):
            print('%12s'%jointname[j], end = ':')
            for a in range(2):
                print('\t%+5.1f'%pose1[j,a], '%+5.1f'%pose2[j,a], end=',')
            print('%+3.1f'%pose1[j,2], '%+3.1f'%pose2[j,2])
    else:
        print('oops!!! unexpected format of poses')
        pass


#######################################################################################
# load dataset dependent files and call the core processing 
#---------------------------------------------------------------
# smpl_mdoel: SMPL 
# inmodel_path : smpl param pkl file (by SMPLify) 
# inimg_path: input image 
# mask image 
# joint 2d
# ind : image index 
#######################################################################################

#
# read smpl param, mask, original images
#
def  _read_ifiles(human):

    # model params 
    with open(human['params'], 'rb') as f:
        if f is None:
            print("cannot open",  human['params']), exit()
        params = pickle.load(f)
    _examine_smpl_params(params)

    #  2d rgb image for texture
    img2D = cv2.imread(human['img'])
    if img2D is None:
        print("cannot open",  human['img']), exit()

    # segmentation mask 
    mask = cv2.imread(human['mask'], cv2.IMREAD_UNCHANGED)
    if mask is None:
        print("cannot open",  human['mask']), exit()

    return params, mask,  img2D


###############################################################################
# restore the Template posed vertices  
# 
# return: vertices (ccoordinates), jtrs' locations
# 
# multistep joint reverse method   
#    
#                           - 13        - 16 [lsh] - 18 [lelb] - 20 [lwr] - 22
#     3        - 6          - 9         - 12       - 15
#                           - 14        - 17 [rsh] - 19 [relb] - 21 [rwr] - 23
# 0 - 
#     1 [lhip] - 4 [lknee] - 7 [lankle] - 10
#     2 [rhip] - 5 [rknee] - 8 [rankle] - 11                
###############################################################################
def restorePose(smpl_model, vertices, j_transformed, pose_s):

    joint_hierarchy = [ [0],    # the level not exactly hierarchy 
                        [1, 2, 3], 
                        [6, 4, 5], 
                        [7, 8, 9, 13, 14], 
                        [10, 11, 12, 16, 17], 
                        [15, 18, 19], 
                        [20, 21], 
                        [22, 23] ] 

    # constant terms 
    weights = ch.array(smpl_model.weights.r)
    kintree_table = smpl_model.kintree_table.copy()

    # 2. back to the default pose
    ##########################################################################
    for step in range(0, len(joint_hierarchy)):

        # 1. get the vertices and paramters 
        if step == 0:
            v_posed = ch.array(vertices)
            J = ch.array(j_transformed)
        else:
            v_posed = ch.array(v_cur.r)
            J = ch.array(jtr.r)

        pose =  ch.zeros(smpl_model.pose.size)

        # 2. LBS setup 
        v_cur, jtr = verts_core(  pose = pose,
                            v    = v_posed,
                            J    = J,
                            weights = weights,
                            kintree_table = kintree_table,
                            xp = ch, #ch vs np
                            want_Jtr = True,
                            bs_style = 'lbs')
        # 3. Renderer 
        #cam_s.v = v_cur

        # 4. repose 
        for joint in joint_hierarchy[step]:
            pose[joint*3:(joint+1)*3] = - pose_s[joint*3:(joint+1)*3]     

    return  v_cur.r, jtr.r 


################################################################################
# building LBS
#
################################################################################
def buildLBS(smpl_model, vertices, jtr):

    # constant terms 
    weights = ch.array(smpl_model.weights.r)
    kintree_table = smpl_model.kintree_table.copy()

    # 1. get the vertices and paramters 
    v_posed = ch.array(vertices)
    J = ch.array(jtr)
    pose =  ch.zeros(smpl_model.pose.size)

    # 2. LBS setup 
    v_cur, jtr = verts_core(  pose = pose,
                            v    = v_posed,
                            J    = J,
                            weights = weights,
                            kintree_table = kintree_table,
                            xp = ch, #ch vs np
                            want_Jtr = True,
                            bs_style = 'lbs')

    return  v_cur, jtr, pose 


def _rotateY(points, angle):
    """Rotate the points by a specified angle."""
    ry = np.array([
        [np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
        [-np.sin(angle), 0., np.cos(angle)]
    ])
    return np.dot(points, ry)




def build_texture_renderer(U, V, f, vt, ft, texture, w, h, ambient=0.0, near=0.5, far=20000, background_image = None):

    from opendr.renderer import TexturedRenderer
    from opendr.lighting import SphericalHarmonics
    from opendr.lighting import LambertianPointLight

    A = SphericalHarmonics(vn=VertNormals(v=V, f=f),
                           components=[0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           light_color=ch.ones(3)) + ambient

    '''
    A = LambertianPointLight(
        f=f,
        v=V,
        num_verts=len(V),
        light_pos=ch.array([-500, -500, -500]),
        vc= np.ones_like(V.r),    #albedo,
        light_color=np.array([0.7, 0.7, 0.7])) + 0.3
    '''

    if background_image is not None:
        R = TexturedRenderer( vc=A, 
                        camera=U, f=f, bgcolor=[0.0, 0.0, 0.0],
                         texture_image=texture, vt=vt, ft=ft,
                         frustum={'width': w, 'height': h, 'near': near, 'far': far}, background_image= background_image)

    else:
        R = TexturedRenderer(vc=A,
                        camera=U, f=f, bgcolor=[0.0, 0.0, 0.0],
                         texture_image=texture, vt=vt, ft=ft,
                         frustum={'width': w, 'height': h, 'near': near, 'far': far})




    return R


def reverse_rendering(smpl_model, source, img_bg):

    params_s, mask_s, img_s = _read_ifiles(source)
    j2d_s =  source['joints']
    idx_s =  source['id']

    #  pre-processing for boundary matching  
    mask_s[mask_s == _clothlabeldict['bag']] = 0  # remove bag
    # cut the connected legs 
    if idx_s == 0:
        mask_s[500:,190] = 0
    elif idx_s == 1:
        mask_s[500:,220] = 0
    else:
        print('Not prepared Yet for the idx!')
        #, exit()


    # 1. SMPL body to cloth  
    cam_s   = params_s['cam']           # camera model, Ch
    betas_s = params_s['betas']
    n_betas_s = betas_s.shape[0]        # 10, tuple to numpy
    pose_s  = params_s['pose']          # angles ((23+1)x3), numpy

    joint_transformed, vt, ft, texture = smpl2cloth3dcore(cam_s,      # camera model, Ch
                 betas_s,    # shape coeff, numpy
                 n_betas_s,  # num of PCA
                 pose_s,     # angles, 27x3 numpy
                 img_s,    # img numpy
                 mask_s,     # mask 
                 j2d_s, 
                 smpl_model,
                 bHead = True,
                 viz =False)

    h, w = mask_s.shape[:2]

    # 2. restore default posed vertices 
    vertices, jtr = restorePose(smpl_model, cam_s.v.r, joint_transformed, pose_s)  
    # 3 build model
    v_cur, jtr_cur, pose = buildLBS(smpl_model, vertices, jtr)

    # 4 build texture renderer 
    cam_s.v = v_cur 
    texture_renderer = build_texture_renderer(cam_s, v_cur, 
            smpl_model.f, vt, ft, texture[::-1, :,:], w, h, 1.0, near=0.5, far = 125., background_image = img_bg[:,:,::-1])  
    return cam_s, pose, texture_renderer

    '''
    # 5 re-pose
    cam_s.v = v_cur  
    pose[:] = pose_t[:]

    # 6 visualize 
    img = (texture_renderer.r*255.0).astype('uint8')
    # joint display 
    cam_s.v = jtr_cur.r  
    j2d = cam_s.r.copy()
    for i in range(j2d.shape[0]):
        cv2.drawMarker(img, (int(j2d[i,0]), int(j2d[i,1])), (255,0,0), cv2.MARKER_DIAMOND, 5, 3)
    plt.subplot(1,2,2)
    plt.imshow(img)
    plt.axis('off')
    plt.draw()
    plt.show()
    _ = raw_input()
    '''




def eulerAnglesToRotationMatrixZXY(theta):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_x, R_y))

    return R


###########################################################################
# Joint mapping between BVH & SMPL
###########################################################################
'''
                - 13 - 16 [lsh] - 18 [lelb] - 20 [lwr] - 22
     3 - 6 - 9 - 12 - 15
               - 14 - 17 [rsh] - 19 [relb] - 21 [rwr] - 23
 0 -
     1 [lhip] - 4 [lknee] - 7 [lankle] - 10
     2 [rhip] - 5 [rknee] - 8 [rankle] - 11
'''
global bvh2smpl_jtmap
bvh2smpl_jtmap = {
    'Hips': 0,
    'hip': 0,
    'mixamorig:Hips': 0,

    # body
    'Chest': 6,
    'chest': 6,
    'Chest2': 9,

    'LeftCollar': 13,
    'lCollar': 13,
    'RightCollar': 14,
    'rCollar': 14,

    # arms // left - right
    'LeftShoulder': 16,
    'lShldr': 16,
    'mixamorig:LeftShoulder': 16,
    'LeftUpArm': 16,
    'mixamorig:LeftArm': 16,
    'RightShoulder': 17,
    'rShldr': 17,
    'mixamorig:RightShoulder': 17,
    'RightUpArm': 17,
    'mixamorig:RightArm': 17,

    # elbow
    'LeftElbow': 18,
    'LeftLowArm': 18,
    'LeftForeArm': 18,
    'lForeArm': 18,
    'mixamorig:LeftForeArm': 18,
    'RightElbow': 19,
    'RightLowArm': 19,
    'RightForeArm': 19,
    'rForeArm': 19,
    'mixamorig:RightForeArm': 19,

    # wrist
    'LeftWrist': 20,
    'LeftHand': 20,
    'lHand': 20,
    'mixamorig:LeftHand': 20,
    'RightWrist': 21,
    'RightHand': 21,
    'rHand': 21,
    'mixamorig:RightHand': 21,

    # fingers
    'LeftFingerBase': 22,
    'mixamorig:LeftHandMiddle1': 22,
    'RightFingerBase': 23,
    'mixamorig:RightHandMiddle1': 23,

    # neck
    'Neck': 12,
    'neck': 12,
    'mixamorig:Neck': 12,
    'Head': 15,  # not correct but..
    'head': 15,  # not correct but..
    'mixamorig:Head': 15,  # not correct but..
    'LowerBack': 3,
    'abdomen': 3,
    'Spine1': 3,
    'mixamorig:Spine2': 3,
    'Spine': 6,
    'mixamorig:Spine1': 6,

    # hip/leg/thigh
    'LeftHip': 1,
    'LeftUpLeg': 1,
    'LeftLeg':  1,  # 4,
    # 'lButtock': 1,
    'lThigh': 1,
    'mixamorig:LeftUpLeg': 1,
    'RightHip': 2,
    # 'rButtock': 2,
    'rThigh': 2,
    'RightLeg': 2,  # 5,
    'RightUpLeg': 2,
    'mixamorig:RightUpLeg': 2,

    # knee
    'LeftKnee': 4,
    'LeftLowLeg': 4,
    'lShin': 4,
    'mixamorig:LeftLeg': 4,
    'RightKnee': 5,
    'RightLowLeg': 5,
    'rShin': 5,
    'mixamorig:RightLeg': 5,

    # ankle/foot/heel
    'LeftAnkle': 7,
    'LeftFoot': 7,
    'lFoot': 7,
    'mixamorig:LeftFoot': 7,
    'RightAnkle': 8,
    'RightFoot': 8,
    'rFoot': 8,
    'mixamorig:RightFoot': 8,

    # toes
    'LeftToeBase': 10,
    'mixamorig:LeftToeBase': 10,
    'mixamorig:RightToeBase': 11
}


################################################################
# play the BVH file
###############################################################
def playBVH(camera, pose_ch, renderer, motion_file_path, video_file_path=None, w = 0, h = 0):

    global bvh2smpl_jtmap

    with open(motion_file_path) as f:
        mocap = Bvh(f.read())

    joints = mocap.get_joints_names()
    print(joints)
    #print('# of joints:', len(joints), joints)
    #print('# of frames:', mocap.nframes)

    if video_file_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        vout = cv2.VideoWriter(video_file_path, fourcc, 20.0, (w, h))
        print('VIDEO_QUAL:', vout.get(1))  # VIDEOWRITER_PROP_QUALITY 
        vout.set(1, 1.0) 
        print('VIDEO_QUAL:', vout.get(1))  # VIDEOWRITER_PROP_QUALITY 

    plt.ion()
    pose = np.zeros([24, 3])
    #allchannel = [0, 1, 2]
    rotchannels = ['Xrotation', 'Yrotation', 'Zrotation']
    poschannels = ['Xposition', 'Yposition', 'Zposition']
    for bvhjoint in joints:
        print(bvhjoint, end='')
    print('')

    #pose[0][0] = np.pi
    """img = poseSMPL(smpl, pose, True)
    if bSave:
        vout.write((img * 255.0).astype('uint8'))"""

    plt.suptitle('BVH player' )
    camera.t[1] = camera.t[1].r - 0.05*20  
    camera.t[2] = camera.t[2].r + 0.1*200  
    # for fn in range(100):  # mocap.nframes):
    for fn in range(200): # mocap.nframes):
        print('fn:', fn)
        # get new pose
        pose[:, :] = 0.0  # reset
        for bvhjoint in joints:

            try:
                if not (bvhjoint in bvh2smpl_jtmap):
                    continue

                # 1.read angles and convert to radian
                angs = mocap.frame_joint_channels(fn, bvhjoint, rotchannels)
                smpljt = bvh2smpl_jtmap[bvhjoint]
                # pose[smpljt, :] = (math.radians(angs[0]), math.radians(angs[1]), math.radians(angs[2]))
                # comp_pose = (math.radians(angs[0]), math.radians(angs[1]), math.radians(angs[2]))
                angles = (math.radians(angs[0]), math.radians(
                    angs[1]), math.radians(angs[2]))
                angles = np.array(angles)

                # 2. euler to rotation and again to rodrigues
                rot_mat = eulerAnglesToRotationMatrixZXY(angles)
                rot_vec = cv2.Rodrigues(rot_mat)
                #rot_vec = cv2.Rodrigues(angles)
                # rot_mat = cv2.Rodrigues(rot_vec)

                # 3. set the new angles for the joint
                pose[smpljt, :] = np.squeeze(rot_vec[0])


                if bvh2smpl_jtmap[bvhjoint] == 0:
                    position = mocap.frame_joint_channels(fn, bvhjoint, poschannels)
                    #print(position)
                    if fn == 0: 
                        position0 = position
                    else:
                        displacement = np.array([position[0] - position0[0], position[1] - position0[1], position[2]-position0[2]])  
                        if False:
                            displacement[0] = 0.0 
                            displacement[1] = -displacement[1]/200 
                            displacement[2] = displacement[2]/600.
                            #print(displacement)
                        else:
                            displacement[0] = 0.0 
                            displacement[1] = -fn/10000.  # y  
                            displacement[2] = fn/900.  # z
                        camera.t = camera.t.r - displacement 

            except Exception as err:
                print(err)
                continue

        # for check
        '''
        for bvhjoint in joints:
            try:
                if not (bvhjoint in bvh2smpl_jtmap):
                    continue
                degs = np.degrees(pose[bvh2smpl_jtmap[bvhjoint], :])
                print('%-10s' % bvhjoint, '%+03.0f,%+03.0f,%+03.0f' %
                      (degs[0], degs[1], degs[2]))
            except Exception as err:
                print(err)
                continue
        '''

        # update posed
        pinningRoot = True
        if pinningRoot:
            pose[0, :] = (np.pi, 0., 0.)  # to make front side showing
            pose[15, :] = (np.pi/6.0, 0., 0.)  # head 

        pose_ch[:] = pose.flatten()[:] 

        # camera control 
        #camera.t[1] = camera.t[1].r + 0.006
        #camera.t[2] = camera.t[2].r - 0.095

        plt.imshow(renderer.r)
        plt.title('fn=%4d' % fn)
        plt.axis('off')
        plt.draw()
        plt.pause(1.0e-4)

        if video_file_path is not None:
            vout.write((renderer.r[:,:,::-1]*255.0).astype('uint8'))
            if fn == 150:
                sample_file_path = video_file_path.replace('mp4', 'png') # mp4 or avi ext to png 
                sample_file_path = sample_file_path.replace('avi', 'png')
                print('img:', sample_file_path)
                cv2.imwrite(sample_file_path, (renderer.r[:,:,::-1]*255.0).astype('uint8'))

        # _ = raw_input()

    if video_file_path is not None:
        vout.release()


if __name__ == '__main__':

    if len(sys.argv) < 5:
       print('usage: %s base_path dataset idx bvhfile [videofile]'% sys.argv[0]), exit()

    # 1. directory check and setting
    base_dir = abspath(sys.argv[1])
    #print(base_dir)
    dataset = sys.argv[2]
    idx_s = int(sys.argv[3])

    bvh_file_path = sys.argv[4]
    video_file_path = None
    if len(sys.argv) > 5:
        video_file_path = sys.argv[5]

    if not exists(base_dir):
        print('No such a directory for base', base_path, base_dir), exit()

    # input Directory: image 
    inp_dir = base_dir + "/images/" + dataset
    if not exists(inp_dir):
        print('No such a directory for dataset', data_set, inp_dir), exit()

    # input directory: preproccesed
    data_dir = base_dir + "/results/" + dataset 
    print(data_dir)
    smpl_param_dir = data_dir + "/smpl"
    if not exists(smpl_param_dir):
        print('No such a directory for smpl param', smpl_param_dir), exit()
    mask_dir = data_dir + "/segmentation"
    if not exists(mask_dir):
        print('No such a directory for mask', mask_dir), exit()

    if not exists(bvh_file_path):
        print('No such file:', bvh_file_path), exit()

    # Output Directory 
    vton_dir = data_dir + "/vton"
    if not exists(vton_dir):
        makedirs(vton_dir)

    # 2. Loading SMPL models (independent from dataset)
    use_neutral = False
    # Assumes 'models' in the 'code/' directory where this file is in.
    MODEL_DIR = join(abspath(dirname(__file__)), 'models')
    MODEL_NEUTRAL_PATH = join(
        MODEL_DIR, 'basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')
    MODEL_FEMALE_PATH = join(
        MODEL_DIR, 'basicModel_f_lbs_10_207_0_v1.0.0.pkl')
    MODEL_MALE_PATH = join(MODEL_DIR,
                           'basicmodel_m_lbs_10_207_0_v1.0.0.pkl')

    if not use_neutral:
        # File storing information about gender 
        with open(join(data_dir, dataset + '_gender.csv')) as f:
            genders = f.readlines()
        model_female = load_model(MODEL_FEMALE_PATH)
        model_male = load_model(MODEL_MALE_PATH)
    else:
        gender = 'neutral'
        smpl_model = load_model(MODEL_NEUTRAL_PATH)

    #_examine_smpl(model_female), exit()

    vton_path = vton_dir + '/%04d.png'%(idx_s)

    # Load joints
    estj2d = np.load(join(data_dir, 'est_joints.npz'))['est_joints']
    #print('est_shape:', est.shape)
    # for i in range(1, 2):
    # if not use_neutral:
    #    gender = 'male' if int(genders[i]) == 0 else 'female'
    #    if gender == 'female':
    smpl_model = model_female

    #### SOURCE 
    joints_s= estj2d[:2, :, idx_s].T
    smpl_param_path_s = smpl_param_dir + '/%04d.pkl'%idx_s 
    inp_path_s = inp_dir + '/dataset10k_%04d.jpg'%idx_s 
    #mask_path = data_dir + '/segmentation/10kgt_%04d.png'%idx
    mask_path_s = mask_dir + '/10kgt_%04d.png'%idx_s
    source  = { 'id': idx_s,
                'params':  smpl_param_path_s,
                'img'  :  inp_path_s,
                'mask' :  mask_path_s,
                'joints':  joints_s}

    img_bg = cv2.imread('background.jpg')
    if img_bg is None:
        print("cannot open",  'backgorund.jpg'), exit()
    img_bg = img_bg.astype('float32')/255.0

    # 1. image to 3d model
    camera, pose, renderer = reverse_rendering(smpl_model, source, img_bg)

    # 2. play the vbhfile 
    if video_file_path is not None:
        playBVH(camera, pose, renderer, bvh_file_path,  video_file_path, 400, 600)
    else:
        playBVH(camera, pose, renderer, bvh_file_path)



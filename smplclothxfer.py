'''
 Image to SMPL to Transfer 
 -----------------------------

 (c) copyright 2019 heejune@seoultech.ac.kr

 Prerequisite: SMPL model 
 In : SMPL paramters(cam, shape, pose) for a image 
      Mask (cloth and body labeled)
      [optionally the input image]
 Out: label array and vertices 3D coordinates array 
      (i.e., the labeled 3-D model for image)
      [optionally the validating images]



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
from opendr.renderer import TexturedRenderer

_LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from smpl2cloth import smpl2cloth3dcore
from smpl2cloth import build_texture_renderer 
from graphutil import  build_labelmask2 

# implimentation plan
def  _plan():

    print(' The implementation plan')
    print('input: smpl,  {smpl paramter cloth vertices vectors, vertices2cloth-label} for source and target')
    #print('1) calculate the dvertices = cloth - body): dvertices: source')
    print('1) paint the faces and skin : texture of original 2D images')
    print('2) apply the source pose and shape to target body, for easy to wear source cloth') 
    print('3) put on the source cloth to taget body')
    print('4) chnage the body shape and pose back to original')
    print('5) apply the pose (maybe not shpae) change to the coth too')
    print('6) handle the dis-occluded part of cloth using in-shop cloth')
    print('7) project 3D model into 2D')

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


#
#
#
#def build_texture_renderer(U, V, f, vt, ft, texture, w, h, ambient=0.0, near=0.5, far=20000):
#
#    A = SphericalHarmonics(vn=VertNormals(v=V, f=f),
#                           components=[0., 0., 0., 0., 0., 0., 0., 0., 0.],
#                           light_color=ch.ones(3)) + ambient
##
#    R = TexturedRenderer(vc=A, camera=U, f=f, bgcolor=[0.0, 0.0, 0.0],
#                         texture_image=texture, vt=vt, ft=ft,
##                         frustum={'width': w, 'height': h, 'near': near, 'far': far})
#
#    return R
#
#


#
# Visualize the surface vertices  with simple color 
#
def render_model_mine(vertices, faces, w, h, cam,  near = 0.5, far = 25) :

    from opendr.renderer import ColoredRenderer
    from opendr.lighting import LambertianPointLight
    from opendr.camera import ProjectPoints

    rn = ColoredRenderer()

    # 1. camera setup 
    rn.camera = cam
    rn.frustum = {'near': near, 'far': far, 'width': w, 'height': h}

    # 2. Construct point light source: random reflection  (0.7) and ambient (0.3)
    rn.vc = LambertianPointLight(
                f=faces,
                #v=rn.v,
                v=vertices,
                num_verts=len(vertices),
                light_pos=np.array([-1000,-1000,-2000]),
                vc=np.ones_like(vertices)*.9,
                light_color=np.array([1., 1., 1.]))  #+ .3

    rn.vc =  rn.vc + LambertianPointLight(
                f=faces,
                #v=rn.v,
                v=vertices,
                num_verts=len(vertices),
                light_pos=np.array([1000,1000,2000]),
                vc=np.ones_like(vertices)*.9,
                light_color=np.array([1., 1., 1.]))  #+ .3

    rn.set(v=vertices, f=faces, bgcolor=np.zeros(3))

    print("V:",  vertices.shape)
    print("F:",  faces.shape)

    return rn.r

#
#
# success with Chumpy model, which is better and works ^^, Bravo!!
#
#  m            :  default smpl (kintreetable, weights) FIXME: can we use the smpl model posed?
#  j_transformed:  joint locations (with c_pose)
#  cam          :  camera parameters 
#  c_pose       :  current human pose 
#  t_pose       :  target human pose 
#  c_vertices   :  current vertices (of cloth) 
#
def buildSrcClothModel(m, j_transformed, cam, c_vertices, c_pose, t_pose, w, h, near, far, vt, ft, texture):

    plt.figure()
    viz = True
    if viz:
        #print('cam.r.shape:', cam.r.shape)
        #print('m.f.shape:', m.f.shape)
        img1 = (render_model(c_vertices, m.f, w, h, cam) * 255.).astype('uint8')
        #img1 = (render_model_mine(c_vertices, m.f, w, h, cam) * 255.).astype('uint8')

        #texture_renderer = build_texture_renderer(cam, c_vertices, m.f, vt, ft,
        #      texture[::-1, :, :], w, h, 1.0, near=0.5, far=25)
        #img1 = texture_renderer.r

        plt.subplot(1, 4, 1)
        plt.imshow(img1)
        plt.title('cloth vertices')

    c_pose_np = np.array(c_pose) #[3:])
    #print_pose(c_pose_np)
    #print(c_pose_np)

    # 2. build a source cloth model with Chumpy (which is easier to operating and ... works ok ^^)
    # 1. get srequired parameters from SMPL
    #pose =  ch.array(c_pose)  # original pose 
    pose = ch.zeros(m.pose.size) # from now on the current pose is default pose, FIMXE 
    #pose[0] = np.pi 
    #pose[1] = np.pi 
    #pose =  ch.array(m.pose.r)
    weights = ch.array(m.weights.r)
    J_transformed = ch.array(j_transformed) 
    print('Transformed.shape:', j_transformed.shape)
    v_posed = ch.array(c_vertices)
    v_cur, jtr = verts_core(  pose = pose,
                            v    = v_posed,
                            J    = J_transformed,
                            weights = weights,
                            kintree_table = m.kintree_table,
                            xp = ch, #ch vs np
                            want_Jtr = True,
                            bs_style = 'lbs')
    if viz:
        #img2 = (render_model_mine(v_cur.r, m.f, w, h, cam) * 255.).astype('uint8')
        #img2 = (render_model(v_cur.r, m.f, w, h, cam) * 255.).astype('uint8')
        texture_renderer = build_texture_renderer(cam, v_cur, m.f, vt, ft,
              texture[::-1, :, :], w, h, 1.0, near=0.5, far=25)  # OpenDR formula
        img2 = (texture_renderer.r*255.0).astype('uint8')
        cam.v = jtr 
        j2d = cam.r.copy()
        for i in range(j2d.shape[0]):
            cv2.drawMarker(img2, (int(j2d[i,0]), int(j2d[i,1])), (255,0,0), cv2.MARKER_DIAMOND, 5, 3)
        plt.subplot(1,4,2)
        plt.imshow(img2)
        plt.title('LBS model')

    # 3. test it works
    # 3.1 back to the default pose
    #print(type(c_pose)) # it is tuple, so have to convert np for use
    #print('c_pose:', c_pose)
    cam.v = v_cur
    pose[3:] =  - c_pose_np[3:]
    #print_pose(pose.r)
    if viz:
        img3 = (render_model(v_cur.r, m.f, w, h, cam) * 255.).astype('uint8')
        #img3 = (render_model_mine(v_cur.r, m.f, w, h, cam) * 255.).astype('uint8')
        #texture_renderer = build_texture_renderer(cam, v_cur.r, m.f, vt, ft,
        #      texture[::-1, :, :], w, h, 1.0, near=0.5, far=25)
        img3 = (texture_renderer.r*255.0).astype('uint8')
        cam.v = jtr 
        j2d = cam.r.copy()
        for i in range(j2d.shape[0]):
            cv2.drawMarker(img3, (int(j2d[i,0]), int(j2d[i,1])), (255,0,0), cv2.MARKER_DIAMOND, 5, 3)
        plt.subplot(1,4,3)
        plt.imshow(img3)
        plt.title('Back to Rest')

    # 3.2 target (random) pose
    print('s-pose:', c_pose)
    print('t-pose:', t_pose)

    if True: # random pose
        pose[3:] = t_pose[3:] 
    else: 
        pose[3:] = (np.random.rand(m.pose.size -3) -0.5) * .4
    #print_pose(pose.r)
    cam.v = v_cur
    if viz:
        #img4 = (render_model(v_cur.r, m.f, w, h, cam) * 255.).astype('uint8')
        #img4 = (render_model_mine(v_cur.r, m.f, w, h, cam) * 255.).astype('uint8')
        #texture_renderer = build_texture_renderer(cam, v_cur.r, m.f, vt, ft,
        #      texture[::-1, :, :], w, h, 1.0, near=0.5, far=25)
        img4 = (texture_renderer.r*255.0).astype('uint8')
        cam.v = jtr 
        j2d = cam.r.copy()
        for i in range(j2d.shape[0]):
            cv2.drawMarker(img4, (int(j2d[i,0]), int(j2d[i,1])), (255,0,0), cv2.MARKER_DIAMOND, 5, 3)
        plt.subplot(1,4,4)
        plt.imshow(img4)
        plt.title('To Target pose')
    

    if viz:
        plt.show()

#
#  m            :  default smpl (kintreetable, weights) FIXME: can we use the smpl model posed?
#  j_transformed:  joint locations (with c_pose)
#  t_cam        :  camera parameters of 'target' 
#  s_pose       :  source human pose 
#  t_pose       :  target human pose 
#  s_vertices   :  source vertices (of cloth) 
#  w, h, near, far, vt, ft, texture : from sources
#
def buildTgtClothModel(m, j_transformed, t_cam, s_vertices, s_pose, t_pose, w, h, near, far, vt, ft, texture):

    plt.figure()
    viz = True
    if viz:
        #print('cam.r.shape:', cam.r.shape)
        #print('m.f.shape:', m.f.shape)
        img1 = (render_model(s_vertices, m.f, w, h, t_cam) * 255.).astype('uint8')
        #img1 = (render_model_mine(c_vertices, m.f, w, h, cam) * 255.).astype('uint8')
        #texture_renderer = build_texture_renderer(cam, c_vertices, m.f, vt, ft,
        #      texture[::-1, :, :], w, h, 1.0, near=0.5, far=25)
        #img1 = texture_renderer.r

        plt.subplot(1, 4, 1)
        plt.imshow(img1)
        plt.title('cloth vertices')

    s_pose_np = np.array(s_pose) 
    #print_pose(c_pose_np)
    # 2. build a source cloth model with Chumpy (which is easier to operating and ... works ok ^^)
    # 1. get srequired parameters from SMPL
    #pose =  ch.array(c_pose)  # original pose 
    pose = ch.zeros(m.pose.size) # from now on the current pose is default pose, FIMXE 
    weights = ch.array(m.weights.r)
    J_transformed = ch.array(j_transformed) 
    print('Transformed.shape:', j_transformed.shape)
    v_posed = ch.array(s_vertices)
    v_cur, jtr = verts_core(  pose = pose,
                            v    = v_posed,
                            J    = J_transformed,
                            weights = weights,
                            kintree_table = m.kintree_table,
                            xp = ch, #ch vs np
                            want_Jtr = True,
                            bs_style = 'lbs')


    if viz:
        #img2 = (render_model_mine(v_cur.r, m.f, w, h, cam) * 255.).astype('uint8')
        #img2 = (render_model(v_cur.r, m.f, w, h, cam) * 255.).astype('uint8')
        t_cam.v = v_cur 
        texture_renderer = build_texture_renderer(t_cam, v_cur, m.f, vt, ft,
              texture[::-1, :, :], w, h, 1.0, near=0.5, far=25)  # OpenDR formula
        img2 = (texture_renderer.r*255.0).astype('uint8')
        t_cam.v = jtr 
        j2d = t_cam.r.copy()
        for i in range(j2d.shape[0]):
            cv2.drawMarker(img2, (int(j2d[i,0]), int(j2d[i,1])), (255,0,0), cv2.MARKER_DIAMOND, 5, 3)
        plt.subplot(1,4,2)
        plt.imshow(img2)
        plt.title('LBS model')

    # 3. test it works
    # 3.1 back to the default pose
    #print(type(c_pose)) # it is tuple, so have to convert np for use
    #print('c_pose:', c_pose)
    t_cam.v = v_cur
    pose[3:] =  - s_pose_np[3:]
    #print_pose(pose.r)
    if viz:
        img3 = (render_model(v_cur.r, m.f, w, h, t_cam) * 255.).astype('uint8')
        #img3 = (render_model_mine(v_cur.r, m.f, w, h, cam) * 255.).astype('uint8')
        #texture_renderer = build_texture_renderer(cam, v_cur.r, m.f, vt, ft,
        #      texture[::-1, :, :], w, h, 1.0, near=0.5, far=25)
        img3 = (texture_renderer.r*255.0).astype('uint8')
        t_cam.v = jtr 
        j2d = t_cam.r.copy()
        for i in range(j2d.shape[0]):
            cv2.drawMarker(img3, (int(j2d[i,0]), int(j2d[i,1])), (255,0,0), cv2.MARKER_DIAMOND, 5, 3)
        plt.subplot(1,4,3)
        plt.imshow(img3)
        plt.title('Back to Rest')

    # 3.2 target (random) pose
    '''
    print('>>> s-pose:')
    print_pose(s_pose)
    print('>>> t-pose:')
    print_pose(t_pose)
    '''
    print_poses(s_pose, t_pose)
    pose[3:] = t_pose[3:] 
    #print_pose(pose.r)
    t_cam.v = v_cur
    if viz:
        #img4 = (render_model(v_cur.r, m.f, w, h, cam) * 255.).astype('uint8')
        #img4 = (render_model_mine(v_cur.r, m.f, w, h, cam) * 255.).astype('uint8')
        #texture_renderer = build_texture_renderer(cam, v_cur.r, m.f, vt, ft,
        #      texture[::-1, :, :], w, h, 1.0, near=0.5, far=25)
        img4 = (texture_renderer.r*255.0).astype('uint8')
        t_cam.v = jtr 
        j2d = t_cam.r.copy()
        for i in range(j2d.shape[0]):
            cv2.drawMarker(img4, (int(j2d[i,0]), int(j2d[i,1])), (255,0,0), cv2.MARKER_DIAMOND, 5, 3)
        plt.subplot(1,4,4)
        plt.imshow(img4)
        plt.title('To Target pose')

    if viz:
        plt.show()


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




#
# combined parts with mask into single image
#
def combineParts(body_pair, model_pair, cloth_pair):


    # preparing the background and model  
    #############################################
    # 1. empty image
    model_img  = model_pair[0]
    model_mask = model_pair[1]
    img = np.zeros_like(model_img)  # .shape, dtype ='uint8') 

    # 2. target SMPL body with skin color
    '''
    body_img  = body_pair[0]
    body_mask = body_pair[1]
    img[body_mask != 0] = bottom_img[body_mask != 0]
    '''
    # TODO: skin color 
    skin_color_b = np.mean(model_img[model_mask == 11, 0])
    skin_color_g = np.mean(model_img[model_mask == 11, 1])
    skin_color_r = np.mean(model_img[model_mask == 11, 2])

    print('skin color:', skin_color_r, skin_color_g, skin_color_b)

    img[:,:,0] = int(skin_color_r) 
    img[:,:,1] = int(skin_color_g) 
    img[:,:,2] = int(skin_color_b)

    # 3. All excluding the clothes to be replaced, background included 
    #img = model_pair[0][:,:,::-1]
    m  = (model_mask == 0)| (model_mask==1)|(model_mask==2)|(model_mask==3)|(model_mask==9)|(model_mask==10)|(model_mask==11)|(model_mask==12)|(model_mask==13)|(model_mask==14)|(model_mask==15)|(model_mask==16)|(model_mask==17)|(model_mask==18)   
    img[m > 0, :]  = model_img[m >0, ::-1]

    # Dressing Start
    #############################################
    # dress the bottom
    bottom_img  = cloth_pair[0]
    bottom_mask = cloth_pair[1]
    img[bottom_mask == 5] = bottom_img[bottom_mask == 5]
    img[bottom_mask == 6] = bottom_img[bottom_mask == 6]

    # dress the upper  
    upper_img  = cloth_pair[0]
    upper_mask = cloth_pair[1]
    img[upper_mask == 4] = upper_img[upper_mask == 4]


    return img


cloth_label_dict = {"background": 0,
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
            "scarf": 17, #          17
            "skin": 18 #  added for skin region from face
        }

def clothxfer(smpl_model, source,  target, outpath):

    plt.ion()

    params_s, mask_s, img_s = _read_ifiles(source)
    params_t, mask_t, img_t = _read_ifiles(target)
    j2d_s =  source['joints']
    j2d_t =  target['joints']

    #  pre-processing for boundary matching  
    mask_s[mask_s == _clothlabeldict['bag']] = 0  # remove bag
    mask_t[mask_t == _clothlabeldict['bag']] = 0  # remove bag
    # cut the connected legs 
    if idx_s == 0:
        mask_s[500:,190] = 0
    elif idx_s == 1:
        mask_s[500:,220] = 0
    else:
        print('Not prepared Yet for the idx!')
        #exit()

    if idx_t == 0:
        mask_t[500:,190] = 0
    elif idx_t == 1:
        mask_t[500:,220] = 0
    else:
        print('Not prepared Yet for the idx!')
        #exit()

    ###########################################################################
    ##### 1. cloth in the second to the first ####################################
    ###########################################################################

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
                 False)
                 #True)      
    clothv3d = cam_s.v.r  

    # target : clothing is not needed
    cam_t   = params_t['cam']       # camera model, Ch
    betas_t = params_t['betas']
    n_betas_t = betas_t.shape[0]    # tuple of numpy to numpy 
    pose_t  = params_t['pose']      # angles, (23+1)x3 numpy

    h, w = mask_s.shape[:2]

    plt.subplot(1,6,1)
    plt.title('source')
    plt.imshow(img_s[:,:,::-1])
    plt.axis('off')
    plt.draw()
    plt.subplot(1,6,6)
    plt.title('target')
    plt.imshow(img_t[:,:,::-1])
    plt.axis('off')
    plt.draw()

    # mapping the labeling info to cloth vertices  
    v2d_int = np.rint(cam_s.r).astype('int32')
    v2label_s = mask_s[v2d_int[:,1], v2d_int[:,0]]

    # 2. restore default posed vertices 
    vertices, jtr = restorePose(smpl_model, clothv3d, joint_transformed, pose_s)  
    # 3 build model
    v_cur, jtr_cur, pose = buildLBS(smpl_model, vertices, jtr)

    # 4 build texture renderer 
    cam_t.v = v_cur 
    texture_renderer = build_texture_renderer(cam_t, v_cur, smpl_model.f, vt, ft, texture[::-1, :,:], w, h, 1.0, near=0.5, far = 25.)  
    img = (texture_renderer.r*255.0).astype('uint8')
    # joint display 
    cam_t.v = jtr_cur.r  
    j2d = cam_t.r.copy()
    for i in range(j2d.shape[0]):
        cv2.drawMarker(img, (int(j2d[i,0]), int(j2d[i,1])), (255,0,0), cv2.MARKER_DIAMOND, 5, 3)
    plt.subplot(1,6,2)
    plt.imshow(img[::-1,:,:])  # flip for easy to see
    plt.title('Rest Posed')
    plt.axis('off')
    plt.draw()

    # 5 transfer pose  
    cam_t.v = v_cur  
    pose[:] = pose_t[:]
    # 6 visualize 
    img_tmp = (texture_renderer.r*255.0).astype('uint8')
    # joint display 
    cam_t.v = jtr_cur.r  
    j2d = cam_t.r.copy()
    for i in range(j2d.shape[0]):
        cv2.drawMarker(img_tmp, (int(j2d[i,0]), int(j2d[i,1])), (255,0,0), cv2.MARKER_DIAMOND, 5, 3)
    plt.subplot(1,6,3)
    plt.imshow(img_tmp)
    plt.title('Target Posed')
    plt.axis('off')
    plt.draw()

    # 4. Merging the projected results or in 3D   
    # 4.1 checking segmented clothes 
    max_cloth_label = len(cloth_label_dict)
    #label_mask1 = graphutil.build_labelmask(clothv3d, model.f, v2label, None, cam,  height=h, width=w, near= 0.5, far= 40)
    reposed_cloth_mask = build_labelmask2(v_cur.r, smpl_model.f, v2label_s, cam_t,  height=h, width=w, near= 0.5, far= 40)
    # TODO  use the texturebasedlabelmask
    #reposed_cloth_mask = build_labelmask_texture(v_cur.r, smpl_model.f, v2label_s, cam_t,  
    # vt, ft, mask, height=h, width=w, near= 0.5, far= 40)
    mask_rgb = cv2.cvtColor(mask_s, cv2.COLOR_GRAY2BGR)  
    mask_rgb_float = mask_rgb.astype('float32')/255.0
    mask_renderer = build_texture_renderer(cam_t, v_cur, smpl_model.f, vt, ft, mask_rgb_float[::-1, :,:], w, h, 1.0, near=0.5, far = 40.)  
    reposed_cloth_mask = (mask_renderer.r *255.0).astype('uint8')
    reposed_cloth_mask = cv2.cvtColor(reposed_cloth_mask, cv2.COLOR_BGR2GRAY)  

    plt.subplot(1,6,4)
    plt.imshow(reposed_cloth_mask)
    plt.axis('off')
    plt.title('reposed mask')
    plt.draw()

    # merge parts into one image
    img_cloth = (texture_renderer.r*255.0).astype('uint8')
    img_combined = combineParts( None,  [img_t, mask_t] , [img_cloth, reposed_cloth_mask])
    plt.subplot(1,6,5)
    plt.imshow(img_combined)
    plt.axis('off')
    plt.title('cloth transferred')
    plt.draw()
    plt.show()
    #_ = raw_input('next?')


    ###########################################################################
    ##### 2. cloth in the second to the first ####################################
    ###########################################################################
    '''
    joint_transformed, vt, ft, texture = smpl2cloth3dcore(cam_t,      # camera model, Ch
                 betas_t,    # shape coeff, numpy
                 n_betas_t,  # num of PCA
                 pose_t,     # angles, 27x3 numpy
                 img_t,    # img numpy
                 mask_t,     # mask 
                 j2d_t, 
                 smpl_model,
                 False)
                 #True)     

    plt.subplot(1,4,1)
    plt.title('source')
    plt.imshow(img_t[:,:,::-1])
    plt.axis('off')
    plt.draw()
    plt.subplot(1,4,4)
    plt.title('target')
    plt.imshow(img_s[:,:,::-1])
    plt.axis('off')
    plt.draw()

    # 2. restore default posed vertices 
    vertices, jtr = restorePose(smpl_model, cam_t.v.r, joint_transformed, pose_t)  
    # 3 build model
    v_cur, jtr_cur, pose = buildLBS(smpl_model, vertices, jtr)

    # 4 build texture renderer 
    cam_t.v = v_cur 
    texture_renderer = build_texture_renderer(cam_t, v_cur, smpl_model.f, vt, ft, texture[::-1, :,:], w, h, 1.0, near=0.5, far = 25.)  
    img = (texture_renderer.r*255.0).astype('uint8')
    # joint display 
    cam_t.v = jtr_cur.r  
    j2d = cam_t.r.copy()
    for i in range(j2d.shape[0]):
        cv2.drawMarker(img, (int(j2d[i,0]), int(j2d[i,1])), (255,0,0), cv2.MARKER_DIAMOND, 5, 3)
    plt.subplot(1,4,2)
    plt.title('Rest Posed')
    plt.imshow(img[::-1,:,:])  # flip for easy to see
    plt.axis('off')
    plt.draw()

    # 5 re-pose
    cam_t.v = v_cur  
    pose[:] = pose_s[:]
    # 6 visualize 
    img = (texture_renderer.r*255.0).astype('uint8')
    # joint display 
    cam_t.v = jtr_cur.r  
    j2d = cam_t.r.copy()
    for i in range(j2d.shape[0]):
        cv2.drawMarker(img, (int(j2d[i,0]), int(j2d[i,1])), (255,0,0), cv2.MARKER_DIAMOND, 5, 3)
    plt.subplot(1,4,3)
    plt.imshow(img)
    plt.title('Target Posed')
    plt.axis('off')
    plt.draw()
    plt.show()
    _ = raw_input()


    # 4. Merging the projected results or in 3D   
    # TODO

    '''


    ##########################################
    # 3.2 save result for checking
    '''
    if outimg_path is not None:
        plt.savefig(outimg_path)
        if img_out is not None:
            cv2.imwrite(outimg_path, img_out)

    '''

if __name__ == '__main__':

    if len(sys.argv) < 5:
       print('usage: %s base_path dataset srcidx targetidx'% sys.argv[0]), exit()

    # 1. directory check and setting
    base_dir = abspath(sys.argv[1])
    #print(base_dir)
    dataset = sys.argv[2]
    idx_s = int(sys.argv[3])
    idx_t = int(sys.argv[4])

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

    vton_path = vton_dir + '/%04d_%04d.png'%(idx_s,idx_t)

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
    #### TARGET 
    joints_t= estj2d[:2, :, idx_t].T
    smpl_param_path_t = smpl_param_dir + '/%04d.pkl'%idx_t 
    inp_path_t = inp_dir + '/dataset10k_%04d.jpg'%idx_t 
    #mask_path = data_dir + '/segmentation/10kgt_%04d.png'%idx
    mask_path_t = mask_dir + '/10kgt_%04d.png'%idx_t
    target  = { 'id': idx_t,
                'params':  smpl_param_path_t,
                'img'  :  inp_path_t,
                'mask' :  mask_path_t,
                'joints':  joints_t}

    clothxfer(smpl_model, source, target, vton_path)

    # plt.pause(10)
    _ = raw_input('quit?')





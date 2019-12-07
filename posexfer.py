'''
  SMPL pose Transfer Demo
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
import math
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

def euler2rotmat(theta) :
     
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
         
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
                 
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
                     
                     
    R = np.dot(R_z, np.dot( R_y, R_x ))
 
    return R


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotMat2ulerAngles(R) :

    assert(isRotationMatrix(R))
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])


def _examine_smpl_params(params):

    #print(type(params))
    #print(params.keys())
    #print('camera params')
    #print(" - type:", type(params['cam']))
    #print(" - members:", dir(params['cam']))
    #print(" - cam.t:",params['cam'].t)
    print(" - cam.t:", params['cam'].t.r)    # none-zero, likely only nonzero z
    print(" - cam.rt:", params['cam'].rt.r)  # zero (fixed)
    print(" - cam.camera_mtx:", params['cam'].camera_mtx)  # 
    print(" - cam.k:", params['cam'].k.r)  #

    #    print(params['f'].shape)      # 2
    #print('pose')
    #print(" - type:", type(params['pose']))
    print(' - pose.hape:', params['pose'].shape)   # 72
    #print(' - pose.values:', params['pose'])
    #print('betas')
    #print(' - type:', type(params['betas']))
    print(' - betas.shape:', params['betas'].shape)  # 10
    #print(' - betas.values:', params['betas'])  # 10


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

    np.set_printoptions(precision =2)
    if pose.shape[0]  == 24*3:
      pose = pose.reshape([-1,3])

    if pose.shape[0] == 24 and pose.shape[1] == 3:
        for j in range(24):
            rotmat = cv2.Rodrigues(pose[j,:])[0]
            euler_angle = rotMat2ulerAngles(rotmat) 
            print('%10s'%jointname[j],  'rodrigues:', pose[j,:], 'euler:', euler_angle)
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
# Visualize the surface vertices with joint markers
#
def visualize( vertices, faces, j3d, w = 400, h = 600):

    from opendr.geometry import VertNormals
    from opendr.lighting import LambertianPointLight
    from opendr.lighting import SphericalHarmonics
    from opendr.camera import ProjectPoints
    from opendr.renderer import ColoredRenderer

    ##  Create OpenDR renderer
    # 1. camera setup (opencv camera definiton)
    camera = ProjectPoints(  v= vertices,
                                rt=np.array([0., np.pi, 0.]),
                                t=np.array([0, 0, 2]),  #  in fact, camera is located in z = -2
                                f=np.array([w,w])/1.1, # forcal length, 1 meter making the target takes half width
                                c=np.array([w,h])/2., # pixel offset = center of image
                                k=np.zeros(5))        # no distortion

    ambient = .2
    if False:
        vc = SphericalHarmonics(vn=VertNormals(v=vertices, f=faces),
                           components=[.5, 0., 0., 0., 0., 0., 0., 0., 0.],
                           light_color=ch.ones(3)) + ambient

    else:

        vc = LambertianPointLight(
                f=faces,
                v=vertices,
                num_verts=len(vertices),
                light_pos=np.array([-1000,-1000,-2000]),
                vc=np.ones_like(vertices)*.9,
                light_color=np.array([1., 1., 1.]))  #+ .3

        vc =  vc +  LambertianPointLight(
                f=faces,
                v=vertices,
                num_verts=len(vertices),
                light_pos=np.array([1000,1000,2000]),
                vc=np.ones_like(vertices)*.9,
                light_color=np.array([1., 1., 1.]))  #+ .3

    rn = ColoredRenderer(vc=vc, camera = camera, f = faces, bgcolor = [0., 0., 0.],
                frustum = {'near': 1., 'far': 10., 'width': w, 'height': h})

    # 2. Construct point light source: random reflection  (0.7) and ambient (0.3)


    #rn.vc = ch.ones(vertices.shape)
    print('vc.shape:', rn.vc.shape)
    #rn.set(v=vertices, f=faces, bgcolor=np.zeros(3))

    # 3. project and marking
    img =  rn.r.copy()
    rn.camera.v = j3d
    j2d = rn.camera.r.copy()
    for i in range(j2d.shape[0]):
        cv2.drawMarker(img, (int(j2d[i,0]), int(j2d[i,1])), (255,0,0), cv2.MARKER_DIAMOND, 5, 3)

    return img


# create V, A, U, f: geom, bright, cam, renderer
def build_color_renderer(camera, vertices, faces, w, h, ambient=0.0, near=0.5, far=20):

    from opendr.geometry import VertNormals
    from opendr.lighting import LambertianPointLight
    from opendr.lighting import SphericalHarmonics
    from opendr.camera import ProjectPoints
    from opendr.renderer import ColoredRenderer

    '''
    A = SphericalHarmonics(vn=VertNormals(v=vertices, f=faces),
                           components=[0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           light_color=ch.ones(3)) + ambient
    '''
    vc = LambertianPointLight(
                f=faces,
                v=vertices,
                num_verts=len(vertices),
                light_pos=np.array([-1000,-1000,-2000]),
                vc=np.ones_like(vertices)*.9,
                light_color=np.array([3., 3., .3]))

    vc = vc +  LambertianPointLight(
                f=faces,
                v=vertices,
                num_verts=len(vertices),
                light_pos=np.array([1000,1000,2000]),
                vc=np.ones_like(vertices)*.9,
                light_color=np.array([.3, .3, .3]))  + ambient


    Renderer = ColoredRenderer(vc= vc, camera=camera, f=faces, bgcolor=[0.0, 0.0, 0.0],
                         frustum={'width': w, 'height': h, 'near': near, 'far': far})

    return Renderer

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
    #_examine_smpl_params(params)

    #  2d rgb image for texture
    img2D = cv2.imread(human['img'])
    if img2D is None:
        print("cannot open",  human['img']), exit()

    # segmentation mask 
    mask = cv2.imread(human['mask'], cv2.IMREAD_UNCHANGED)
    if mask is None:
        print("cannot open",  human['mask']), exit()

    return params, mask,  img2D

def checkSMPLModel(smpl_model, cam_s, pose_s):

    # 1. check source SMPL body model (pose)
    cam_s.v = smpl_model # cameera <= smpl vertices

    ##########################################################################################
    # ZERO POSE
    ##########################################################################################
    smpl_model.pose[:] = 0 
    #img1 = visualize(smpl_model.r, smpl_model.f, smpl_model.J_transformed.r)
    #plt.imshow(img1)
    smpl_renderer =  build_color_renderer(cam_s, cam_s.v, smpl_model.f, w = 400, h = 600, ambient=0.2, near=0.5, far=30)
    #plt.imshow(smpl_renderer.r)
    img_t = (smpl_renderer.r*255.0).astype('uint8')
    # to check the 3D vertices directions 
    v2d = cam_s.r.copy()
    v3d = cam_s.v.r.copy()
    x_mid =  np.mean(v3d[:,0])
    x_max =  np.amax(v3d[:,0])
    x_min =  np.amin(v3d[:,0])
    y_mid =  np.mean(v3d[:,1])
    y_max =  np.amax(v3d[:,1])
    y_min =  np.amin(v3d[:,1])
    z_mid =  np.mean(v3d[:,2])
    z_max =  np.amax(v3d[:,2])
    z_min =  np.amin(v3d[:,2])
    print('x mean:', x_mid, 'max:', x_max, 'min:', x_min)
    print('y mean:', y_mid, 'max:', y_max, 'min:', y_min)
    print('z mean:', z_mid, 'max:', z_max, 'min:', z_min)
    plt.subplot(2,4,1)
    for i in range(v3d.shape[0]):
        if v3d[i,2] < 0:  # z < 0, front 
            if v3d[i,1] > y_mid:  # y > 0, bottom, so it is also up-side down 
                if v3d[i,0] > x_mid:  # y > 0, bottom, so it is also up-side down 
                    cv2.drawMarker(img_t,(int(v2d[i,0]), int(v2d[i,1])), (0, 255, 0), markerSize=3) # green
                else:
                    cv2.drawMarker(img_t,(int(v2d[i,0]), int(v2d[i,1])), (0, 0, 0), markerSize=3) # black 
            else:
                 cv2.drawMarker(img_t,(int(v2d[i,0]), int(v2d[i,1])), (0, 0, 255), markerSize=3) # blue
        else:
            pass

    plt.imshow(img_t)
    plt.title('SMPL 0 pose 3D coord')
    plt.draw()

    x2d_mid =  np.mean(v2d[:,0])
    x2d_max =  np.amax(v2d[:,0])
    x2d_min =  np.amin(v2d[:,0])
    y2d_mid =  np.mean(v2d[:,1])
    y2d_max =  np.amax(v2d[:,1])
    y2d_min =  np.amin(v2d[:,1])
    plt.subplot(2,4,2)
    img_t = (smpl_renderer.r*255.0).astype('uint8')
    for i in range(v2d.shape[0]):
        if v3d[i,2] < 0:  # z < 0, front 
            if v2d[i,1] > y2d_mid:  # y > 0, bottom, so it is also up-side down 
                if v2d[i,0] > x2d_mid:  # y > 0, bottom, so it is also up-side down 
                    cv2.drawMarker(img_t,(int(v2d[i,0]), int(v2d[i,1])), (0, 255, 0), markerSize=3) # green
                else:
                    cv2.drawMarker(img_t,(int(v2d[i,0]), int(v2d[i,1])), (0, 0, 0), markerSize=3) # black 
            else:
                 cv2.drawMarker(img_t,(int(v2d[i,0]), int(v2d[i,1])), (0, 0, 255), markerSize=3) # blue
        else:
            pass

    plt.imshow(img_t)
    plt.title('SMPL 0 pose  2D coord')
    plt.draw()

    ##########################################################################################
    # SIMPLIFY  POSE
    ##########################################################################################
    smpl_model.pose[:] = pose_s[:] 

    t = cv2.Rodrigues(pose_s[:3])[0]
    print('body rotmat:', t)
    root_pose_recovered = cv2.Rodrigues(t)[0].ravel()
    print('body rodrigues recovered:', root_pose_recovered)
    smpl_model.pose[:3] = root_pose_recovered[:]  ###### WORKED #############################

    #img1 = visualize(smpl_model.r, smpl_model.f, smpl_model.J_transformed.r)
    #plt.imshow(img1)
    #smpl_renderer =  build_color_renderer(cam_s, cam_s.v, smpl_model.f, w = 400, h = 600, ambient=0.3, near=0.5, far=30)
    #plt.imshow(smpl_renderer.r)
    img_t = (smpl_renderer.r*255.0).astype('uint8')
    # to check the 3D vertices directions 
    v2d = cam_s.r.copy()
    v3d = cam_s.v.r.copy()
    x_mid =  np.mean(v3d[:,0])
    x_max =  np.amax(v3d[:,0])
    x_min =  np.amin(v3d[:,0])
    y_mid =  np.mean(v3d[:,1])
    y_max =  np.amax(v3d[:,1])
    y_min =  np.amin(v3d[:,1])
    z_mid =  np.mean(v3d[:,2])
    z_max =  np.amax(v3d[:,2])
    z_min =  np.amin(v3d[:,2])
    print('x mean:', x_mid, 'max:', x_max, 'min:', x_min)
    print('y mean:', y_mid, 'max:', y_max, 'min:', y_min)
    print('z mean:', z_mid, 'max:', z_max, 'min:', z_min)
    plt.subplot(2,4,3)
    for i in range(v3d.shape[0]):
        if v3d[i,2] < 0:  # z < 0, front 
            if v3d[i,1] > y_mid:  # y > 0, bottom, so it is also up-side down 
                if v3d[i,0] > x_mid:  # y > 0, bottom, so it is also up-side down 
                    cv2.drawMarker(img_t,(int(v2d[i,0]), int(v2d[i,1])), (0, 255, 0), markerSize=3) # green
                else:
                    cv2.drawMarker(img_t,(int(v2d[i,0]), int(v2d[i,1])), (0, 0, 0), markerSize=3) # black 
            else:
                 cv2.drawMarker(img_t,(int(v2d[i,0]), int(v2d[i,1])), (0, 0, 255), markerSize=3) # blue
        else:
            pass

    plt.imshow(img_t)
    plt.title('SMPL-posed 3D coord')
    plt.draw()

    x2d_mid =  np.mean(v2d[:,0])
    x2d_max =  np.amax(v2d[:,0])
    x2d_min =  np.amin(v2d[:,0])
    y2d_mid =  np.mean(v2d[:,1])
    y2d_max =  np.amax(v2d[:,1])
    y2d_min =  np.amin(v2d[:,1])
    plt.subplot(2,4,4)
    img_t = (smpl_renderer.r*255.0).astype('uint8')
    for i in range(v2d.shape[0]):
        if v3d[i,2] < 0:  # z < 0, front 
            if v2d[i,1] > y2d_mid:  # y > 0, bottom, so it is also up-side down 
                if v2d[i,0] > x2d_mid:  # y > 0, bottom, so it is also up-side down 
                    cv2.drawMarker(img_t,(int(v2d[i,0]), int(v2d[i,1])), (0, 255, 0), markerSize=3) # green
                else:
                    cv2.drawMarker(img_t,(int(v2d[i,0]), int(v2d[i,1])), (0, 0, 0), markerSize=3) # black 
            else:
                 cv2.drawMarker(img_t,(int(v2d[i,0]), int(v2d[i,1])), (0, 0, 255), markerSize=3) # blue
        else:
            pass

    plt.imshow(img_t)
    plt.title('SMPL-posed 2D coord')
    plt.draw()


    ###########################################################
    # 2. test angle and lef-right body part mapping
    ###########################################################
    euler_z90 = np.array([0.,         0.,        np.pi/2.0])
    euler_x90 = np.array([np.pi/2.0,  0.,        0.])
    euler_y90 = np.array([0,          np.pi/2.0, 0.])

    rotmat_z45 = euler2rotmat(euler_z90/2.)
    rodrig_z45 = cv2.Rodrigues(rotmat_z45)[0].ravel() # 1 is for Jacobian 
    print('rodrigues :', rodrig_z45)
    # right hip 
    plt.subplot(2,4,5)
    #img_t = (smpl_renderer.r*255.0).astype('uint8')
    #plt.imshow(img_t)
    #plt.draw()

    plt.subplot(2,4,5)
    smpl_model.pose[:] =  0 
    smpl_model.pose[2*3:3*3] = rodrig_z45[:]  # right hip, + 45: outter   
    img_t = (smpl_renderer.r*255.0).astype('uint8')
    plt.imshow(img_t)
    plt.title('0 pose Right HIP, +z45')
    plt.draw()

    plt.subplot(2,4,6)
    smpl_model.pose[:] =  0 
    smpl_model.pose[1*3:2*3] = rodrig_z45[:]  # left hip, + 45:  insider   
    img_t = (smpl_renderer.r*255.0).astype('uint8')
    plt.title('0 pose Left HIP, +z45')
    plt.imshow(img_t)
    plt.draw()

    ############################
    # with Pose 
    ############################
    # right hip 
    plt.subplot(2,4,5)
    #img_t = (smpl_renderer.r*255.0).astype('uint8')
    #plt.imshow(img_t)
    #plt.draw()

    plt.subplot(2,4,7)
    smpl_model.pose[:] =  0 
    smpl_model.pose[:3] = root_pose_recovered[:]  ###### WORKED #############################
    smpl_model.pose[2*3:3*3] = rodrig_z45[:]  # right hip, + 45: outter   
    img_t = (smpl_renderer.r*255.0).astype('uint8')
    plt.imshow(img_t)
    plt.title('Posed, Right HIP, +z45')
    plt.draw()

    plt.subplot(2,4,8)
    smpl_model.pose[:] =  0 
    smpl_model.pose[:3] = root_pose_recovered[:]  ###### WORKED #############################
    smpl_model.pose[1*3:2*3] = rodrig_z45[:]  # left hip, + 45:  insider   
    img_t = (smpl_renderer.r*255.0).astype('uint8')
    plt.imshow(img_t)
    plt.title('Posed, Left HIP, +z45')
    plt.draw()

    plt.show()

###############################################################################
# LBS model with Chumpy
# 
# singlestep method test for multijoint case   
###############################################################################
def checkLBSModel(smpl_model, cam_s, pose_s, cam_t, pose_t, synthetic, Flip, img_s = None, img_t = None):

    if synthetic: 
        numFigures = 4
        startFigure = 0
    else:
        numFigures = 6
        startFigure = 1
        plt.subplot(1, numFigures, 1)
        plt.imshow(img_s)
        plt.title('source image')
        plt.subplot(1, numFigures, numFigures)
        plt.imshow(img_t)
        plt.title('Target image')


    # constant terms 
    weights = ch.array(smpl_model.weights.r)
    kintree_table = smpl_model.kintree_table.copy()

    # 1. set the source pose 
    if Flip:
        smpl_model.pose[:] = pose_s[:]
    else:
        smpl_model.pose[:] = np.zeros(pose_s.shape) 
        smpl_model.pose[3:] = pose_s[3:]
   
    if synthetic:
        euler_z90 = np.array([0.,           0.,  np.pi/2.0])
        euler_x90 = np.array([np.pi/2.0,    0.,  0.0])
        euler_y90 = np.array([0,     np.pi/2.0,  0.0])
        rotmat_z45 = euler2rotmat(euler_z90/2.)
        rotmat_x45 = euler2rotmat(euler_x90/2.)
        rotmat_y45 = euler2rotmat(euler_y90/2.)
        rotmat_z90 = euler2rotmat(euler_z90/1.)
        rotmat_x90 = euler2rotmat(euler_x90/1.)
        rotmat_y90 = euler2rotmat(euler_y90/1.)
        rodrig_z45 = cv2.Rodrigues(rotmat_z45)[0].ravel() # 1 is for Jacobian 
        rodrig_x45 = cv2.Rodrigues(rotmat_x45)[0].ravel() # 1 is for Jacobian 
        rodrig_y45 = cv2.Rodrigues(rotmat_y45)[0].ravel() # 1 is for Jacobian 
        rodrig_z90 = cv2.Rodrigues(rotmat_z90)[0].ravel() # 1 is for Jacobian 
        rodrig_x90 = cv2.Rodrigues(rotmat_x90)[0].ravel() # 1 is for Jacobian 
        rodrig_y90 = cv2.Rodrigues(rotmat_y90)[0].ravel() # 1 is for Jacobian 
        smpl_model.pose[4:] =  0 
        #smpl_model.pose[1*3:2*3] = rodrig_z45[:]  # left hip, + 45: outsider   
        smpl_model.pose[16*3:17*3] = - rodrig_z90[:]  # lsh, - 45:   outsider   
        smpl_model.pose[18*3:19*3] = - rodrig_y90[:]  # lelbow, - 45:     
        smpl_model.pose[20*3:21*3] = - rodrig_z90[:]  # lwrist, - 45:    

    # 2. get the vertices and paramters 
    v_posed = ch.array(smpl_model.r)
    J = ch.array(smpl_model.J_transformed.r)

    pose =  ch.zeros(smpl_model.pose.size)
    #pose =  ch.array(pose_s)
    v_cur, jtr = verts_core(  pose = pose,
                            v    = v_posed,
                            J    = J,
                            weights = weights,
                            kintree_table = kintree_table,
                            xp = ch, #ch vs np
                            want_Jtr = True,
                            bs_style = 'lbs')
    # 3. Renderer 
    cam_s.v = v_cur
    Renderer = build_color_renderer(cam_s, cam_s.v, smpl_model.f, w = 400, h = 600, ambient = .2, near=0.5, far = 25.) 


    #img2 = visualize(cam_s, v_cur.r,  smpl_model.f, jtr.r) # mark the joints
    plt.subplot(1,numFigures,startFigure + 1)
    #plt.imshow(img2)
    plt.imshow(Renderer.r) #.copy()[::-1,:,:])
    plt.title('Source pose')

    ##########################################################################
    # 3. back to the default pose
    ##########################################################################
    if not Flip:
        if not synthetic:
            pose[3:] = - pose_s[3:] 
        else:
            #pose[1*3:2*3] = - rodrig_z45[:]  # left hip, + 45:   outsider   

            ##############################################################
            # important ####################################################
            ##############################################################
            # first rotation joint can use 'negative forward angle' 
            pose[16*3:17*3] =  rodrig_z90[:]  # lshoulder:  pose-z => pose+z    
            # elbow use +x90 not +y90, which is the rotation axis for forward pose
            ################################################################
            pose[18*3:19*3] =  rodrig_x90[:]  # lelbow: pose-y => pose+x   
            pose[20*3:21*3] =  rodrig_y90[:]  # lwrist: pose-x => pose+y   

            # I don't know why it doesnot work !!
            '''
            rot = np.matmul(rotmat_y90,rotmat_z90)
            print('rot    :', rot)
            print('rot_x90:', rotmat_x90)
            rodrig = cv2.Rodrigues(rot)[0].ravel()
            pose[18*3:19*3] =  rodrig[:]  # lelbow, + 45:  outsider   
            '''
    else:
        pose_tmp = np.zeros(smpl_model.pose.size)
        if not synthetic:
            #pose_tmp[0:3] = pose_s[0:3]
            for i in range(1, 24):
                pose_tmp[i*3 + 0] = -  pose_s[i*3 + 0]
                pose_tmp[i*3 + 1] = + pose_s[i*3 + 1]
                pose_tmp[i*3 + 2] = + pose_s[i*3 + 2]
        else:
            #pose_tmp[1*3:2*3] = rodrig_z45[:]  # left hip, + 45:  outsider   
            pose[16*3:17*3] = - rodrig_z90[:]  # lshouler, + 45:  outsider   
            pose[18*3:19*3] = rodrig_x90[:]  # lelbow, + 45:  outsider   
        pose[:] = pose_tmp[:]

    plt.subplot(1,numFigures,startFigure + 2)
    plt.imshow(Renderer.r) #.copy()[::-1,:,:])
    plt.title('to Rest pose')

    # 4.  target pose transfer 
    v_posed = ch.array(v_cur.r)
    J = ch.array(jtr.r)
    pose =  ch.zeros(smpl_model.pose.size)
    #pose =  ch.array(pose_s)
    v_cur, jtr = verts_core(  pose = pose,
                            v    = v_posed,
                            J    = J,
                            weights = weights,
                            kintree_table = kintree_table,
                            xp = ch, #ch vs np
                            want_Jtr = True,
                            bs_style = 'lbs')
    # 3. new Renderer 
    cam_s.v = v_cur
    Renderer = build_color_renderer(cam_s, cam_s.v, smpl_model.f, w = 400, h = 600, ambient = .2, near=0.5, far = 25.) 

    plt.subplot(1,numFigures,startFigure + 3)
    plt.imshow(Renderer.r) #.copy()[::-1,:,:])
    plt.title('2nd & final step to Rest')

    # 4.  
    if not Flip:
        pose[:] =  pose_t[:]

    else:    
        pose_tmp = np.zeros(smpl_model.pose.size)
        for i in range(1, 24):
            pose_tmp[i*3 + 0] =  pose_t[i*3 + 0]
            pose_tmp[i*3 + 1] =  - pose_t[i*3 + 1]
            pose_tmp[i*3 + 2] =  - pose_t[i*3 + 2]
        pose[3:] = pose_tmp[3:]   

    plt.subplot(1,numFigures,startFigure + 4)
    plt.imshow(Renderer.r) #.copy()[::-1,:,:])
    plt.title('Target Posed')

    plt.suptitle('Linear Blending Skin: Pose Transfer SMPL, one shot')
    plt.show()

###############################################################################
# LBS model with Chumpy
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
def checkLBSModel_multistep(smpl_model, cam_s, pose_s, cam_t, pose_t, img_s = None, img_t = None):


    joint_hierarchy = [ [0],    # the level not exactly hierarchy 
                        [1, 2, 3], 
                        [6, 4, 5], 
                        [7, 8, 9, 13, 14], 
                        [10, 11, 12, 16, 17], 
                        [15, 18, 19], 
                        [20, 21], 
                        [22, 23] ] 

    numFigures = 10  
    startFigure = 0

    if img_s is not None and img_t is not None:  # SMPL only model  
        numFigures = numFigures  + 2 
        startFigure = 1
        plt.subplot(1, numFigures, 1)
        plt.imshow(img_s)
        plt.axis('off')
        plt.title('source image')
        plt.subplot(1, numFigures, numFigures)
        plt.imshow(img_t)
        plt.axis('off')
        plt.title('Target image')

    # constant terms 
    weights = ch.array(smpl_model.weights.r)
    kintree_table = smpl_model.kintree_table.copy()

    # 1. set the source pose 
    ##########################################################################
    smpl_model.pose[:] = pose_s[:]

    # 2. back to the default pose
    ##########################################################################
    for step in range(0, len(joint_hierarchy)):

        # 1. get the vertices and paramters 
        if step == 0: # first step 
            v_posed = ch.array(smpl_model.r)
            J = ch.array(smpl_model.J_transformed.r)
            pose =  ch.zeros(smpl_model.pose.size)
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
        cam_s.v = v_cur
        Renderer = build_color_renderer(cam_s, cam_s.v, smpl_model.f, w = 400, h = 600, ambient = .2, near=0.5, far = 25.) 

        if step == 0: # first step 
            plt.subplot(1,numFigures,startFigure + 1)
            plt.imshow(Renderer.r) 
            plt.axis('off')
            plt.title('Source pose')

        # 4. repose 
        for joint in joint_hierarchy[step]:
            pose[joint*3:(joint+1)*3] = - pose_s[joint*3:(joint+1)*3]     

        # 5. check results
        plt.subplot(1,numFigures,startFigure + step +2)
        plt.imshow(Renderer.r) #.copy()[::-1,:,:])
        plt.axis('off')
        plt.title('Step %d'%(step +1))


    # 4.  target pose transfer 
    ##########################################################################
     # 4.1 build LBS model
    v_posed = ch.array(v_cur.r)
    J = ch.array(jtr.r)
    pose =  ch.zeros(smpl_model.pose.size)
    #pose =  ch.array(pose_s)
    v_cur, jtr = verts_core(  pose = pose,
                            v    = v_posed,
                            J    = J,
                            weights = weights,
                            kintree_table = kintree_table,
                            xp = ch, #ch vs np
                            want_Jtr = True,
                            bs_style = 'lbs')
    # 4.2. new Renderer 
    cam_t.v = v_cur
    Renderer = build_color_renderer(cam_t, cam_t.v, smpl_model.f, w = 400, h = 600, ambient = .2, near=0.5, far = 25.) 

    # 4.3 repose
    pose[:] =  pose_t[:]

    # 4.4 check 
    plt.subplot(1,numFigures,startFigure + 10)
    plt.imshow(Renderer.r) #.copy()[::-1,:,:])
    plt.axis('off')
    plt.title('Target Posed')

    plt.suptitle('Linear Blending Skin: Pose Transfer SMPL: step by step')
    plt.show()


def posexfer(smpl_model, source,  target, outpath):

    plt.ion()

    params_s, _, img_s = _read_ifiles(source)
    params_t, _, img_t = _read_ifiles(target)

    # 3. SMPL Model Params   
    cam_s   = params_s['cam']           # camera model, Ch
    betas_s = params_s['betas']
    n_betas_s = betas_s.shape[0]        # 10, tuple to numpy
    pose_s  = params_s['pose']          # angles ((23+1)x3), numpy
    print('Source')
    _examine_smpl_params(params_s)
    print_pose(pose_s)

    # target : clothing is not needed
    cam_t   = params_t['cam']       # camera model, Ch
    betas_t = params_t['betas']
    n_betas_t = betas_t.shape[0]    # tuple of numpy to numpy 
    pose_t  = params_t['pose']      # angles, (23+1)x3 numpy
    print('Target')
    _examine_smpl_params(params_s)
    print_pose(pose_t)


    # angle values 
    euler_z90 = np.array([0.,           0.,  np.pi/2.0])
    euler_x90 = np.array([np.pi/2.0,    0.,  0.0])
    euler_y90 = np.array([0,     np.pi/2.0,  0.0])
    rotmat_z45 = euler2rotmat(euler_z90/2.)
    rotmat_x45 = euler2rotmat(euler_x90/2.)
    rotmat_y45 = euler2rotmat(euler_y90/2.)
    rotmat_z90 = euler2rotmat(euler_z90/1.)
    rotmat_x90 = euler2rotmat(euler_x90/1.)
    rotmat_y90 = euler2rotmat(euler_y90/1.)
    rodrig_z45 = cv2.Rodrigues(rotmat_z45)[0].ravel() # 1 is for Jacobian 
    rodrig_x45 = cv2.Rodrigues(rotmat_x45)[0].ravel() # 1 is for Jacobian 
    rodrig_y45 = cv2.Rodrigues(rotmat_y45)[0].ravel() # 1 is for Jacobian 
    rodrig_z90 = cv2.Rodrigues(rotmat_z90)[0].ravel() # 1 is for Jacobian 
    rodrig_x90 = cv2.Rodrigues(rotmat_x90)[0].ravel() # 1 is for Jacobian 
    rodrig_y90 = cv2.Rodrigues(rotmat_y90)[0].ravel() # 1 is for Jacobian 

    # This one check the coordinate direction with SMPL Joints, and Camera 
    #checkSMPLModel(smpl_model, cam_s, pose_s)
    #_ = raw_input()
   
    
    # This one test with synthetic Joint Pose case 
    #checkLBSModel(smpl_model, cam_s, pose_s, cam_t, pose_t, True, False, img_s, img_t)
    #_ = raw_input()

    ### sythetic ################################################################
    smpl_model.pose[4:] =  0 
    smpl_model.pose[16*3:17*3] = - rodrig_z90[:]  # lsh, - 45:   outsider   
    smpl_model.pose[18*3:19*3] = - rodrig_y90[:]  # lelbow, - 45:     
    smpl_model.pose[20*3:21*3] = - rodrig_x90[:]  # lwrist, - 45:    
    checkLBSModel_multistep(smpl_model, cam_s, smpl_model.pose.r, cam_t, pose_t )
    _ = raw_input()

    ### from SMPLIFY  ############################################################
    smpl_model.pose[:] =  pose_s[:] 
    checkLBSModel_multistep(smpl_model, cam_s, pose_s, cam_t, pose_t, img_s, img_t)
    _ = raw_input()

    # This one test Real Pose (especially, ARM and ELBOW angles are large)
    #checkLBSModel(smpl_model, cam_s, pose_s, cam_t, pose_t, False, False, img_s, img_t)


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

    posexfer(smpl_model, source, target, vton_path)

    # plt.pause(10)
    _ = raw_input('quit?')


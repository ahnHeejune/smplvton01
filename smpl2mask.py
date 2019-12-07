"""
  SMPL model to body mask  
 -----------------------------

 (c) copyright 2019 heejune@seoultech.ac.kr

 Prerequisite: SMPL model 
 In : SMPL paramters(cam, shape, pose) for a image 
 Out: body mask  (binary or labeled) 
      updated joint json file 
      [optionally the validating images]

    1.1 pre-calcuated fit data (camera, pose, shape)
   
"""
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
from smpl_webuser.verts import verts_decorated
from render_model import render_model
import inspect  # for debugging
import matplotlib.pyplot as plt
from opendr.lighting import SphericalHarmonics
from opendr.geometry import VertNormals, Rodrigues
from opendr.renderer import TexturedRenderer


import json
from smpl_webuser.lbs import global_rigid_transformation

_LOGGER = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)


import boundary_matching
import graphutil as graphutil

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

# To understand and verify the SMPL itself 
def _examine_smpl(model):

    print(">>>> SMPL <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print(type(model))
    print(dir(model))
    #print('kintree_table', model.kintree_table)
    #print('J:', model.J)
    #print('v_template:', model.v_template)
    #print('J_regressor:', model.J_regressor)
    #print('shapedirs:', model.shapedirs)
    #print('weights:', model.weoptimize_on_jointsights)
    #print('bs_style:', model.bs_style)
    #print('f:', model.f)
    print('V template :', type(model.v_template))
    print('V template :', model.v_template.shape)
    print('W type:', type(model.weights))
    print('W value:', type(model.weights.r))
    print('W shape:', model.weights.r.shape)
    #parts = np.count_nonzero(model.weights.r, axis =1)
    parts = np.argmax(model.weights.r, axis=1)
    print("        :",  parts.shape, parts[:6000])



# To understand and verify the paramters

def _examine_smpl_params(params):

    print(type(params))
    print(params.keys())
    print('camera params')
    camera = params['cam']
    print(" - type:", type(camera))
    #print(" - members:", dir(camera))
    print(" - cam.t:", camera.t.r)    # none-zero, likely only nonzero z
    print(" - cam.rt:", camera.rt.r)  # zero (fixed)
#    print(" - cam.camera_mtx:", camera.camera_mtx)  # 
    print(" - cam.k:", camera.k.r)  #
    print(" - cam.c:", camera.c.r)  #
    print(" - cam.f:", camera.f.r)  #

    #    print(params['f'].shape)      # 2
    print('>> pose')
    pose = params['pose']
    print("\t\ttype:", type(pose))
    print('\t\tshape:', pose.shape)   # 72
    
    # convert  within 
    #pose = pose % (2.0*np.pi)

    print('\t\tvalues:', pose*180.0/np.pi) # degree
    print('>> betas')
    betas = params['betas']
    print('\ttype:', type(betas))
    print('\tshape:', betas.shape)  # 10
    # print('\tvalues:', params['betas'])  # 10


# connvert 
# 1) uint8 image to float texture image 
# 2) normalize the vertices 
# optionally, 
# 3) coloring the backsize if face visibiltiy is not None)
# ***note ****:   texture coordinate is UP-side Down, and x-y  nomalized 
#j
def prepare_texture(pjt_v, pjt_f, img, face_visibility = None):

    # texture  = overlayed images of 2d and projected.
    pjt_texture = img.astype(float)/255.0  #  uint8 to float 
    #pjt_texture[:, :, :] = pjt_texture[:, :, :]/255.0
    #print('dtype of img:',  img.dtype)
    #print('dtype of pjt_texture:',  pjt_texture.dtype)
    th, tw = pjt_texture.shape[0:2]
    '''
    pjt_texture[:,:,:] = (1.0, .0, .0)  #  
    #pjt_texture[:,:int(tw/2),:] = (1.0, 0., 0.)  # B, G, R 
    pjt_texture[:,int(tw/4):int(3*tw/4),:] = (1.0, 1.0, 1.0)  # B, G, R 
    '''
    #print("th, tw:", th, tw)
    # vt
    #pjt_v = cam.r.copy()
    pjt_v[:, 0] = pjt_v[:, 0]/tw  # uv normalize
    pjt_v[:, 1] = pjt_v[:, 1]/th  # uv normalize
    # ft
    #pjt_ft = model.f.copy()
    #print("ft:", pjt_ft.shape)

    # 5. project the body model with texture renderer
    # 3. reprojection
    #print(type(cam.v))
    #print(cam.v.r.shape)

    #print("textured:",  type(pjt_texture), 'dtype:', pjt_texture.dtype, "shape:",  pjt_texture.shape)
    #print('max:', np.amax(pjt_texture[:, :, 0]), np.amax(
    #    pjt_texture[:, :, 1]), np.amax(pjt_texture[:, :, 2]))
    #print('meam:', np.mean(pjt_texture[:, :, 0]), np.mean(
    #    pjt_texture[:, :, 1]), np.mean(pjt_texture[:, :, 2]))
    #  apply the visibility map for texturing
    if face_visibility is not None:
        v_end = cam.r.shape[0]
        pjt_vt = np.append(
            pjt_vt, [[0./tw,  0./th], [1.0/tw, 0./th], [0./tw, 1.0/th]], axis=0)
        pjt_texture[th-50:th, 0:50] = (1.0, 1.0, 1.0)
        pjt_texture[0:50, 0:50] = (1.0, 1.0, 1.0)
        for i in range(f_normal.shape[0]):
            if face_visibility[i] < 0:
                pjt_ft[i] = (v_end, v_end+1, v_end+2)  # (0, 1, 2)

    return pjt_texture 


#
# texture processing with alpha blending 
def prepare_texture_with_alpha(pjt_v, pjt_f, img, mask, target_label):


    alpha = np.zeros(mask.shape)
    alpha[mask == target_label]  = 1.0  # 1.0 for fully opaque, 0.0 for transparent 

    rgb = img.astype(float)/255.0  #  uint8 to float 
    rgba = cv2.merge((rgb, alpha)) 
    print('shapes:', img.shape, rgb.shape, alpha.shape, rgba.shape)
    
    th, tw = rgba.shape[0:2]
    pjt_v[:, 0] = pjt_v[:, 0]/tw  # uv normalize
    pjt_v[:, 1] = pjt_v[:, 1]/th  # uv normalize

    return rgba #[:,:,:3] 

# create V, A, U, f: geom, bright, cam, renderer
def build_texture_renderer(U, V, f, vt, ft, texture, w, h, ambient=0.0, near=0.5, far=20000, background_image = None):

    A = SphericalHarmonics(vn=VertNormals(v=V, f=f),
                           components=[0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           light_color=ch.ones(3)) + ambient

    if background_image is not None:
        R = TexturedRenderer(vc=A, camera=U, f=f, bgcolor=[0.0, 0.0, 0.0],
                         texture_image=texture, vt=vt, ft=ft,
                         frustum={'width': w, 'height': h, 'near': near, 'far': far}, background_image= background_image)

    else:
        R = TexturedRenderer(vc=A, camera=U, f=f, bgcolor=[0.0, 0.0, 0.0],
                         texture_image=texture, vt=vt, ft=ft,
                         frustum={'width': w, 'height': h, 'near': near, 'far': far})

    return R





# convert numpy to json for a single person joint
def cvt_joints_np2json(joints_np):

    # 1. re-ordering
    # same as viton2lsp_joint and reamining
    order = [13,12,8, 7, 6, 9, 10, 11, 2, 1, 0, 3, 4, 5, 14, 15, 16, 17]

    # 2. build dictionary
    oneperson = { "face_keypoints": [],
                  "pose_keypoints": joints_np[order].flatten().tolist(),
                  "hand_right_keypoints": [],
                  "hand_left_keypoints":[]}

    people   = [oneperson]
    joints_json =  { "version": 1.0, "people": people }

    return joints_json



#
# SMPL => mask  
#
def smpl2maskcore(cam,      # camera model, Chv
          betas,    # shape coef, numpy
          n_betas,  # num of PCA
          pose,     # angles, 27x3 numpy
          imRGB,   # img numpy
          model,    # SMPL model 
          viz = False):     # visualize or not  

    for which in [cam,  betas,  pose, imRGB, model]:
        if which  is None:
            print( retrieve_name(which) ,  'is  None')
            exit()

    h, w = imRGB.shape[0:2]

    print('betas:', type(betas), betas)
    print('pose:', type(pose), pose)

    # 1. build body model  
    sv = verts_decorated(  # surface vertices
        trans=ch.zeros(3),
        pose=ch.array(pose),
        v_template=model.v_template,
        J=model.J_regressor,
        betas=ch.array(betas),
        shapedirs=model.shapedirs[:, :, :n_betas],
        weights=model.weights,
        kintree_table=model.kintree_table,
        bs_style=model.bs_style,
        f=model.f,
        bs_type=model.bs_type,
        posedirs=model.posedirs,
        want_Jtr = not viz) # need J_transformed for reposing based on vertices 

    #sv_r = sv.r.copy()

    # 2. Pose to standard pose   
    if True:   # make standard pose for easier try-on 
        sv.pose[:] = 0.0    
        sv.pose[0] = np.pi  
        # lsh = 16 rsh = 17 67.5 degree rotation around z axis
        sv.pose[16*3+2] = -7/16.0*np.pi  
        sv.pose[17*3+2] = +7/16.0*np.pi 

    # 3. render the model with parameter
    im3CGray = cv2.cvtColor(cv2.cvtColor(imRGB, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)  # 3 channel gray 
    im3CBlack = np.zeros_like(imRGB)  # uint8 type, right? 
    imBackground = im3CBlack

    dist = np.abs(cam.t.r[2] - np.mean(sv.r, axis=0)[2])
    im = (render_model(
        sv.r, model.f, w, h, cam, far= 20 + dist, img=imBackground[:, :, ::-1]) * 255.).astype('uint8')
    if False:
        plt.imshow(im[:,:,::-1])  # , cmap='gray')
        plt.suptitle('rendered')
        _ = raw_input('next?')

    # 4. binary mask 
    imBinary = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # gray silhouette  
    imBinary[imBinary > 0] = 255  # binary (0, 1)
    if False:
        plt.imshow(imBinary)  # , cmap='gray')
        plt.suptitle('binary mask')
        _ = raw_input('next?')

    ###############################
    # 5. new 2d joints 
    smpl_ids = [8, 5, 2, 1, 4, 7, 21, 19, 17, 16,   18,   20] # ,   12 ] # index in Jtr # @TODO correct neck  
    #                                         lsh,lelb,   lwr, neck

    # make the SMPL joints depend on betas
    Jdirs = np.dstack([model.J_regressor.dot(model.shapedirs[:, :, i]) for i in range(len(betas))])
    J_onbetas = ch.array(Jdirs).dot(betas) + model.J_regressor.dot( model.v_template.r)

    # get joint positions as a function of model pose, betas and trans
    (_, A_global) = global_rigid_transformation( sv.pose, J_onbetas, model.kintree_table, xp=ch)
    Jtr = ch.vstack([g[:3, 3] for g in A_global]) + sv.trans

    # add joints, with corresponding to a vertex...
    neck_id = 3078 #2951 #3061 # viton's bewtween shoulder
    Jtr = ch.vstack((Jtr, sv[neck_id]))
    smpl_ids.append(len(Jtr) - 1)
    # head_id = 411
    nose_id = 331  # nose vertex id
    Jtr = ch.vstack((Jtr, sv[nose_id]))
    smpl_ids.append(len(Jtr) - 1)
    lear_id  = 516
    Jtr = ch.vstack((Jtr, sv[lear_id]))
    smpl_ids.append(len(Jtr) - 1)
    rear_id  = 3941 # 422# 226 #396
    Jtr = ch.vstack((Jtr, sv[rear_id]))
    smpl_ids.append(len(Jtr) - 1)
    leye_id  = 125 #220 # 125
    Jtr = ch.vstack((Jtr, sv[leye_id]))
    smpl_ids.append(len(Jtr) - 1)
    reye_id  = 3635
    Jtr = ch.vstack((Jtr, sv[reye_id]))
    smpl_ids.append(len(Jtr) - 1)

    # project SMPL joints on the image plane using the estimated camera
    cam.v = Jtr

    joints_np_wo_confidence = cam.r[smpl_ids]  # get the projected value  
    print(joints_np_wo_confidence)
    joints_np = np.zeros([18, 3])
    joints_np[:,:2] =  joints_np_wo_confidence
    joints_np[:,2] =  1.0 

    for i in range(joints_np.shape[0]):
        if  joints_np[i,0] < 0 or joints_np[i,0] > (w-1) or joints_np[i,1] < 0 or joints_np[i,1] > (h-1):  
            joints_np[i, 2]  = 0.0 

    #print(joints_np)

    # check the joints 
    joints_np_int = joints_np.astype(int)
    for i in range(joints_np_int.shape[0]):   
        cv2.circle(im, tuple(joints_np_int[i,:2]), 2, (0, 0, 255), -1) # 2D joint White

    plt.imshow(im[:,:,::-1])  # , cmap='gray')
    plt.suptitle('joint check')
    #_ = raw_input('next?')


    # 6. convert format 
    joints_json = cvt_joints_np2json(joints_np)  # json joints 
    #print(joints_json)
    #json.dumps(joints_json) 

    return imBinary, joints_json


    # checking the redering result, but we are not using this.
    # we could drawing the points on it
    #print('th:', th,  '  tw:', tw)
    # plt.figure()
    img2 = img3CGray.copy()
    '''
    plt.imshow(img2)
    plt.hold(True)
    # now camera use only joints
    plt.plot(cam.r[:,0], cam.r[:, 1], 'r+', markersize=10) # projected pts 
    '''
    # project all vertices using camera
    cam.v = sv.r  
    #print('>>>>SV.R:',  type(sv.r))
    '''
    print('sv.r:', sv.r.shape)
    plt.plot(cam.r[:,0], cam.r[:, 1], 'b.', markersize=1) # projected pts 
    plt.show()
    plt.hold(False)
    plt.pause(3)
    '''
    # 1.2 vertices
    vertices = np.around(cam.r).astype(int)
    for idx, coord in enumerate(vertices):
        cv2.drawMarker(img2, tuple(coord), [0, 255, 0], markerSize=1)
        # cv2.circle(im, (int(round(uv[0])), int(round(uv[1]))), 1, [0, 255, 0]) # Green

    # Partmap for vertices
    ###################################################
    body_colormap = {   0: (255, 0, 0),
                        1: (0,255, 0),  # lhip
                        2: (0,0, 255),  # rhip
                        3: (255, 0, 0),     
                        4: (0, 0, 255),
                        5: (0,255, 0),  # lknee
                        6: (0,255, 0),  # rknee
                        7: (0, 255, 0), # lankle
                        8: (0,0, 255),  # rankle
                        9: (255, 0, 0),
                        10: (0,255, 0), # lfoot
                        11: (0,0, 255), # rfoot
                        12: (0,255, 0), # neck ******
                        # arms  
                        13: (0,255, 0), # shoulder
                        14: (0,255, 0), # shoulder 
                        15: (255,  0, 0),#  head *****
                        16: (0, 0, 255), # back arm 
                        17: (0, 0, 255),
                        18: (0,255, 0), # fore-arm
                        19: (0,255, 0),
                        20: (0,0, 255), # wrist
                        21: (0,0, 255),
                        22: (0,255, 0), # hands
                        23: (0,255, 0)  }

    use_partmap = True
    check_partmap  =  True 
    if use_partmap:
        bodyparts = np.argmax(model.weights.r, axis=1)
        if check_partmap:
            #bodypartmap = graphutil.build_bodypartmap(sv.r, cam, bodyparts, h, w, False)
            bodypartmap = graphutil.build_bodypartmap_2d(img3CGray, cam.r, bodyparts, body_colormap, h, w, False)
            print('part-max:', np.amax(bodyparts))
            plt.suptitle('body partmap')
            plt.subplot(1, 2, 1)
            plt.imshow(img3CGray[:, :, ::-1])  # , cmap='gray')
            plt.subplot(1, 2, 2)
            plt.imshow(bodypartmap[:,:,::-1])  # , cmap='gray')
            _ = raw_input('quit?')
            #exit()


    '''

    # DEPTH MAP  

    use_depthmap = True
    check_depthmap = True 
    if use_depthmap:
        bodydepth = graphutil.build_depthmap2(sv.r, cam)
        if check_depthmap:
            # depth in reverse way
            plt.suptitle('depthmap')
            plt.subplot(1, 2, 1)
            plt.imshow(img[:, :, ::-1])  # , cmap='gray')
            plt.subplot(1, 2, 2)
            depthmap = graphutil.build_depthimage(sv.r,  model.f,  bodydepth, cam,  height=h, width=w) #, near= 0.5, far= 400)
            #plt.imshow(depthmap, cmap='gray')
            plt.imshow(depthmap)
            plt.draw()
            plt.show()

            #plt.imshow(depthmap, cmap='gray_r') # the closer to camera, the brighter 
            _ = raw_input('quit?')
            exit()

    '''

#######################################################################################
# load dataset dependent files and call the core processing 
#---------------------------------------------------------------
# smpl_mdoel: SMPL 
# inmodel_path : smpl param pkl file (by SMPLify) 
# inimg_path: input image 
# out mask image 
# ind : image index 
#######################################################################################
def smpl2mask_single(smpl_model, inmodel_path, inimg_path, outimg_path, outjoint_path, ind):

    if smpl_model is None or inmodel_path is None or inimg_path is None or outimg_path is None:
        print('There is None inputs'), exit()

    plt.ion()

    # model params 
    with open(inmodel_path, 'rb') as f:
        if f is None:
            print("cannot open",  inmodel_path), exit()
        params = pickle.load(f)

    #params['pose'] = params['pose'] % (2.0*np.pi) # modulo 

    cam = ProjectPoints(f = params['cam_f'], rt=params['cam_rt'], t=params['cam_t'], k=params['cam_k'], c= params['cam_c'])
    params['cam'] = cam

    _examine_smpl_params(params)

    #  2d rgb image for texture
    #inimg_path = img_dir + '/dataset10k_%04d.jpg'%idx
    img2D = cv2.imread(inimg_path)
    if img2D is None:
        print("cannot open",  inimg_path), exit()

    # segmentation mask 
    '''
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        print("cannot open",  mask_path), exit()

    #  pre-processing for boundary matching  
    mask[mask == cloth_label_dict['bag']] = 0  # remove bag,  FIXME:  occlusion recovery 

    # remark skin (not face)
    neck_y = j2d[12,1]
    lsh_y  = j2d[9,1]
    rsh_y  = j2d[8,1]
    cut_y  = int((neck_y + lsh_y +rsh_y)/3.0) 
    #print(type(neck_y), neck_y)
    mask_skin = mask.copy()
    mask_skin[mask == cloth_label_dict['face']] = cloth_label_dict['skin'] # 18  
    mask[cut_y:,:]  = mask_skin[cut_y:, :]   # replace only non face region bellow the neck
    
    # cut the connected legs 
    if idx == 0:
        mask[500:,190] = 0
    elif idx == 1:
        mask[500:,220] = 0
    else:
        print('Not prepared Yet for the idx!')
        # exit()

    '''

    # 3. run the SMPL body to cloth processing 
    cam   = params['cam']      # camera model, Ch
    betas = params['betas']
    n_betas = betas.shape[0] #10
    pose  = params['pose']    # angles, 27x3 numpy
    img_mask, joints_json  = smpl2maskcore(params['cam'],      # camera model, Ch
                 betas,    # shape coeff, numpy
                 n_betas,  # num of PCA
                 pose,     # angles, 27x3 numpy
                 img2D,    # img numpy
                 smpl_model, # SMPL
                 viz = True)    # display   

    # 3.2 save result for checking
    if outimg_path is not None:
       cv2.imwrite(outimg_path, img_mask)
    if outjoint_path is not None:
        with  open(outjoint_path, 'w') as joint_file:
            json.dump(joints_json, joint_file)


if __name__ == '__main__':

    if len(sys.argv) < 5:
       print('usage: %s  ase_path dataset start_idx end_idx(exclusive)'% sys.argv[0]), exit()

    # 1. directory check and setting
    base_dir = abspath(sys.argv[1])
    #print(base_dir)
    dataset = sys.argv[2]
    idx_s = int(sys.argv[3])
    idx_e= int(sys.argv[4])

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
    '''
    mask_dir = data_dir + "/segmentation"
    if not exists(mask_dir):
        print('No such a directory for mask', mask_dir), exit()
    '''

    # Output Directory 
    smplmask_dir = data_dir + "/smplmask"
    if not exists(smplmask_dir):
        makedirs(smplmask_dir)

    smpljson_dir = data_dir + "/smpljson"
    if not exists(smpljson_dir):
        makedirs(smpljson_dir)

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


    # Load joints
    '''
    estj2d = np.load(join(data_dir, 'est_joints.npz'))['est_joints']
    #print('est_shape:', est.shape)
    joints = estj2d[:2, :, idx].T
    '''
    for idx in range(idx_s, idx_e):

        # for i in range(1, 2):
        # if not use_neutral:
        #    gender = 'male' if int(genders[i]) == 0 else 'female'
        #    if gender == 'female':
        smpl_model = model_female
        smpl_param_path = smpl_param_dir + '/%04d.pkl'%idx 
        if dataset == '10k':
            inp_path = inp_dir + '/' + 'dataset10k' + '_%04d.jpg'%idx 
        else:
            inp_path = inp_dir + '/' + dataset + '_%06d.jpg'%idx 

        '''
        mask_path = mask_dir + '/10kgt_%04d.png'%idx
        cloth_path = cloth_dir + '/%04d.png'% idx 
        #print(smpl_model, smpl_param_path, inp_path, mask_path, cloth_path, idx)
        '''
        smplmask_path = smplmask_dir + '/%06d_0.png'% idx 
        jointfile_path = smpljson_dir + '/%06d_0.json'% idx 
        smpl2mask_single(smpl_model, smpl_param_path, inp_path, smplmask_path,  jointfile_path, idx)


    # plt.pause(10)
    _ = raw_input('quit?')

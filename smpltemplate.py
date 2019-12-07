"""
  SMPL Standard Image and Mask Generation 
 -----------------------------

 (c) copyright 2019 heejune@seoultech.ac.kr

 Prerequisite: SMPL model 
 In : SMPL params fixed 
 Out: body mask (binary and labeled) 
      camera, pose and shape paramters  

camera
 - cam.t: [ 0. 0.  20.]  # [-3.12641449e-03  4.31656201e-01  2.13035413e+01]
 - cam.rt: [0. 0. 0.]
 - cam.k: [0. 0. 0. 0. 0.]
 - cam.c: [ 96. 128.]    # depending on the image size 
 - cam.f: [5000. 5000.]  # fixed 

betas type: <type 'numpy.ndarray'> shape: (10,)
all zeros used (default) 

pose: <type 'numpy.ndarray'> shape: (72,)

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
def _examine_smpl_template(model, detail = False):

    print(">> SMPL Template  <<<<<<<<<<<<<<<<<<<<<<")
    print(type(model))
    print(dir(model))
    #print('kintree_table', model.kintree_table)
    print('pose:', model.pose)
    if detail:
        print('posedirs:', model.posedirs)
    print('betas:', model.betas)
    print('shape(model):', model.shape)
    if detail:
        print('shapedirs:', model.shapedirs)

    #print('bs_style:', model.bs_style)  # f-m-n
    #print('f:', model.f)
    print('V template :', type(model.v_template))
    print('V template :', model.v_template.shape)
    #print('weights:', model.weoptimize_on_jointsights)
    print('W type:', type(model.weights))
    print('W shape:', model.weights.r.shape)
    if detail:
        print('W value:')
        print(model.weights.r)
        #parts = np.count_nonzero(model.weights.r, axis =1)
        parts = np.argmax(model.weights.r, axis=1)
        print("        :",  parts.shape, parts[:6000])

    #print('J:', model.J)
    #print('v_template:', model.v_template)
    #print('J_regressor:', model.J_regressor)

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

    print('\t\tvalues (in degree):')
    print(pose*180.0/np.pi) # degree
    print('>> betas')
    betas = params['betas']
    print('\ttype:', type(betas))
    print('\tshape:', betas.shape)  # 10
    # print('\tvalues:', params['betas'])  # 10


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
# calculate pixel position of SMPL joints 
#
#  cam: camera ie. projector 
#  model: smpl basic mdoel 
#  sv:  surfac vectors  (opendr)
#  betas : body shape, why needed?
#  h: projection image height
#  w: projection image width 
def calculate_joints(cam, model, sv, betas, h , w):

    # 1. get the joint locations
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

    # 2. project SMPL joints on the image plane using the estimated camera
    cam.v = Jtr

    joints_np_wo_confidence = cam.r[smpl_ids]  # get the projected value  
    #print(joints_np_wo_confidence)
    joints_np = np.zeros([18, 3])
    joints_np[:,:2] =  joints_np_wo_confidence
    joints_np[:,2] =  1.0 

    for i in range(joints_np.shape[0]):
        if  joints_np[i,0] < 0 or joints_np[i,0] > (w-1) or joints_np[i,1] < 0 or joints_np[i,1] > (h-1):  
            joints_np[i, 2]  = 0.0 

    #print(joints_np)
    return joints_np



# rendering with color  
# 
# @TODO: Texture rendering might be better for clearer segmentation 
#
# no light  setting needed for ColorRenderer
# vertices:  3D position 
# faces    : triangles 
# labelmap : map  from vertex to label 
# cam,
# height, width:  projection  size  
# near = 0.5,  far = 25, 
# bDebug = False):
def render_with_label(vertices, faces, labelmap,
                    cam, height, width, near = 0.5,  far = 25, bDebug = False):

    # 1. check labelmap 
    if bDebug:
        print('label :min:', np.amin(labelmap), 'max:', np.amax(labelmap), 'avg:', np.mean(labelmap))
        print('labelshape:', labelmap.shape)

    # 2. setup color renderer
    from opendr.renderer import ColoredRenderer
    rn = ColoredRenderer()
    rn.camera = cam
    rn.frustum = {'near': near, 'far': far, 'width': width, 'height': height}
    rn.bgcolor = ch.zeros(3)

    # 3. VC become the brightness of vertices 
    # OpenGL uses float for processing, so  convert it to float and then revert it to integer
    # in this rendering process, boundary gets blurred,  so be carefull if you need clear boundary 
    vc = np.zeros(vertices.shape)
    labelmap_float = labelmap.astype(np.float)/23.0
    vc[:,0], vc[:,1],vc[: ,2] = labelmap_float, labelmap_float, labelmap_float #  gray color
    rn.vc = vc   # giving the albera, FIXME: far -> bright? No! so you should  use gray_r for display
    rn.set(v=vertices, f=faces)

    # get one channel for segmentation 
    img = (rn.r[:,:,0]*23.0).astype('uint8') 
    return img

#
# SMPL => binary mask, part mask, joint location   
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

    # 1. Pose to standard pose   
    if True:   # make standard pose for easier try-on 
        pose[:] = 0.0    
        pose[0] = np.pi  
        # lsh = 16 rsh = 17 67.5 degree rotation around z axis
        pose[16*3+2] = -7/16.0*np.pi  
        pose[17*3+2] = +7/16.0*np.pi 
        betas[:] = 0.0
        #cam.t = [0. , 0., 20.] - cam.t: [ 0. 0.  20.]  # [-3.12641449e-03  4.31656201e-01  2.13035413e+01]
        cam.t = [0., 0.4, 25.]
        cam.rt =  [0.,  0.,  0.]
        cam.k = [0.,  0., 0.,  0.,  0.]
        cam.f = [5000.,  5000.]
        cam.c =  [ 96., 128.]    # depending on the image size 

    print('Final pose and betas ')
    print('pose:', type(pose), pose)
    print('betas:', type(betas), betas)

    # 2. build body model  
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
    dist = np.abs(cam.t.r[2] - np.mean(sv.r, axis=0)[2])

    # 3. render the model with parameter
    h = h*3//2  # extended for full body 
    print("output size (hxw):", h,  w)
    im3CGray = cv2.cvtColor(cv2.cvtColor(imRGB, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)  # 3 channel gray 
    im3CBlack = np.zeros([h, w, 3], dtype = np.uint8)   
    imBackground = im3CBlack
    im = (render_model(
        sv.r, model.f, w, h, cam, far= 20 + dist, img=imBackground[:, :, ::-1]) * 255.).astype('uint8')



    # 4. binary mask 
    imBinary = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # gray silhouette  
    imBinary[imBinary > 0] = 255  # binary (0, 1)
    # @TODO: check any noisy pixels,  if yes, filtering or get contours and redrawing with contours 



    # 4'. segmentation information of  human body 
    bodyparts = np.argmax(model.weights.r, axis=1)
    print(np.unique(bodyparts))  # will list all the joints nmbers 
    imPart = render_with_label(sv.r,  model.f,  bodyparts, cam,  height=h, width=w, near= 0.5, far= 40, bDebug = True)
    print("impart:", np.amax(imPart), np.amin(imPart), np.mean(imPart))
    print("       ", np.unique(imPart))


    # 5. new 2d joints for template 
    joints_np = calculate_joints(cam, model, sv, betas, h, w)
    # check the joints 
    joints_np_int = joints_np.astype(int)
    imJoint = im.copy()
    for i in range(joints_np_int.shape[0]):   
        cv2.circle(imJoint, tuple(joints_np_int[i,:2]), 2, (0, 0, 255), -1) # 2D joint White

    if True:
        plt.subplot(2,2,1)
        plt.imshow(im[:,:,::-1])  # , cmap='gray')
        plt.title('rendered')
        plt.subplot(2,2,2)
        plt.imshow(imBinary)  # , cmap='gray')
        plt.title('binary mask')
        plt.subplot(2,2,3)
        plt.imshow(imPart) # , cmap='gray')
        plt.title('part mask')
        plt.subplot(2,2,4)
        plt.imshow(imJoint[:,:,::-1])  # , cmap='gray')
        plt.title('joints')
        plt.suptitle('SMPL Template Check')
    #_ = raw_input('next?')

    # 6. convert format 
    joints_json = cvt_joints_np2json(joints_np)  # json joints 
    #print(joints_json)
    #json.dumps(joints_json) 

    params = {'cam_t': cam.t.r,   
              'cam_rt': cam.rt.r,   
              'cam_f': cam.f.r,   
              'cam_k': cam.k.r,   
              'cam_c': cam.c.r,   
              'pose': sv.pose.r,
              'betas': sv.betas.r}

    return imBinary, imPart, joints_json, params


# load dataset dependent files and call the core processing 
#---------------------------------------------------------------
# smpl_mdoel: SMPL 
# inmodel_path : smpl param pkl file (by SMPLify) 
# inimg_path: input image 
# out mask image 
# ind : image index 
def smpl2mask_single(smpl_model, inmodel_path, inimg_path, outbinimg_path,  outpartimg_path, outjoint_path,  outparam_path, ind):

    if smpl_model is None or inmodel_path is None or inimg_path is None or outbinimg_path is None or outpartimg_path is None:
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

    # 3. run the SMPL body to cloth processing 
    cam   = params['cam']      # camera model, Ch
    betas = params['betas']
    n_betas = betas.shape[0] #10
    pose  = params['pose']    # angles, 27x3 numpy
    img_mask, img_part, joints_json, params  = smpl2maskcore(params['cam'],      # camera model, Ch
                 betas,    # shape coeff, numpy
                 n_betas,  # num of PCA
                 pose,     # angles, 27x3 numpy
                 img2D,    # img numpy
                 smpl_model, # SMPL
                 viz = True)    # display   

    # 3.2 save output result
    if outbinimg_path is not None:
       cv2.imwrite(outbinimg_path, img_mask)
    if outpartimg_path is not None:
       cv2.imwrite(outpartimg_path, img_part)
    if outjoint_path is not None:
        with  open(outjoint_path, 'w') as joint_file:
            json.dump(joints_json, joint_file)

    if outparam_path is not None:
        with open(outparam_path, 'w') as outf:
            pickle.dump(params, outf)



if __name__ == '__main__':

    if len(sys.argv) < 4:
       print('usage: %s base_path dataset start_idx'% sys.argv[0]), exit()

    # 1. directory check and setting
    base_dir = abspath(sys.argv[1])
    #print(base_dir)
    dataset = sys.argv[2]
    idx_s = int(sys.argv[3])
    #idx_e= int(sys.argv[4])

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

    _examine_smpl_template(model_female) #, exit()

    # Load joints
    '''
    estj2d = np.load(join(data_dir, 'est_joints.npz'))['est_joints']
    #print('est_shape:', est.shape)
    joints = estj2d[:2, :, idx].T
    '''
    idx = idx_s

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

    #smplmask_path = smplmask_dir + '/%06d_0.png'% idx 
    #jointfile_path = smpljson_dir + '/%06d_0.json'% idx 
    smplmask_path = './templatemask.png' 
    smplpart_path = './templatepart.png' 
    jointfile_path = './templatejoint.json' 
    param_path = './templateparam.pkl'
    smpl2mask_single(smpl_model, smpl_param_path, inp_path, smplmask_path, smplpart_path, jointfile_path, param_path, idx)


    # plt.pause(10)
    _ = raw_input('quit?')

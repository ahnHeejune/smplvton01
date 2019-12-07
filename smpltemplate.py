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
#[ 0.86333032  0.66983643  0.03516618 -0.0108483   0.08504395 -0.03625412
 -0.14589336  0.05147145 -0.09457806 -0.23076016]
#  what is the  default betas?  all zeros?

pose: <type 'numpy.ndarray'> shape: (72,)
   values: [ 2.23741497e+05  1.76197090e+04 -2.87053045e+04 -3.11457680e+01
 -1.03651321e+00  1.07745803e+01 -3.30462125e+01 -8.75933329e-02
 -8.42849899e+00  1.95213130e+01 -4.77103449e-01 -6.65209831e-01
  4.66690547e+01 -1.91283890e+00 -9.85186186e+00  5.53364988e+01
  2.60572518e+00  5.70655214e+00 -4.78080913e+00  9.61015082e-01
  8.05987811e-01 -5.61919083e+00  5.05691714e+00 -3.50657313e+00
 -1.09920714e+01 -8.33753237e+00  1.14504502e+01  5.36324998e-01
 -1.87318192e+00  3.58527364e-01 -1.63665236e+01  1.15114489e+01
  1.20531329e+01 -9.78285335e+00  5.56664737e+00 -2.08030024e+01
 -2.07777378e+01  1.07819986e+01  1.71939020e+00  2.91112580e+00
  1.33954258e+00 -2.73712556e+01  6.52049103e-01  4.28826740e+00
  2.79391512e+01  1.72933816e+00  2.74343466e+00 -3.25453137e+00
  8.51945524e+00 -1.07478619e+01 -5.39806102e+01  1.46659045e+01
  1.64629554e+01  5.45966113e+01  1.25128178e+01 -4.29796685e+01
  9.27542674e+00  3.19900858e+00  4.31000189e+01 -8.98971514e+00
 -4.56694457e-01  4.03708262e-01  7.01004288e+00 -7.40812882e-02
  1.92158331e+00 -3.77941487e+00 -1.32085765e+01 -4.55730154e+00
 -1.27710507e+01 -8.56575127e+00  6.76882057e+00  1.25808406e+01]



pose: <type 'numpy.ndarray'> [ 3.90502580e+03  3.07521935e+02 -5.01002076e+02 -5.43596199e-01
 -1.80905683e-02  1.88051902e-01 -5.76765214e-01 -1.52879206e-03
 -1.47105058e-01  3.40711186e-01 -8.32702606e-03 -1.16101018e-02
  8.14528663e-01 -3.33853369e-02 -1.71947427e-01  9.65804101e-01
  4.54784837e-02  9.95981237e-02 -8.34408602e-02  1.67728773e-02
  1.40671410e-02 -9.80733813e-02  8.82598541e-02 -6.12012466e-02
 -1.91847837e-01 -1.45517391e-01  1.99848057e-01  9.36063708e-03
 -3.26931920e-02  6.25748297e-03 -2.85649724e-01  2.00912686e-01
  2.10366854e-01 -1.70743001e-01  9.71563249e-02 -3.63080885e-01
 -3.62639935e-01  1.88181376e-01  3.00090202e-02  5.08087301e-02
  2.33794285e-02 -4.77718530e-01  1.13804037e-02  7.48443854e-02
  4.87630179e-01  3.01826448e-02  4.78819676e-02 -5.68022880e-02
  1.48692544e-01 -1.87585578e-01 -9.42139379e-01  2.55968321e-01
  2.87332777e-01  9.52890628e-01  2.18389869e-01 -7.50136726e-01
  1.61886736e-01  5.58332325e-02  7.52237238e-01 -1.56900128e-01
 -7.97082195e-03  7.04603839e-03  1.22348329e-01 -1.29296239e-03
  3.35379557e-02 -6.59632333e-02 -2.30533150e-01 -7.95399168e-02
 -2.22896883e-01 -1.49500562e-01  1.18138205e-01  2.19577092e-01]

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



###############################################################################
# rendering with color  
# 
# @TODO: Texture rendering might be better for clearer segmentation 
#
# no light  setting needed for ColorRenderer
###############################################################################
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


#######################################################################################
# load dataset dependent files and call the core processing 
#---------------------------------------------------------------
# smpl_mdoel: SMPL 
# inmodel_path : smpl param pkl file (by SMPLify) 
# inimg_path: input image 
# out mask image 
# ind : image index 
#######################################################################################
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

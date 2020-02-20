"""
  cloth 3d model reconstruction (smpl3dclothrecon.py) and transfer to a human model  
 ------------------------------------------------------------------------------------------

 (c) copyright 2019 heejune@seoultech.ac.kr

 In : used in smpl3dclothrec.py 
       - SMPL template model params file (pkl) 
       - 2D matched cloth image file and mask (png) 
      used for transfering
       - SMPL target model params file (pkl)
      helper file
       - list.npy for re-ordering the smpl human image number to viton image number 

        Note: re-ordering needed for SMPLify code

 Out: 
        3D warped cloth and mask (png)

 Note: the Texture (2D warped  cloth) and related 2D vertex and face information is obtained
       with  original SMPL and camera parameters

 For in-advance tesrt purpose of part 3.  we could move the pose and apply the displacement vector 
       we apply the pose and shape params for target user but with same texture and vertices and faces defitnion


                   template (source: pose and shape)   target (pose and shape)
      --------------------------------------------------------------------------
      SMPL- p      smpltemplate.pkl                    results/viton/smpl/000000.pkl
      camera-p     smpltemplate.pkl                    results/viton/smpl/000000.pkl
      3D body-v    smpl  with template param           smpl with target params
      3D cloth-v   displacement obtained               use displacemt obtained at template 
      texture      results/viton/2dwarp/00000_1.png    same 
      texture-v    cam projected onto the texture      same as template (not new vertices)
      texture-f    model.f                             same 
      lightening   only for cloth-related vertices     same 
 
 
 """
from __future__ import print_function
import smpl3dclothrec_v7
import graphutil as graphutil
import boundary_matching
import sys
from os.path import join, exists, abspath, dirname
from os import makedirs
import logging
import cPickle as pickle
import time
import cv2
from PIL import Image
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


# To understand and verify the SMPL itself
def _examine_smpl_template(model, detail=False):

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

    # print('bs_style:', model.bs_style)  # f-m-n
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
    print(pose*180.0/np.pi)  # degree
    print('>> betas')
    betas = params['betas']
    print('\ttype:', type(betas))
    print('\tshape:', betas.shape)  # 10
    # print('\tvalues:', params['betas'])  # 10


def construct_clothed3d_from_clothed2d_depth(body_sv, cam, clothed2d):

    # 1. get the dept for body vertex
    bodydepth = graphutil.build_depthmap2(body_sv.r, cam)

    check_depthmap = False
    if check_depthmap:
        # depth in reverse way
        plt.suptitle('depthmap')
        plt.subplot(1, 2, 1)
        plt.imshow(img[:, :, ::-1])  # , cmap='gray')
        plt.subplot(1, 2, 2)
        depthmap = graphutil.build_depthimage(
            body_sv.r,  model.f,  bodydepth, cam,  height=h, width=w)
        #plt.imshow(depthmap, cmap='gray')
        plt.imshow(depthmap)
        plt.draw()
        plt.show()
        # plt.imshow(depthmap, cmap='gray_r') # the closer to camera, the brighter
        _ = raw_input('quit?')
        exit()

    # 2. modify the depth for clothed
    # @TODO

    # 3. unproject to 3D
    # uv space? pixels coordinated!!
    clothuvd = np.zeros(body_sv.r.shape)
    clothuvd[:, 0] = clothed2d[:, 0]
    clothuvd[:, 1] = clothed2d[:, 1]
    # @TODO for now simply use the same depth as body ^^;
    clothuvd[:, 2] = bodydepth
    cloth3d = cam.unproject_points(clothuvd)
    # sv.r = cloth3d  #  now the model is not body but cloth

    return cloth3d


# calcuated the local coordinates at each vetex.
#
#  z : normal to the vertex
#  x : the  smallest  indexed neighbor vertex based unit vector
#  y : the remianing  axis in right handed way, ie.  z x x => y
def setup_vertex_local_coord(faces, vertices):

    # 1.1 normal vectors (1st axis) at each vertex
    _, axis_z = graphutil.calc_normal_vectors(vertices, faces)
    # 1.2 get 2nd axis
    axis_x = graphutil.find2ndaxis(faces, axis_z, vertices)
    # 1.3 get 3rd axis
    # matuir contribution. np.cross support row-vectorization
    axis_y = np.cross(axis_z[:, :], axis_x[:, :])

    return axis_x, axis_y, axis_z

#
# reporesent the displacement (now in global coord) into local coordinates
#
# model: smpl mesh structure
# v0   : reference vertex surface, ie. the body
# v*****array: vertext index array for interest
# d    : displacement, ie. v = v0 + d
#


def compute_displacement_at_vertex(model, v0, d_global):

    debug = False

    # 1.setup local coordinate system to each vertex
    axis_x, axis_y, axis_z = setup_vertex_local_coord(model.f, v0)

    # 2. express displacement in 3 axises
    #dlocal = np.concatenate(np.dot(d, axis_x), np.dot(d, axis_y), np.dot(d, axis_z))
    xl = np.sum(d_global*axis_x, axis=1)
    yl = np.sum(d_global*axis_y, axis=1)
    zl = np.sum(d_global*axis_z, axis=1)
    d_local = np.stack((xl, yl, zl), axis=-1)
    print('dlocal shape:', xl.shape, yl.shape, zl.shape,  d_local.shape)

    if debug:  # verifying d_global =  xs * axis_x + ys* axis_y  + z*axis_z
        # get global coorindate vector
        xg = xl[:, None]*axis_x
        yg = yl[:, None]*axis_y
        zg = zl[:, None]*axis_z
        dg = xg + yg + zg

        # check the error
        err = np.absolute(dg - d_global)
        print('d, e x:',  np.amax(d_global[:, 0]), np.amax(
            err[:, 0]), np.mean(d_global[:, 0]), np.mean(err[:, 0]))
        print('d, e y:',  np.amax(d_global[:, 1]), np.amax(
            err[:, 1]), np.mean(d_global[:, 1]), np.mean(err[:, 1]))
        print('d, e z:',  np.amax(d_global[:, 2]), np.amax(
            err[:, 2]), np.mean(d_global[:, 2]), np.mean(err[:, 2]))
        '''
        print('d    0:',  np.amax(d_global[:,0]), np.amin(d_global[:,0]))
        print('error0:',  np.amax(err[:,0]), np.amin(err[:,0]))
        print('d    1:',  np.amax(d_global[:,1]), np.amin(d_global[:,1]))
        print('error1:',  np.amax(err[:,1]), np.amin(err[:,1]))
        print('d    2:',  np.amax(d_global[:,2]), np.amin(d_global[:,2]))
        print('error2:',  np.amax(err[:,2]), np.amin(err[:,2]))
        '''

    return d_local


#
# @TODO: Do this !! the most key part combining with displacement generatrion
#
# model   : the body surface structure
# body    : body surface vertices
# vi4cloth: vertex index for the cloth surface
# d       : displacement vector in local coordinate
#
# def transfer_body2clothed(cam_tgt, betas_tgt, n_betas_tgt, pose_tgt, v4cloth, d):
def transfer_body2clothed(model, body, d_local):

    # 1.setup local coordinate system to each vertex
    axis_x, axis_y, axis_z = setup_vertex_local_coord(model.f, body)

    # 2. express local to global
    # 2.1 select vectices under interest
    #axis_x, axis_y, axis_z = axis_x[vi4cloth], axis_y[vi4cloth], axis_z[vi4cloth]
    # 2.2 displacement in global coordinate
    xg = (d_local[:, 0])[:, None]*axis_x
    yg = (d_local[:, 1])[:, None]*axis_y
    zg = (d_local[:, 2])[:, None]*axis_z
    dg = xg + yg + zg

    # 3. adding them to the base/body vertices
    clothed = body + dg

    return clothed


# display 3d model
def render_cloth(cam, _texture, texture_v2d, faces, imHuman):

    #h, w = imTexture.shape[:2]
    h_ext, w = _texture.shape[:2]   # full body
    h, _ = imHuman.shape[:2]         # half body

    texture = _texture[:, :, :]

    # 1. texture rendering
    dist = 20.0
    cloth_renderer = smpl3dclothrec_v7.build_texture_renderer(cam, cam.v, faces, texture_v2d, faces,
                                                           texture[::-1, :, :], w, h_ext, 1.0, near=0.5, far=20 + dist)
    imCloth = (cloth_renderer.r * 255.).astype('uint8')
    imCloth = imCloth[:h, :, ::-1]

    # 2. mask  generation
    im3CBlack = np.zeros([h, w, 3], dtype=np.uint8)
    imModel = (render_model(
        cam.v, faces, w, h, cam, far=20 + dist, img=im3CBlack) * 255.).astype('uint8')
    imMask = cv2.cvtColor(imModel, cv2.COLOR_BGR2GRAY)  # gray silhouette
    imMask[imMask > 0] = 255  # binary (0, 1)

    # 3. image overlay to check result
    imBG = imHuman[:, :, ::-1].astype('float32')/255.0
    overlay_renderer = smpl3dclothrec_v7.build_texture_renderer(cam, cam.v, faces, texture_v2d, faces, texture[::-1, :, :], w, h, 1.0, near=0.5, far=20 + dist, background_image=imBG)
    imOverlayed = overlay_renderer.r.copy()

    # plt.figure()
    plt.subplot(1, 4, 1)
    plt.axis('off')
    plt.imshow(texture[:h, :, ::-1])
    plt.title('texture')

    plt.subplot(1, 4, 2)
    plt.imshow(imCloth[:, :, ::-1])
    plt.axis('off')
    plt.title('transfered')

    plt.subplot(1, 4, 3)
    # @TODO use color render for mask or all whilte color for the cloth area texture
    plt.imshow(imMask)
    plt.axis('off')
    plt.title('mask')

    plt.subplot(1, 4, 4)
    plt.imshow(imOverlayed[:, :, :])  # @overlay with human image
    plt.axis('off')
    plt.title('target human')
    plt.show()

    return imCloth, imMask


def save_rendered_textures(imCloth3dWarped, imClothMask3dWarped, human_segm_path, ocloth_path, oclothfull_path, omask_path):

    """
        LIP labels

        [(0, 0, 0),  # 0=Background
         (128, 0, 0),  # 1=Hat
         (255, 0, 0),  # 2=Hair
         (0, 85, 0),  # 3=Glove
         (170, 0, 51),  # 4=Sunglasses
         (255, 85, 0),  # 5=UpperClothes
         (0, 0, 85),  # 6=Dress
         (0, 119, 221),  # 7=Coat
         (85, 85, 0),  # 8=Socks
         (0, 85, 85),  # 9=Pants
         (85, 51, 0),  # 10=Jumpsuits
         (52, 86, 128),  # 11=Scarf
         (0, 128, 0),  # 12=Skirt
         (0, 0, 255),  # 13=Face
         (51, 170, 221),  # 14=LeftArm
         (0, 255, 255),  # 15=RightArm
         (85, 255, 170),  # 16=LeftLeg
         (170, 255, 85),  # 17=RightLeg
         (255, 255, 0),  # 18=LeftShoe
         (255, 170, 0)  # 19=RightShoe
         (189, 170, 160)  # 20=Skin/Neck
         ]
    """

    im_parse = Image.open(human_segm_path)
    # im_parse_2d = Image.open(human_segm_path).convert('L')
    parse_array = np.array(im_parse)
    # parse_array_2d = np.array(im_parse_2d)

    parse_cloth = (parse_array == 0) + \
                  (parse_array == 5) + \
                  (parse_array == 6) + \
                  (parse_array == 7) + \
                  (parse_array == 14) + \
                  (parse_array == 15) + \
                  (parse_array == 20)

    """parse_mask = (parse_array_2d == 0) + \
                 (parse_array_2d == 5) + \
                 (parse_array_2d == 6) + \
                 (parse_array_2d == 7) + \
                 (parse_array_2d == 14) + \
                 (parse_array_2d == 15) + \
                 (parse_array_2d == 20)"""

    im_cloth = imCloth3dWarped * parse_cloth - (1 - parse_cloth)  # [-1,1], fill 0 for other parts
    # im_cloth_mask = imClothMask3dWarped * parse_mask - (1 - parse_mask)  # [-1,1], fill 0 for other parts
    im_cloth_mask = np.zeros_like(im_cloth)
    im_cloth_mask[im_cloth > 0] = 255

    # make white bg
    im_cloth[im_cloth <= 0] = 255

    plt.subplot(1, 3, 1)
    plt.axis('off')
    plt.imshow(imCloth3dWarped[:, :, ::-1])
    plt.title('raw')

    plt.subplot(1, 3, 2)
    plt.axis('off')
    plt.imshow(im_cloth[:, :, ::-1])
    plt.title('final')

    plt.subplot(1, 3, 3)
    plt.axis('off')
    plt.imshow(im_cloth_mask[:, :, ::-1])
    plt.title('mask')

    plt.show()
    _ = raw_input("Save?")

    # save result
    if oclothfull_path is not None:
        cv2.imwrite(oclothfull_path, imCloth3dWarped)
    if ocloth_path is not None:
        cv2.imwrite(ocloth_path, im_cloth)
    if omask_path is not None:
        cv2.imwrite(omask_path, im_cloth_mask)


def cloth3dxfer_single(smpl_model, src_param_path, tgt_param_path, cloth_path, clothmask_path, human_path, human_segm_path, ocloth_path, oclothfull_path, omask_path):

    # 1. reconstruct 3D cloth from template
    params_src, body, diff_cloth_body, texture, texture_v2d, face4cloth = smpl3dclothrec_v7.cloth3drec_single(
        smpl_model, src_param_path, cloth_path, clothmask_path, human_path, human_segm_path)

    # 2. express the displacement in vertices specific coordinate.
    diff_cloth_body_local = compute_displacement_at_vertex(
        smpl_model, body, diff_cloth_body)

    # 3. transfer to a new human parameters
    # 3.1 load the SMPL params
    with open(tgt_param_path, 'rb') as f:
        if f is None:
            print("cannot open",  tgt_param_path), exit()
        params_tgt = pickle.load(f)

    # 3.2 construct the model
    cam_tgt = ProjectPoints(f=params_tgt['cam_f'], rt=params_tgt['cam_rt'],
                            t=params_tgt['cam_t'], k=params_tgt['cam_k'], c=params_tgt['cam_c'])
    betas_tgt = params_tgt['betas']
    n_betas_tgt = betas_tgt.shape[0]  # 10
    pose_tgt = params_tgt['pose']    # angles, 27x3 numpy

    # 3.3 build a new body
    body_tgt_sv = smpl3dclothrec_v7.build_smplbody_surface(
        smpl_model, pose_tgt, betas_tgt, cam_tgt)

    # 3.4 build the corresponding clothed
    clothed3d = transfer_body2clothed(
        smpl_model, body_tgt_sv.r, diff_cloth_body_local)
    cam_tgt.v = clothed3d
    #cam_tgt.v = body_tgt_sv.r

    # 4.5 check by viewing
    imHuman = cv2.imread(human_path)

    # smpl_model.f) # cam_tgt has all the information
    # imCloth3dWarped, imClothMask3dWarped = render_cloth(cam_tgt, texture, texture_v2d, smpl_model.f, imHuman)  # fixed cloth with all faces
    imCloth3dWarped, imClothMask3dWarped = render_cloth(cam_tgt, texture, texture_v2d, face4cloth, imHuman)  # fixed cloth with all faces
    # _, imClothMask3dWarped = render_cloth(cam_tgt, texture, texture_v2d, face4cloth, imHuman)  # fixed mask from cloth vertices only
    _ = raw_input("next?")

    # save combined warped rendered texture
    save_rendered_textures(imCloth3dWarped, imClothMask3dWarped, human_segm_path, ocloth_path, oclothfull_path, omask_path)

    # smpl3dclothrec.show_3d_model(cam_tgt, texture, texture_v2d, face4cloth)  # smpl_model.f) # cam_tgt has all the information
    _ = raw_input("next sample?")
    plt.subplot(1, 1, 1)  # restore the plot section
    # plt.close() # not to draw in subplot()


if __name__ == '__main__':

    # 1. command argument checking
    if len(sys.argv) != 3:
        print('usage for batch  test: %s base_path dataset' % sys.argv[0])
        #print('usage for test: %s base_path smpl_param clothimg maskimg'% sys.argv[0]), exit()
        exit()

    base_dir = abspath(sys.argv[1])
    dataset = sys.argv[2]

    # 2. input and output directory check and setting
    # 2.1 base dir
    base_dir = abspath(sys.argv[1])
    if not exists(base_dir):
        print('No such a directory for base', base_path, base_dir), exit()

    # 2.2.1 human image dir
    human_dir = base_dir + "/images/" + dataset
    if not exists(human_dir):
        print('No such a directory for human images',
              data_set, human_dir), exit()

    data_dir = base_dir + "/results/" + dataset
    # print(data_dir)
    # 2.2.2 target human info
    human_smpl_param_dir = data_dir + "/smpl"
    if not exists(human_smpl_param_dir):
        print('No such a directory for smpl param', smpl_param_dir), exit()
    # 2.2.3 source cloth
    cloth_dir = data_dir + "/c2dw"
    if not exists(cloth_dir):
        print('No such a directory for cloth images', cloth_dir), exit()
    # 2.2.4 source cloth mask
    cloth_mask_dir = data_dir + "/c2dwmask"
    if not exists(cloth_mask_dir):
        print('No such a directory for cloth mask', cloth_mask_dir), exit()

    # 2.2.5 human segmentation dir
    human_segm_dir = data_dir + "/segmentation"
    if not exists(human_segm_dir):
        print('No such a directory for human segmentation',
              human_segm_dir), exit()

    # 2.2.4 test pair file
    testpair_filepath = data_dir + "/" + dataset + "_test_pairs.txt"
    if not exists(testpair_filepath):
        print('No test pair file: ', cloth_mask_dir), exit()

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
        # with open(join(data_dir, dataset + '_gender.csv')) as f:
        #    genders = f.readlines()
        model_female = load_model(MODEL_FEMALE_PATH)
        model_male = load_model(MODEL_MALE_PATH)
    else:
        gender = 'neutral'
        smpl_model = load_model(MODEL_NEUTRAL_PATH)

    #_examine_smpl(model_female), exit()

    '''
    # Load joints
    estj2d = np.load(join(data_dir, 'est_joints.npz'))['est_joints']
    #print('est_shape:', est.shape)
    joints = estj2d[:2, :, idx].T
    '''

    # 2.3. Output Directory

    oclothfull_dir = data_dir + "/c3dwfull"
    if not exists(oclothfull_dir):
        makedirs(oclothfull_dir)
    ocloth_dir = data_dir + "/c3dw"
    if not exists(ocloth_dir):
        makedirs(ocloth_dir)
    ocloth_mask_dir = data_dir + "/c3dwmask"
    if not exists(ocloth_mask_dir):
        makedirs(ocloth_mask_dir)

    #smplmask_path = smplmask_dir + '/%06d_0.png'% idx
    #jointfile_path = smpljson_dir + '/%06d_0.json'% idx
    '''
    smpl_model = model_female
    # 3D reconstruction and tranfer it to a define smpl model
    cloth3dxfer_single(smpl_model, smplparam_path, cloth_path, clothmask_path)
        
    '''

    test_pair_lines = open(testpair_filepath).read().splitlines()
    test_pairs = []

    for i in range(len(test_pair_lines)):
        # loading batch data
        pair = test_pair_lines[i].split()
        # print(pair)
        test_pairs.append([pair[0], pair[1]])  # 0: human 1: cloth

    #print(test_pairs), exit()

    # Might each cloth have different verison of template used
    template_smpl_param_path = './templateparam1.pkl'
    # We have to take into account this later
    template_jointfile_path = './templatejoints1.json'

    for i in range(len(test_pairs)):

        # for i in range(1, 2):
        # if not use_neutral:
        #    gender = 'male' if int(genders[i]) == 0 else 'female'
        #    if gender == 'female':
        smpl_model = model_female
        human_smpl_param_path = human_smpl_param_dir + \
            '/' + test_pairs[i][0] + '.pkl'
        human_image_path = human_dir + '/' + test_pairs[i][0] + '.jpg'
        human_segm_path = human_segm_dir + '/' + test_pairs[i][0] + '.png'
        cloth_path = cloth_dir + '/' + test_pairs[i][1] + '.png'
        clothmask_path = cloth_mask_dir + '/' + test_pairs[i][1] + '.png'
        oclothfull_path = oclothfull_dir + '/' + \
            test_pairs[i][1] + '_' + test_pairs[i][0] + '.png'  # '.png'
        ocloth_path = ocloth_dir + '/' + \
                      test_pairs[i][1] + '_' + test_pairs[i][0] + '.jpg'  # '.png'
        oclothmask_path = ocloth_mask_dir + '/' + \
            test_pairs[i][1] + '_' + test_pairs[i][0] + '.jpg'  # '.png'
        cloth3dxfer_single(smpl_model, template_smpl_param_path, human_smpl_param_path,
                           cloth_path, clothmask_path, human_image_path, human_segm_path, ocloth_path, oclothfull_path, oclothmask_path)

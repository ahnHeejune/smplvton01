"""
  cloth 3d model reconstruction based on SMPL body model   
 ------------------------------------------------------------

 (c) copyright 2019 heejune@seoultech.ac.kr

 In : 2D VTON of cloth to SMPL silhouette
      SMPL template model params file (pkl) 
      2D matched  cloth image file and mask 

 Out: plk or npz file for subset of SMPL vertices and displacement vector

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
from opendr.lighting import SphericalHarmonics, LambertianPointLight
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

#
#
#


def build_smplbody_surface(model, pose, betas, cam):

    n_betas = betas.shape[0]
    viz = False

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
        want_Jtr=not viz)  # need J_transformed for reposing based on vertices

    return sv


#
# 3D Body shape => 2D Body shape => cloth 2D Shape
#
# in order to have the cooresponding body vertex  for each cloth vertex, we deform body shape
# here we do the 2D mask based and then later will get the 3D coordinates
#
# model : smpl  mesh  structure (esp. faces)
# body_sv :  3d  body vertices
# cam  : camera paramter
# imMask :  mask for body  + cloth area
# return: clothed2dvt : for body + cloth area
#
def construct_clothed2d_from_body(model, body_sv, j2d, cam, mask):

    h, w = mask.shape
    # 1. extract edge vertices

    # cam  should be set with cam.v = body_sv.r
    # 1.1 construct face_visibility map in 3D body shape
    f_normal, v_normal = graphutil.calc_normal_vectors(cam.v.r, model.f)
    face_visibility = graphutil.build_face_visibility(f_normal, cam)

    # 1.2. extract edge vertices
    check_edge_vertices = True
    '''
         graph analysis data structure:  vertex, edge, vs face 

         face to edges: mdoel.f
         face to vertices: model.f
         vertex to edges: graph
         vertex to faces: graph
         edge to vertices: graph
         edge to faces: graph


        graph[start_v][end_v][0, 1,  2 (contour label)] 
    '''
    graph, longest_contour_label, con_length = graphutil.build_edge_graph_dict(
        cam.v, model.f, face_visibility)

    # num_body_vertices = np.count_nonzero(
    #    graph[:, :, 2] == longest_contour_label)
    num_edge_vertices = np.amax(con_length)
    #print("edge v number:", num_edge_vertices)
    # if save_edge_vertices:
    edge_vertices = np.zeros([num_edge_vertices, 2], dtype='int32')

    # visualization of contour
    img_contour = np.zeros([h, w], dtype='uint8')
    i = 0
    if check_edge_vertices or save_edge_vertices:
        for v_s in range(cam.v.shape[0]):
            # for v_e in range(v_s):
            for v_e in graph[v_s]:
                # if graph[v_s, v_e, 2] == longest_contour_label:  # > 0:
                if graph[v_s][v_e][2] == longest_contour_label:  # > 0:
                    if check_edge_vertices:
                        sx, sy = cam.r[v_s]  # projected coordinate
                        ex, ey = cam.r[v_e]
                        edge_vertices[i, 0], edge_vertices[i, 1] = int(
                            sx), int(sy)
                        i = i + 1
                        cv2.line(img_contour, (int(sx), int(sy)), (int(
                            ex), int(ey)), graph[v_s][v_e][2], thickness=1)

    # boudnary matching
    # Body Part
    ##############################################
    # 1.1 read boundary matching input files
    #img_idx = 1
    #maskfile = "../results/10k/segmentation/10kgt_%04d.png"%img_idx
    #mask = cv2.imread(maskfile, cv2.IMREAD_UNCHANGED)
    # if mask is None:
    #    print("cannot open",  maskfile), exit()
    '''
    edge_vertices_path  ='edge_vertices_%04d.pkl'%img_idx
    with  open(edge_vertices_path, 'rb')  as f:
        edge_vertices  = pickle.load(f)
    '''
    # 1.3 boudnary matching
    #neck_xy = (j2d[12,0], j2d[12,1])
    neck_y = j2d[12, 1]
    lsh_y = j2d[9, 1]
    rsh_y = j2d[8, 1]
    top_y = int((neck_y + lsh_y + rsh_y)/3.0)
    nearest_list, img_allcontours = boundary_matching.boundary_match(
        mask, edge_vertices, top_y, step=5)
    # print(j2d), print(j2d[12,:2]), exit()

    # joints matching added
    # j2d will be added for source and tgt ....
    njoints = j2d.shape[0]

    # 2. transform
    # 2.1 adaptation of matching data
    nboundarypts = len(nearest_list)
    npts = nboundarypts + njoints
    srcPts = np.zeros([1, npts, 2], dtype='float32')
    tgtPts = np.zeros([1, npts, 2], dtype='float32')
    for i in range(nboundarypts):
        srcPts[0, i, :] = nearest_list[i][0]
        tgtPts[0, i, :] = nearest_list[i][1]
        #print(tgtPts[0,i,:], srcPts[0,i,:])

    for i in range(nboundarypts, npts):
        srcPts[0, i, :] = j2d[i-nboundarypts, :]
        tgtPts[0, i, :] = j2d[i-nboundarypts, :]

    # 2.2 estimate TPS params
    tps = boundary_matching.estimateTPS(srcPts, tgtPts, 10)

    # 3. deform the boundary 2D vertices using TPS
    '''
    body2dvt1 =  edge_vertices.astype('float32').reshape(1, -1,2)
    print('>>> Edge vertices <<<;-')
    print('type:', body2dvt1.dtype)
    print('shape:', body2dvt1.shape)
    print('x:', np.amin(body2dvt1[:,:,0]),  np.amax(body2dvt1[:,:,0]))
    print('y:', np.amin(body2dvt1[:,:,1]),  np.amax(body2dvt1[:,:,1]))
    print(body2dvt1)
    transformed = tps.applyTransformation(body2dvt1)
    print(transformed)
    '''
    body2dvt_save = cam.r.copy()
    body2dvt = cam.r.copy().reshape(1, -1, 2).astype('float32')
    '''
    print('>>> all vertices <<<;-')
    print('type:', body2dvt.dtype)
    print('shape:', body2dvt.shape)
    print('x:', np.amin(body2dvt[:,:,0]),  np.amax(body2dvt[:,:,0]))
    print('y:', np.amin(body2dvt[:,:,1]),  np.amax(body2dvt[:,:,1]))
    print(body2dvt)
    '''
    transformed = tps.applyTransformation(body2dvt)
    # print(transformed)

    clothed2dvt = transformed[1].reshape(-1, 2)
    body2dvt = body2dvt.reshape(-1, 2)

    return clothed2dvt


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
        #exit()

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


# convert numpy to json for a single person joint
# connvert
# 1) uint8 image to float texture image
# 2) normalize the vertices
# optionally,
# 3) coloring the backsize if face visibiltiy is not None)
# ***note ****:   texture coordinate is UP-side Down, and x-y  normalized
# j
# def prepare_texture(imv2d, faces, im4texture, skin_color_b, skin_color_g, skin_color_r):
def prepare_texture(imv2d, faces, im4texture):

    # add arms to textures
    """img_skin = np.zeros(
        [im4texture.shape[0], im4texture.shape[1], 3], dtype=np.uint8)

    # img_skin.fill(255)  # make skin color
    img_skin[:, :, 0] = int(skin_color_b)
    img_skin[:, :, 1] = int(skin_color_g)
    img_skin[:, :, 2] = int(skin_color_r)

    # target cloth + skin-colored painting
    im4texture = im4texture + img_skin * (im4texture == 0)"""

    # texture  = overlayed images of 2d and projected.
    texture = im4texture.astype(float)/255.0  # uint8 to float

    #print('dtype of img:',  img.dtype)
    #print('dtype of pjt_texture:',  pjt_texture.dtype)
    th, tw = texture.shape[0:2]
    '''
    pjt_texture[:,:,:] = (1.0, .0, .0)  #  
    #pjt_texture[:,:int(tw/2),:] = (1.0, 0., 0.)  # B, G, R 
    pjt_texture[:,int(tw/4):int(3*tw/4),:] = (1.0, 1.0, 1.0)  # B, G, R 
    '''
    #print("th, tw:", th, tw)
    texture_v2d = np.stack(
        (imv2d[:, 0]/tw, imv2d[:, 1]/th), axis=-1)  # uv normalize

    # 5. project the body model with texture renderer
    # 3. reprojection
    # print(type(cam.v))
    # print(cam.v.r.shape)

    #print("textured:",  type(pjt_texture), 'dtype:', pjt_texture.dtype, "shape:",  pjt_texture.shape)
    # print('max:', np.amax(pjt_texture[:, :, 0]), np.amax(
    #    pjt_texture[:, :, 1]), np.amax(pjt_texture[:, :, 2]))
    # print('meam:', np.mean(pjt_texture[:, :, 0]), np.mean(
    #    pjt_texture[:, :, 1]), np.mean(pjt_texture[:, :, 2]))
    #  apply the visibility map for texturing

    return texture, texture_v2d


def prepare_body_texture(imv2d, c3dw, im_path, im_parse_path):

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

    # get full target body texture except upper-clothes
    im = Image.open(im_path)
    im_parse = Image.open(im_parse_path)
    parse_array = np.array(im_parse)

    parse_top = (parse_array == 1) + \
                 (parse_array == 2) + \
                 (parse_array == 4) + \
                 (parse_array == 8) + \
                 (parse_array == 13) + \
                 (parse_array == 20)

    im_top = im * parse_top - (1 - parse_top)  # [-1,1], fill 0 for other parts

    parse_cloth = (parse_array == 0) + \
                  (parse_array == 3) + \
                  (parse_array == 5) + \
                  (parse_array == 6) + \
                  (parse_array == 7) + \
                  (parse_array == 10) + \
                  (parse_array == 11) + \
                  (parse_array == 14) + \
                  (parse_array == 15) + \
                  (parse_array == 20)

    im_cloth = c3dw * parse_cloth - (1 - parse_cloth)  # [-1,1], fill 0 for other parts

    parse_bottom = (parse_array == 9) + \
                 (parse_array == 12) + \
                 (parse_array == 16) + \
                 (parse_array == 17) + \
                 (parse_array == 18) + \
                 (parse_array == 19) + \
                 (parse_array == 20)

    im_bottom = im * parse_bottom - (1 - parse_bottom)  # [-1,1], fill 0 for other parts

    # texture  = overlayed images of 2d and projected.
    full_body = im_top + im_cloth + im_bottom

    h, w = full_body.shape[0:2]
    h_ext = h * 3//2
    im_body_ext = np.zeros([h_ext, w, 3], dtype='uint8')
    im_body_ext[:h, :, :] = full_body

    # texture = im_body_ext.astype(float)/255.0  # uint8 to float
    imHuman = cv2.imread(im_path)
    texture = imHuman.astype(float)/255.0  # uint8 to float

    th, tw = texture.shape[0:2]

    texture_v2d = np.stack(
        (imv2d[:, 0]/tw, imv2d[:, 1]/th), axis=-1)  # uv normalize

    """plt.subplot(1, 3, 1)
    plt.axis('off')
    plt.imshow(im_cloth)
    plt.title('cloth')

    plt.subplot(1, 3, 2)
    plt.axis('off')
    plt.imshow(im_top)
    plt.title('body')

    plt.subplot(1, 3, 3)
    plt.axis('off')
    plt.imshow(full_body)
    plt.title('texture')

    plt.show()
    _ = raw_input("next?")"""

    return texture, texture_v2d


#
# texture processing with alpha blending
def prepare_texture_with_alpha(pjt_v, pjt_f, img, mask, target_label):

    alpha = np.zeros(mask.shape)
    # 1.0 for fully opaque, 0.0 for transparent
    alpha[mask == target_label] = 1.0

    rgb = img.astype(float)/255.0  # uint8 to float
    rgba = cv2.merge((rgb, alpha))
    print('shapes:', img.shape, rgb.shape, alpha.shape, rgba.shape)

    th, tw = rgba.shape[0:2]
    pjt_v[:, 0] = pjt_v[:, 0]/tw  # uv normalize
    pjt_v[:, 1] = pjt_v[:, 1]/th  # uv normalize

    return rgba  # [:,:,:3]

# create V, A, U, f: geom, bright, cam, renderer


def build_texture_renderer(U, V, f, vt, ft, texture, w, h, ambient=0.0, near=0.5, far=20000, background_image=None):

    # add lighting
    A = SphericalHarmonics(vn=VertNormals(v=V, f=f),
                           components=[0., 0., 0., 0., 0., 0., 0., 0., 0.],
                           light_color=ch.ones(3)) + ambient

    """A = LambertianPointLight(
        f=f,
        v=V,
        num_verts=len(V),
        light_pos=ch.array([-500,-500,-500]),
        vc=np.ones_like(V.r),
        # light_color=ch.array([0.7, 0.7, 0.7])) + 0.3
        light_color=ch.array([1.0, 1.0, 1.0])) + 0.5  # brighter"""

    if background_image is not None:
        R = TexturedRenderer(vc=A, camera=U, f=f, bgcolor=[0.0, 0.0, 0.0],
                             texture_image=texture, vt=vt, ft=ft,
                             frustum={'width': w, 'height': h, 'near': near, 'far': far}, background_image=background_image)

    else:
        R = TexturedRenderer(vc=A, camera=U, f=f, bgcolor=[0.0, 0.0, 0.0],
                             texture_image=texture, vt=vt, ft=ft,
                             frustum={'width': w, 'height': h, 'near': near, 'far': far})

    return R


# display 3d model

def show_3d_model(cam, _texture, texture_v2d, faces, normalImage=False):

    #h, w = imTexture.shape[:2]
    h, w = _texture.shape[:2]
    dist = 20.0

    if normalImage:
        texture = prepare_texture(texture_v2d, faces, _texture)
    else:
        texture = _texture

    # 1. build texture renderer
    texture_renderer = build_texture_renderer(cam, cam.v, faces, texture_v2d, faces,
                                              texture[::-1, :, :], w, h, 1.0, near=0.5, far=20 + dist)
    #textured_cloth2d = texture_renderer.r

    # plt.figure()
    plt.subplot(1, 5, 1)
    plt.axis('off')
    plt.imshow(texture[:, :, ::-1])
    plt.title('input')

    rot_axis = 1
    rotation = ch.zeros(3)
    rotation[rot_axis] = np.pi/4
    img0 = texture_renderer.r[:, :, ::-1]*255.0
    img0 = img0.astype('uint8')
    for i in range(4):
        plt.subplot(1, 5, i+2)
        # plt.imshow(pjt_R.r)
        plt.imshow(texture_renderer.r)
        plt.axis('off')
        # plt.draw()
        # plt.show()
        #plt.title('angle =%f'%yaw)
        plt.title('%.0f degree' % (i*45))
        cam.v = cam.v.dot(Rodrigues(rotation))

    plt.show()


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
# get subset numpy 2-D array of triangles
# whose all 3 vertices or one or them are included in the target vertices set
#
#
def getSubsetFaces(ifaces, set_v, smpl_model, allinclusion):

    # get arm/hand vertices
    """set_hand = []
    for i in range(smpl_model.shape[0]):
        if smpl_model.weights_prior[i][13] > 0 or smpl_model.weights_prior[i][14] > 0 or smpl_model.weights_prior[i][16] > 0 or smpl_model.weights_prior[i][17] > 0 or smpl_model.weights_prior[i][18] > 0 or smpl_model.weights_prior[i][19] > 0 or smpl_model.weights_prior[i][20] > 0 or smpl_model.weights_prior[i][21] > 0 or smpl_model.weights_prior[i][22] > 0 or smpl_model.weights_prior[i][23] > 0:
            set_hand.append(i)"""

    # get wrist vertices
    set_wrist = []
    for i in range(smpl_model.shape[0]):
        if smpl_model.weights_prior[i][22] > 0 or smpl_model.weights_prior[i][23] > 0:
            set_wrist.append(i)

    flags = np.zeros(ifaces.shape[0], dtype=np.bool)
    for i in range(ifaces.shape[0]):

        # no need to check for cloth, we will take all faces
        # v1, v2, v3  = ifaces[i]
        """mask_v = np.isin(ifaces[i], set_v)
        if (mask_v[0] == True) and (mask_v[1] == True) and (mask_v[2] == True):
            flags[i] = True"""
        # else:
        #   flags[i]  = False

        # add the hand parts
        """hand_v = np.isin(ifaces[i], set_hand)
        if (hand_v[0] == True) and (hand_v[1] == True) and (hand_v[2] == True):
            flags[i] = True"""

        # remove the wrist parts
        wrist_v = np.isin(ifaces[i], set_wrist)
        if (wrist_v[0] == True) and (wrist_v[1] == True) and (wrist_v[2] == True):
            flags[i] = False
        else:
            flags[i] = True

    return ifaces[flags, :]


#
# calculate pixel position of SMPL joints
#
#  cam: camera ie. projector
#  model: smpl basic mdoel
#  sv:  surfac vectors  (opendr)
#  betas : body shape, why needed?
#  h: projection image height
#  w: projection image width
def calculate_joints(cam, model, sv, betas, h, w):

    # 1. get the joint locations
    # ,   12 ] # index in Jtr # @TODO correct neck
    smpl_ids = [8, 5, 2, 1, 4, 7, 21, 19, 17, 16,   18,   20]
    #                                         lsh,lelb,   lwr, neck

    # make the SMPL joints depend on betas
    Jdirs = np.dstack([model.J_regressor.dot(model.shapedirs[:, :, i])
                       for i in range(len(betas))])
    J_onbetas = ch.array(Jdirs).dot(betas) + \
        model.J_regressor.dot(model.v_template.r)

    # get joint positions as a function of model pose, betas and trans
    (_, A_global) = global_rigid_transformation(
        sv.pose, J_onbetas, model.kintree_table, xp=ch)
    Jtr = ch.vstack([g[:3, 3] for g in A_global]) + sv.trans

    # add joints, with corresponding to a vertex...
    neck_id = 3078  # 2951 #3061 # viton's bewtween shoulder
    Jtr = ch.vstack((Jtr, sv[neck_id]))
    smpl_ids.append(len(Jtr) - 1)
    # head_id = 411
    nose_id = 331  # nose vertex id
    Jtr = ch.vstack((Jtr, sv[nose_id]))
    smpl_ids.append(len(Jtr) - 1)
    lear_id = 516
    Jtr = ch.vstack((Jtr, sv[lear_id]))
    smpl_ids.append(len(Jtr) - 1)
    rear_id = 3941  # 422# 226 #396
    Jtr = ch.vstack((Jtr, sv[rear_id]))
    smpl_ids.append(len(Jtr) - 1)
    leye_id = 125  # 220 # 125
    Jtr = ch.vstack((Jtr, sv[leye_id]))
    smpl_ids.append(len(Jtr) - 1)
    reye_id = 3635
    Jtr = ch.vstack((Jtr, sv[reye_id]))
    smpl_ids.append(len(Jtr) - 1)

    # 2. project SMPL joints on the image plane using the estimated camera
    cam.v = Jtr

    joints_np_wo_confidence = cam.r[smpl_ids]  # get the projected value
    # print(joints_np_wo_confidence)
    joints_np = np.zeros([18, 3])
    joints_np[:, :2] = joints_np_wo_confidence
    joints_np[:, 2] = 1.0

    for i in range(joints_np.shape[0]):
        if joints_np[i, 0] < 0 or joints_np[i, 0] > (w-1) or joints_np[i, 1] < 0 or joints_np[i, 1] > (h-1):
            joints_np[i, 2] = 0.0

    # print(joints_np)
    return joints_np


def cvt_joints_np2json(joints_np):

    # 1. re-ordering
    # same as viton2lsp_joint and reamining
    order = [13, 12, 8, 7, 6, 9, 10, 11, 2, 1, 0, 3, 4, 5, 14, 15, 16, 17]

    # 2. build dictionary
    oneperson = {"face_keypoints": [],
                 "pose_keypoints": joints_np[order].flatten().tolist(),
                 "hand_right_keypoints": [],
                 "hand_left_keypoints": []}

    people = [oneperson]
    joints_json = {"version": 1.0, "people": people}

    return joints_json


def get_skin_color_from_image(im_path, im_parse_path):
    imBody = cv2.imread(im_path)
    imBodySegm = cv2.imread(im_parse_path)

    # neck/skin
    skin_color_b = np.mean(imBody[imBodySegm[:, :, 0] == 20, 0])
    skin_color_g = np.mean(imBody[imBodySegm[:, :, 1] == 20, 1])
    skin_color_r = np.mean(imBody[imBodySegm[:, :, 2] == 20, 2])

    print('skin color:', skin_color_r, skin_color_g, skin_color_b)
    skin_color = np.mean([skin_color_b, skin_color_g, skin_color_r])

    # face
    kin_color_b = np.mean(imBody[imBodySegm[:, :, 0] == 13, 0])
    kin_color_g = np.mean(imBody[imBodySegm[:, :, 1] == 13, 1])
    kin_color_r = np.mean(imBody[imBodySegm[:, :, 2] == 13, 2])

    print('face color:', kin_color_r, kin_color_g, kin_color_b)
    face_color = np.mean([kin_color_b, kin_color_g, kin_color_r])

    # left-arm
    in_color_b = np.mean(imBody[imBodySegm[:, :, 0] == 14, 0])
    in_color_g = np.mean(imBody[imBodySegm[:, :, 1] == 14, 1])
    in_color_r = np.mean(imBody[imBodySegm[:, :, 2] == 14, 2])

    print('left-arm color:', in_color_r, in_color_g, in_color_b)
    larm_color = np.mean([in_color_b, in_color_g, in_color_r])

    # right-arm
    n_color_b = np.mean(imBody[imBodySegm[:, :, 0] == 15, 0])
    n_color_g = np.mean(imBody[imBodySegm[:, :, 1] == 15, 1])
    n_color_r = np.mean(imBody[imBodySegm[:, :, 2] == 15, 2])

    print('right-arm color:', n_color_r, n_color_g, n_color_b)
    rarm_color = np.mean([n_color_b, n_color_g, n_color_r])

    brightest_color = np.max([skin_color, face_color, larm_color, rarm_color])

    if face_color == brightest_color:
        skin_color_b = kin_color_b
        skin_color_g = kin_color_g
        skin_color_r = kin_color_r
        print("face color is chosen.")
    elif larm_color == brightest_color:
        skin_color_b = in_color_b
        skin_color_g = in_color_g
        skin_color_r = in_color_r
        print("left-arm color is chosen.")
    elif rarm_color == brightest_color:
        skin_color_b = n_color_b
        skin_color_g = n_color_g
        skin_color_r = n_color_r
        print("right-arm color is chosen.")
    else:
        print("skin color is chosen.")

    return skin_color_b, skin_color_g, skin_color_r


#
#  cloth 3D model reconstrction  using  2d cloth (mapped onto template) and template
#
def cloth3drec_core(model,    # SMPL model
                    cam,      # camera model, Chv
                    betas,    # shape coef, numpy
                    n_betas,  # num of PCA
                    pose,     # angles, 27x3 numpy
                    imCloth,   # img numpy
                    imClothMask,  # img numpy
                    human_path,  # img numpy
                    human_segm_path,  # img numpy
                    viz=False):     # visualize or not

    # get human image and segm
    imBody = cv2.imread(human_path)
    if imBody is None:
        print("cannot open",  human_path), exit()

    imBodySegm = cv2.imread(human_segm_path, cv2.IMREAD_UNCHANGED)
    if imBodySegm is None:
        print("cannot open",  human_segm_path), exit()

    for which in [cam,  betas,  pose, imCloth, imClothMask, imBody, imBodySegm, model]:
        if which is None:
            print(retrieve_name(which),  'is  None')
            exit()

    h, w = imCloth.shape[0:2]
    h_ext = h * 3//2

    print(imClothMask.shape)
    print(len(imClothMask.shape))
    if len(imClothMask.shape) > 2:  # ie. 3 ch  to 1 ch
        imClothMask = cv2.cvtColor(imClothMask, cv2.COLOR_BGR2GRAY)

    # get skin/face color for hand/skin area painting
    # skin_color_b, skin_color_g, skin_color_r = get_skin_color_from_image(imBody, imBodySegm)

    print(imBodySegm.shape)
    print(len(imBodySegm.shape))
    if len(imBodySegm.shape) > 2:  # ie. 3 ch  to 1 ch
        imBodySegm = cv2.cvtColor(imBodySegm, cv2.COLOR_BGR2GRAY)

    # h vs h_ext
    # half body image has size of h x w
    # rendering needs full body texture image : size of h_ext x w
    '''
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
    print('pose:',  pose.reshape([-1,3]))
    print('betas:', betas)

   '''

    # 1. Prepare input images and masks
    # 1.1 build template body model
    body_sv = build_smplbody_surface(model, pose, betas, cam)
    dist = np.abs(cam.t.r[2] - np.mean(body_sv.r, axis=0)[2])
    im3CBlack = np.zeros([h_ext, w, 3], dtype=np.uint8)
    imBackground = im3CBlack
    imBodyRGB = (render_model(
        body_sv.r, model.f, w, h_ext, cam, far=20 + dist, img=imBackground[:, :, ::-1]) * 255.).astype('uint8')

    imBodyRGB_body = imBodyRGB.copy()

    # 1.2 source cloth image and mask  (extension for the same size as body silhouette)
    imClothedMask = cv2.cvtColor(
        imBodyRGB, cv2.COLOR_BGR2GRAY)  # gray silhouette
    imClothedMask[imClothedMask > 0] = 255  # binary (0, 1)
    # imClothedMask[imClothMask[:,:] > 0]  = 255   # union of body and .....
    # blank background image
    imClothMask_ext = np.zeros([h_ext, w], dtype='uint8')
    imClothMask_ext[:h, :] = imClothMask[:, :]
    imClothedMask[imClothMask_ext > 0] = 255   # union of body and .....
    # blank background image
    imCloth_ext = np.zeros([h_ext, w, 3], dtype='uint8')
    imCloth_ext[:h, :, :] = imCloth[:, :, :]
    imCloth_ext[imClothMask_ext <= 0] = 0  # black out
    imBodyRGB[imClothMask_ext > 0] = imCloth_ext[imClothMask_ext > 0]

    if viz:  # show cloth  overlayed on smpl
        plt.imshow(imBodyRGB[:h, :, ::-1])
        # plt.imshow(imClothedMask)
        # print("overlaid")
        plt.draw()
        plt.show()
        _ = raw_input('next?')

    # 2. Derive 2D cloth vertics position from body vertices using 2-D deformation
    j2d = calculate_joints(cam, model, body_sv, betas, h_ext, w)
    j2d_wo_confidence = j2d[:, :2]
    cam.v = body_sv
    clothed2d = construct_clothed2d_from_body(
        model, body_sv, j2d_wo_confidence, cam, imClothedMask)
    if viz:  # show the clothed 2d vertices
        marksize = 1
        # blank background image
        imClothed2d = np.zeros([h_ext, w], dtype='uint8')
        for i in range(clothed2d.shape[0]):
            x, y = int(clothed2d[i, 0]), int(clothed2d[i, 1])
            imClothed2d[y-marksize:y+marksize, x-marksize:x+marksize] = 255
            #cv2.drawMarker(img[:,:, parts[i]], (x,y), 255, markerSize = 5)
        plt.imshow(imClothed2d)
        plt.draw()
        plt.show()
        _ = raw_input('next?')

    # 3. 3D cloth vertices from 2d position and depth
    clothed3d = construct_clothed3d_from_clothed2d_depth(
        body_sv, cam, clothed2d)
    cam.v = clothed3d  # now camera  project clothed 3D vertex not body's

    # 4. Rendering texture
    # 4.1 updatng the cloth mask and image at boundary
    '''
    imClothMask1d = imClothMask[t].flatten()  
    print(imClothMask1d.shape)
    print(imClothMask1dv4Cloth)
    '''
    # extend the mask boundary  for hide the mismatch beween mask and image for rendering texture
    kernel = np.ones((3, 3), np.uint8)
    imClothMask_ext_raw = imClothMask_ext.copy()
    imClothMask_ext = cv2.dilate(imClothMask_ext, kernel, iterations=2)

    # modify the boundary
    imClothMask_ext_bndry = imClothMask_ext.copy()
    imClothMask_ext_bndry[imClothMask_ext_raw > 0] = 0

    imCloth_ext_bndry = imCloth_ext.copy()
    imCloth_ext_bndry[:, :, 0] = cv2.dilate(
        imCloth_ext[:, :, 0], kernel, iterations=2)
    imCloth_ext_bndry[:, :, 1] = cv2.dilate(
        imCloth_ext[:, :, 1], kernel, iterations=2)
    imCloth_ext_bndry[:, :, 2] = cv2.dilate(
        imCloth_ext[:, :, 2], kernel, iterations=2)
    # imCloth_ext[imClothMask_ext_bndry > 0, :] = (255, 0, 0) # draw the boundary with Blue
    imCloth_ext[imClothMask_ext_bndry > 0,
                :] = imCloth_ext_bndry[imClothMask_ext_bndry > 0, :]

    if viz:
        plt.subplot(1, 2, 1)
        plt.imshow(imCloth_ext[:, :, ::-1])
        plt.subplot(1, 2, 2)
        # plt.imshow(imCloth_ext_not_modified[:,:,::-1])
        plt.imshow(imClothMask_ext_bndry)
        plt.show()
        _ = raw_input('next?')

    # 4.2 Prepare Rendering Texture
    # texture, texture_v2d = prepare_texture(cam.r, model.f, imCloth_ext, skin_color_b, skin_color_g, skin_color_r)
    texture, texture_v2d = prepare_texture(cam.r, model.f, imCloth_ext)

    # 5. Collect the vertices for the cloth  surface
    # print(cam.r.shape)
    # print(imauto-normals.ClothMask.shape)
    # print(imClothMask[cam.r.astype(np.uint8)].shape)
    pjt_positions = cam.r.astype(np.uint8)
    # print(imClothMask[pjt_positions].shape)
    # Matiur contribution
    imClothMask1d = imClothMask_ext[pjt_positions[:, 1], pjt_positions[:, 0]]
    # print(imClothMask1d.shape)
    v4Cloth = np.argwhere(imClothMask1d > 0).flatten()

    # check v4Cloth is right?, ie. it mapped onto the cloth
    if False:
        marksize = 1
        imTest = np.zeros([h_ext, w], dtype='uint8')  # blank background image
        for i in range(v4Cloth.shape[0]):
            iv = v4Cloth[i]
            x, y = int(pjt_positions[iv, 0]), int(pjt_positions[iv, 1])
            imTest[y-marksize:y+marksize, x-marksize:x+marksize] = 255
        plt.imshow(imTest)
        plt.draw()
        plt.show()
        _ = raw_input('next?')

    # 6. get the displacement vectors
    #print('vertices for cloth area:', v4Cloth.shape,  v4Cloth)
    # getting all value is easier to coding
    diffClothminusBody = clothed3d - body_sv.r
    #diffClothminusBody  =  clothed3d[v4Cloth]  - body_sv.r[v4Cloth]
    #print('diff cloth and body:', diffClothminusBody.shape, diffClothminusBody)

    return body_sv.r, clothed3d, diffClothminusBody, v4Cloth, texture, texture_v2d


#
# add the cloth displacement to the body surafce
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


# load dataset dependent files and call the core processing
# ---------------------------------------------------------------
# smpl_mdoel: SMPL
# inmodel_path : smpl param pkl file (by SMPLify)
# cloth_path: input image
# clothmask_path: mask 1-channel image
def cloth3drec_single(smpl_model, inmodel_path, cloth_path, clothmask_path, human_path, human_segm_path, viz=False):

    if smpl_model is None or inmodel_path is None or cloth_path is None or clothmask_path is None:
        print('There is None inputs'), exit()

    plt.ion()

    # model params
    with open(inmodel_path, 'rb') as f:
        if f is None:
            print("cannot open",  inmodel_path), exit()
        params = pickle.load(f)

    # params['pose'] = params['pose'] % (2.0*np.pi) # modulo

    cam = ProjectPoints(f=params['cam_f'], rt=params['cam_rt'],
                        t=params['cam_t'], k=params['cam_k'], c=params['cam_c'])
    params['cam'] = cam

    _examine_smpl_params(params)

    #  2d rgb image for texture
    imCloth = cv2.imread(cloth_path)
    if imCloth is None:
        print("cannot open",  cloth_path), exit()

    imClothMask = cv2.imread(clothmask_path)
    if imClothMask is None:
        print("cannot open",  clothmask_path), exit()

    # 3. run the SMPL body to cloth processing
    cam_src = params['cam']      # camera model, Ch
    betas_src = params['betas']
    n_betas_src = betas_src.shape[0]  # 10
    pose_src = params['pose']    # angles, 27x3 numpy
    body, clothed, diff_cloth_body, vertex4cloth, texture, texture_v2d = cloth3drec_core(smpl_model,  # SMPL
                                                                                         cam_src,      # camera model, Ch
                                                                                         betas_src,    # shape coeff, numpy
                                                                                         n_betas_src,  # num of PCA
                                                                                         pose_src,     # angles, 27x3 numpy
                                                                                         imCloth,    # img numpy
                                                                                         imClothMask,  # mask
                                                                                         human_path,  # human img
                                                                                         human_segm_path,  # human segmentation
                                                                                         viz=viz)    # display

    face4cloth = getSubsetFaces(smpl_model.f, vertex4cloth, smpl_model, True)
    show_3d_model(cam_src, texture, texture_v2d, face4cloth)  # smpl_model.f)
    # show_3d_model(cam_src, texture, texture_v2d, smpl_model.f)
    _ = raw_input('next?')

    return params, body, diff_cloth_body, texture, texture_v2d, face4cloth
    # return params, body, diff_cloth_body, texture, texture_v2d, smpl_model.f


def cloth3drec_single_xfer_test(smpl_model, inmodel_path, cloth_path, clothmask_path):

    params, body, diff_cloth_body, texture, texture_v2d, face4cloth = cloth3drec_single(
        smpl_model, inmodel_path, cloth_path, clothmask_path, True)

    # express the displacement in vertice specific coordinate.
    diff_cloth_body_local = compute_displacement_at_vertex(
        smpl_model, body, diff_cloth_body)

    # 4. (should separate into another script file)
    # try to another shape and posed person, kind of trabsfer
    # for test purpose, simple copy the template and repose and reshape as you like
    # 4.1 initial body params
    cam_tgt = ProjectPoints(f=params['cam_f'], rt=params['cam_rt'],
                            t=params['cam_t'], k=params['cam_k'], c=params['cam_c'])
    betas_tgt = params['betas']
    n_betas_tgt = betas_tgt.shape[0]  # 10
    pose_tgt = params['pose']    # angles, 27x3 numpy
    #pose_tgt = pose_src.copy()
    #n_betas_tgt = n_betas_src
    #betas_tgt = betas_src.copy()
    # 4.2 repose and reshape

    # pose_tgt[16*3] =  np.pi/4  # left shoulder # rotate
    pose_tgt[16*3+1] = -np.pi/3  # front/back
    # pose_tgt[16*3+2] =  -np.pi/3  # side
    # pose_tgt[17*3] =  np.pi/4  # right  shoulder
    pose_tgt[17*3+1] = np.pi/3
    #pose_tgt[17*3+2] =  -np.pi/3
    # pose_tgt[18*3] =  np.pi/4  # left elbow
    #pose_tgt[18*3+1] =  np.pi/3
    #pose_tgt[18*3+2] =   np.pi/6
    # pose_tgt[19*3] =  np.pi/4  # right elbow
    #pose_tgt[19*3+1] =  np.pi/4
    #pose_tgt[19*3+2] =  np.pi/6

    betas_tgt[1] = 0.0

    # 4.3 build a new body
    body_tgt_sv = build_smplbody_surface(
        smpl_model, pose_tgt, betas_tgt, cam_tgt)

    # 4.4 build the corresponding clothed
    clothed3d = transfer_body2clothed(
        smpl_model, body_tgt_sv.r, diff_cloth_body_local)
    cam_tgt.v = clothed3d
    #cam_tgt.v = body_tgt_sv.r

    # 4.5 check by viewing
    # smpl_model.f) # cam_tgt has all the information
    show_3d_model(cam_tgt, texture, texture_v2d, face4cloth)

    '''
    # save result for checking
    if outimg_path is not None:
       cv2.imwrite(outimg_path, img_mask)
    if outjoint_path is not None:
        with  open(outjoint_path, 'w') as joint_file:
            json.dump(joints_json, joint_file)
    '''


if __name__ == '__main__':

    '''
    if len(sys.argv) < 5:
       print('usage: %s  ase_path dataset start_idx end_idx(exclusive)'% sys.argv[0]), exit()

    # 1. directory check and setting
    base_dir = abspath(sys.argv[1])
    #print(base_dir)
    dataset = sys.argv[2]
    idx_s = int(sys.argv[3])
    idx_e= int(sys.argv[4])
    '''
    if len(sys.argv) < 5:
        print('usage: %s base_path smpl_param clothimg maskimg' %
              sys.argv[0]), exit()

    # 1. directory check and setting
    base_dir = abspath(sys.argv[1])
    if not exists(base_dir):
        print('No such a directory for base', base_path, base_dir), exit()
    smplparam_path = abspath(sys.argv[2])
    if not exists(smplparam_path):
        print('No such a file  for ', smplparam_path), exit()
    cloth_path = abspath(sys.argv[3])
    if not exists(cloth_path):
        print('No such a file  for ', cloth_path), exit()
    clothmask_path = abspath(sys.argv[4])
    if not exists(clothmask_path):
        print('No such a file for ', clothmask_path), exit()

    '''
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
    '''

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

    # Load joints
    '''
    estj2d = np.load(join(data_dir, 'est_joints.npz'))['est_joints']
    #print('est_shape:', est.shape)
    joints = estj2d[:2, :, idx].T
    '''

    # Output Directory
    '''
    smplmask_dir = data_dir + "/smplmask"
    if not exists(smplmask_dir):
        makedirs(smplmask_dir)

    smpljson_dir = data_dir + "/smpljson"
    if not exists(smpljson_dir):
        makedirs(smpljson_dir)
    smplmask_path = smplmask_dir + '/%06d_0.png'% idx 
    #jointfile_path = smpljson_dir + '/%06d_0.json'% idx 
    '''
    smpl_model = model_female
    if True:  # 3D reconstrction only
        cloth3drec_single(smpl_model, smplparam_path,
                          cloth_path, clothmask_path, True)
    else:     # 3D reconstruction and tranfer it to a define smpl model
        cloth3drec_single_xfer_test(
            smpl_model, smplparam_path, cloth_path, clothmask_path)

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

        smplmask_path = smplmask_dir + '/%06d_0.png'% idx 
        jointfile_path = smpljson_dir + '/%06d_0.json'% idx 
        smpl2mask_single(smpl_model, smpl_param_path, inp_path, smplmask_path,  jointfile_path, idx)
    '''

    # plt.pause(10)
    _ = raw_input('quit?')

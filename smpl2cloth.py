"""
  SMPL model to Cloth Model 
 -----------------------------

 (c) copyright 2019 heejune@seoultech.ac.kr

 Prerequisite: SMPL model 
 In : SMPL paramters(cam, shape, pose) for a image 
      Mask (cloth and body labeled)
      [optionally the input image]
 Out: label array and vertices 3D coordinates array 
      (i.e., the labeled 3-D model for image)
      [optionally the validating images]


  1. 2D image to 3D SMPL body model
    1.1 pre-calcuated fit data (camera, pose, shape)
    1.2 some demo (like texture mapping etc) 

  2. SMPL body model to  2D cloth matching   
    2.1 extract 2D projected edges
    2.2 matching the edge vertices and image bounday 
    2.3 morphing the body shape to cloth shape

  3. reconstrcut 3D cloth model from 2-D coordinate and depth 
    3.1 prepare depth 
    3.2 calcaute the x, y coordinates of  2D cloth model
    3.3 combined with Depth
   
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
    print(" - cam.camera_mtx:", camera.camera_mtx)  # 
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

#
# SMPL, 2dIMG => 3D cloth 
#
def smpl2cloth3dcore(cam,      # camera model, Chv
          betas,    # shape coef, numpy
          n_betas,  # num of PCA
          pose,     # angles, 27x3 numpy
          imgRGB,   # img numpy
          mask,     # label of img 
          j2d,      # joint 2d
          model,    # SMPL model 
          bHead = False,
          viz = False):     # visualize or not  

    for which in [cam,  betas,  pose, imgRGB, mask, model]:
        if which  is None:
            print( retrieve_name(which) ,  'is  None')
            exit()

    print('pose:', type(pose), pose)
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

    sv_r = sv.r.copy()
    num_axes = 0 
    axis_idx = 1

    #################################################################
    #  Step 1: data preparation 
    #################################################################
    # render the model with paramter
    h, w = imgRGB.shape[0:2]
    img = cv2.cvtColor(cv2.cvtColor(imgRGB, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)  # 3 channel gray 
    dist = np.abs(cam.t.r[2] - np.mean(sv.r, axis=0)[2])
    im = (render_model(
        sv.r, model.f, w, h, cam, far=20 + dist, img=img[:, :, ::-1]) * 255.).astype('uint8')

    # checking the redering result, but we are not using this.
    # we could drawing the points on it
    #print('th:', th,  '  tw:', tw)
    # plt.figure()
    img2 = img.copy()
    '''
    plt.imshow(img2)
    plt.hold(True)
    # now camera use only joints
    plt.plot(cam.r[:,0], cam.r[:, 1], 'r+', markersize=10) # projected pts 
    '''
    # project all vertices using camera
    cam.v = sv.r  # is this a vertices ?
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
            bodypartmap = graphutil.build_bodypartmap_2d(img, cam.r, bodyparts, body_colormap, h, w, False)
            num_axes = num_axes + 1
            '''
            print('part-max:', np.amax(bodyparts))
            plt.suptitle('body partmap')
            plt.subplot(1, 2, 1)
            plt.imshow(img[:, :, ::-1])  # , cmap='gray')
            plt.subplot(1, 2, 2)
            plt.imshow(bodypartmap[:,:,::-1])  # , cmap='gray')
            _ = raw_input('quit?')
            exit()
            '''

    #################################################################
    #  Step 2: Body shape to cloth Shape
    #################################################################
    # 1. extract edge vertices
    # 1.1 construct face_visibility map
    f_normal, v_normal = graphutil.calc_normal_vectors(cam.v, model.f)
    face_visibility = graphutil.build_face_visibility(f_normal, cam)

    # 1.3. extract edge vertices
    use_edge_vertices = True
    check_edge_vertices = True
    save_edge_vertices = False
    if use_edge_vertices:
        '''
         graph analysis data structure
         face to edges: mdoel.f
         face to vertices: model.f  
         vertex to edges: graph 
         vertex to faces: graph
         edge to vertices: graph   
         edge to faces: graph
        '''
        #graph, longest_contour_label = graphutil.build_edge_graph_matrix(
        graph, longest_contour_label, con_length = graphutil.build_edge_graph_dict(cam.v, model.f, face_visibility)

        #num_body_vertices = np.count_nonzero(
        #    graph[:, :, 2] == longest_contour_label)
        num_edge_vertices = np.amax(con_length)   
        #print("edge v number:", num_edge_vertices)
        #if save_edge_vertices:
        edge_vertices = np.zeros([num_edge_vertices, 2], dtype='int32')

        # visualization of contour
        img_contour = np.zeros([h, w], dtype='uint8')
        i = 0
        if check_edge_vertices or save_edge_vertices:
            num_axes = num_axes + 1
            for v_s in range(cam.v.shape[0]):
                #for v_e in range(v_s):
                for v_e in graph[v_s]:
                    #if graph[v_s, v_e, 2] == longest_contour_label:  # > 0:
                    if graph[v_s][v_e][2] == longest_contour_label:  # > 0:
                        if check_edge_vertices:
                            sx, sy = cam.r[v_s]  # projected coordinate
                            ex, ey = cam.r[v_e]
                            edge_vertices[i,0],edge_vertices[i,1]=int(sx),int(sy)
                            i = i + 1
                        cv2.line(img_contour, (int(sx), int(sy)), (int(
                            ex), int(ey)), graph[v_s][v_e][2], thickness=1)


        #############################################
        # edges in head region 
        ##############################################
        if bHead:
            # 1. get the contour edge in head area
            graph_head, _, head_con_lengths = graphutil.build_edge_graph_dict_part(cam.v, model.f, face_visibility, bodyparts, 15)
            print('Head vertex contours length:', head_con_lengths[:20])
            # 2. a head has quite complicated shape, so we have to outline of face 
            max_s, max_c = -1, -1
            for c in range(head_con_lengths.shape[0]):
                if head_con_lengths[c] > 0:  # non-zero
                    s = graphutil.calc_contour_area(cam.v.r, graph_head, c)
                    #print(c, s)
                    if s > max_s:
                        max_s = s
                        max_c = c
            print('Largest Head contours', max_s, max_c)
            head_contour_label = max_c
            num_head_edge_vertices = np.amax(head_con_lengths)   
            head_edge_vertices = np.zeros([num_head_edge_vertices, 2], dtype='int32')

            i = 0
            for v_s in range(cam.v.shape[0]):
                for v_e in graph_head[v_s]:
                    if graph_head[v_s][v_e][2] == head_contour_label: 
                        sx, sy = cam.r[v_s]  
                        head_edge_vertices[i,:]= (int(sx),int(sy))
                        i = i + 1
                        
            # 3. using the head contours, reverse rendering the head part
            #    deformation (mapping to mask)
            #    3D reconstrction (maybe together with body part.


        # save for reated use and slowness
        if save_edge_vertices:
            edge_vertices_path = "edge_vertices.pkl"
            with open(edge_vertices_path, 'w') as outf:
                pickle.dump(edge_vertices, outf)


    ##############################################
    # boudnary matching
    ##############################################
    # Body Part
    ##############################################
    # 1.1 read boundary matching input files
    #img_idx = 1
    #maskfile = "../results/10k/segmentation/10kgt_%04d.png"%img_idx
    #mask = cv2.imread(maskfile, cv2.IMREAD_UNCHANGED)
    #if mask is None:
    #    print("cannot open",  maskfile), exit()
    '''
    edge_vertices_path  ='edge_vertices_%04d.pkl'%img_idx
    with  open(edge_vertices_path, 'rb')  as f:
        edge_vertices  = pickle.load(f)
    '''
    # 1.3 boudnary matching
    #neck_xy = (j2d[12,0], j2d[12,1])
    neck_y = j2d[12,1]
    lsh_y  = j2d[9,1]
    rsh_y  = j2d[8,1]
    top_y  = int((neck_y + lsh_y +rsh_y)/3.0) 
    nearest_list, img_allcontours = boundary_matching.boundary_match(mask, edge_vertices, top_y, step = 5)
    #print(j2d), print(j2d[12,:2]), exit()
    
    # joints matching added 
    # j2d will be added for source and tgt ....
    njoints = j2d.shape[0]

    # 2. transform 
    # 2.1 adaptation of matching data 
    nboundarypts = len(nearest_list)
    npts = nboundarypts + njoints
    srcPts = np.zeros([1, npts, 2], dtype ='float32')
    tgtPts = np.zeros([1, npts, 2], dtype ='float32')
    for i in range(nboundarypts):
        srcPts[0,i,:] = nearest_list[i][0]
        tgtPts[0,i,:] = nearest_list[i][1]
        #print(tgtPts[0,i,:], srcPts[0,i,:])

    for i in range(nboundarypts, npts):
        srcPts[0,i,:] = j2d[i-nboundarypts,:] 
        tgtPts[0,i,:] = j2d[i-nboundarypts,:] 

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
    body2dvt_save =  cam.r.copy()
    body2dvt =  cam.r.copy().reshape(1, -1, 2).astype('float32')
    '''
    print('>>> all vertices <<<;-')
    print('type:', body2dvt.dtype)
    print('shape:', body2dvt.shape)
    print('x:', np.amin(body2dvt[:,:,0]),  np.amax(body2dvt[:,:,0]))
    print('y:', np.amin(body2dvt[:,:,1]),  np.amax(body2dvt[:,:,1]))
    print(body2dvt)
    '''
    transformed = tps.applyTransformation(body2dvt)
    #print(transformed)
    cloth2dvt = transformed[1].reshape(-1,2)
    body2dvt = body2dvt.reshape(-1,2)

    ###########################################################
    # HEAD PART (optional for single image animation purpose)
    # TODO: consider apply the deformation together with body part.
    #       Pro: more natural Con: unnecesary for cloth swapping
    ##########################################################
    if bHead:
        # 1. cooresponding points 
        nearest_list2, img_allcontours2 = boundary_matching.boundary_match_head(mask, head_edge_vertices, top_y, step = 5)
        #print(j2d), print(j2d[12,:2]), exit()

        # 2 adaptation of matching data 
        nboundarypts2 = len(nearest_list2)
        npts2 = nboundarypts2 
        srcPts2 = np.zeros([1, npts2, 2], dtype ='float32')
        tgtPts2 = np.zeros([1, npts2, 2], dtype ='float32')
        for i in range(nboundarypts2):
            srcPts2[0,i,:] = nearest_list2[i][0]
            tgtPts2[0,i,:] = nearest_list2[i][1]
            #print(tgtPts[0,i,:], srcPts[0,i,:])

        # 3 estimate TPS params
        tps_head = boundary_matching.estimateTPS(srcPts2, tgtPts2, 10)

        # 4. deform the bodypartcloth 2D vertices to fullcloth 
        halfcloth2dvt =  cloth2dvt.reshape(1, -1, 2) # 
        transformed2 = tps_head.applyTransformation(halfcloth2dvt)
        #print(transformed)
        temp = transformed2[1].reshape(-1,2)  
        cloth2dvt[bodyparts == 15, :] = temp[bodyparts==15,:]  # head-part only 
        #
        # end of head part 2D deformation 
        #

    # 3.2  check  
    check_body2cloth2d = True 
    check_body2cloth2d_detail = False 
    if  check_body2cloth2d:
        cloth2dmap = img.copy() 
        for i in range(cloth2dvt.shape[0]):
            cv2.drawMarker(cloth2dmap,(int(cloth2dvt[i,0]), int(cloth2dvt[i,1])), body_colormap[bodyparts[i]], markerSize=3)
        num_axes = num_axes + 1

        if check_body2cloth2d_detail:  
            # show the correspodning points before deformation 
            before = cv2.cvtColor(mask*28, cv2.COLOR_GRAY2BGR)  # to 3 channel gray 
            for i in range(body2dvt.shape[0]):
                #print(body2dvt[i,0])
                #rint(body2dvt[i,1])
                #cv2.drawMarker(before,(int(body2dvt[i,0]), int(body2dvt[i,1])), (255,255,255), markerSize=3)
                if bodyparts[i] == 15 : # or bodyparts[i] == 15: # neck or head
                    cv2.drawMarker(before,(int(body2dvt[i,0]), int(body2dvt[i,1])), (255,255,255), markerSize=3)
                    #print(i)
            '''
            for i in range(npts):
                sx, sy = int(srcPts[0,i,0]), int(srcPts[0,i,1])
                tx, ty = int(tgtPts[0,i,0]), int(tgtPts[0,i,1]) 
                cv2.drawMarker(before,(sx, sy), (0,255, 0), markerSize=3)
                cv2.drawMarker(before,(tx, ty), (0,0, 255), markerSize=3)
            '''
            for i in range(nboundarypts -1):
                sx0, sy0 = nearest_list[i][0][0], nearest_list[i][0][1] 
                sx1, sy1 = nearest_list[i+1][0][0], nearest_list[i+1][0][1] 
                tx0, ty0 = nearest_list[i][1][0], nearest_list[i][1][1] 
                tx1, ty1 = nearest_list[i+1][1][0], nearest_list[i+1][1][1] 
                # drawing boundary 
                #cv2.line(before, (sx0, sy0), (sx1,sy1), (0, 255,0), 1)
                #cv2.line(before, (tx0, ty0), (tx1,ty1), (0, 0, 255), 1)

                # draw mapping
                cv2.drawMarker(before,(sx0, sy0), (0,255, 0), markerType = cv2.MARKER_SQUARE, markerSize=5)
                cv2.drawMarker(before,(tx0, ty0), (0,0, 255), markerSize=5)
                cv2.line(before, (sx0, sy0), (tx0,ty0), (0, 255,0), 1)

            # head mapping
            if bHead:
                for i in range(nboundarypts2 -1):
                    sx0, sy0 = nearest_list2[i][0][0], nearest_list2[i][0][1] 
                    sx1, sy1 = nearest_list2[i+1][0][0], nearest_list2[i+1][0][1] 
                    tx0, ty0 = nearest_list2[i][1][0], nearest_list2[i][1][1] 
                    tx1, ty1 = nearest_list2[i+1][1][0], nearest_list2[i+1][1][1] 
                    # drawing boundary 
                    #cv2.line(before, (sx0, sy0), (sx1,sy1), (0, 255,0), 1)
                    #cv2.line(before, (tx0, ty0), (tx1,ty1), (0, 0, 255), 1)
                    # draw mapping
                    cv2.drawMarker(before,(sx0, sy0), (0,255, 0), markerType = cv2.MARKER_SQUARE, markerSize=5)
                    cv2.drawMarker(before,(tx0, ty0), (0,0, 255), markerSize=5)
                    cv2.line(before, (sx0, sy0), (tx0,ty0), (0, 255,0), 1)
                
            # showing all contours edge vertices 
            '''
            color_map = {1: (255, 255, 255), # this is face 
                   2: (255,  0, 255),
                   3: (0, 255, 255),
                   5: (128, 128, 128),
                   18: (255, 0, 0),
                   27: (0,  255, 0),
                   29: (0, 0, 255)}

            for s in range(graph.shape[0]):
                for t in graph[s]:
                    #if graph[s][t][2] != -1: # kind of edge
                    if  graph[s][t][2] in [1,2,3,5,18,27,29]:
                        sx, sy  = int(cam.r[s,0]), int(cam.r[s,1])
                        cv2.drawMarker(before,(sx, sy), color_map[graph[s][t][2]], markerType = cv2.MARKER_SQUARE, markerSize=3)
            '''
            # marking face area in RED color 
            for s in range(graph_head.shape[0]):
                for t in graph_head[s]:
                    #if graph_head[s][t][2] in [1,5,7,14]: # kind of edge
                    if graph_head[s][t][2] == head_contour_label: 
                        sx, sy  = int(cam.r[s,0]), int(cam.r[s,1])
                        cv2.drawMarker(before,(sx, sy), (0, 0, 255), markerType = cv2.MARKER_SQUARE, markerSize=3)

            # marking joints in GREEN 
            for i in range(j2d.shape[0]):
                sx, sy  = j2d[i,:] 
                #tx, ty  = j2d[i,:] 
                cv2.drawMarker(before,(int(sx), int(sy)), (0,255, 0), markerSize=5)
                #cv2.drawMarker(before,(int(tx), int(ty)), (0,0, 255), markerSize=5)

            plt.subplot(1,2,1)
            plt.imshow(before[:,:,::-1])
            plt.title('before 2D warping')
            plt.draw()

            # after deformation 
            after = cv2.cvtColor(mask*28, cv2.COLOR_GRAY2BGR)  # to 3 channel gray 
            for i in range(cloth2dvt.shape[0]):
                cv2.drawMarker(after,(int(cloth2dvt[i,0]), int(cloth2dvt[i,1])), (255,255,255), markerSize=3)
            '''
            for i in range(npts):
                sx, sy = int(srcPts[0,i,0]), int(srcPts[0,i,1])
                tx, ty = int(tgtPts[0,i,0]), int(tgtPts[0,i,1]) 
                cv2.drawMarker(after,(sx, sy), (0,255, 0), markerSize=3)
                cv2.drawMarker(after,(tx, ty), (0,0, 255), markerSize=3)
            '''
            # FIXME use the transformed results of sources and desired targets 
            for i in range(nboundarypts -1):
                sx0, sy0 = nearest_list[i][0][0], nearest_list[i][0][1] ## FIXME  
                sx1, sy1 = nearest_list[i+1][0][0], nearest_list[i+1][0][1] ## FIXME 
                tx0, ty0 = nearest_list[i][1][0], nearest_list[i][1][1] 
                tx1, ty1 = nearest_list[i+1][1][0], nearest_list[i+1][1][1] 
                # draw mapping
                cv2.drawMarker(before,(sx0, sy0), (0,255, 0), markerType = cv2.MARKER_SQUARE, markerSize=5)
                cv2.drawMarker(before,(tx0, ty0), (0,0, 255), markerSize=5)
                '''
                cv2.line(after, (sx0, sy0), (sx1,sy1), (0, 255,0), 1)
                # FIXME use the transformed 
                cv2.line(after, (tx0, ty0), (tx1,ty1), (0, 0, 255), 1)
                '''

            if bHead:
                for i in range(nboundarypts2 -1):
                    sx0, sy0 = nearest_list2[i][0][0], nearest_list2[i][0][1] 
                    sx1, sy1 = nearest_list2[i+1][0][0], nearest_list2[i+1][0][1] 
                    tx0, ty0 = nearest_list2[i][1][0], nearest_list2[i][1][1] 
                    tx1, ty1 = nearest_list2[i+1][1][0], nearest_list2[i+1][1][1] 
                    # drawing boundary 
                    #cv2.line(before, (sx0, sy0), (sx1,sy1), (0, 255,0), 1)
                    #cv2.line(before, (tx0, ty0), (tx1,ty1), (0, 0, 255), 1)
                    # draw mapping
                    cv2.drawMarker(before,(sx0, sy0), (0,255, 0), markerType = cv2.MARKER_SQUARE, markerSize=5)
                    cv2.drawMarker(before,(tx0, ty0), (0,0, 255), markerSize=5)
                    cv2.line(before, (sx0, sy0), (tx0,ty0), (0, 255,0), 1)
                
            for i in range(j2d.shape[0]):
                sx, sy  = j2d[i,:] # FIXME  
                tx, ty  = j2d[i,:] 
                cv2.drawMarker(after,(int(sx), int(sy)), (0,255, 0), markerSize=5)
                cv2.drawMarker(after,(int(tx), int(ty)), (0,0, 255), markerSize=5)

            plt.subplot(1,2,2)
            plt.imshow(after[:,:,::-1])
            plt.title('after 2D warping')
            _ = raw_input('quit?')
            exit()

    #################################################################
    #  Step 3: Reconstruction of 3D cloth model 
    #################################################################
    # 1. Depthmap at vertices
    # @TODO we should modify the depth for cloth
    use_depthmap = True
    check_depthmap = False 
    if use_depthmap:
        bodydepth = graphutil.build_depthmap2(sv.r, cam)
        if check_depthmap:
            # depth in reverse way
            plt.suptitle('depthmap')
            plt.subplot(1, 2, 1)
            plt.imshow(img[:, :, ::-1])  # , cmap='gray')
            plt.subplot(1, 2, 2)
            depthmap = graphutil.build_depthimage(sv.r,  model.f,  bodydepth, cam,  height=h, width=w, near= 0.5, far= 40)
            plt.imshow(depthmap, cmap='gray')
            plt.draw()
            plt.show()

            #plt.imshow(depthmap, cmap='gray_r') # the closer to camera, the brighter 
            _ = raw_input('quit?')
            exit()

    # 2. 2D cloth to 3D vertices   
    # check with body vertices, it works 
    '''
    body3d =  sv.r.copy()
    bodyuvd  = np.zeros(sv.r.shape)
    cam.v = body3d
    bodyuvd[:, :2] = cam.r
    bodyuvd[:, 2]  = bodydepth
    body3d_r = cam.unproject_points(bodyuvd)
    diff = body3d_r - body3d
    print( 'diff-x:',  np.amin(diff[:,0]),np.amax(diff[:,0])) 
    print( 'diff-y:',  np.amin(diff[:,1]),np.amax(diff[:,1])) 
    print( 'diff-z:',  np.amin(diff[:,2]),np.amax(diff[:,2])) 
    '''
    # uv space? pixels coordinated!! 
    clothuvd  = np.zeros(sv.r.shape)
    clothuvd[:,0] = cloth2dvt[:,0] #/ /w
    clothuvd[:,1] = cloth2dvt[:,1] #/h 
    clothuvd[:,2] = bodydepth    # we should calcuated clothdepth 
    clothv3d = cam.unproject_points(clothuvd)
    #sv.r = clothv3d  #  now the model is not body but cloth  
    cam.v =  clothv3d  # ths is body !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
    check_clothv3d = True
    num_axes = num_axes + 1
    #tt = graphutil.build_bodypartmap2(None, sv.r, cam, bodyparts, body_colormap, h, w, False)


    ########################################################
    # vertex2label and face2label mapping
    ########################################################
    cloth_colormap = {  0: (0, 0, 0),  # FIXME
                        1: (128,128, 0),  
                        2: (0,128, 128),  
                        3: (128, 0, 128),     
                        4: (255, 255, 0),
                        5: (0,255, 255),  
                        6: (255, 0, 255), 
                        7: (128, 255, 128), 
                        8: (128,128, 255),  
                        9: (255, 128, 128),
                        10: (128, 128, 128), 
                        11: (128, 128, 255), 
                        12: (128,255, 128),
                        13: (0,255, 128), 
                        14: (128,255, 0),  
                        15: (255, 128 , 0),
                        16: (0, 128, 255), 
                        17: (255, 255, 255),
                        18: (128, 000000000, 255)}

    cloth_label_list = ['background', 'hat', 'hair', 'sunglass', 'upper', 'skirt', 'pants', 'dress', 'belt',
                    'leftshoe', 'rightshoe', 'face', 'leftleg', 'rightleg', 'leftarm', 'rightarm', 'bag', 'scarf', 'skin']


    # vertices2label mapping
    v2d_int = np.rint(cam.r).astype('int32')
    v2label = mask[v2d_int[:,1], v2d_int[:,0]]
    check_v2label = False 
    if check_v2label:
        labelmap_img = graphutil.build_bodypartmap_2d(img, cam.r, v2label, cloth_colormap, h, w, False)
        plt.imshow(labelmap_img)  # FIXME: color_map 
        plt.title('SMPL: cloth and body labeled')
        _ = raw_input('quit?')
        #exit()
        cam.v =  clothv3d  # restore  
    # faces to label mapping
    check_f2label = False 
    f2label, _, fcenter2d = graphutil.build_face2label(cam, clothv3d, mask, model.f)
    if check_f2label:
        # face2label mapping and display
        labelmap_img = graphutil.build_bodypartmap_2d(img, fcenter2d, f2label, cloth_colormap, h, w, False)
        plt.imshow(labelmap_img)  # FIXME: color_map 
        plt.title('Cloth 3D projected:labeled')
        _ = raw_input('next?')

    check_clothbycloth = False 
    if check_clothbycloth:

        # method 1 controlling the faces 
        '''
        # display only the specific labeled faces
        all_v3d = cam.v.r
        all_vt = cam.r.copy()
        all_ft = model.f.copy()
        pjt_texture = prepare_texture(all_vt,all_ft, imgRGB) 

        # filtering only the specific area faces 
        max_cloth_label = len(cloth_label_dict) 
        for target_label in range(1, max_cloth_label): # except background  

            print('label:', target_label, cloth_label_list[target_label])
            fidx4cloth = np.nonzero(f2label == target_label)[0]
            if fidx4cloth.shape[0] < 10: # FIXME
                print('skipping too small segement')
                continue

            print(fidx4cloth)
            f4cloth    = all_ft[fidx4cloth]
            print('filtering:', all_ft.shape, '=>', f4cloth.shape, 'by', fidx4cloth.shape)
            cloth_renderer   = build_texture_renderer(cam, all_v3d, f4cloth, all_vt, f4cloth,  # cam.v is ok or filtering it also
                                 pjt_texture[::-1, :, :], w, h, 1.0, near=0.5, far=20 + dist)

        '''
        # method 2. controlling the alpha channels
        '''
        cam.v =  clothv3d  # restore  
        max_cloth_label = len(cloth_label_dict) 
        for target_label in range(1, max_cloth_label): # except background  
            all_vt = cam.r.copy()
            all_ft = model.f.copy()
            pjt_texture = prepare_texture_with_alpha(all_vt, all_ft, imgRGB, mask, target_label) 
            cloth_renderer   = build_texture_renderer(cam, cam.v, model.f, all_vt, all_ft,  # cam.v is ok or filtering it also
                                 pjt_texture[::-1, :, :], w, h, 1.0, near=0.5, far=20 + dist)

        '''
        # method 3: not implemented 

        # method 4: making a mask 
        cam.v =  clothv3d  # restore  
        max_cloth_label = len(cloth_label_dict) 
        label_mask1 = graphutil.build_labelmask(clothv3d, model.f, v2label, None, cam,  height=h, width=w, near= 0.5, far= 40)
        label_mask2 = graphutil.build_labelmask2(clothv3d, model.f, v2label, cam,  height=h, width=w, near= 0.5, far= 40)

        plt.subplot(1,4,1)
        plt.imshow(imgRGB[:,:,::-1])
        plt.title('original mask')
        plt.subplot(1,4,2)
        plt.imshow(mask)
        plt.title('original mask')
        plt.subplot(1,4,3)
        plt.imshow(label_mask1)
        plt.title('cloth vertices projected')
        plt.subplot(1,4,4)
        plt.imshow(label_mask2)
        plt.title('separate & merged')
        plt.draw()
        plt.show()
        _ = raw_input('next?')
        '''
        for target_label in range(1, max_cloth_label): # except background  

            label_mask = graphutil.build_labelmask(sv.r, model.f, v2label, target_label, cam,  height=h, width=w, near= 0.5, far= 40)
            #num_v = np.count_nonzero(v2label == target_label)
            num_source_pixels = np.count_nonzero(mask == target_label)
            num_target_pixels = np.count_nonzero(label_mask)
            print('pixels for ',  cloth_label_list[target_label], ':',  num_source_pixels, '->',  num_target_pixels)

            plt.subplot(1,3,1)
            plt.imshow(imgRGB[:,:,::-1])
            plt.title('original')
            plt.subplot(1,3,2)
            plt.imshow(label_mask, cmap='gray')
            plt.title('mask')
            plt.subplot(1,3,3)
            imgtmp = imgRGB.copy()
            imgtmp[label_mask == 0, :] = 0
            plt.imshow(imgtmp[:,:,::-1])
            plt.title('extracted')
            plt.suptitle(cloth_label_list[target_label])# + ':' +  str(num_v) + ',' + str(num_p))
            plt.draw()
            plt.show()
            _ = raw_input('next?')
        '''

        exit()
        cam.v =  clothv3d  # restore  

    cam.v =  clothv3d  # restore  

    #  Step 4:  Texuring with 2d image 
    ##############################################
    pjt_vt = cam.r.copy()
    pjt_ft = model.f.copy()
    pjt_texture = prepare_texture(pjt_vt,pjt_ft, imgRGB) 
    texture_renderer = build_texture_renderer(cam, cam.v, model.f, pjt_vt, pjt_ft,
                                 pjt_texture[::-1, :, :], w, h, 1.0, near=0.5, far=20 + dist)
    textured_cloth2d = texture_renderer.r
    check_clothv3d_texture = True

    if not viz:
        print('sv.J_transformed.r :', sv.J_transformed.r.shape)
        return sv.J_transformed.r, pjt_vt, pjt_ft, pjt_texture #, body2dvt_save
        #return None


    #########################################################################
    # 4. visualize
    #########################################################################
    fig = plt.gcf()
    #fig  =  plt.figure(1)
    fig.suptitle('Img2SMPL2Cloth')

    # 4.1 texture image
    if check_partmap: 
        plt.subplot(1, num_axes, axis_idx)
        axis_idx = axis_idx + 1
        plt.axis('off')
        plt.imshow(bodypartmap[:,:,::-1])
        plt.title('smpl body part')
    # 4.1.1. vertices
    '''
    for idx, uv in enumerate(pjt_vt):
        #print('idx:', idx, ':',  uv)
        cv2.circle(pjt_texture, (int(round(uv[0]*tw)), int(round(uv[1]*th))), 5, [0, 255, 0]) # Green
    '''

    # 4.1.2 face triangles
    '''
    for i in range(pjt_ft.shape[0]):
        pt0 = [pjt_vt[pjt_ft[i,0],0]*tw, pjt_vt[pjt_ft[i,0],1]*th]
        pt1 = [pjt_vt[pjt_ft[i,1],0]*tw, pjt_vt[pjt_ft[i,1],1]*th]
        pt2 = [pjt_vt[pjt_ft[i,2],0]*tw, pjt_vt[pjt_ft[i,2],1]*th]
        triangle = np.array([[pt0, pt1, pt2]], 'int32')
        cv2.polylines(pjt_texture, triangle, True, (255,0,0), 1)
    '''
    # 4.1.3 annotaation of texture mesh
    #plt.imshow(pjt_texture[:, :, ::-1])

    # contour
    #plt.imshow(img_contour, cmap='gray')
    # lt.hold(True)
    # plt.plot(cam.r[:,0], cam.r[:, 1], 'b.', markersize=1) # projected points
    # plt.title('texture')
    # plt.hold(False)

    if check_body2cloth2d: 
        plt.subplot(1, num_axes, axis_idx)
        axis_idx = axis_idx + 1
        plt.axis('off')
        plt.imshow(cloth2dmap[:,:,::-1])
        plt.title('2D Cloth deformed')

    if check_clothv3d:
        plt.subplot(1, num_axes, axis_idx)
        axis_idx = axis_idx + 1
        plt.axis('off')
        #tt = graphutil.build_bodypartmap2(img, sv.r, cam, bodyparts, body_colormap, h, w, False)
        img_clothpart3d = graphutil.build_bodypartmap_2d(img, cam.r, bodyparts, body_colormap, h, w, False)
        plt.imshow(img_clothpart3d[:,:,::-1])
        plt.title('3D Cloth Projected')

    if check_clothv3d_texture:
        plt.subplot(1, num_axes, axis_idx)
        axis_idx = axis_idx + 1
        plt.axis('off')
        plt.imshow(textured_cloth2d)
        plt.title('3D cloth rendered')

    #########################################
    # 3D multiview check by camera rotation 
    #########################################
    img0 = None 

    check_multiviews = False 
    if check_multiviews:

        plt.subplot(1, 5, 1)
        plt.axis('off')
        plt.imshow(imgRGB[:,:,::-1])
        plt.title('input')

        rot_axis = 1
        rotation = ch.zeros(3)
        rotation[rot_axis] = 3.14/4
        #img0 = pjt_R.r[:, :, ::-1]*255.0
        img0 = texture_renderer.r[:, :, ::-1]*255.0
        img0 = img0.astype('uint8')
        for i in range(4):
            plt.subplot(1, 5, i+2)
            #plt.imshow(pjt_R.r)
            plt.imshow(texture_renderer.r)
            plt.axis('off')
            plt.draw()
            plt.show()
            #plt.title('angle =%f'%yaw)
            plt.title('%.0f degree' % (i*45))
            cam.v = cam.v.dot(Rodrigues(rotation))
    plt.show()

    return img0


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
def smpl2cloth(smpl_model, inmodel_path, inimg_path, mask_path, outimg_path,  j2d, ind):

    if smpl_model is None or inmodel_path is None or inimg_path is None or mask_path is None or outimg_path is None:
        print('There is None inputs'), exit()

    plt.ion()

    # model params 
    with open(inmodel_path, 'rb') as f:
        if f is None:
            print("cannot open",  inmodel_path), exit()
        params = pickle.load(f)

    #params['pose'] = params['pose'] % (2.0*np.pi) # modulo 

    _examine_smpl_params(params)

    #  2d rgb image for texture
    #inimg_path = img_dir + '/dataset10k_%04d.jpg'%idx
    img2D = cv2.imread(inimg_path)
    if img2D is None:
        print("cannot open",  inimg_path), exit()

    # segmentation mask 
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

    # 3. run the SMPL body to cloth processing 
    cam   = params['cam']      # camera model, Ch
    betas = params['betas']
    n_betas = betas.shape[0] #10
    pose  = params['pose']    # angles, 27x3 numpy
    _  = smpl2cloth3dcore(params['cam'],      # camera model, Ch
                 betas,    # shape coeff, numpy
                 n_betas,  # num of PCA
                 pose,     # angles, 27x3 numpy
                 img2D,    # img numpy
                 mask,     # mask 
                 j2d, 
                 smpl_model, # SMPL
                 bHead = True,   # 
                 viz = True)    # display   

    # 3.2 save result for checking
    '''
    if outimg_path is not None:
        plt.savefig(outimg_path)
        if img_out is not None:
            cv2.imwrite(outimg_path, img_out)
    '''


if __name__ == '__main__':

    if len(sys.argv) < 4:
       print('usage: %s  ase_path dataset idx'% sys.argv[0]), exit()

    # 1. directory check and setting
    base_dir = abspath(sys.argv[1])
    #print(base_dir)
    dataset = sys.argv[2]
    idx = int(sys.argv[3])

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
    cloth_dir = data_dir + "/cloth"
    if not exists(cloth_dir):
        makedirs(cloth_dir)

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
    estj2d = np.load(join(data_dir, 'est_joints.npz'))['est_joints']
    #print('est_shape:', est.shape)
    joints = estj2d[:2, :, idx].T

    # for i in range(1, 2):
    # if not use_neutral:
    #    gender = 'male' if int(genders[i]) == 0 else 'female'
    #    if gender == 'female':
    smpl_model = model_female
    smpl_param_path = smpl_param_dir + '/%04d.pkl'%idx 
    inp_path = inp_dir + '/dataset10k_%04d.jpg'%idx 
    #mask_path = data_dir + '/segmentation/10kgt_%04d.png'%idx
    mask_path = mask_dir + '/10kgt_%04d.png'%idx
    cloth_path = cloth_dir + '/%04d.png'% idx 
    #print(smpl_model, smpl_param_path, inp_path, mask_path, cloth_path, idx)
    smpl2cloth(smpl_model, smpl_param_path, inp_path, mask_path, cloth_path,joints, idx)

    # plt.pause(10)
    _ = raw_input('quit?')

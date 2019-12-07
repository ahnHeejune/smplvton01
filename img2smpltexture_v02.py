"""
  1. fit 3d SMPL mesh model into 2D again 
 
  2. extract boundary of 2-D prjected mesh 

  3. morping the body shape to cloth shape 

  4. recover 3-D cloth mesh model 
   
"""
from __future__ import print_function
from os.path import join, exists, abspath, dirname
from os import makedirs
import logging
import cPickle as pickle

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

import sys

_LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


############################################################################
# examine smpl model 
############################################################################
def examine_smpl(model):

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
    parts = np.argmax(model.weights.r, axis =1)
    print("        :",  parts.shape, parts[:6000])




##############################################################################
#  normal vector of triangle 
#  https://sites.google.com/site/dlampetest/python/calculating-normals-of-a-triangle-mesh-using-numpy
##############################################################################

def normalize_v3(arr):
    # Normalize a numpy array of 3 component vectors shape=(n,3) 
    lens = np.sqrt( arr[:,0]**2 + arr[:,1]**2 + arr[:,2]**2 )
    arr[:,0] /= lens
    arr[:,1] /= lens
    arr[:,2] /= lens
    return arr
def calc_normal_vectors(vertices, faces):

    ''' 
       return face normal, and vertex normal 

    '''
    #Create a zero array with the same type and shape as our vertices i.e., per vertex normal
    _norm = np.zeros( vertices.shape, dtype=vertices.dtype )
    #Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    #Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
    n = np.cross( tris[::,1 ] - tris[::,0]  , tris[::,2 ] - tris[::,0] )
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(n)
    #n = norm(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    _norm[ faces[:,0] ] += n
    _norm[ faces[:,1] ] += n
    _norm[ faces[:,2] ] += n
    normalize_v3(_norm)
    #norm(_norm)

    return n, _norm


''' 
   cos(theta) =   v1 * v2 
                 ------------
                 |v1|*|v2|

'''
def cosine_similarity(v1, v2, isnorm = False):

    from numpy import dot
    from numpy.linalg import norm

    cos_sim = dot(v1, v2)
    if not isnorm:
       cos_sim = cos_sim/(norm(v1)*norm(v2))

    return cos_sim



# create V, A, U, f: geom, bright, cam, renderer
def build_remap_renderer(U, V, f, vt, ft, texture, w, h, ambient = 0.0, near = 0.5, far = 20000):

    A = SphericalHarmonics(vn=VertNormals(v=V, f=f),
        components=[0., 0., 0., 0., 0., 0., 0., 0., 0.],
        light_color = ch.ones(3)) + ambient

    R = TexturedRenderer(vc=A, camera=U, f=f, bgcolor = [0.0, 0.0,0.0],
        texture_image = texture, vt = vt, ft=ft,
        frustum={'width':w, 'height': h, 'near':near, 'far':far})

    return R


########################################################
# We make the back size white/skin color#
# 1. calculate normal vector to the faces
# 2. map the face into a artifically define texture triangle
# 3. calcuate cosine value +: front, -:back
# 4. map to special textured triangle
#
#     \ ----------------------------->camera 
#     | -----> n
#     /
#
#######################################################
def build_face_visibility(f_normal, cam):

    from numpy.linalg import norm
    visibility =  np.zeros(f_normal.shape[0], dtype = 'int64')
    camera_pos = cam.t.r/norm(cam.t.r)
    for i in range(f_normal.shape[0]):
        if cosine_similarity(f_normal[i], cam.t.r, True) < 0: 
            visibility[i] = 1 
        else:
            visibility[i] = -1 

    return visibility

def build_edge_graph(vertices, faces, face_visibility):

    n_vertices = vertices.shape[0]
    print('n_vertices:', n_vertices)
    print('n_faces:', faces.shape)  #model.f.shape)
    print('face_vis:', face_visibility.shape)

    # 1) construct edge map (no directed) from faces 
    # the most typical ds of graph is  bidirectional dictionaly 
    # it is memory efficient, and  computationaly efficient 
    # but  now implemented in 2-D and 3 channel format
    # graph[s,t,0:2] = -1, -1,  not connected
    #                  =  f1, f2 two faces 
    #                  =  f1, -1  single face (impossible)
    # note: only low half triangle is used (no direction) 
    graph  =  - np.ones([n_vertices, n_vertices, 3], dtype='int64')
    max_label =  1000
    label_equiv =  max_label*np.ones(max_label, dtype ='int64')  #  the smallest label equivalent 
    for i in range(1,1000):
            label_equiv[i] = i
    con_length =  np.zeros(max_label, dtype='int64') 
     
    #print('graph:', graph[:5, :5, :])
    # based in faces, fill the link from edge to faces
    #                 already have face to edges, and vertexs
    for fidx, fv in enumerate(faces):
        [v1, v2, v3] = sorted([fv[0], fv[1], fv[2]]) # increasing order
        #if fidx < 20:
        #    print('vs:', v1, v2, v3)
        # do we needs any polarity of faces 
        pos = 0 if graph[v3,v1,0]==-1 else 1
        graph[v3,v1, pos] = fidx
        pos = 0 if graph[v3,v2,0]==-1 else 1 
        graph[v3,v2, pos] = fidx
        pos = 0 if graph[v2,v1,0]==-1 else 1
        graph[v2,v1, pos] = fidx

    #print('finished graph')
    #print('graph:', graph[:10, :10, :])

    # 1) mark the contour edges 
    n_contour_edge = 0
    for s  in range(n_vertices):
        for t  in range(n_vertices):
            if  graph[s,t,0] > 0 and graph[s,t,1] > 0:
                #print('edge:', graph[s,t,0], graph[s,t,1]) 
                if face_visibility[graph[s,t,0]]*face_visibility[graph[s,t,1]] < 0:
                    graph[s,t,2] = 0   # 1: contour edge, >1: contour index 
                    n_contour_edge = n_contour_edge +1 
                    #print('(', s, '->', t, ')', end=' ')
    print('finished contour detection : ', n_contour_edge)


    # 2) extract connected contour edges
    longest_contour_label, longest_contour_len  = -1,  0
    contour_label = 0 
    found_contour_edge = True 
    while found_contour_edge == True:

        # 2.1) search a starting edge/vertex
        found_contour_edge = False
        cur_v = 0
        next_v = -1
        while found_contour_edge == False and  cur_v  < n_vertices:
            for v in range(n_vertices):  # lower half
                (v_big, v_small) = (cur_v, v )if cur_v > v else (v, cur_v)
                if graph[v_big, v_small,2] == 0: # contour edge, not used 
                    found_contour_edge = True
                    next_v = v
                    break

            if found_contour_edge == True:
                # 2) path through connected contour edge
                num_edges = 0
                contour_label = contour_label + 1
                (v_big, v_small) = (cur_v, v )if cur_v > v else (v, cur_v)
                graph[v_big, v_small,2] = contour_label
                num_edges = num_edges + 1
                print(contour_label, '-th contour:', '(' , cur_v, '-', next_v, ')', end='')
                cur_v = next_v
                found_next_v = True 
                while found_next_v == True:
                    found_next_v = False
                    for v in range(n_vertices): 
                        (v_big, v_small) = (cur_v, v)if cur_v > v else (v, cur_v)
                        if graph[v_big, v_small, 2]  == 0: # unused c edge
                            graph[v_big, v_small, 2] = contour_label
                            print('(' , cur_v, '-', v, ')', end='')
                            num_edges = num_edges + 1
                            found_next_v = True
                            cur_v = v
                            break
                        elif graph[v_big, v_small, 2] > 0: # for later merging
                            if label_equiv[contour_label] > graph[v_big, v_small, 2]:
                                print("connected:", contour_label,  "=>", graph[v_big,v_small, 2])
                                label_equiv[contour_label] = graph[v_big, v_small, 2]
                print(' [total=', num_edges, ']') 
                con_length[contour_label] = num_edges

                if longest_contour_len < num_edges:
                    longest_contour_len = num_edges
                    longest_contour_label = contour_label 
            else:
                cur_v = cur_v +1 # check next cur


    #print("labels:", label_equiv[:50])
    #print("length:", con_length[:50])
    for i in range(50):
        print(i,  label_equiv[i], con_length[i])


    # merge  the connected ones 
    # @TODO add the length , but  needed now becuase  the longest is far longer
    for label in range(contour_label,0,-1):
        graph[graph[:,:,2]== label, 2]  = label_equiv[label]

    return  graph, longest_contour_label



# construct visualizing image for depth of all front vertices 
def  build_depthmap( vertices, cam,  height, width, zonly = True):

    # 1. blank background image 
    depthmap = np.ones([height, width], dtype= 'float32')
    depthmap = depthmap*cam.t.r[2]*10 # very far 

    # 2. depth  value for vertices and projection 
    if zonly:  # simplified depth utilizing cam.t.x/y and cam.rt = 0
        cam_z  = cam.t.r[2]
        for i in range(vertices.shape[0]):
            new_depth = cam_z - vertices[i,2]
            #print(type(new_depth))
            y, x = int(cam.r[i,1]), int(cam.r[i,0]) 
            if  new_depth < depthmap[y,x]:
                depthmap[y,x] = new_depth 
    else:
        from numpy import dot
        normalized_camera_pos = cam.t.r/norm(cam.t.r)
        # make depthmap = zeros(th,tw)
        for i in range(vertices.shape[0]):
            displacement = vertices[i,:] - camera_pos
            depth  = dot(displacement, camera_pos)
            #depthmap(pjt_vt[i]) = depth

    return depthmap

# visualize  depth using opendr renderer
# 1. calcuate the depth for each vertex
# 2. apply the depth on the vc value 
#
def  build_depthmap2( vertices, faces,  cam,  height, width):

    vc1 =  np.zeros(vertices.shape)

    # 1. blank background image 
    #depthmap = np.ones([height, width], dtype= 'float32')
    #depthmap = depthmap*cam.t.r[2]*10 # very far 

    # 2. depth  value for vertices and projection 
    cam_z  = cam.t.r[2]
    for i in range(vertices.shape[0]):
        depth = cam_z - vertices[i,2]
        vc1[i,:] = depth

    vc_min = np.amin(vc1)
    print('vc_min:', vc_min)
    vc1 = vc1 - vc_min
    vc_max = np.amax(vc1)
    print('vc_max:', vc_max)
    vc1 = vc1/vc_max        #  for the range : (0, 1)
    vc_max = np.amax(vc1)
    print('vc_max:', vc_max)
    vc_min = np.amin(vc1)
    print('vc_min:', vc_min)
    vc_mean = np.mean(vc1)
    print('vc_mean:', vc_mean)

    #vc1  = np.ones(vertices.shape)
    print('vc1.shape:', vc1.shape)

    from opendr.renderer import ColoredRenderer
    rn = ColoredRenderer()
    rn.camera = cam 
    rn.frustum = {'near': 1., 'far': 40., 'width': width, 'height': height}
    rn.v = vertices
    rn.f = faces
    rn.bgcolor = ch.zeros(3)
    rn.vc = ch.array(vc1)  # giving the albera

    depthmap = rn.r.copy()
    return depthmap

#  bod part assgined  
def  build_bodypartmap(vertices, cam, parts, height, width, separated = False):

    # 1. blank background image 
    if not separated:
        img = np.zeros([height, width], dtype= 'uint8')
    else:
        img = np.zeros([height, width, 24], dtype= 'uint8')

    # 2. depth  value for vertices and projection 
    cam_z  = cam.t.r[2]
    for i in range(vertices.shape[0]):
        y, x = int(cam.r[i,1]), int(cam.r[i,0]) 
        if not separated:
            img[y-5:y+5, x-5:x+5] = parts[i] + 64 
        else:
            img[y-5:y+5, x-5:x+5, parts[i]] = 1
            #cv2.drawMarker(img[:,:, parts[i]], (x,y), 255, markerSize = 5)

    return  img

#####################################################################
#
#
#
####################################################################
def remap(cam,      # camera model, Chv 
          betas,    # shape coef, numpy  
          n_betas,  # num of PCA 
          pose,     # angles, 27x3 numpy 
          img,      # img numpy  
          model):   # what type? 

    #print(type(model.posedirs))
    sv = verts_decorated(
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
            posedirs=model.posedirs)

    # 3. render the model with paramter
    #img = cv2.imread(img_dir + "/dataset10k_" + "0000" + ".jpg")
    h, w = img.shape[0:2]
    #rt  = np.zeros(3)
    #center =  np.zeros(2)
    #cam = ProjectPoints(
    #    f=params['f'], rt=rt, t=params['cam_t'], k=np.zeros(5), c=center)
    dist = np.abs(cam.t.r[2] - np.mean(sv.r, axis=0)[2])
    im = (render_model(
            sv.r, model.f, w, h, cam, far=20 + dist, img=img[:,:,::-1]) * 255.).astype('uint8')

    # checking the redering result, but we are not using this.
    # we could drawing the points on it 
    #print('th:', th,  '  tw:', tw)
    #plt.figure()
    pjt_vt = cam.r.copy()
    img2 = img.copy()

    '''
    plt.imshow(img2)
    plt.hold(True)
    # now camera use only joints
    plt.plot(cam.r[:,0], cam.r[:, 1], 'r+', markersize=10) # projected points 
    '''
    # project all vertices using camera  
    cam.v = sv.r # is this a vertices ?

    '''
    print('sv.r:', sv.r.shape)
    plt.plot(cam.r[:,0], cam.r[:, 1], 'b.', markersize=1) # projected points 
    plt.show()
    plt.hold(False)
    plt.pause(3)
    '''
    
    # 1.2 vertices
    vertices = np.around(cam.r).astype(int)
    for idx, coord in enumerate(vertices):
       cv2.drawMarker(img2, tuple(coord), [0, 255, 0], markerSize = 1)
       #cv2.circle(im, (int(round(uv[0])), int(round(uv[1]))), 1, [0, 255, 0]) # Green

    # vertexes 
    #pjt_v = sv.r.copy() 
    # faceses
    #pjt_f = model.f.copy()
    # texture  = overlayed images of 2d and projected.
    pjt_texture = img.astype(float)
    pjt_texture[:,:,:] = pjt_texture[:,:,:]/255.0
    print('dtype of img:',  img.dtype)
    print('dtype of pjt_texture:',  pjt_texture.dtype)
    th , tw = pjt_texture.shape[0:2]
    '''
    pjt_texture[:,:,:] = (1.0, .0, .0)  #  
    #pjt_texture[:,:int(tw/2),:] = (1.0, 0., 0.)  # B, G, R 
    pjt_texture[:,int(tw/4):int(3*tw/4),:] = (1.0, 1.0, 1.0)  # B, G, R 
    '''
    print("th, tw:", th , tw)
    # vt
    pjt_vt = cam.r.copy()
    pjt_vt[:,0] = pjt_vt[:,0]/tw  # uv normalize
    pjt_vt[:,1] = pjt_vt[:,1]/th  # uv normalize
    # ft 
    pjt_ft = model.f.copy()
    print("ft:", pjt_ft.shape)

    # 5. project the body model with texture renderer
# 3. reprojection
    print(type(cam.v))
    print(cam.v.r.shape)


    print("textured:",  type(pjt_texture), 'dtype:',  pjt_texture.dtype, "shape:",  pjt_texture.shape)
    print('max:', np.amax(pjt_texture[:,:,0]), np.amax(pjt_texture[:,:,1]), np.amax(pjt_texture[:,:,2]))
    print('meam:', np.mean(pjt_texture[:,:,0]), np.mean(pjt_texture[:,:,1]), np.mean(pjt_texture[:,:,2]))


    # 1. construct face_visibility map
    f_normal, v_normal = calc_normal_vectors(cam.v, model.f)
    face_visibility  = build_face_visibility(f_normal, cam)

    # 2.  apply the visibility map for texturing 
    backSideTexture = False
    if not backSideTexture: 
        v_end = cam.r.shape[0]
        pjt_vt = np.append(pjt_vt, [[0./tw,  0./th], [1.0/tw, 0./th], [0./tw, 1.0/th]], axis=0)
        pjt_texture[th-50:th,0:50] = (1.0, 1.0, 1.0) 
        pjt_texture[0:50,0:50] = (1.0,1.0,1.0)
        for i in range(f_normal.shape[0]):
            if face_visibility[i] <  0: 
                pjt_ft[i] = (v_end, v_end+1, v_end+2) #(0, 1, 2)  


    # 3. extract edge vertices 
    use_edge_vertices = check_edge_vertices = save_edge_vertices = True 
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
        graph, longest_contour_label = build_edge_graph(cam.v, model.f, face_visibility)

        num_body_vertices = np.count_nonzero(graph[:,:,2] == longest_contour_label) 
        print("Body V Number:", num_body_vertices)
        if save_edge_vertices:
            body_vertices =np.zeros([num_body_vertices, 2], dtype ='int32') 

        #visualization of contour 
        img_contour = np.zeros([th,tw], dtype='uint8')
        i = 0
        if check_edge_vertices or save_edge_vertices:
            for v_s in range(cam.v.shape[0]):
                for v_e in range(v_s):
                    if graph[v_s,v_e,2]  == longest_contour_label: # > 0:
                        if check_edge_vertices:
                            sx, sy = cam.r[v_s] # projected coordinate
                            ex, ey = cam.r[v_e]
                        if save_edge_vertices:
                            body_vertices[i,0], body_vertices[i,1] = int(sx), int(sy)
                            i = i + 1
                        cv2.line(img_contour, (int(sx),int(sy)), (int(ex),int(ey)), graph[v_s,v_e,2], thickness=1)

        if save_edge_vertices:
            body_vertices_path = "edge_vertices.pkl" 
            with open(body_vertices_path, 'w') as outf:
                pickle.dump(body_vertices, outf)

    # 4. Partmap for vertices
    use_partmap = check_partmap = False 
    if  use_partmap:
        parts = np.argmax(model.weights.r, axis =1)
        bodypartmap = build_bodypartmap(sv.r, cam, parts, th, tw, False)
        if check_partmap:
            print('part-max:', np.amax(parts))
            plt.suptitle('body partmap')
            plt.subplot(1,2,1)
            plt.imshow(img[:,:,::-1]) #, cmap='gray')
            plt.subplot(1,2,2)
            plt.imshow(bodypartmap) #, cmap='gray')
            '''
            for i in range(24):
                plt.subplot(6,4,i+1)
                plt.imshow(bodypartmap[:,:, i] )
            '''
            _ = raw_input('quit?')
            exit()
    
    # 5. Depthmap at vertices
    use_depthmap = check_depthmap = False 
    if use_depthmap:
        #depthmap = build_depthmap(sv.r, cam, th, tw)
        depthmap = build_depthmap2(sv.r, model.f, cam, th, tw)
        if check_depthmap:
            # depth in reverse way 
            plt.suptitle('depthmap')
            plt.subplot(1,2,1)
            plt.imshow(img[:,:,::-1]) #, cmap='gray')
            plt.subplot(1,2,2)
            plt.imshow(depthmap)
            #plt.imshow(depthmap, cmap='gray_r')
            _ = raw_input('quit?')
            exit()
    # we should modify the depth for cloth 


    # ***note ****:   texture coordinate is UP-side Down
    pjt_R = build_remap_renderer(cam, cam.v, model.f, pjt_vt, pjt_ft, pjt_texture[::-1,:, :], w, h, 1.0, near  =0.5, far = 20 + dist)
    # 4. visualize
    cam.v = sv.r


    fig = plt.gcf()
    #fig  =  plt.figure(1)
    fig.suptitle('2D to SMPLify with Texuring')


    # 4.1 texture image 
    plt.subplot(1,5,1)
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
    plt.imshow(pjt_texture[:,:,::-1])
    # contour  
    #plt.imshow(img_contour, cmap='gray')



    #lt.hold(True)
    #plt.plot(cam.r[:,0], cam.r[:, 1], 'b.', markersize=1) # projected points 
    plt.title('texture')
    #plt.hold(False)

    '''
    plt.subplot(1,5,2)
    plt.imshow(pjt_R.r)
    plt.title('reprojected')
    '''

    rot_axis = 1
    rotation = ch.zeros(3)
    rotation[rot_axis] =  3.14/4
    img0 = pjt_R.r[:, :, ::-1]*255.0
    img0 =  img0.astype('uint8') 
    for i in range(4):
        plt.subplot(1, 5, i+2)
        plt.imshow(pjt_R.r)
        plt.axis('off')
        plt.draw()
        plt.show()
        #plt.title('angle =%f'%yaw)
        plt.title('%.0f degree'%(i*45))
        cam.v = cam.v.dot(Rodrigues(rotation))

    plt.show()

    return img0

def single_image_smplify (inimg_path, inmodel_path, outimg_path):

    plt.ion()

    use_neutral = False
    use_interpenetration  = True
    base_dir = '..'  # parent directory 
    n_betas  = 10
    dataset  = '10k'
    img_dir = join(abspath(base_dir), 'images/' + dataset)
    data_dir = join(abspath(base_dir), 'results/' + dataset)

    # 1. load SMPL models (independent upon dataset)
    # Assumes 'models' in the 'code/' directory where this file is in.
    MODEL_DIR = join(abspath(dirname(__file__)), 'models')
    # Model paths:
    MODEL_NEUTRAL_PATH = join(
        MODEL_DIR, 'basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')
    MODEL_FEMALE_PATH = join(
        MODEL_DIR, 'basicModel_f_lbs_10_207_0_v1.0.0.pkl')
    MODEL_MALE_PATH = join(MODEL_DIR,
                           'basicmodel_m_lbs_10_207_0_v1.0.0.pkl')

    if use_interpenetration:
        # paths to the npz files storing the regressors for capsules
        SPH_REGS_NEUTRAL_PATH = join(MODEL_DIR,
                                     'regressors_locked_normalized_hybrid.npz')
        SPH_REGS_FEMALE_PATH = join(MODEL_DIR,
                                    'regressors_locked_normalized_female.npz')
        SPH_REGS_MALE_PATH = join(MODEL_DIR,
                                  'regressors_locked_normalized_male.npz')

    # 1. SMPL model
    sph_regs = None
    if not use_neutral:
        # File storing information about gender in LSP
        with open(join(data_dir, dataset + '_gender.csv')) as f:
            genders = f.readlines()
        model_female = load_model(MODEL_FEMALE_PATH)
        model_male = load_model(MODEL_MALE_PATH)
        if use_interpenetration:
            sph_regs_male = np.load(SPH_REGS_MALE_PATH)
            sph_regs_female = np.load(SPH_REGS_FEMALE_PATH)
    else:
        gender = 'neutral'
        model = load_model(MODEL_NEUTRAL_PATH)
        if use_interpenetration:
            sph_regs = np.load(SPH_REGS_NEUTRAL_PATH)

    # 3.  handle images
    ind = 0
    #if not use_neutral:
    #    gender = 'male' if int(genders[ind]) == 0 else 'female'
    #     if gender == 'female':
    model = model_female
    if use_interpenetration:
        sph_regs = sph_regs_female
    #       elif gender == 'male':
    #            model = model_male
    #            if use_interpenetration:
    #                sph_regs = sph_regs_male

    # examine_smpl(model), exit()


    # 2. loading the image specific model paramter 
    with  open(inmodel_path, 'rb')  as f:
        params = pickle.load(f)   

    #img_path = join(img_dir, 'dataset10k_'+  '0000.jpg')
    #_LOGGER.info('processing input:' + img_path)
    #img_path = sys.argv[2]
    img = cv2.imread(inimg_path)

    print(type(params))
    print(params.keys())
    print('camera params')
    print(" - type:", type(params['cam']))   
    #print(" - members:", dir(params['cam']))   
    #print(" - cam.t:",params['cam'].t)   
    print(" - cam.t:",params['cam'].t.r)    # none-zero, likely only nonzero z  
    print(" - cam.rt:",params['cam'].rt.r)  # zero (fixed)
    

#    print(params['f'].shape)      # 2 
    print('pose')
    print(" - type:", type(params['pose']))   
    print(' - shape:', params['pose'].shape)   # 72
    #print(' - values:', params['pose'])   
    print('betas')
    print(' - type:', type(params['betas']))   
    print(' - shape:', params['betas'].shape)  # 10
    #print(' - values:', params['betas'])  # 10
   

    img0 =  remap(params['cam'],      # camera model, Ch 
          params['betas'],    # shape coef, numpy  
          n_betas,  # num of PCA 
          params['pose'],     # angles, 27x3 numpy 
          img,      # img numpy  
          model)    # what type? 

    if outimg_path is not None:
        plt.savefig(outimg_path)
        cv2.imwrite('z' + outimg_path, img0)


if __name__ == '__main__':
 
    if False:
        if len(sys.argv) < 4:
            print('usage: %s  input_img input_model out_img' %sys.argv[0])
            exit()            

        single_image_smplify(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        inp   =    "../images/10k/dataset10k_" 
        model =  "/tmp/smplify_10k/"
        out   =  "out"

        for i in range(1,2):
            single_image_smplify(inp + '%04d.jpg'%i, model + '%04d.pkl'%i, 'out%04d.png'%i)

    #plt.pause(10)
    _ = raw_input('quit?')


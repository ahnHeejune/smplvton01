from __future__ import print_function
#from os.path import join, exists, abspath, dirname
#from os import makedirs
import logging
#import cPickle as pickle
import time

import cv2
import numpy as np
import chumpy as ch

#from opendr.camera import ProjectPoints
#from smpl_webuser.serialization import load_model
#from smpl_webuser.verts import verts_decorated
#from render_model import render_model
#import inspect  # for debugging
import matplotlib.pyplot as plt

from opendr.lighting import SphericalHarmonics
from opendr.geometry import VertNormals, Rodrigues
from opendr.renderer import TexturedRenderer

import sys

_LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

###########################################################
# Graph manipulation
###########################################################

# dictionary based
def _test_graph_ds():
    nvertices = 5
    graph = np.zeros(nvertices, dtype=object)  # numpy array of empty dict 
    graph[:] = None   
    print(graph)
    for s in range(nvertices):
        graph[s] = {}   
        for t in range(s):
            print('s:', s, 't:', t)
            graph[s][t] = [s, t, s + t]
        #print(graph)

    print(graph)

    for s in range(nvertices):
        for t in graph[s]:
            print('s:', s, 't:', t)
            print(graph[s][t]) 


################################################################################
#  normal vector of triangle
#  https://sites.google.com/site/dlampetest/python/calculating-normals-of-a-triangle-mesh-using-numpy
################################################################################
def normalize_v3(arr):
    # Normalize a numpy array of 3 component vectors shape=(n,3)
    lens = np.sqrt(arr[:, 0]**2 + arr[:, 1]**2 + arr[:, 2]**2)
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr


################################################################################
#  re:  face normal, and vertex normal 
################################################################################
def calc_normal_vectors(vertices, faces):

    # Create a zero array with the same type and shape as our vertices i.e., per vertex normal
    _norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    # Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(n)
    #n = norm(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    _norm[faces[:, 0]] += n
    _norm[faces[:, 1]] += n
    _norm[faces[:, 2]] += n
    normalize_v3(_norm)
    # norm(_norm)

    return n, _norm


###############################################################################
#   calcuate: 
#   cos(theta) =   v1 * v2 
#                 ------------
#                 |v1|*|v2|
#
#   > 0 when  -90 < thetha < 90 degree 
#   < 0 o.w.  
#
#   v1, v2 : 2 vector inputs 
#   return : cosine value 
###############################################################################
def cosine_similarity(v1, v2, normalized=False):
    from numpy import dot
    from numpy.linalg import norm

    #print('shape -comp:', v1.shape, v2.shape)
    cos_sim = dot(v1, v2)
    if not normalized: # not normaled yet, normalized 
        cos_sim = cos_sim/(norm(v1)*norm(v2))

    return cos_sim

###############################################################################
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
###############################################################################
def build_face_visibility(f_normal, cam):

    from numpy.linalg import norm
    visibility = np.zeros(f_normal.shape[0], dtype='int64')
    
    #camera_pos = cam.t.r/norm(cam.t.r)

    camera_vec =  - cam.t.r # due to the opencv camera matrix defintion
    for i in range(f_normal.shape[0]):
        #if cosine_similarity(f_normal[i], cam.t.r, True) < 0:
        if cosine_similarity(f_normal[i], cam.t.r, True) > 0:
            visibility[i] = 1
        else:
            visibility[i] = -1

    return visibility


# compute the second local axis for the vertices
# the local axis should be orthonormal to v_normal and aligned to specific edge, i.e., the neighbor vertex

def find2ndaxis(faces, v_normal, v_ref):

    debug  = True
    n_vertex = v_ref.shape[0] 


    # 1. first find the smalles t-indexed neighbor vertex 
    # @TODO any way to speed up this step?
    ngbr_vertex =  n_vertex*np.ones(v_ref.shape[0], dtype = np.int64)
    for fidx, fv in enumerate(faces):
         v0, v1, v2 =  fv
         # v0 
         if ngbr_vertex[v0] > v1:
             ngbr_vertex[v0] = v1
         if ngbr_vertex[v0] > v2:
             ngbr_vertex[v0] = v2
         # v1
         if ngbr_vertex[v1] > v0:
             ngbr_vertex[v1] = v0
         if ngbr_vertex[v1] > v2:
             ngbr_vertex[v1] = v2
         # v2 
         if ngbr_vertex[v2] > v1:
             ngbr_vertex[v2] = v1
         if ngbr_vertex[v2] > v0:
             ngbr_vertex[v2] = v0

    # check results 
    if debug:
       for idx in range(n_vertex):
          if ngbr_vertex[idx] >= n_vertex:
              print('This vertex has no neighbor hood:',  idx)


    # 2. compute the tangential vector component 
    #    vec -   dot(normal, vec) * normal
    from numpy import dot
    from numpy.linalg import norm

    vec1 = v_ref[ngbr_vertex] - v_ref       # get the edge vector 
    print('shape comp: ',  v_normal.shape, vec1.shape)
    coefs = np.sum(v_normal*vec1, axis=1) # coef = dot(v_normal, vec1)
    vec2 = vec1 - coefs[:, None]*v_normal  # remove the normal components

    axis = normalize_v3(vec2)

    return axis


###############################################################################
# build edge graph 
# return graph 
# version 1: 2D vertex to vertex matrix 
###############################################################################
def build_edge_graph_matrix(vertices, faces, face_visibility):

    n_vertices = vertices.shape[0]
    print('n_vertices:', n_vertices)
    print('n_faces:', faces.shape)  # model.f.shape)
    print('face_vis:', face_visibility.shape)

    # 1) construct edge map (no directed) from faces
    # inefficient both in memory and computation
    # but now implemented in 2-D and 3 channel format
    # graph[s,t,0:2] = -1, -1,  not connected
    #                  =  f1, f2 two faces
    #                  =  f1, -1  single face (impossible)
    # note: only low half triangle is used (no direction)
    graph = - np.ones([n_vertices, n_vertices, 3], dtype='int64')
    max_label = 1000
    # the smallest label equivalent
    label_equiv = max_label*np.ones(max_label, dtype='int64')
    for i in range(1, 1000):
        label_equiv[i] = i
    con_length = np.zeros(max_label, dtype='int64')

    # contour_edges
    # indices: 0: big vertex, 1: small vertex, 2: face1, 3: face2, 4: edge label
    contours = []

    #print('graph:', graph[:5, :5, :])
    # based in faces, fill the link from edge to faces
    #                 already have face to edges, and vertices
    for fidx, fv in enumerate(faces):
        [v1, v2, v3] = sorted([fv[0], fv[1], fv[2]])  # increasing order
        # if fidx < 20:
        #    print('vs:', v1, v2, v3)
        # do we needs any polarity of faces
        pos = 0 if graph[v3, v1, 0] == -1 else 1
        graph[v3, v1, pos] = fidx
        pos = 0 if graph[v3, v2, 0] == -1 else 1
        graph[v3, v2, pos] = fidx
        pos = 0 if graph[v2, v1, 0] == -1 else 1
        graph[v2, v1, pos] = fidx

    #print('finished graph')
    #print('graph:', graph[:10, :10, :])

    c_start = time.time()
    # 1) mark the contour edges
    n_contour_edge = 0
    for s in range(n_vertices):
        # for t in range(n_vertices):
        for t in range(s):    # To iterate through the half of the graph
            if graph[s, t, 0] > 0 and graph[s, t, 1] > 0:
                #print('edge:', graph[s,t,0], graph[s,t,1])
                if face_visibility[graph[s, t, 0]]*face_visibility[graph[s, t, 1]] < 0:
                    graph[s, t, 2] = 0   # 1: contour edge, >1: contour index
                    n_contour_edge = n_contour_edge + 1
                    # print('(', s, '->', t, ')', end=' ')

                    # Add contour
                    contours.append([s, t, graph[s, t, 0], graph[s, t, 1], 0])

    print('finished contour detection : ', n_contour_edge)
    print("contour time: ", time.time() - start, "seconds")

    # 2) extract connected contour edges
    longest_contour_label, longest_contour_len = -1, 0
    contour_label = 0
    found_contour_edge = True

    while found_contour_edge is True:

        # 2.1) search a starting edge/vertex
        found_contour_edge = False
        cur_v = 0
        next_v = -1

        while found_contour_edge is False and cur_v < n_vertices:
            # for v in range(n_vertices):  # lower half
            for c_idx, each in enumerate(contours):  # contours
                if each[0] == cur_v:
                    v = each[1]
                    (v_big, v_small) = (cur_v, v) if cur_v > v else (v, cur_v)
                    if graph[v_big, v_small, 2] == 0:  # contour edge, not used
                        found_contour_edge = True
                        next_v = v
                        break

            if found_contour_edge is True:
                # 2) path through connected contour edge
                num_edges = 0
                contour_label = contour_label + 1
                (v_big, v_small) = (cur_v, v) if cur_v > v else (v, cur_v)
                graph[v_big, v_small, 2] = contour_label
                num_edges = num_edges + 1
                print(contour_label, '-th contour:', '(', cur_v, '-', next_v, ')', end='')
                cur_v = next_v
                found_next_v = True

                while found_next_v is True:
                    found_next_v = False
                    for v in range(n_vertices):
                        (v_big, v_small) = (cur_v, v) if cur_v > v else (v, cur_v)
                        if graph[v_big, v_small, 2] == 0:  # unused c edge
                            graph[v_big, v_small, 2] = contour_label
                            print('(', cur_v, '-', v, ')', end='')
                            num_edges = num_edges + 1
                            found_next_v = True
                            cur_v = v
                            break
                        elif graph[v_big, v_small, 2] > 0:  # for later merging
                            if label_equiv[contour_label] > graph[v_big, v_small, 2]:
                                print("connected:", contour_label, "=>", graph[v_big, v_small, 2])
                                label_equiv[contour_label] = graph[v_big, v_small, 2]

                print(' [total=', num_edges, ']')
                con_length[contour_label] = num_edges

                if longest_contour_len < num_edges:
                    longest_contour_len = num_edges
                    longest_contour_label = contour_label
            else:
                cur_v = cur_v + 1  # check next cur

    #print("labels:", label_equiv[:50])
    #print("length:", con_length[:50])
    for i in range(50):
        print(i,  label_equiv[i], con_length[i])

    # merge  the connected ones
    # @TODO add the length, but needed now because  the longest is far longer
    for label in range(contour_label, 0, -1):
        graph[graph[:, :, 2] == label, 2] = label_equiv[label]

    return graph, longest_contour_label

###############################################################################
# build edge graph 
#
# return : graph (np of one dimentional objects)  
#        : lengths of contours (1-d numpy array) 
#        : longest contour's label 
#
# version 2: 1-D (source vertex) of dictionaries (target vertices)
# 1. construct edge (vertex to vetex) graph 
# 2. calcuate the face polarities
# 3. detect the contour edges 
# 4. labeling the connected contour edges 
# 5. merge the connected contours  
#
###############################################################################
def build_edge_graph_dict(vertices, faces, face_visibility, partmap = None, interested = -1, bDebug  = False):

    n_vertices = vertices.shape[0]
    if bDebug: 
        print('n_vertices:', n_vertices)
        print('n_faces:', faces.shape)  # model.f.shape)
        print('face_vis:', face_visibility.shape)

    # 1.construct edge map (no directed) from faces
    # the most typical ds of graph is  bidirectional dictionary
    # note: only low half triangle is used (no direction)
    #
    # but now implemented in 2-D and 3 channel format
    # graph[s] : dict[t] = [face1, face2, status)
    #            faces : -1, -1,  not connected
    #                     f1, f2 two faces
    #                     f1, -1  single face (impossible for now)

    graph =  np.zeros(n_vertices, dtype=object)
    for v in range(n_vertices):
        graph[v] = {} 

    max_edge_label = 1000
    # the smallest label equivalent
    label_equiv = -1*np.ones(max_edge_label, dtype='int64')
    for i in range(1, 1000):
        label_equiv[i] = i
    con_length = np.zeros(max_edge_label, dtype='int64')

    # contour_edges
    # indices: 0: big vertex, 1: small vertex, 2: face1, 3: face2, 4: edge label
    contours = []

    # print('graph:', graph[:5])
    # 2. face poloarity 
    #    based in faces, fill the link from edge to faces
    #                 already have face to edges, and vertices
    for fidx, fv in enumerate(faces):

        [v1, v2, v3] = sorted([fv[0], fv[1], fv[2]])  # increasing order

        edge_info = graph[v3].get(v2)
        if edge_info is None:
            graph[v3][v2] = [fidx, -1, -1]
        else:
            graph[v3][v2][1] = fidx  
        edge_info = graph[v3].get(v1)
        if edge_info is None:
            graph[v3][v1] = [fidx, -1, -1]
        else:
            graph[v3][v1][1] = fidx  
        edge_info = graph[v2].get(v1)
        if edge_info is None:
            graph[v2][v1] = [fidx, -1, -1]
        else:
            graph[v2][v1][1] = fidx  

    if bDebug:  
        print('finished graph')
        print('graph:', graph[:10])

    c_start = time.time()

    # 3. detect the contour edges  
    n_cedge = 0 # num of contour edge for checking
    for s in range(n_vertices):
        for t in graph[s]: 
                           
            if graph[s][t][0] > -1 and graph[s][t][1] > -1:
                #print('edge:', graph[s][t][0], graph[s][t][1])
                if face_visibility[graph[s][t][0]]*face_visibility[graph[s][ t][1]] < 0:
                    if partmap is not None:
                        graph[s][t][2] = 0    # contour edge detected, initialized, not index assinged
                        n_cedge = n_cedge + 1
                        # print('(', s, '->', t, ')', end=' ')
                    else:
                        graph[s][t][2] = 0    # contour edge detected, initialized, not index assinged
                        n_cedge = n_cedge + 1
                        # print('(', s, '->', t, ')', end=' ')
                else:
                    pass # same direction faced edge 

            else: # single faced edge ... impossible 
                print('should not happend:' '(', s, '->', t, ')')
                print(graph[s][t][0], graph[s][t][1])

    if bDebug:
        print('finished contour detection : ', n_cedge)
        #print("contour time: ", time.time() - start, "seconds")

    # 4. extract 'connected' contour edges
    contour_label = 0
    found_cedge = True # if found contour edge, ie. remained any 
    while found_cedge is True:

        # 2.1) search a starting edge/vertex
        found_cedge = False
        cur_v = 0
        next_v = -1
        while (found_cedge is False) and (cur_v < n_vertices):  
            # not yet found any seed one # not all edge checked 
            for v in graph[cur_v]:
                #(v_big, v_small) = (cur_v, v) if cur_v > v else (v, cur_v)
                #if graph[v_big][v_small][2] == 0:  # not used c edge 
                if graph[cur_v][v][2] == 0:  # not used c edge 
                    found_cedge = True
                    next_v = v
                    break
            if found_cedge:
                break
            else:
                cur_v = cur_v + 1

        # yes, dound seed contour, path through connected contour edges
        if found_cedge is True:
            num_edges = 0
            contour_label = contour_label + 1
            (v_big, v_small) = (cur_v, v) if cur_v > v else (v, cur_v)
            graph[v_big][v_small][2] = contour_label
            num_edges = num_edges + 1
            if bDebug: 
                print(contour_label, '-th contour:', '(', cur_v, '-', next_v, ')', end='')
            cur_v = next_v
            found_next_v = True

            while found_next_v is True:
                found_next_v = False
                for v in range(n_vertices):
                    (v_big, v_small) = (cur_v, v) if cur_v > v else (v, cur_v)
                    if graph[v_big].get(v_small) is not None:  
                        if graph[v_big][v_small][2] == 0:  # unused c edge
                            graph[v_big][v_small][2] = contour_label
                            if bDebug:
                                print('(', cur_v, '-', v, ')', end='')
                            num_edges = num_edges + 1
                            found_next_v = True
                            cur_v = v
                            break
                        elif graph[v_big][v_small][2] > 0:  # for later merging
                            if label_equiv[contour_label] > graph[v_big][v_small][2]:
                                if bDebug:
                                    print("connected:", contour_label, "=>", graph[v_big][v_small][ 2])
                                label_equiv[contour_label] = graph[v_big][v_small][2]

            if bDebug:
                print(' [total=', num_edges, ']')
            con_length[contour_label] = num_edges

    if bDebug:
        #print("labels:", label_equiv[:50])
        #print("length:", con_length[:50])
        for i in range(50):
            print(i,  label_equiv[i], con_length[i])

    # 5. merge  the connected ones
    # @TODO add the length, but needed now because  the longest is far longer
    for label in range(contour_label, 0, -1):
        if label_equiv[label] != label:
            con_length[label_equiv[label]] = con_length[label_equiv[label]] + con_length[label] 
            con_length[label] =  0 
            # BUG FIXED 2019.07.10, should chnage the graph label too here
            for s in range(n_vertices):
                for t in graph[s]: 
                    newlabel = label_equiv[label]
                    if graph[s][t][2] == label:
                        graph[s][t][2] = newlabel

    if bDebug:
        print('connected edges len:')
        for i in range(50):
             print(i,  label_equiv[i], con_length[i])

    longest_contour_label = np.argmax(con_length)
    #longest_contour_len = np.amax(con_length) 

    return graph, longest_contour_label, con_length

#
# use  part of vertices 
#
# 1. construct edge (vertex to vetex) graph 
# 2. calcuate the face polarities
# 3. detect the contour edges   ***** HERE check the body part 
# 4. labeling the connected contour edges 
# 5. merge the connected contours  
#
def build_edge_graph_dict_part(vertices, faces, face_visibility, partmap = None, interestpart = -1, bDebug = False):

    n_vertices = vertices.shape[0]
    if bDebug: 
        print('n_vertices:', n_vertices)
        print('n_faces:', faces.shape)  # model.f.shape)
        print('face_vis:', face_visibility.shape)

    # 1. construct edge map (no directed) from faces
    # the most typical ds of graph is  bidirectional dictionary
    # note: only low half triangle is used (no direction)
    #
    # but now implemented in 2-D and 3 channel format
    # graph[s] : dict[t] = [face1, face2, status)
    #            faces : -1, -1,  not connected
    #                     f1, f2 two faces
    #                     f1, -1  single face (impossible for now)

    graph =  np.zeros(n_vertices, dtype=object)
    for v in range(n_vertices):
        graph[v] = {} 

    max_edge_label = 1000
    # the smallest label equivalent
    label_equiv = -1*np.ones(max_edge_label, dtype='int64')
    for i in range(1, 1000):
        label_equiv[i] = i
    con_length = np.zeros(max_edge_label, dtype='int64')

    # contour_edges
    # indices: 0: big vertex, 1: small vertex, 2: face1, 3: face2, 4: edge label
    contours = []

    # 2. face polarity 
    # print('graph:', graph[:5])
    # based in faces, fill the link from edge to faces
    #                 already have face to edges, and vertices
    for fidx, fv in enumerate(faces):

        [v1, v2, v3] = sorted([fv[0], fv[1], fv[2]])  # increasing order

        edge_info = graph[v3].get(v2)
        if edge_info is None:
            graph[v3][v2] = [fidx, -1, -1]
        else:
            graph[v3][v2][1] = fidx  
        edge_info = graph[v3].get(v1)
        if edge_info is None:
            graph[v3][v1] = [fidx, -1, -1]
        else:
            graph[v3][v1][1] = fidx  
        edge_info = graph[v2].get(v1)
        if edge_info is None:
            graph[v2][v1] = [fidx, -1, -1]
        else:
            graph[v2][v1][1] = fidx  

    if bDebug:  
        print('finished graph')
        print('graph:', graph[:10])

    c_start = time.time()

    # 3. mark the contour edges as 0 
    n_cedge = 0 # num of contour edge for checking
    for s in range(n_vertices):
        for t in graph[s]: 

            # vertex filtering : consider only interestting vertices 
            if  partmap is not None and partmap[s] != interestpart or partmap[t] != interestpart:
                continue
                           
            if graph[s][t][0] > -1 and graph[s][t][1] > -1:
                #print('edge:', graph[s][t][0], graph[s][t][1])
                if face_visibility[graph[s][t][0]]*face_visibility[graph[s][ t][1]] < 0:
                    if partmap is not None:
                        graph[s][t][2] = 0    # contour edge detected, initialized, not index assinged
                        n_cedge = n_cedge + 1
                        # print('(', s, '->', t, ')', end=' ')
                    else:
                        graph[s][t][2] = 0    # contour edge detected, initialized, not index assinged
                        n_cedge = n_cedge + 1
                        # print('(', s, '->', t, ')', end=' ')
                else:
                    pass # same direction faced edge 

            else: # single faced edge ... impossible 
                print('should not happend:' '(', s, '->', t, ')')
                print(graph[s][t][0], graph[s][t][1])
                    

    if bDebug:
        print('finished contour detection : ', n_cedge)
        #print("contour time: ", time.time() - start, "seconds")

    # 4. extract 'connected' contour edges
    contour_label = 0
    found_cedge = True # if found contour edge, ie. remained any 

    while found_cedge is True:

        # 2.1) search a starting edge/vertex
        found_cedge = False
        cur_v = 0
        next_v = -1
        while (found_cedge is False) and (cur_v < n_vertices):  
            # not yet found any seed one # not all edge checked 
            for v in graph[cur_v]:
                #(v_big, v_small) = (cur_v, v) if cur_v > v else (v, cur_v)
                #if graph[v_big][v_small][2] == 0:  # not used c edge 
                if graph[cur_v][v][2] == 0:  # not used c edge 
                    found_cedge = True
                    next_v = v
                    break
            if found_cedge:
                break
            else:
                cur_v = cur_v + 1

        # yes, dound seed contour, path through connected contour edges
        if found_cedge is True:
            num_edges = 0
            contour_label = contour_label + 1
            (v_big, v_small) = (cur_v, v) if cur_v > v else (v, cur_v)
            graph[v_big][v_small][2] = contour_label
            num_edges = num_edges + 1
            if bDebug: 
                print(contour_label, '-th contour:', '(', cur_v, '-', next_v, ')', end='')
            cur_v = next_v
            found_next_v = True

            while found_next_v is True:
                found_next_v = False
                for v in range(n_vertices):
                    (v_big, v_small) = (cur_v, v) if cur_v > v else (v, cur_v)
                    if graph[v_big].get(v_small) is not None:  
                        if graph[v_big][v_small][2] == 0:  # unused c edge
                            graph[v_big][v_small][2] = contour_label
                            if bDebug:
                                print('(', cur_v, '-', v, ')', end='')
                            num_edges = num_edges + 1
                            found_next_v = True
                            cur_v = v
                            break
                        elif graph[v_big][v_small][2] > 0:  # for later merging
                            if label_equiv[contour_label] > graph[v_big][v_small][2]:
                                if bDebug:
                                    print("connected:", contour_label, "=>", graph[v_big][v_small][ 2])
                                label_equiv[contour_label] = graph[v_big][v_small][2]

            if bDebug:
                print(' [total=', num_edges, ']')
            con_length[contour_label] = num_edges

    if bDebug:
        #print("labels:", label_equiv[:50])
        #print("length:", con_length[:50])
        for i in range(50):
            print(i,  label_equiv[i], con_length[i])

    # 5. merge  the connected ones
    # @TODO add the length, but needed now because  the longest is far longer
    for label in range(contour_label, 0, -1):
        if label_equiv[label] != label:
            con_length[label_equiv[label]] = con_length[label_equiv[label]] + con_length[label] 
            con_length[label] =  0 
            # BUG FIXED 2019.07.10, should chnage the graph label too here
            for s in range(n_vertices):
                for t in graph[s]: 
                    newlabel = label_equiv[label]
                    if graph[s][t][2] == label:
                        graph[s][t][2] = newlabel

     
    if bDebug:
        print('connected edges len:')
        for i in range(50):
             print(i,  label_equiv[i], con_length[i])

    # caculate the longest contours
    longest_contour_label = np.argmax(con_length)
    #longest_contour_len = np.amax(con_length) 

    return graph, longest_contour_label, con_length


##################################################################################
# calcuate the area  of vertexes in graph of label 
# now simply the bounding box 
# only consider x, y not z coordinate
##################################################################################
def calc_contour_area(v3d, graph_dic, clabel):

    min_x = +10000.0 
    max_x = -10000.0 
    min_y = +10000.0 
    max_y = -10000.0 
    #print('len of graph_dic:', len(graph_dic))
    for s in range(len(graph_dic)):
        for t in graph_dic[s]:
            if graph_dic[s][t][2] == clabel:
                v_x, v_y = v3d[s,:2]
                if max_x < v_x:
                    max_x = v_x
                elif min_x > v_x:
                    min_x = v_x
                if max_y < v_y:
                    max_y = v_y
                elif min_y > v_y:
                    min_y = v_y

    if min_x < max_x and min_y < max_y:
        return (max_x - min_x)*(max_y - min_y)
    else:
        return -1  # something wrong 0?  
                

'''
   project vertices with depth values as brightness
'''
def build_depthmap(vertices, cam,  height, width, zonly=True):

    # 1. blank background image
    depthmap = np.ones([height, width], dtype='float32')
    depthmap = depthmap*cam.t.r[2]*10  # very far

    # 2. depth  value for vertices and projection
    if zonly:  # simplified depth utilizing cam.t.x/y and cam.rt = 0
        cam_z = cam.t.r[2]
        for i in range(vertices.shape[0]):
            new_depth = cam_z - vertices[i, 2]
            # print(type(new_depth))
            y, x = int(cam.r[i, 1]), int(cam.r[i, 0])
            if new_depth < depthmap[y, x]:
                depthmap[y, x] = new_depth
    else:
        from numpy import dot
        normalized_camera_pos = cam.t.r/norm(cam.t.r)
        # make depthmap = zeros(th,tw)
        for i in range(vertices.shape[0]):
            displacement = vertices[i, :] - camera_pos
            depth = dot(displacement, camera_pos)
            # depthmap(pjt_vt[i]) = depth

    return depthmap

###############################################################################
# calculate the depth for each vertex
# 
# in: vertices (3d coord) in numpy, camera (rt, t) in Ch instance 
# re: depth in numpy  
#
# notice: the camera in OpenDL/SMPL is OpenCV Camera, not OpenGL 
#         that means t and rt is actually inverse movement of camera
#         check the openCV documents on the details
###############################################################################
def build_depthmap2(vertices, cam):

    cam_z = - cam.t.r[2] # real camera postion 
    if  cam_z > 0:
        depth =  - cam_z + vertices[:, 2]
    else:
        depth = vertices[:, 2] - cam_z

    '''
    # for testing front and back: it shows +z for back side, -z for front 
    depth = np.zeros(vertices.shape[0])
    cam_z = cam.t.r[2]
    # FIXME: vectorization, absoute distance 
    for i in range(vertices.shape[0]):
        if vertices[i,2] > 0:  # only for + z 
            depth[i] = cam_z - vertices[i, 2]
    '''
    return depth

###############################################################################
# rendering depth into brightness image
# depth: distance from camera 
# no light  setting needed for ColorRenderer
#
###############################################################################
def build_depthimage(vertices, faces,  depth,  
                    cam, height, width, near = 0.5,  far = 25):
    
    # 1. normalization to (0,1)
    depth_min = np.amin(depth)
    depth_max = np.amax(depth)
    depth = (depth - depth_min)/(depth_max - depth_min) # for the range : (0, 1)
    bDebug =True 
    if bDebug:
        print('depth_min:', depth_min,'depth_max:', depth_max, '=>', end =' ')
        print('depth_min:', np.amin(depth), 'vc_max:', np.amax(depth), 'avg:', np.mean(depth))
        print('depth.shape:', depth.shape)

    # 2. rendering 
    from opendr.renderer import ColoredRenderer
    # blank background image
    #depthmap = np.ones([height, width], dtype= 'float32')
    # depthmap = depthmap*cam.t.r[2]*10 # very far
    rn = ColoredRenderer()
    rn.camera = cam
    rn.frustum = {'near': near, 'far': far, 'width': width, 'height': height} 

    vc = np.zeros(vertices.shape)
    vc[:,0], vc[:,1],vc[: ,2] = depth, depth, depth   #  gray color 
    if True: # 
        rn.vc = vc   # giving the albera, FIXME: far -> bright? No! so you should  use gray_r for display
    else: 
        from opendr.lighting import LambertianPointLight
        rn.vc = LambertianPointLight(
                f=faces,
                v=vertices,
                num_verts=len(vertices),
                light_pos=np.array([0,0,+2000]),
                vc=np.ones_like(vertices)*.9,
                #vc = vc,
                light_color=np.array([1., 1., 1.]))  #+ .3

    #rn.bgcolor = np.ones(3) 
    rn.bgcolor = ch.zeros(3)
    rn.set(v=vertices, f=faces)

    depthimg = cv2.cvtColor((rn.r*255.).astype('uint8'), cv2.COLOR_BGR2GRAY)
    return depthimg 



###############################################################################
# build mask for a target label  
# 
# Using Color-Rednerer, mapping the label of a vetex to vc value of it  
# 
# v2label: vertice to label mapping 
# target_label: the label in our interest
#               None:  all label mask 
#
###############################################################################
def build_labelmask(vertices, faces,  v2label, target_label,  cam, height, width, near = 0.5,  far = 25):
    
    from opendr.renderer import ColoredRenderer
    rn = ColoredRenderer()
    rn.camera = cam
    rn.frustum = {'near': near, 'far': far, 'width': width, 'height': height} 

    rn.bgcolor = ch.zeros(3)
    rn.set(v=vertices, f=faces)

    vc = np.zeros(vertices.shape)
    if target_label is not None:
        vc[v2label == target_label,:] = 1.0  #  gray color 
    else:
        max_label = float(np.amax(v2label))
        v2label_float = v2label.astype('float32')
        # can keep the difference? Otherwise, we have to run each label and merge again 
        vc[:,0] = v2label_float[:]/max_label  # 0.0 to 1.0 range  
        vc[:,1] = v2label_float[:]/max_label  #   
        vc[:,2] = v2label_float[:]/max_label  #   
        #print(np.mean(vc[:,0]),  np.mean(vc[:, 1]), np.mean(vc[:,2]))

    rn.vc = vc   # giving the albera, FIXME: far -> bright? No! so you should  use gray_r for display
    mask_rgb = (rn.r*max_label).astype('uint8')
    mask_gray = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2GRAY)
    if target_label is not None: # binarization 
        mask_gray[mask_gray > 0] = 255

    return mask_gray 

def build_labelmask2(vertices, faces, v2label, cam, height, width, near = 0.5, far = 25):

    # 1. set up color-renderer
    from opendr.renderer import ColoredRenderer
    rn = ColoredRenderer()
    rn.camera = cam
    rn.frustum = {'near': near, 'far': far, 'width': width, 'height': height} 
    rn.bgcolor = ch.zeros(3)
    rn.set(v=vertices, f=faces)

    # 2. merged mask
    max_label = np.amax(v2label)
    vc = np.zeros(vertices.shape)
    mask_merged = np.zeros((height, width), dtype='uint8')

    # 3. each labels
    for t in range(1, max_label+1):
        vc[:,:] = 0.0  
        vc[v2label == t,:] = 1.0   
        rn.vc = vc   # giving the albera, FIXME: far -> bright? No! so you should  use gray_r for display
        mask_rgb = (rn.r*max_label).astype('uint8')
        mask_gray = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2GRAY)
        mask_merged[mask_gray > 0] = t  # what if conflicts between 

    return mask_merged 

###############################################################################
#  body part assgined
#  
# parts:  vertex to part mapping (in numpy)
# cam : for 2d projection
# height, width: projection range (should be in range)
# serparated: make a channel for each parts or not 
#
# re: label images (single or N channels)
#
###############################################################################
def build_bodypartmap(vertices, cam, parts, height, width, separated=False):

    # 1. blank background image
    if not separated:
        img = np.zeros([height, width], dtype='uint8')
    else:
        img = np.zeros([height, width, 24], dtype='uint8')

    # 2. depth  value for vertices and projection
    for i in range(vertices.shape[0]):
        y, x = int(cam.r[i, 1]), int(cam.r[i, 0])
        if not separated:
            img[y-5:y+5, x-5:x+5] = parts[i] + 64
        else:
            img[y-5:y+5, x-5:x+5, parts[i]] = 1
            #cv2.drawMarker(img[:,:, parts[i]], (x,y), 255, markerSize = 5)

    return img

###############################################################################
#  body part assgined
#
#
#
###############################################################################
def build_bodypartmap_2d(img, vertices2d, parts, colormap, height, width, separated=False):

    # 1. blank background image
    if img is None:
        _img = np.zeros([height, width, 3], dtype='uint8')
    else:
        _img = img.copy()

    # 2. depth  value for vertices and projection
    for i in range(vertices2d.shape[0]):
        y, x = int(vertices2d[i, 1]), int(vertices2d[i, 0])
        cv2.drawMarker(_img, (x,y), colormap[parts[i]], markerSize = 3)

    return _img

############################################################################
# cloth label assignment to vertices
# 
# in: vertices (3d)
# in:
# side effect: cam.v changes
###########################################################################
def build_3dvertex_label(vertices, cam, mask):

    # 1. project 3d to 2d
    cam.v = vertices
    v_2d = cam.r 
    
    # 2. get the cloth label
    v_2d_int = np.rint(v_2d).astype(int32)
    v_labels = mask[v_2d_int[:,1], v_2d_int[:,0]] 
    #v_labels = label[v_2d_int]

    return v_labels


# build face to label map
# 
# we use 3d face-centers label for simplicity 
# we also could use 2d vertex centers ... buttttttttt
# side effects: cam.v changed
#
def build_face2label(cam, v3d, mask, faces):

    # 1. calcuate the centroid of faces 
    fcenter3d = np.zeros(faces.shape, dtype='float32')
    fcenter3d[:,0] = (v3d[faces[:,0], 0] + v3d[faces[:,1], 0] + v3d[faces[:,2], 0])/3.0  # x 
    fcenter3d[:,1] = (v3d[faces[:,0], 1] + v3d[faces[:,1], 1] + v3d[faces[:,2], 1])/3.0  # y 
    fcenter3d[:,2] = (v3d[faces[:,2], 2] + v3d[faces[:,2], 2] + v3d[faces[:,2], 2])/3.0  # z 

    # 2. project it into mask   
    cam.v = fcenter3d

    # 3. get the labels 
    fcenter2d = np.rint(cam.r).astype('int32')
    f2label = mask[fcenter2d[:,1], fcenter2d[:,0]]

    return f2label, fcenter3d, fcenter2d

#
# Test code
#
def test_edge_extraction():

    # 1. extract edge vertices
    # 1.1 construct face_visibility map
    f_normal, v_normal = calc_normal_vectors(cam.v, model.f)
    face_visibility = build_face_visibility(f_normal, cam)

    # 1.3. extract edge vertices
    use_edge_vertices = check_edge_vertices = save_edge_vertices = False
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
        graph, longest_contour_label = build_edge_graph(
            cam.v, model.f, face_visibility)

        num_body_vertices = np.count_nonzero(
            graph[:, :, 2] == longest_contour_label)
        print("Body V Number:", num_body_vertices)
        if save_edge_vertices:
            body_vertices = np.zeros([num_body_vertices, 2], dtype='int32')

        # visualization of contour
        img_contour = np.zeros([th, tw], dtype='uint8')
        i = 0
        if check_edge_vertices or save_edge_vertices:
            for v_s in range(cam.v.shape[0]):
                for v_e in range(v_s):
                    if graph[v_s, v_e, 2] == longest_contour_label:  # > 0:
                        if check_edge_vertices:
                            sx, sy = cam.r[v_s]  # projected coordinate
                            ex, ey = cam.r[v_e]
                        if save_edge_vertices:
                            body_vertices[i, 0], body_vertices[i, 1] = int(
                                sx), int(sy)
                            i = i + 1
                        cv2.line(img_contour, (int(sx), int(sy)), (int(
                            ex), int(ey)), graph[v_s, v_e, 2], thickness=1)



if __name__ =='__main__':

   _test_graph_ds()



# end of code

""" 

    1. TPS Transform demo
    2. Correspondences demo
    

"""
from __future__ import print_function
import math
import random
import time
import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import cPickle as pickle
#from matplotlib.colors import ListedColormap, BoundaryNorm

labeldict = {"background": 0,
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
        
def  body2human():
    """
     1. load SMPL model 
     
     2. project to 2-D
     
     3. matching points (vertices)   
     
     4. estimated 2D TPS
     
     5. apply TPS algorithm
     
     6. get vertices positions
     
     7. display the vertices onto the images 
     
     8. get the cloth vertices
     
     9. 
    """
    pass

def estimateTPS(srcPts, tgtPts, regulation = 0):
    """
      srcPoints: numpy array of 1x-1x2 
      tgtPoints: numpy array of 1x-1x2 
      regulation : tolerance?
    """
    # 4.1 create TPS warper
    tps = cv2.createThinPlateSplineShapeTransformer(regulation)    
    
    # 1. corresponding points  
    matches = list()
    for i in range(srcPts.shape[1]):
        matches.append(cv2.DMatch(i,i,0))

    #print('len of match:',  len(matches))
    
    # 2. estimate TPS paramters using correspondings
    #  @TODO: some problem: points mapping is fine, but image not ^^;; 
    #    https://github.com/opencv/opencv/issues/7084
    tps.estimateTransformation(srcPts, tgtPts, matches)
    regParam= tps.getRegularizationParameter()
    print('TPS param: reg', regParam)
    #print('TPS param:',tps.tpsParameters) 
    
    return tps
  
  
def testTPS():    

    h, w = 640, 480
    srcimg = np.zeros([h,w], dtype = 'uint8')
    tgtimg  = np.zeros([h,w], dtype = 'uint8')
    # body-like
    cv2.ellipse(srcimg, (w//2, h//2),  (w//4, h//3), 0.0, 0.0,
                        360.0,  128, -1)
    # cloth-like
    cv2.ellipse(tgtimg, (w//2, h//2),  (w//3, h//3), 0.0, 0.0,
                        360.0,  128, -1)
   
   
    # 2. get the corresponding points 
    srcPoints = []
    tgtPoints = []
 
    #srcimg2 = srcimg.copy()
    #tgtimg2 = tgtimg.copy()  
    
    step = 20
    markersize = 5
    for y in range(0, h, step):
        c = np.count_nonzero(srcimg[y, :])
        if c == 0:
            continue
        x1 = np.argmax(srcimg[y, :] !=0, axis=0)
        x2 = x1 + c
        print(x1, x2)
        srcPoints.append([y, x1])
        srcPoints.append([y, x2])
    
        # target points
        c = np.count_nonzero(tgtimg[y, :])
        x1 = np.argmax(tgtimg[y, :] !=0, axis=0)
        x2 = x1 + c
        print(x1, x2)
        tgtPoints.append([y, x1])
        tgtPoints.append([y, x2])
       
    
    for i in range(len(srcPoints)):
        cv2.circle(srcimg, tuple(srcPoints[i][::-1]), markersize, 255, -1)
        cv2.circle(tgtimg, tuple(tgtPoints[i][::-1]), markersize, 255, -1)
       
    # grid 
    for y in range(step, h, step):
        srcimg[y-1:y+1,:] = 255
        tgtimg[y-1:y+1,:] = 255
    for x in range(step, w, step):
        srcimg[:,x-1:x+1] = 255
        tgtimg[:,x-1:x+1] = 255    
        
               
    plt.subplot(1,3,1)     
    plt.imshow(srcimg[:,:], cmap='gray')
    plt.title('src')
    plt.subplot(1,3,2)  
    plt.imshow(tgtimg[:,:], cmap='gray')
    plt.title('target')
    #plt.show()        
   
    # 2. create TPS warper 
    tps = cv2.createThinPlateSplineShapeTransformer()
     
    # 3. corresponding points  
    srcPts = np.array(srcPoints,np.float32)
    srcPts = srcPts.reshape(1,-1,2)
    tgtPts = np.array(tgtPoints,np.float32)
    tgtPts = tgtPts.reshape(1,-1,2)
    matches = list()
    for i in range(len(srcPoints)):
        matches.append(cv2.DMatch(i,i,0))
    
    # 4. estimate TPS paramters using correspondings
    #  @TODO: some problem: points mapping is fine, but image not ^^;; 
    #    https://github.com/opencv/opencv/issues/7084
   
    reverse = False
    
    if reverse:
        param = tps.estimateTransformation(tgtPts, srcPts, matches)
    else:
        param = tps.estimateTransformation(srcPts, tgtPts, matches)
    
    
    # 5. check points  
    if reverse:
        estTgtPts = tps.applyTransformation(srcPts)
    else:    
        estTgtPts = tps.applyTransformation(srcPts)
    print('shapes:', srcPts.shape, tgtPts.shape, estTgtPts[1].shape)
    
    if reverse:
        print(' tgt      est         src')
        for i in range(len(srcPoints)):
            print(tgtPts[0, i, :], "=>", estTgtPts[1][0, i, :], ":", srcPts[0, i,:])
    else:        
        print(' src      tgt         gt')
        for i in range(len(srcPoints)):
            print(srcPts[0, i, :], "=>", estTgtPts[1][0, i, :], ":", tgtPts[0, i,:])
                
    
    # 6. warp image 
    warpedimg = tps.warpImage(srcimg)
    for i in range(len(srcPoints)):
        cv2.circle(warpedimg, tuple(estTgtPts[1][0, i, ::-1]), markersize, 255, -1)
          
    plt.subplot(1,3,3) 
    plt.imshow(warpedimg[:,:], cmap='gray')
    plt.title('warped')
    plt.show()     
    # 6. warp points 

#
#  find the index of  cloest pixel at contours of 2D image from the SMPL  boundary vertex 
#
def find_nearest_contourpixel(pt, contour):
    """ pt = (x,y) source point
        contour : list of (x, y) target 
    """
    """
    import sys
    mindist = sys.float_info.max
    minidx = -1
    for i in range(contour.shape[0]):
        dx = (pt[0] - contour[i, 0, 0])
        dy = (pt[1] - contour[i, 0, 1])
        dist = dx*dx + dy*dy
        if dist < mindist:
            mindist = dist
            minidx = i
    """
    dists = (contour[:, 0, 0] -pt[0])**2 + (contour[:,0, 1] -pt[1])**2
    minidx =  np.argmin(dists)
    mindist = dists[minidx]
            
    return minidx, math.sqrt(mindist)    
#
#  find the index of cloeset vertex 
# 
def find_nearest_smpl(pt, vertices):

    """
    import sys
    mindist = sys.float_info.max
    minidx = -1
    # @TODO vectorization 
    for i in range(vertices.shape[0]):
        dx = pt[0] - vertices[i][0]
        dy = pt[1] - vertices[i][1]
        dist = dx*dx + dy*dy
        if dist < mindist:
            mindist = dist
            minidx = i
    """
    dists = (vertices[:,0] -pt[0])**2 + (vertices[:,1] -pt[1])**2
    minidx =  np.argmin(dists)
    mindist = dists[minidx]
    #print(mindist)

    return minidx, math.sqrt(mindist)    

    
#
# extract  contour pixels in the mask 
#
# mask :  binary input (0 or  non zeor)
# annotation :  flag for  showing the boundary or not (mostly debugging purpose) 
#
def extractContours(mask, annotation = False):

    bDebug = False

    img_allcontours = None
    # 3.1 extract contours    
    #_, 
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if bDebug:
        print(len(contours))  # list 
        for i in range(len(contours)):
            print(i, ";", len(contours[i]))
            print(i, ";", type(contours[i]), contours[i].shape)
        
        print(hierarchy) # 3 dimentional list ( different from manual)
    #cv2.drawContours(img, contours, -1, (0,255,0), 3) # To draw all the contours in an image    
    #cnt = contours[0]   But most of the time, below method will be useful:
    #cv2.drawContours(img, [cnt], 0, (0,255,0), 3)
    
    outtest = -1
    maxval = -1
    for i in range(len(contours)):
        if hierarchy[0][i][3] == -1:
            if maxval < len(contours[i]):
                outtest, maxval = i, len(contours[i])        
    if bDebug:
        print('the outest contour:', outtest, maxval)
    
    if annotation == True:
        # 3.2. draw all contours (outer and interiors with different colors)
        img_allcontours = mask.copy()
        for i in range(len(contours)):
            color = 255 if hierarchy[0][i][3] == -1 else 180
            cv2.drawContours(img_allcontours, contours, i, color, 1) # To draw an individual contour, 
    
    return contours, outtest, img_allcontours 
  

# 
# matching mask boudnary to projected edge vertices
#
# mask: bi-level (o: bg, 1:fg)
# edge_vertices: verticies coordinates in numpy [Nx2]
# neck  (x,y)  is used for cutting the head part 
# return : list of (edges, contours, dist)
#
# algorithm
#
# for matching, 
# - mask boundary to vertices 
# - the reverse is difficult : TODO
# - use the distance as a measure: TODO
#
# To handle the head and body separatley, 
# 1) seprate the mask into above and below neck 
# 2) b_h: smpl body boundary vertices of head
#    b_b: smpl body boundary vertices of body
#    m_h: 2d segmentation boundary of head 
#    m_b: 2d segmentation boudnary of head
#
#   
# 
def boundary_match (mask, edge_vertices = None, top_y = None, step = 5) :

    # 1.  extract contours from mask 
    contours, outtest, img_allcontours = extractContours(mask, annotation=True)  

    nearest_list = []
    edge2contour = False  # direction of matching  
    ext_contour = contours[outtest]


    # 2. boundary matching (assuming we have quite good joint matching
    if edge2contour:   
        # case 1: from body to mask  
        if edge_vertices is not None:
            for i in range(edge_vertices.shape[0]): 
                pt = (edge_vertices[i,0], edge_vertices[i,1])
                minidx, dist = find_nearest_contourpixel(pt, ext_contour)
                nearest_list.append([(int(pt[0]), int(pt[1])), (ext_contour[minidx, 0,0], ext_contour[minidx, 0,1]), dist])
                #if  i % 25 == 0:
                #    print(pt, coord, dist)
        else: 
            print("need edge vertices")
            return 

    else:  
        # case 2: from mask contour to body boundary vertices 
        if edge_vertices is not None:
            ext_contour = contours[outtest]
            # 1. adding matches for body 
            for i in range(0, ext_contour.shape[0], step):  #47) :
                if ext_contour[i, 0,  1] > top_y :  # only  below of neck, ie. body 
                     minidx, dist = find_nearest_smpl((ext_contour[i, 0, 0], ext_contour[i, 0, 1]), edge_vertices)
                     nearest_list.append([(int(edge_vertices[minidx,0]), int(edge_vertices[minidx,1])), (ext_contour[i, 0, 0], ext_contour[i,0, 1]),  dist])
                     #print('diff:', i,   edge_vertices[minidx,:] - ext_contour[i, 0, :])

            # 2. add head part 
            head_idxes = np.argwhere(edge_vertices[:,1] < top_y) # head part  
            for i in range(0, head_idxes.shape[0], step): # all?
                idx = head_idxes[i]
                pt = (int(edge_vertices[idx,0]), int(edge_vertices[idx,1]))
                nearest_list.append([pt, pt, 0]) # same position, i.e., pinning not to move it.
                #print('head part:', i, pt) #  edge_vertices[i,:])

        else: # just  for testing purpose 
            deviation = 30 # ! sensitive
            for i in range(0, ext_contour.shape[0], step): # sampling the real boundary 
                dx = random.randint(-deviation, +deviation)
                dy = random.randint(-deviation, +deviation)
                pt = (ext_contour[i, 0, 0] + dx, ext_contour[i, 0, 1] + dy)  # decreasing, inrease? offset?
                minidx, dist = find_nearest_contourpixel(pt, ext_contour)
                nearest_list.append([(int(pt[0]), int(pt[1])), (ext_contour[minidx, 0,0], ext_contour[minidx, 0,1]), dist])

    return nearest_list, img_allcontours


#
# head part 
# TODO: Can we combine head and body part transform? 
#       Maybe it is possible we can pin the boundary of region not to move into different region 
#
def boundary_match_head (mask, edge_vertices = None, top_y = None, step = 5) :

    nearest_list = []

    # 1.  extract contours from mask 
    contours, outtest, img_allcontours = extractContours(mask, annotation=True)  
    ext_contour = contours[outtest]

    # 2. boundary matching (assuming we have quite good joint matching
    # case 2: from mask contour to body boundary vertices 
    # 1. adding matches for body 
    for i in range(0, ext_contour.shape[0], step):  #47) :
        if ext_contour[i, 0,  1] <= top_y :  # above the neck 
             minidx, dist = find_nearest_smpl((ext_contour[i, 0, 0], ext_contour[i, 0, 1]), edge_vertices)
             nearest_list.append([(int(edge_vertices[minidx,0]), int(edge_vertices[minidx,1])), (ext_contour[i, 0, 0], ext_contour[i,0, 1]),  dist])
             #print('diff:', i,   edge_vertices[minidx,:] - ext_contour[i, 0, :])

    # 2. add body part 
    body_idxes = np.argwhere(edge_vertices[:,1] > top_y) # body part  FIXME: USE PARTMAP!! 
    for i in range(0, body_idxes.shape[0]) :#, step): # all?
        idx = body_idxes[i]
        pt = (int(edge_vertices[idx,0]), int(edge_vertices[idx,1]))
        nearest_list.append([pt, pt, 0]) # same position, i.e., pinning not to move it.
        #print('body part:', i, pt) #  edge_vertices[i,:])

    return nearest_list, img_allcontours

def tpsMorph(img_org, mask, edge_vertices = None):

    """
        img_org: 
        mask: bi-level (o: bg, 1:fg)
        edge_vertices: verticies coordinates in numpy [Nx2]
    """

    numplot = 1 + 1 # rgb,  contour 
    # 1. boudnary matching 
    nearest_list, img_allcontours = boundary_match(mask, edge_vertices)

    # visualization 
    img_cor2 = mask.copy()
    for i in range(len(nearest_list)):
        cv2.drawMarker(img_cor2, nearest_list[i][0], 255, markerType=cv2.MARKER_STAR, markerSize=5) # source
        cv2.drawMarker(img_cor2, nearest_list[i][1], 255, markerType=cv2.MARKER_CROSS,markerSize=5) # dest 
        cv2.line(img_cor2, nearest_list[i][0], nearest_list[i][1], 255) # line from src to dest 
    #cv2.drawContours(img_cor2, contours, outtest, 255, 1) # To draw an individual contour, say 4th    
    numplot = numplot +1       
   
    # 2. estimate transform
    # 2.1 reformat corresponding points
    npts = len(nearest_list)
    srcPts = np.zeros([1, npts, 2], dtype ='float32')   
    tgtPts = np.zeros([1, npts, 2], dtype ='float32')   
    for i in range(npts):
        srcPts[0,i,:] = nearest_list[i][0]
        tgtPts[0,i,:] = nearest_list[i][1]
        
    # 2.2 estimate TPS params    
    tps = estimateTPS(srcPts, tgtPts, 20)
    
    # 3. apply it to check it works 
    estTgtPts = tps.applyTransformation(srcPts)
    #print('estTgtPts:', estTgtPts)
    
    # 3.2 check result
    print("distance average: ", estTgtPts[0])
    print(' src   =>    result     :   tgt')
    for i in range(srcPts.shape[1]):
        print(srcPts[0, i, :], "=>", estTgtPts[1][0, i, :], ":", tgtPts[0, i,:])
       
    # 3.3 apply to all vertices  
    img_cor3 = mask.copy()
    if edge_vertices is not None: 
        npts = edge_vertices.shape[0]
        srcPts =  edge_vertices.astype('float32')
        srcPts =  srcPts.reshape(1, -1, 2)
        estTgtPts = tps.applyTransformation(srcPts)
        '''
        srcPts2 = np.zeros([1, npts, 2], dtype ='float32')   
        for  i in range(npts):
            srcPts2[0,i,:] = edge_vertices[i, :]
        print("all edge vertices num:", srcPts.shape, srcPts2.shape)
        print("all edge vertices type:", srcPts.dtype, srcPts2.dtype)
        print("copied ", srcPts)
        print("reshaped", srcPts)
        print('estTgtPts:', estTgtPts)
        '''
        for i in range(srcPts.shape[1]):
            cv2.drawMarker(img_cor3, (int(estTgtPts[1][0,i,0]), int(estTgtPts[1][0,i,1])), 255, markerType=cv2.MARKER_STAR, markerSize=5) # source

    else:
        for i in range(srcPts.shape[1]):
            cv2.drawMarker(img_cor3, (int(estTgtPts[1][0,i,0]), int(estTgtPts[1][0,i,1])), 255, markerType=cv2.MARKER_STAR, markerSize=5) # source
            #cv2.drawMarker(img_cor3, (int(tgtPts[0,i,0]), int(tgtPts[0,i,1])), 255, markerType=cv2.MARKER_CROSS,markerSize=1) # dest 
        
    numplot = numplot +1 
    '''
    plt.imshow(img_cor2)          # show correspondings 
    plt.title('matching')    
    
    #imwrite('contour.png', mask)
    '''
    
    # 4. display all results
    plt.suptitle('correspondence')
    plt.subplot(1,numplot,1)
    plt.imshow(img_org[:,:,::-1])  # show all masks
    plt.title('input')    
    
    plt.subplot(1,numplot,2)
    plt.imshow(img_allcontours)  # show all masks
    plt.title('contours')    
   
    plt.subplot(1,numplot,3)  
    plt.imshow(img_cor2)          # show correspondings 
    plt.title('matching')    
    
    plt.subplot(1,numplot,4)  
    plt.imshow(img_cor3)          # show correspondings 
    plt.title('Morphed Points')    
    plt.show()
    
def testMorph(img_idx):
    """
        get the contour of segmentaion mask
        define a shape and matching the nearest contour pixel
    """    
    # 1. read images 
    infile = "../images/10k/dataset10k_%04d.jpg"%img_idx 
    maskfile = "../results/10k/segmentation/10kgt_%04d.png"%img_idx 
    img_org = cv2.imread(infile, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(maskfile, cv2.IMREAD_UNCHANGED)
    if img_org is None:
        print("cannot open",  infile)
        exit()
    if mask is None:
        print("cannot open",  maskfile)
        exit()

    # 2. edge vertices file (pre-calcuated)
    use_edge_vertices = True 
    edge_vertices = None   
    if use_edge_vertices:

        edge_vertices_path  ='edge_vertices_%04d.pkl'%img_idx
        with  open(edge_vertices_path, 'rb')  as f:
            edge_vertices  = pickle.load(f)   

        print(type(edge_vertices))
        print(edge_vertices.shape)
        print(np.amax(edge_vertices[:,0]))
        print(np.amax(edge_vertices[:,1]))

        '''
        #img_cor3 = mask.copy()
        img_cor3 = img_org.copy()
        for i in range(edge_vertices.shape[0]):
            #print(edge_vertices[i])
            cv2.drawMarker(img_cor3, (edge_vertices[i,0], edge_vertices[i,1]), 255, markerType=cv2.MARKER_STAR, markerSize=3) # source
        plt.imshow(img_cor3[:,:,::-1])
        plt.show()
        exit()
        '''

    '''
    img_org = cv2.imread('in_%d.png'%img_idx, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread('gt_%d.png'%img_idx, cv2.IMREAD_UNCHANGED)
    '''
    # 2. pre-processing 
    mask[mask == labeldict['bag']] = 0  # remove bag 
    if True:  # proprocessing 
        if img_idx == 0:
            mask[500:,190] = 0
            #mask[200:,106] = 0
        elif img_idx == 1:
            mask[500:,220] = 0
            #mask[180:,120] = 0

    tpsMorph(img_org, mask, edge_vertices)
   
if  __name__ == '__main__':

    testMorph(1)
    #testTPS()
    
    

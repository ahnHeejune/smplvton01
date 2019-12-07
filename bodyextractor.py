import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import sys

def extract(srcimg, xoffset = 50, yoffset = 50, tobgcolor = (0,0,0)):
    """ 
       grabcut with offsets 
       assuming that the forground can be bounded by a rectangle.      
       code from :https://docs.opencv.org/3.4.3/d8/d83/tutorial_py_grabcut.html
    """
    
    img = srcimg.copy()
    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    
    w, h = img.shape[1], img.shape[0]
    
    rect = ( xoffset, yoffset, w - xoffset*2, h - yoffset*2)  #(50,50,450,290)
    cv.rectangle(img, (xoffset, yoffset), (w-xoffset, h -yoffset),(0,255,0),3)
    
    cv.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    masked = img*mask2[:,:,np.newaxis]
    
    '''
    plt.subplot(1,3,1)
    plt.imshow(img[:,:, ::-1])
    plt.title('original')
    plt.subplot(1,3,2)
    plt.imshow(masked[:, :, ::-1])
    plt.title('segmented')
    plt.subplot(1,3,3)
    '''
    img[mask2 == 0] = tobgcolor
    
    return img

    '''
    plt.imshow(img)
    plt.title('bg color changed')
    plt.show()
    '''        
                
                
if __name__ == "__main__":

    if len(sys.argv) < 4:
        print("usage: python ", sys.argv[0], " filename xoffset, yoffset")
    
    img = cv.imread(sys.argv[1])
    extract(img, int(sys.argv[2]), int(sys.argv[3]))
    

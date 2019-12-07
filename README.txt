

SMPL model based 3D cloth shape recovery and virtual Try on Project
------------------------------------------------------------------------

PI: Heejune AHN (SeoulTech)
CoI:  Matiur R  (SeoulTech), Paul Rosin (Cardiff U), Yukun Lai (Cardiff U)




 copy smpl directory  inside of code


1. The programs 
--------------


   First verion: model  cloth swapping   


 s.mask ---------------------+
                             :
 s.img --> +----------+      +--> +------------+             
           : img2smpl : --------> : smpl2cloth : cloth_v3d, label 
 s.joint-> :          : s.smpl_p  :            :--------+     
       +-> +----------+ s.cam     +------------+        : 
       :                                                +-->+-------------+
 smpl -+                                                    :smplxlothxfer:-> vtoned img
       :                                                +-->+-------------+ 
       +-> +----------+           +------------+        :     
           : img2smpl : --------> : smpl2body  : -------+ 
 t.joint-> :          : t.smpl_p  :            : body_v3d, label     
 t.img  -> +----------+ t.cam +-> +------------+ 
                              :
 t.mask ----------------------+       



   Inshop cloth VTON
   -------------------

  
    s.img --> +----------+            +------------+             
               : img2smpl : --------> : smpltoseg  |        smplmask_%6d_0.png
     s.joint-> : viton    : s.smpl_p  :            :----->  %6d_0_keypoints.json     
        +-> +----------+ s.cam        +------------+      


                                                                 
 
 
1) 2D image with Joint to SMPL model

for  10k dataset 
    python img2smpl.py .. --dataset=10k --viz

for  viton dataset
    python img2smplviton.py .. --dataset=viton --viz

Note: 
    the codes are simple modification from fit_3d.py for our application


2') SMPL model to shillouette mask 

    python smpl2mask.py .. viton 1

    1. load pkl file 
    2. pose it into standard pose 
    3. rendering it into (binary) mask
    4. the json joint gneration 

2'')  resorting the image and json joint  files 

    1. when saving use the viton_list file for re-naming the images 


2) SMPL model to Cloth model
   python smpl2cloth.py .. 10k 1 

   graphuitl.py used for vertices operations
   boundary_maching.py used for matching boudnary with TPS algorithm

3) transfer cloth (not implemented yet) 
   python smpl_cloth_xfer.py  1 .. 10k 

   
 2. Directory and data prepration  
-------------- -

    mpips-smplify_public_v2 
           |
           + code  # python scripts 
           |    +  fid_3d.py, render_model.py, show_humaneva.py   # original SMPLify release
           |    +  img2smpl.py, smpl2cloth.py (and graphutil.py, boundary_matching.py), smplclothxfer.py  # my work  
           |    + ---  model  # smpl model files
           |             +--- basicmodel_f/m_lbs_10_207_0_v1.0.0..pkl
           |             +--- gmm_08.pkl
           |             +--- regression_locked_nromalized_female/male/hybrid.npz 
           |    +---- library 
           | 
           + results 
           |    +--- <10k> or <viton>   
           |          +---smpl
           |          |    + %04d.pkl
           |          |    + %04d.png
           |          +---segmentation
           |          |    + %04d.png
           |          +---cloth
           |          |    + %04d.pkl
           |          +---vton
           |               + %04d.pkl
           |
           + images
                +--- 10k
                      +--- dataset10k_%04d.jpg
                +--- viton 
                      +--- %06d.jpg
                        


 Data prepration for viton  dataset 
 --------------------------------------------------------

 1. dataset preparation 

   location: Work/VTON/Dataset

  1.1 rename images 

    test images should be renamed for easy use (because image number is not continous).

    sort_and_unsort.py 

    * FORWARD 

             images/0000xx_0.jpg =>  viton_%06d.jpg

	     viton_list.npy

    * BACKWARD 
             images/viton_%06d.jpg <=  00000n.jpg

  1.2 generate est_joints.npz 

    * FORWARD
      cvtjint_viton2smpl.py 


       1. sort the json files as above 
       2. read each file and  extract  the required joint and  make a SMPL style format 
       3. save it  into npz file 

    * BACKWARD

      Also back to json format?

       1. read viton.list.npy file
       for each 
       2. load the joint from pkl file and  get the updated joint from SMPL model
       3. load the json file (corresponding)
       4. update the values in json file 
       5. save the updated values into another json file 

  1.3. gender.cvs 

      created manually: all 1 (woman) 

  2.  im2smplviton.py 

      moddifed  joint locations and optimization cost detials for difference 
      in viton joint and LSP joints 


 


                           



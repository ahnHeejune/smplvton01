

SMPL model based 3D cloth shape recovery and virtual Try on Project
------------------------------------------------------------------------

PI: Heejune AHN (SeoulTech)
CoI:  Matiur Rahman Minar (SeoulTech), Thai Thanh Tuan (SeoulTech)  Paul Rosin (Cardiff U), Yukun Lai (Cardiff U)
Project page: https://minar09.github.io/c3dvton/
Paper: 3D Reconstruction of Clothes using a Human Body Model and its Application to Image-based Virtual Try-On (CVPRW 2020)

------------------------------------------------------------------------

0. directory contents and sub project 

  0. original simplify project (reference purpose) 
  1. in-shop cloth VTON implementation (the current main project)
  2. human image cloth swapping (finised first version, but not published work)
  3. fashion show projects (partially published) 


1. dependency  

   copy smpl directory under the './code' directory.
   othewise, add pathonpath to the smpl project 'web_user' directory


///////////////////////////////////////////////////////////////////////////////////////////////////

 sub-project 1: Inshop cloth VTON

///////////////////////////////////////////////////////////////////////////////////////////////////

  1) pipeline

    Note: cnum : viton orignal cloth number, e.g. 000010_1
          vitonhnum : viton orignal human number, e.g. 000010_0
          hnum : sorted human image number  e.g. viton_000010 

   cloth/<vitoncnum>.jpg
   cloth-mask/<vitoncnum>.jpg --------------------------------+
                                                              |
                                                              v  
                   +--------------+  smplparam.pkl       +-----------------+    results/viton/ 
                   | smpltemplate |--------------------> | cloth2smplmask.m|--->  /c2dw/<cnum>.png  
                   | (python)     |  smpltemlatemask.png | (matlab)        |      /c2dwmask/<cnum>.png       
                   +--------------+  smpltmeplate.pkl    +-----------------+              | 
                                           |                                              |
  pose/<vitonhnum>.json's                  |                                              |
    |  image/<vitonhnum>.jpg               |                                              | 
    |              |                       |                                              | 
    |          +---v---------+             |                                              |
    |          | sort_unsort |------------------------+                                  |
    |          +---|---------+             |          |                                  |
    |              v                       |          |                                  |
    |            <hnum>.png -----+         +----------------------+       +--------------+
    |                            |                    |           |       | 
    |                         +--v------------+       |       +---v--------v---+   results/<dataset>        
  +-v---------+               : img2smplviton :-------v------>: smpl3dclothxfer|     /c3dw/<cnum>_<vitonhnum>.png 
  |cvtjoint_  ->est_joints.npz->  (python)    :<hnum> => <vitonhnum>           |---> /c3dwmask/<cnum>_<viton_hnum>.png  
  |viton2smpl*|               +---------------+               +--------^-------+           | 
  +-----------+                                                       |                   |
                               +--------------------------------------|-------------------+
                               |                                      |
  image-parse/<vitonhnum>.jpg  |   +------------------------- viton_test_pair.txt
  image/<vitonhum>.jpg         |   |
                |     +--------v---v---+
                ----> |  TON+          | -------------> VTON image
                      |                |
                      +----------------+

   
  1. Data prepration from viton to SMPL 

    viton data location: Work/VTON/Dataset

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


  1) cvtjoint_viton2smpl.py 

     sort and re-format (joint) all viton joint file (json) into a single smplify (LSP) npz format
     > python cvtjoint_viton2smpl.py tonp
     input: hard-coded for the viton joint directory 
     output:  viton.npz

     Note: already run and distributed 

  1') sort_unsort.py 
      > python sort_unsort.py 
     
      output>  <hnum>.png  
               mask needed to be sorted? 

      Note: hard-codded for Cardiff Desktop. 
            already done 

  2) Template SMPL mask, joints (jsonfile), smpl and camera file used for mask and joints 
  
    python smpltemplate.py  <1>  

    input  :  smpl model files 
    output :  templatemask.png   wxh = 196x(254*3/2) if 0,  196x254 if 1  
              templatejoints.json (same format as viton )
              templateparam.pkl   (smpl parameter, camera file) 

    Note 1: in fact , the index  (0) is not very important, the params are fixed, not using the specific index  
    Note 2:  you generate the file or you can get it from google drive too

  3) img2smplviton.py  

    python img2smplviton.py .. --dataset=viton --viz

    Note: the codes are  modification from fit_3d.py for our application
          moddifed joint locations (*add detail here)
          optimization cost details 


  4) smpl3dclothxfer.py 

    python smpl3dclothxfer.py .. <template.pkl> cnum hnum 

    input:  <result_dir>/cloth2dwarped/<cnum>.png 
            <result_dir>/cloth2dwarpedmask/<cnum>.png
    output: <result_dir>/cloth3dwarped/<cnum>.png
            <result_dir>/cloth3dwarpedmask/<cnum>.png


////////////////////////////////////////////////////////////////////////////////////////////
  
   subProject 2: Old files for human cloth swapping and Fashion show with 10k dataset

////////////////////////////////////////////////////////////////////////////////////////////


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



 
 
1) 2D image with Joint to SMPL model

for  10k dataset 
    python img2smpl.py .. --dataset=10k --viz

for  viton dataset
    python img2smplviton.py .. --dataset=viton --viz

Note: 
    the codes are simple modification from fit_3d.py for our application

2) Template SMPL mask, joints (jsonfile), smpl and camera file used for mask and joints 
  
    python smpl2mask.py .. viton 0

    input:  smpl parameter file (pkl),  SMPL template model files

    output :  templatemask.png ( w x  h ) :   we can turn on full-size mask with size_ext = True in  the  code.  
              templatejoints.json (same format as viton )
              templateparam.pkl  (parmater file)

    Note 1: in fact , the index  (0) is not very important, the params are fixed, not using the specific index  
    Note 2:  you generate the file or you can get it from google drive too

  
2') SMPL model to shillouette mask 

    python smpl2mask.py .. viton 1

    1. load pkl file 
    2. pose it into standard pose 
    3. rendering it into (binary) mask
    4. the json joint gneration 

2'')  resorting the image and json joint  files 

    1. when saving use the viton_list file for re-naming the images 


3) cloth  3d reconstuction
   python  smpl3dclothrec.py  ..  smpltemplate.pkl  cloth.png  clothmask.png

   Note 1: The script can be run for testing purpose as above.
           Or can be used other script, i.e., smpl3dclothxfer.py 

3') SMPL model to Cloth model
   python smpl2cloth.py .. 10k 1 

   graphuitl.py used for vertices operations
   boundary_maching.py used for matching boudnary with TPS algorithm

4) 3D warping/transfer cloth 

   python smpl3dcothxfer.py .. <cloth_number> <human_number>

   note 1: the script uses smpl3dclothrec.py for reconstruct 3d model of cloth from 2d matching 
   note 2:  
    



4') transfer cloth from src human image to dst human image
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
                        




 


////////////////////////////////////////////////////////////////////////////////////////////

 Git Usage

////////////////////////////////////////////////////////////////////////////////////////////
  
 // setup local reposity 
 git init
 touch *.py
 touch *.txt

 // add files to local  repository 
 git status
 git add *.py
 git add README.txt 

 // commit them  
 2106  git status
 2107  git commit -m 'first commit for sharing update'

 // push to github site
 2108  git remote add origin https://github.com/ahnHeejune/smplvton01.git
 2109  git push -u origin master



////////////////////////////////////////////////////////////////////////////////////////////

 Citation

////////////////////////////////////////////////////////////////////////////////////////////
Please cite our paper in your publications if it helps your research:


@InProceedings{Minar_C3DVTON_2020_CVPR_Workshops,
    title={3D Reconstruction of Clothes using a Human Body Model and its Application to Image-based Virtual Try-On},
    author={Minar, Matiur Rahman and Thai Thanh Tuan and Ahn, Heejune and Rosin, Paul and Lai, Yu-Kun},
    booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month = {June},
    year={2020}
}

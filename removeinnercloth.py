"""
  Remove inner cloth parts

  (c) 2020 Matiur Rahman Minar and Thai Thanh Tuan @ icomlab.seoutech.ac.kr


  Description
  ============
    Remove inner cloth part from cloth image and mask based on the mask labels.

"""
import os
import cv2
import numpy as np


def remove_inner_cloth(img_dir, mask_dir):
    img_files = os.listdir(img_dir)

    for each in img_files:

        try:
            print("processing ", each)

            img_file = os.path.join(img_dir, each)
            mask_file = os.path.join(mask_dir, each)

            img = cv2.imread(img_file)
            mask = cv2.imread(mask_file)

            img_white = np.zeros([img.shape[0], img.shape[1], 3], dtype=np.uint8)
            img_white.fill(255)

            img = img * (mask != 2) + img_white * (mask == 2)
            new_mask = mask * (mask == 1)

            cv2.imwrite(img_file, img)
            cv2.imwrite(mask_file, new_mask)
        except Exception as err:
            print("issue with ", each, err)


if __name__ == '__main__':
    img_dir_path = "./results/viton/c2dw/"
    mask_dir_path = "./results/viton/c2dwmask/"

    remove_inner_cloth(img_dir_path, mask_dir_path)


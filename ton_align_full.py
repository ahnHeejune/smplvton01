import os
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


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


def ton_align(im_path, im_parse_path, im_parse_vis_path, c_name, c_path, save_dir, viz=False):
	im = Image.open(im_path)
	im_parse = Image.open(im_parse_path)
	im_parse_vis = Image.open(im_parse_vis_path)
	c = Image.open(c_path)

	if viz:
		plt.subplot(1, 3, 1)
		plt.imshow(im)
		plt.axis('off')
		plt.title('im')
		plt.draw()

		plt.subplot(1, 3, 2)
		plt.imshow(im_parse_vis)
		plt.axis('off')
		plt.title('parse')
		plt.draw()

		plt.subplot(1, 3, 3)
		plt.imshow(c)
		plt.axis('off')
		plt.title('cloth')
		plt.draw()

		plt.show()

	parse_array = np.array(im_parse)

	parse_bg = (parse_array == 0)

	im_bg = np.zeros_like(im)
	im_bg[:] = 255
	im_bg = im_bg * parse_bg - (1 - parse_bg)  # [-1,1], fill 0 for other parts

	parse_top = (parse_array == 1) + \
					(parse_array == 2) + \
					(parse_array == 4) + \
					(parse_array == 13)
					
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
					
	im_cloth = c * parse_cloth - (1 - parse_cloth)  # [-1,1], fill 0 for other parts

	parse_bottom = (parse_array == 8) + \
					(parse_array == 9) + \
					(parse_array == 12) + \
					(parse_array == 16) + \
					(parse_array == 17) + \
					(parse_array == 18) + \
					(parse_array == 19)
					
	im_bottom = im * parse_bottom - (1 - parse_bottom)  # [-1,1], fill 0 for other parts

	# ton_img = im_bg + im_top + im_cloth + im_bottom
	ton_img = im_top + im_bottom + im_cloth
	ton_img[ton_img <= 0] = 255
	
	if viz:
		plt.subplot(1, 4, 1)
		plt.imshow(im_top)
		plt.axis('off')
		plt.title('top')
		plt.draw()

		plt.subplot(1, 4, 2)
		plt.imshow(im_cloth)
		plt.axis('off')
		plt.title('cloth')
		plt.draw()

		plt.subplot(1, 4, 3)
		plt.imshow(im_bottom)
		plt.axis('off')
		plt.title('bottom')
		plt.draw()

		plt.subplot(1, 4, 4)
		plt.imshow(ton_img)
		plt.axis('off')
		plt.title('ton')
		plt.draw()

		plt.show()

	# Save final result
	Image.fromarray(ton_img.astype('uint8')).save(os.path.join(save_dir, c_name))


def main():
	save_dir = "D:/Research/Fashion-Project-SeoulTech/9. 3D VTON/Results/SMPL-VTON-v2/TON-align-full"
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	
	im_dir = "./images/viton"
	c_dir = "./results/viton/c3dw-transparent"
	im_parse_dir = "./results/viton/segmentation"
	im_parse_vis_dir = "./results/viton/segmentation-vis"
	
	pairs_filepath = "./results/viton/viton_test_pairs.txt"
	f = open(pairs_filepath, 'r')
	pairs_list = f.readlines()
	
	for each in pairs_list:
		print("processing pair:", each)
		pair = each.split(" ")
		c_name = pair[1].strip() + "_" + pair[0] + ".png"
		c_path = os.path.join(c_dir, c_name)
		im_path = os.path.join(im_dir, pair[0] + ".jpg")
		im_parse_path = os.path.join(im_parse_dir, pair[0] + ".png")
		im_parse_vis_path = os.path.join(im_parse_vis_dir, pair[0] + ".png")
		
		ton_align(im_path, im_parse_path, im_parse_vis_path, c_name, c_path, save_dir, viz=True)


if __name__ == "__main__":
	main()

import os
import cv2 
import Equirec2Perspec as E2P 
import equirect2perspective as E2PC
import argparse
from os import walk
import shutil



def parse_text_file(text_file, image_file_dir=None):
	'''Get image names and poses from text file.'''
	poses = []
	images = []

	with open(text_file) as f:
		next(f)  # skip the 3 header lines
		next(f)
		next(f)
		for line in f:
			fname, p0, p1, p2, p3, p4, p5, p6 = line.split()
			p0 = float(p0)
			p1 = float(p1)
			p2 = float(p2)
			p3 = float(p3)
			p4 = float(p4)
			p5 = float(p5)
			p6 = float(p6)
			poses.append((p0, p1, p2, p3, p4, p5, p6))
			if image_file_dir == None:
				images.append(fname)
			else:
				images.append(image_file_dir + fname)


	return images, poses

def use_code_chelhwon(images, image_width, image_height, fx, fy, y, p, r, file_save):

	for idx,img in enumerate(images):
		print(img)
		src = cv2.imread(img)
		if idx % 100:
			print("processed images {}".format(idx))
		for i in range(len(y)):
			perspective_img = E2PC.run(src, image_width, image_height, fx, fy, y[i], p, r[0])
			# print(y[i])
			if y[i] == -90:
				temp = img.split('/')
				print(temp)
				save_path = file_save + 'left/' + temp[-1]
				cv2.imwrite(save_path, perspective_img,[cv2.IMWRITE_PNG_COMPRESSION, 0])
			elif y[i] == 0:
				temp = img.split('/')
				print(temp)
				save_path = file_save + 'center/' + temp[-1]
				cv2.imwrite(save_path, perspective_img,[cv2.IMWRITE_PNG_COMPRESSION, 0])
			elif y[i] == 90:
				temp = img.split('/')
				print(temp)
				save_path = file_save + 'right/' + temp[-1]
				cv2.imwrite(save_path, perspective_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])


def use_other_lib(images, FOV, theta_range, phi_range, image_width, image_height, file_save):
	# print(cmdargs.FOV,cmdargs.theta_range,cmdargs.phi_range,cmdargs.image_height,cmdargs.image_width,cmdargs.folder_name)
	for idx,img in enumerate(images):
		if idx % 100 == 0:
			print("processed images {}".format(idx))
		# print(img)
		equ = E2P.Equirectangular(img)    # Load equirectangular image
		for i in range(len(theta_range)):
			# print(theta_range[i])

			# FOV unit is degree 
			# theta is z-axis angle(right direction is positive, left direction is negative)
			# phi is y-axis angle(up direction positive, down direction negative)
			# height and width is output image dimension 
			#
			# perspective_img = equ.GetPerspective(FOV, theta_range[i], phi_range[0], image_width, image_height) # Specify parameters(FOV, theta, phi, height, width)
			perspective_img = equ.GetPerspective(FOV, theta_range[i], phi_range[0], image_height, image_width) # Specify parameters(FOV, theta, phi, height, width)
			if theta_range[i] == -90:
				temp = img.split('/')
				save_path = file_save + 'left/' + temp[-1]
				cv2.imwrite(save_path, perspective_img,[cv2.IMWRITE_PNG_COMPRESSION, 0])
			elif theta_range[i] == 0:
				temp = img.split('/')
				save_path = file_save + 'center/' + temp[-1]
				cv2.imwrite(save_path, perspective_img,[cv2.IMWRITE_PNG_COMPRESSION, 0])
			elif theta_range[i] == 90:
				temp = img.split('/')
				save_path = file_save + 'right/' + temp[-1]
				cv2.imwrite(save_path, perspective_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])



if __name__ == '__main__':
	aparse = argparse.ArgumentParser(
		prog="extract perspective image from equirectangular image",
		description="todo ..")
	aparse.add_argument('--FOV',
						action='store',
						dest='FOV',
						default=80)
	aparse.add_argument('--fx',
						action='store',
						dest='fx',
						default=600)
	aparse.add_argument('--fy',
						action='store',
						dest='fy',
						default=250)
	aparse.add_argument('--y',
						action='store',
						dest='y',
						default=[-90,0,90])
	aparse.add_argument('--r',
						action='store',
						dest='r',
						default=[0, 90])
	aparse.add_argument('--p',
						action='store',
						dest='p',
						default=0)
	aparse.add_argument('--theta_range',
						action='store',
						dest='theta_range',
						default=[-90,0,90])
	aparse.add_argument('--phi_range',
						action='store',
						dest='phi_range',
						default=[0, 90])
	aparse.add_argument('--image_height',
						action='store',
						dest='image_height',
						default=720)
	aparse.add_argument('--image_width',
						action='store',
						dest='image_width',
						default=1080,
					   type=int)
	aparse.add_argument('--file_name',
						action='store',
						dest='file_name',
						default=1,
					   type=int)
	aparse.add_argument('--use_lib_flag',
						action='store',
						dest='use_lib_flag',
						default=-1,
					   type=int)
	aparse.add_argument('--copy_files_flag',
						action='store',
						dest='copy_files_flag',
						default=True,
					   type=bool)
	
	cmdargs = aparse.parse_args()
	file_path = '/ssd/data/fxpal_yolo_train/yoloTrainData/'
	file_name_ext = file_path + '{:02d}'.format(cmdargs.file_name) + '.txt'
	file_save = file_path + 'perspective/' + '{:02d}'.format(cmdargs.file_name) + '/'
	FOV = cmdargs.FOV
	theta_range = cmdargs.theta_range
	phi_range = cmdargs.phi_range
	image_width = cmdargs.image_width
	image_height = cmdargs.image_height
	use_lib_flag = cmdargs.use_lib_flag # 0 = chelhwon and 1 = other
	y = cmdargs.y
	r = cmdargs.r
	p = cmdargs.p
	fx = cmdargs.fx
	fy = cmdargs.fy
	copy_files_flag = cmdargs.copy_files_flag



	print(file_save, file_name_ext, file_save)
	images, poses = parse_text_file(file_name_ext, file_path)
	print('paramters are: {},{},{},{},{}'.format(FOV, theta_range,phi_range,image_width,image_height))
	if copy_files_flag:
		## used for copying files from one folder to ohter
		
		folders = ['left', 'center', 'right']
		everynImg = 50
		
		for fold in folders:
			mypath = file_save + fold + '/'#'center/'	
			f = []
			for (dirpath, dirnames, filenames) in walk(mypath):
				f.extend(filenames)
				# print(dirpath)
				# print(dirnames)
			for idx, fname in enumerate(f):
				if idx % everynImg == 0:
					temp = fname.split('.')
					print(idx, '\t', temp[-2] + '_' + fold + '.' + temp[-1] )
					shutil.copy( (mypath + fname) , (file_save + 'sample/' + temp[-2] + '_' + fold + '.' + temp[-1] ) )
				# img = cv2.imread(mypath + img)
		print(len(f))

			
	else:
		if use_lib_flag == 0:
			print('using chelhwon library')
			use_code_chelhwon(images, image_width, image_height, fx, fy, y, p, r, file_save)
		elif use_lib_flag == 1:
			print('using other library')
			use_other_lib(images, FOV, theta_range, phi_range, image_width, image_height, file_save)
		else:
			print('select which library you want to use: 0 is for chelhwon code and 1 is for other library')
	
	# #
	# # FOV unit is degree 
	# # theta is z-axis angle(right direction is positive, left direction is negative)
	# # phi is y-axis angle(up direction positive, down direction negative)
	# # height and width is output image dimension 
	# #
	# img = equ.GetPerspective(80, -90, 0, 720, 1080) # Specify parameters(FOV, theta, phi, height, width)
	# cv2.imwrite('perspective1280x960_80_-90.jpg', img)



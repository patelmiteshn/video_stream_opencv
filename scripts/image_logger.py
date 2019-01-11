#!/usr/bin/env python

'''
Read data published through rosbag and converts it into posenet format <image_name, label>

'''

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from sensor_msgs.msg import Joy
import cv_bridge
import cv2
import numpy as np
import equirect2perspective


#output_image_directory = '/home/brendan/projects/data_backup/fxpal_kinect_run2/'
output_image_directory = '/home/mitesh/Documents/catkin_ws/src/magni_log_data/'
current_pose = None
output_msg_list = []
img_width = 1280
img_height = 720
count = 0



#f = open('/home/brendan/projects/data_backup/fxpal_kinect_run2/dataset_test.txt', 'w')
f = open(output_image_directory + 'dataset_train.txt', 'w')

def imageCallback(image):
	
   
	pose = '{} {} {} {} {} {} {} {} {}'.format(current_pose.pose.position.x, 
										 current_pose.pose.position.y,
										 current_pose.pose.position.z, 
										 current_pose.pose.orientation.x, 
										 current_pose.pose.orientation.y, 
										 current_pose.pose.orientation.z, 
										 current_pose.pose.orientation.w)
									
	# Convert sensor_msgs/Image into numpy array.
	bridge = cv_bridge.CvBridge()
	local_image = bridge.imgmsg_to_cv2(image, desired_encoding="bgr8")

	# for i in range(2): # for roll
	# 	for j in range(12): # for yaw
	# # run(src, args.dst_width, args.dst_height, args.f, args.y, args.p, args.r)
	# 		yaw = j*30 # generate perspective image in 30 degree increments
	# 		roll = i*30
	# 		pitch = 0
	# 		focal = 250
	# 		prespective_img = equirect2perspective.run(local_image, img_width, img_height, focal, yaw, pitch, roll)

	image_name = output_image_directory + 'images/image_{:07d}.png'.format(count)
	cv2.imwrite(image_name, local_image)

	output_msg = 'images/image_{:010d}.png'.format(count) + ' ' + pose #images/image_%09d.png' % count + ' ' + pose
	f.write(output_msg + '\n')
	print output_msg

	global count
	count += 1


def poseCallback(pose):
	global current_pose
	current_pose = pose

def joyCallback(joy):
	global joy_data
	joy_data = joy
	 
def listener():
	rospy.init_node('image_logger', anonymous=True)

	global pub
	rospy.Subscriber("/camera/image_raw", Image,imageCallback)
	# rospy.Subscriber("/slam_out_pose", PoseStamped, poseCallback)

	# rospy.Subscriber("/joy",Joy, joyCallback)
	# spin() simply keeps python from exiting until this node is stopped
	rospy.spin()

if __name__ == '__main__':
	listener()

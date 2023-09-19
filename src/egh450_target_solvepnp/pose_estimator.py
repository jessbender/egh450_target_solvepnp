#!/usr/bin/env python3

import sys
import rospy
import cv2
import math
import numpy as np
import tf2_ros
import tf.transformations as tf_trans
from geometry_msgs.msg import TransformStamped, PoseStamped, Point
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Bool, String, Time

class PoseEstimator():
	aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)
	aruco_params = cv2.aruco.DetectorParameters_create()

	def __init__(self):

		self.aruco_pub = rospy.Publisher(
            '/processed_aruco/image/compressed', CompressedImage, queue_size=10)
		self.pub_tts = rospy.Publisher('/depthai_node/detection/tts', String, queue_size=10)
		
		# Set up the CV Bridge
		self.bridge = CvBridge()
		self.pubrefresh = False

		# Load in parameters from ROS
		self.param_use_compressed = rospy.get_param("~use_compressed", False)
		self.param_circle_radius = rospy.get_param("~circle_radius", 1.0)
		self.param_hue_center = rospy.get_param("~hue_center", 170)
		self.param_hue_range = rospy.get_param("~hue_range", 20) / 2
		self.param_sat_min = rospy.get_param("~sat_min", 50)
		self.param_sat_max = rospy.get_param("~sat_max", 255)
		self.param_val_min = rospy.get_param("~val_min", 50)
		self.param_val_max = rospy.get_param("~val_max", 255)

		# Set additional camera parameters
		self.got_camera_info = False
		self.camera_matrix = None
		self.dist_coeffs = None

		# init UAV pose
		self.uav_pose = []
		self.x_p = "-1"
		self.y_p = "-1"

		# Set up landing site publisher
		self.land_pub = rospy.Publisher('landing_site', Bool, queue_size=2)
		self.landing = False

		self.aruco_pose_pub = rospy.Publisher('/depthai_node/detection/aruco_pose', PoseStamped, queue_size=10)

		# Set up the publishers, subscribers, and tf2
		self.sub_info = rospy.Subscriber("~camera_info", CameraInfo, self.callback_info)
		
		self.pub_found = rospy.Publisher('/uavasr/target_found', Time, queue_size=1)

		self.sub_topic_coord = rospy.Subscriber('/depthai_node/detection/target_coord',String, self.callback_coord)
		# TODO: change back for flight
		# self.sub_uav_pose = rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self.callback_uav_pose)
		self.sub_uav_pose = rospy.Subscriber('/uavasr/pose', PoseStamped, self.callback_uav_pose)

		if self.param_use_compressed:
			self.sub_img = rospy.Subscriber("~image_raw/compressed", CompressedImage, self.callback_img)
			self.pub_mask = rospy.Publisher("~debug/image_raw/compressed", CompressedImage, queue_size=1)
			self.pub_overlay = rospy.Publisher("~overlay/image_raw/compressed", CompressedImage, queue_size=1)
		else:
			self.sub_img = rospy.Subscriber("~image_raw", Image, self.callback_img)
			self.pub_mask = rospy.Publisher("~debug/image_raw", Image, queue_size=1)
			self.pub_overlay = rospy.Publisher("~overlay/image_raw", Image, queue_size=1)

		self.tfbr = tf2_ros.TransformBroadcaster()

		# Generate the model for the pose solver
		# For this example, draw a square around where the circle should be
		# There are 5 points, one in the center, and one in each corner
		r = self.param_circle_radius
		self.model_object = np.array([(0.0, 0.0, 0.0),
										(r, r, 0.0),
										(r, -r, 0.0),
										(-r, r, 0.0),
										(-r, -r, 0.0)])

	def shutdown(self):
		# Unregister anything that needs it here
		self.sub_info.unregister()
		self.sub_img.unregister()

	def callback_uav_pose(self, msg_in):
		self.current_location = msg_in.pose.position
		self.uav_pose = [self.current_location.x, self.current_location.y, self.current_location.z, 0.0]
		self.x_p = self.uav_pose[0]
		self.y_p = self.uav_pose[1]
	
	def callback_coord(self, msg_in):		
		if msg_in.data != '':
			msg = msg_in.data.split('-')
			if msg[0] != '1':
				self.x_p = msg[1]
				self.y_p = msg[2]
				self.pubrefresh = True


	# Collect in the camera characteristics
	def callback_info(self, msg_in):
		self.dist_coeffs = np.array([[msg_in.D[0], msg_in.D[1], msg_in.D[2], msg_in.D[3], msg_in.D[4]]], dtype="double")

		self.camera_matrix = np.array([
                 (msg_in.P[0], msg_in.P[1], msg_in.P[2]),
                 (msg_in.P[4], msg_in.P[5], msg_in.P[6]),
                 (msg_in.P[8], msg_in.P[9], msg_in.P[10])],
				 dtype="double")

		if not self.got_camera_info:
			rospy.loginfo("Got camera info")
			self.got_camera_info = True

	def callback_img(self, msg_in):
		# Don't bother to process image if we don't have the camera calibration
		if self.got_camera_info:
			#Convert ROS image to CV image
			cv_image = None

			try:
				if self.param_use_compressed:
					cv_image = self.bridge.compressed_imgmsg_to_cv2( msg_in, "bgr8" )
				else:
					cv_image = self.bridge.imgmsg_to_cv2( msg_in, "bgr8" )
			except CvBridgeError as e:
				rospy.loginfo(e)
				return
			
			if self.landing == False:
				aruco = self.find_aruco(cv_image, msg_in)
				self.publish_to_ros(aruco)

			# Perform a colour mask for detection
			mask_image = self.process_image(cv_image)

			# # Find circles in the masked image
			# min_dist = mask_image.shape[0]/8
			# circles = cv2.HoughCircles(mask_image, cv2.HOUGH_GRADIENT, 1, min_dist, param1=50, param2=20, minRadius=0, maxRadius=0)

			# If circles were detected
			if (self.sub_topic_coord is not None) and self.pubrefresh == True:
				self.pubrefresh = False
				# Just take the first detected circle
				# px = circles[0,0,0]
				# py = circles[0,0,1]
				# pr = circles[0,0,2]

				# Centre of bounding box detection 
				px = int(self.x_p)
				py = int(self.y_p)
				pr = int(0.5 * self.current_location.z) # Current Altitude (m)
				# pr = 30

				# Calculate the pictured the model for the pose solver
				# For this example, draw a square around where the circle should be
				# There are 5 points, one in the center, and one in each corner
				self.model_image = np.array([
											(px, py),
											(px+pr, py+pr),
											(px+pr, py-pr),
											(px-pr, py+pr),
											(px-pr, py-pr)])
				self.model_image = self.model_image.astype('float32')

				# Do the SolvePnP method
				(success, rvec, tvec) = cv2.solvePnP(self.model_object, self.model_image, self.camera_matrix, self.dist_coeffs)
				
				# If a result was found, send to TF2
				if success:
					msg_out = TransformStamped()
					msg_out.header = msg_in.header
					msg_out.child_frame_id = "target"
					msg_out.transform.translation.x = tvec[0]
					msg_out.transform.translation.y = tvec[1]
					msg_out.transform.translation.z = tvec[2]
					msg_out.transform.rotation.w = 1.0	# Could use rvec, but need to convert from DCM to quaternion first
					msg_out.transform.rotation.x = 0.0
					msg_out.transform.rotation.y = 0.0
					msg_out.transform.rotation.z = 0.0

					# time_found = rospy.Time.now()
					time_found = rospy.Time(0)
					self.pub_found.publish(time_found)
					self.tfbr.sendTransform(msg_out)
					

				# Draw the circle for the overlay
				cv2.circle(cv_image, (px,py), 2, (255, 0, 0), 2)	# Center
				cv2.circle(cv_image, (px,py), pr, (0, 0, 255), 2)	# Outline
				cv2.rectangle(cv_image, (px-pr,py-pr), (px+pr,py+pr), (0, 255, 0), 2)	# Model

			#Convert CV image to ROS image and publish the mask / overlay
			try:
				if self.param_use_compressed:
					self.pub_mask.publish( self.bridge.cv2_to_compressed_imgmsg( mask_image, "png" ) )
					self.pub_overlay.publish( self.bridge.cv2_to_compressed_imgmsg( cv_image, "png" ) )
				else:
					self.pub_mask.publish( self.bridge.cv2_to_imgmsg( mask_image, "mono8" ) )
					self.pub_overlay.publish( self.bridge.cv2_to_imgmsg( cv_image, "bgr8" ) )
			except (CvBridgeError,TypeError) as e:
				rospy.loginfo(e)

	def process_image(self, cv_image):
		#Convert the image to HSV and prepare the mask
		hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
		mask_image = None

		hue_lower = (self.param_hue_center - self.param_hue_range) % 180
		hue_upper = (self.param_hue_center + self.param_hue_range) % 180

		thresh_lower = np.array([hue_lower, self.param_val_min, self.param_val_min])
		thresh_upper = np.array([hue_upper, self.param_val_max, self.param_val_max])


		if hue_lower > hue_upper:
			# We need to do a wrap around HSV 180 to 0 if the user wants to mask this color
			thresh_lower_wrap = np.array([180, self.param_sat_max, self.param_val_max])
			thresh_upper_wrap = np.array([0, self.param_sat_min, self.param_val_min])

			mask_lower = cv2.inRange(hsv_image, thresh_lower, thresh_lower_wrap)
			mask_upper = cv2.inRange(hsv_image, thresh_upper_wrap, thresh_upper)

			mask_image = cv2.bitwise_or(mask_lower, mask_upper)
		else:
			# Otherwise do a simple mask
			mask_image = cv2.inRange(hsv_image, thresh_lower, thresh_upper)

		# Refine image to get better results
		kernel = np.ones((5,5),np.uint8)
		mask_image = cv2.morphologyEx(mask_image, cv2.MORPH_OPEN, kernel)

		return mask_image

	def find_aruco(self, frame, msg_in):
		if self.got_camera_info:
			(corners, ids, _) = cv2.aruco.detectMarkers(
				frame, self.aruco_dict, parameters=self.aruco_params)

			tts_target = String()
			tts_target.data = ''

			if len(corners) > 0:
				ids = ids.flatten()

				for (marker_corner, marker_ID) in zip(corners, ids):
					# TODO: if marker_ID == land_aruco
					aruco_corners = corners
					corners = marker_corner.reshape((4, 2))
					(top_left, top_right, bottom_right, bottom_left) = corners

					top_right = (int(top_right[0]), int(top_right[1]))
					bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
					bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
					top_left = (int(top_left[0]), int(top_left[1]))

					cv2.line(frame, top_left, top_right, (0, 255, 0), 2)
					cv2.line(frame, top_right, bottom_right, (0, 255, 0), 2)
					cv2.line(frame, bottom_right, bottom_left, (0, 255, 0), 2)
					cv2.line(frame, bottom_left, top_left, (0, 255, 0), 2)

					rospy.loginfo("Aruco detected, ID: {} a coordinate: {}, {}".format(marker_ID, self.x_p, self.y_p))
					self.landing = True

					tts_target.data = "Landing Site: ArUco Marker {}".format(marker_ID)
					self.pub_tts.publish(tts_target)

					cv2.putText(frame, str(
						marker_ID), (top_left[0], top_right[1] - 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
					
					# Estimate the pose of the ArUco marker
					rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(aruco_corners, 0.3, self.camera_matrix, self.dist_coeffs)
					
					if rvec is not None and tvec is not None:

						roll, pitch, yaw = rvec[0][0]
						# Convert roll and pitch to degrees
						roll_degrees = math.degrees(roll)
						pitch_degrees = math.degrees(pitch)
						
						rospy.loginfo('Roll: {} degrees, Pitch: {} degrees'.format(roll_degrees, pitch_degrees))
						
						# Check if both roll and pitch are close to zero (within a tolerance)
						tolerance_degrees = 5.0  # Adjust this tolerance as needed
						if abs(roll_degrees) < tolerance_degrees and abs(pitch_degrees) < tolerance_degrees:
							rospy.loginfo('Marker is safe to land on.')
						else:
							rospy.loginfo('Marker is not suitable for landing.')

						# Create a rotation matrix from roll, pitch, and yaw
						rotation_matrix = tf_trans.euler_matrix(roll, pitch, yaw, 'sxyz')

						# Extract the quaternion from the rotation matrix
						quaternion = tf_trans.quaternion_from_matrix(rotation_matrix)

						msg_out = TransformStamped()
						msg_out.header = msg_in.header
						msg_out.child_frame_id = "target"
						msg_out.transform.translation.x = tvec[0,0,0]
						msg_out.transform.translation.y = tvec[0,0,1]
						msg_out.transform.translation.z = tvec[0,0,2]
						
						msg_out.transform.rotation.x = quaternion[0]
						msg_out.transform.rotation.y = quaternion[1]
						msg_out.transform.rotation.z = quaternion[2]
						msg_out.transform.rotation.w = quaternion[3]

						# time_found = rospy.Time.now()
						time_found = rospy.Time(0)
						self.pub_found.publish(time_found)
						self.tfbr.sendTransform(msg_out)
			return frame

			
	def publish_to_ros(self, frame):
            msg_out = CompressedImage()
            msg_out.header.stamp = rospy.Time.now()
            msg_out.format = "jpeg"
            msg_out.data = np.array(cv2.imencode('.jpg', frame)[1]).tostring()

            self.aruco_pub.publish(msg_out)

            msg = Bool()
            msg.data = self.landing
            
            self.land_pub.publish(msg)
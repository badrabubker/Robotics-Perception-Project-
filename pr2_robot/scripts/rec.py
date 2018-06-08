#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

	#Convert ROS msg to PCL data
	cloud = ros_to_pcl(pcl_msg)
	
	#Voxel Grid Downsampling
	vox = cloud.make_voxel_grid_filter()
	LEAF_SIZE = 0.01  
	vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
	cloud_filtered = vox.filter()

	#Statistical Outlier Removal Filter
	outlier_filter = cloud_filtered.make_statistical_outlier_filter()
	# Set the number of neighboring points to analyze for any given point
	outlier_filter.set_mean_k(20)
	# Set threshold scale factor
	x = 0.1
	# Any point with a mean distance larger than global (mean distance+x*std_dev) will be considered outlier
	outlier_filter.set_std_dev_mul_thresh(x)
	# Finally call the filter function for magic
	cloud_filtered = outlier_filter.filter()
	test.publish(pcl_to_ros(cloud_filtered))
	#PassThrough Filter
	passthrough = cloud_filtered.make_passthrough_filter()
	passthrough.set_filter_field_name('z')
	axis_min = 0.6
	axis_max = 1.1
	passthrough.set_filter_limits(axis_min, axis_max)
	cloud_filtered = passthrough.filter()
	#for y too we have wide not useable area 
	passthroughY = cloud_filtered.make_passthrough_filter()
	passthroughY.set_filter_field_name('y')
	axis_min = -0.4
	axis_max = 0.4
	passthroughY.set_filter_limits(axis_min, axis_max)
	cloud_filtered = passthroughY.filter()
	
	#RANSAC Plane Segmentation
	seg = cloud_filtered.make_segmenter()
	seg.set_model_type(pcl.SACMODEL_PLANE)
	seg.set_method_type(pcl.SAC_RANSAC)
	max_distance = 0.01
	seg.set_distance_threshold(max_distance)
	inliers, coefficients = seg.segment()
	#Extract inliers and outliers
	extracted_inliers = cloud_filtered.extract(inliers, negative=False)
	extracted_outliers = cloud_filtered.extract(inliers, negative=True)

	#Euclidean Clustering
	white_cloud = XYZRGB_to_XYZ (extracted_outliers)
	tree = white_cloud.make_kdtree()
	
	#Create a cluster extraction object
	ec = white_cloud.make_EuclideanClusterExtraction()
	# Set tolerances for distance threshold 
	# as well as minimum and maximum cluster size (in points)
	ec.set_ClusterTolerance(0.03)
	ec.set_MinClusterSize(100)
	ec.set_MaxClusterSize(3000)
	# Search the k-d tree for clusters
	ec.set_SearchMethod(tree)
	# Extract indices for each of the discovered clusters
	cluster_indices = ec.Extract()

	#Create Cluster-Mask Point Cloud to visualize each cluster separately
	#Assign a color corresponding to each segmented object in scene
	cluster_color = get_color_list(len(cluster_indices))
	color_cluster_point_list = []

	for j, indices in enumerate(cluster_indices):
	    for i, indice in enumerate(indices):
		color_cluster_point_list.append([white_cloud[indice][0],
		                                white_cloud[indice][1],
		                                white_cloud[indice][2],
		                                 rgb_to_float(cluster_color[j])])

	#Create new cloud containing all clusters, each with unique color
	cluster_cloud = pcl.PointCloud_PointXYZRGB()
	cluster_cloud.from_list(color_cluster_point_list)





	# Classify the clusters! (loop through each detected cluster one at a time)
	detected_objects_labels = []
	detected_objects = []
        # Grab the points for the cluster
	for index, pts_list in enumerate(cluster_indices):
		# Grab the points for the cluster from the extracted outliers (cloud_objects)
		pcl_cluster = extracted_outliers.extract(pts_list)
		
		#convert the cluster from pcl to ROS using helper function
		ros_cluster = pcl_to_ros(pcl_cluster)
		# Extract histogram features
		chists = compute_color_histograms(ros_cluster , using_hsv=True)
		normals = get_normals(ros_cluster)
          	nhists = compute_normal_histograms(normals)
            	feature = np.concatenate((chists, nhists))
		# Make the prediction, retrieve the label for the result
		# and add it to detected_objects_labels list
		prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
		label = encoder.inverse_transform(prediction)[0]
		detected_objects_labels.append(label)

		# Publish a label into RViz
		label_pos = list(white_cloud[pts_list[0]])
		label_pos[2] += .4
		object_markers_pub.publish(make_label(label,label_pos, index))

		# Add the detected object to the list of detected objects.
		do = DetectedObject()
		do.label = label
		do.cloud = ros_cluster
		detected_objects.append(do)

	rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))
	# Publish the list of detected objects
	detected_objects_pub.publish(detected_objects)
	
	# Suggested location for where to invoke your pr2_mover() function within pcl_callback()
	# Could add some logic to determine whether or not your object detections are robust
	# before calling pr2_mover()
	try:
		pr2_mover(detected_objects)
	except rospy.ROSInterruptException:
		pass

# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    #Initialize variables
	SCENE_NUM = Int32()
	OBJECT_NAME = String()
	WHICH_ARM = String() 
	PICK_POSE = Pose()
	PLACE_POSE = Pose()
	centroid = []
	output_yaml = []


    #Get/Read parameters
	object_list_param = rospy.get_param('/object_list')
	dropboxs = rospy.get_param('dropbox')
	rospy.loginfo('Starting with {} objects'.format(len(object_list)))
	#SCENE_NUM.data = rospy.get_param('/world_name')
	SCENE_NUM.data = 2

    # TODO: Rotate PR2 in place to capture side tables for the collision map
	#rotater = rospy.Publisher('/pr2/world_joint_controller/command', Float64, queue_size=10)
	#rotater.publish(-np.pi/2) 
	#rotater.publish(-np.pi)
	#rotater.publish(np.pi)
    #  Loop through the pick list

	for counter in range(0,len(object_list_param)):
		OBJECT_NAME.data = object_list_param[counter]['name']
		found = False

        # Get the PointCloud for a given object and obtain it's centroid
		for detected in object_list:

			if (OBJECT_NAME.data == detected.label):

				points_arr = ros_to_pcl(detected.cloud).to_array()
				centroids = (np.mean(points_arr,axis=0)[:3])        
				found = True

		if (found):
			PICK_POSE.position.x = float(centroids[0])
			PICK_POSE.position.y = float(centroids[1])
			PICK_POSE.position.z = float(centroids[2])

			for dropbox in dropboxs:
				if object_list_param[counter]['group'] == dropbox['group']:
					WHICH_ARM.data = dropbox['name']
					PLACE_POSE.position.x = dropbox['position'][0]
					PLACE_POSE.position.y = dropbox['position'][1]
					PLACE_POSE.position.z = dropbox['position'][2]
			# Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format

			yaml_dict = make_yaml_dict(SCENE_NUM,WHICH_ARM,OBJECT_NAME,PICK_POSE,PLACE_POSE)

			output_yaml.append(yaml_dict)



		# Wait for 'pick_place_routine' service to come up

			rospy.wait_for_service('pick_place_routine')



			try:

				pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)
		
                		# Insert your message variables to be sent as a service request

                 		#resp = pick_place_routine(SCENE_NUM, OBJECT_NAME, WHICH_ARM, PICK_POSE, PLACE_POSE)

                		#print ("Response: ",resp.success)



			except rospy.ServiceException, e:

				print "Service call failed: %s"%e

		else:

			rospy.loginfo('Can\'t Find Object: {}'.format(object_list_param[counter]['name'] ))


    # Output your request parameters into output yaml file

	send_to_yaml("output_" + str(SCENE_NUM.data) + ".yaml",output_yaml)

if __name__ == '__main__':

	#ROS node initialization
	rospy.init_node('Perception', anonymous=True)
	#Create Subscribers
	pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

	#Create Publishers
	test = rospy.Publisher("/test", PointCloud2, queue_size=1)
	object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
	detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)

	#Load Model From disk
	model = pickle.load(open('model.sav', 'rb'))
    	clf = model['classifier']
    	encoder = LabelEncoder()
    	encoder.classes_ = model['classes']
   	scaler = model['scaler']

	# Initialize color_list
	get_color_list.color_list = []

	#Spin while node is not shutdown
	while not rospy.is_shutdown():
 		rospy.spin()

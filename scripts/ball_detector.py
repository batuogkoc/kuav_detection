import cv2
import numpy as np
import rospkg
import os
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import TransformStamped, Point
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import rospy
import tf2_ros
from collections import deque
from kudrone_py_utils import *
from kudrone_py_utils.transformation import Transformation
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Byte
from scipy.spatial.transform import Rotation
from closest_point import detect_center, RayCenterCalculator
from lib import ROSCamera, ROSMultiCamera

tf_buffer = tf2_ros.Buffer()

QUEUE_SIZE=100
QUEUE_TIME_LIMIT=rospy.Duration(secs=5)

fwd_cam_img_queue=deque(maxlen=QUEUE_SIZE)
fwd_cam_transform_queue=deque(maxlen=QUEUE_SIZE)

fwd_cam_info=None
fwd_cam_img=None
marker_array_pub=None
marker_pub=None
ray_center_calculator = RayCenterCalculator(1000)

fwd_cam = None
multicamera = None

rays = []

def publish_markers(rays):
    markers = []
    for i, ray in enumerate(rays):
        ray_origin_map = ray[0]
        ray_direction_map = ray[1]
        stamp = ray[2]
        start_point = Point()
        start_point.x = ray_origin_map[0][0]
        start_point.y = ray_origin_map[1][0]
        start_point.z = ray_origin_map[2][0]

        ray_end_map = ray_direction_map*300 + ray_origin_map
        end_point = Point()
        end_point.x = ray_end_map[0][0]
        end_point.y = ray_end_map[1][0]
        end_point.z = ray_end_map[2][0]

        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = stamp
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.id = i
        marker.scale.x = 0.1
        marker.scale.y = 0.02
        marker.scale.z = 0.02
        marker.color.a = 1.0 
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        marker.points = [start_point, end_point]
        markers.append(marker)

    marker_array = MarkerArray()
    marker_array.markers = markers
    marker_array_pub.publish(marker_array)


def detect_balls(img, red_threshold=10):
    img_inverse = cv2.bitwise_not(img)

    hsv_img = cv2.cvtColor(img_inverse, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv_img, (90-red_threshold, 0, 0), (90+red_threshold, 255,255))

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    for contour in contours:
        top_left = contour.min(axis=0)
        bottom_right = contour.max(axis=0)
        bounding_boxes.append((top_left.T, bottom_right.T))

    return bounding_boxes

def get_K_matrix(info:CameraInfo):
    return np.array(info.K).reshape(3,3)

def fwd_cam_transform_callback(data: Odometry):
    global fwd_cam
    global fwd_cam_img
    global multicamera
    global ray_center_calculator
    times = Times()
    # fwd_cam.fill_transforms()
    multicamera.fill_transforms()
    times.add("fill")
    img, center = process_camera(multicamera.cameras["cam_zero"])
    if img is not None:
        fwd_cam_img = img
    print(ray_center_calculator.ray_directions.shape)
    # process_camera(multicamera.cameras["cam_zero"])
    # process_camera(multicamera.cameras["cam_one"])
    # process_camera(multicamera.cameras["cam_two"])
    # process_camera(multicamera.cameras["cam_three"])
    # for camera in multicamera.cameras.values():
    #     img, center = process_camera(camera)
    #     if img is not None and center is not None:
    #         fwd_cam_img = img
    times.add("loop")
    print(times)

def process_camera(cam: ROSCamera):
    ret_img = None
    ret_center = None
    global markers
    global ray_center_calculator

    if len(cam.img_queue) == 0:
        return ret_img, ret_center
    print(cam.num_images_with_tf())
    start = time.perf_counter()
    while cam.num_images_with_tf() > 0:
        msg = cam.pop_message(True)
        if msg is None:
            continue
        image, map_to_fwd_cam_trans = msg

        img = CvBridge().imgmsg_to_cv2(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        bounding_boxes = detect_balls(img)

        K_inv = np.linalg.inv(cam.get_K_matrix())
        # print(K_inv)
        center = None
        for bounding_box in bounding_boxes:
            center = ((bounding_box[1] + bounding_box[0])/2).reshape(2,1)
            pixel_noise = 10
            center = center + (np.random.rand(2,1)*pixel_noise - pixel_noise/2)
            img = cv2.circle(img, center.reshape(-1).astype(int), 3, (255,0,0), thickness=-1)
            center = np.vstack((center, np.ones((1,1))))
            ray_direction = K_inv @ center

            ray_direction /= np.linalg.norm(ray_direction)
            ray_direction = np.array([[ray_direction[2][0], -ray_direction[0][0], -ray_direction[1][0]]])
            # print(ray_direction.reshape(3))

            ray_direction_map = map_to_fwd_cam_trans.rotation.apply(ray_direction.reshape(3)).reshape(3,1)
            ray_origin_map = map_to_fwd_cam_trans.translation

            ray_map = (ray_origin_map, ray_direction_map, image.header.stamp)
            rays.append(ray_map)
            
            center = ray_center_calculator.add_ray(ray_map)
        publish_markers(rays) 

        if center is not None:
            print(center)
            ret_center = center
            point = Point()
            point.x = center[0]
            point.y = center[1]
            point.z = center[2]

            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = image.header.stamp
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.id = 0
            marker.pose.position = point
            marker.scale.x = 2
            marker.scale.y = 2
            marker.scale.z = 2
            marker.color.a = 1.0 
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0

            marker_pub.publish(marker)
        
        ret_img = img

    # print(time.perf_counter()-start)
    return ret_img, ret_center

if __name__=="__main__":
    node = rospy.init_node("ball_detector")
    listener = tf2_ros.TransformListener(tf_buffer)
    rospy.Subscriber("/mavros/global_position/local", Odometry, fwd_cam_transform_callback)
    marker_array_pub = rospy.Publisher("visualization_marker_array", MarkerArray, queue_size=10)
    marker_pub = rospy.Publisher("visualization_marker", Marker, queue_size=10)
    kuav_detection_path = rospkg.RosPack().get_path("kuav_detection")
    
    multicamera = ROSMultiCamera("map", QUEUE_SIZE, QUEUE_TIME_LIMIT)
    multicamera.add_camera("fwd_cam", "/fwd_cam/raw", "/fwd_cam/info", "zephyr_delta_wing_ardupilot_camera__camera_pole__fwd_cam_link")
    
    multicamera.add_camera("cam_zero", "/cam_zero/raw", "/cam_zero/info", "zephyr_delta_wing_ardupilot_camera__camera_pole__cam_zero_link")
    multicamera.add_camera("cam_one", "/cam_one/raw", "/cam_one/info", "zephyr_delta_wing_ardupilot_camera__camera_pole__cam_one_link")
    multicamera.add_camera("cam_two", "/cam_two/raw", "/cam_two/info", "zephyr_delta_wing_ardupilot_camera__camera_pole__cam_two_link")
    multicamera.add_camera("cam_three", "/cam_three/raw", "/cam_three/info", "zephyr_delta_wing_ardupilot_camera__camera_pole__cam_three_link")

    #wait for tf buffer to fill up
    time.sleep(0.5)
    while not rospy.is_shutdown():
        if fwd_cam_img is not None:
            cv2.imshow("fwd_cam", fwd_cam_img)
        else:
            pass
            # print(fwd_cam_img)
        if cv2.waitKey(1) == ord('q'):
            break

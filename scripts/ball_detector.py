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
from closest_point import detect_center

tf_buffer = tf2_ros.Buffer()

QUEUE_SIZE=100
QUEUE_TIME_LIMIT=rospy.Duration(secs=5)

fwd_cam_img_queue=deque(maxlen=QUEUE_SIZE)
fwd_cam_transform_queue=deque(maxlen=QUEUE_SIZE)
base_to_fwd_cam_transform=None
fcu_frame_to_base_transform=Transformation.identity()
fwd_cam_info=None
fwd_cam_img=None
marker_array_pub=None
marker_pub=None


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
        marker.pose.position = start_point
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

def fwd_cam_img_callback(data: Image):
    global fwd_cam_img_queue
    fwd_cam_img_queue.append(data)
    while (data.header.stamp - fwd_cam_img_queue[0].header.stamp)>QUEUE_TIME_LIMIT:
        fwd_cam_img_queue.popleft()
    
    process_data()

def fwd_cam_transform_callback(data: Odometry):
    global fwd_cam_transform_queue
    fwd_cam_transform_queue.append(data)
    while (data.header.stamp - fwd_cam_transform_queue[0].header.stamp)>QUEUE_TIME_LIMIT:
        fwd_cam_transform_queue.popleft()
    process_data()

def fwd_cam_info_callback(data:CameraInfo):
    global fwd_cam_info
    fwd_cam_info = data

def process_data():
    global fwd_cam_img
    global fwd_cam_img_queue
    global fwd_cam_transform_queue
    global markers

    if base_to_fwd_cam_transform is None or fwd_cam_info is None:
        return
    if len(fwd_cam_img_queue) < 1 or len(fwd_cam_transform_queue) < 2:
        return
    if fwd_cam_img_queue[-1].header.stamp>fwd_cam_transform_queue[-1].header.stamp:
        return

    image = fwd_cam_img_queue[-1]
    o1 = fwd_cam_transform_queue[-1] #right after image
    
    i=-1
    while fwd_cam_transform_queue[i].header.stamp>image.header.stamp:
        o1 = fwd_cam_transform_queue[i]
        i -= 1

    i=-1
    while fwd_cam_transform_queue[i].header.stamp>image.header.stamp:
        i -= 1
    o0 = fwd_cam_transform_queue[i] #right before image

    t = duration_to_sec(image.header.stamp-o0.header.stamp)/duration_to_sec(o1.header.stamp-o0.header.stamp)

    trans1 = Transformation.from_Odometry(o1)
    trans0 = Transformation.from_Odometry(o0)

    trans = Transformation.lerp(trans0, trans1, t)

    img = CvBridge().imgmsg_to_cv2(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    bounding_boxes = detect_balls(img)

    K_inv = np.linalg.inv(get_K_matrix(fwd_cam_info))
    # print(K_inv)
    map_to_fwd_cam_trans = trans @ fcu_frame_to_base_transform @ base_to_fwd_cam_transform

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

    publish_markers(rays)

    start = time.perf_counter()
    center = detect_center([(ray[0].reshape(3), (ray[0] + ray[1]).reshape(3)) for ray in rays])
    print(time.perf_counter()-start)
    if center is not None:
        print(center)

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
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.5
        marker.color.a = 1.0 
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        marker_pub.publish(marker)
    
    fwd_cam_img = img

def vtol_state_callback(data: Byte):
    global fcu_frame_to_base_transform
    MAV_VTOL_STATE_UNDEFINED = 0	
    MAV_VTOL_STATE_TRANSITION_TO_FW	= 1 
    MAV_VTOL_STATE_TRANSITION_TO_MC	= 2 
    MAV_VTOL_STATE_MC = 3
    MAV_VTOL_STATE_FW = 4
    matrix = np.array([[0, 0, -1],
                       [0, 1, 0],
                       [1, 0, 0]])

    if data.data == MAV_VTOL_STATE_FW:
        fcu_frame_to_base_transform = Transformation.identity()
    elif data.data == MAV_VTOL_STATE_MC:
        fcu_frame_to_base_transform = Transformation(Rotation.from_matrix(matrix), np.zeros((3,1)))
    elif data.data == MAV_VTOL_STATE_TRANSITION_TO_MC:
        fcu_frame_to_base_transform = Transformation(Rotation.from_matrix(matrix), np.zeros((3,1)))
    elif data.data == MAV_VTOL_STATE_TRANSITION_TO_FW:
        fcu_frame_to_base_transform = Transformation.identity()
    elif data.data == MAV_VTOL_STATE_UNDEFINED:
        fcu_frame_to_base_transform = Transformation.identity()
    else:
        print("Invalid vtol state")
        fcu_frame_to_base_transform = Transformation.identity()


if __name__=="__main__":
    node = rospy.init_node("ball_detector")
    listener = tf2_ros.TransformListener(tf_buffer)
    rospy.Subscriber("/fwd_cam/raw", Image, fwd_cam_img_callback)
    rospy.Subscriber("/fwd_cam/info", CameraInfo, fwd_cam_info_callback)
    rospy.Subscriber("/mavros/global_position/local", Odometry, fwd_cam_transform_callback)
    rospy.Subscriber("/vtol_state", Byte, vtol_state_callback)
    marker_array_pub = rospy.Publisher("visualization_marker_array", MarkerArray, queue_size=10)
    marker_pub = rospy.Publisher("visualization_marker", Marker, queue_size=10)
    kuav_detection_path = rospkg.RosPack().get_path("kuav_detection")
    # img = cv2.imread(os.path.join(kuav_detection_path, "test_data/ball_120deg_720p_100m.png"))
    
    #wait for tf buffer to fill up
    time.sleep(0.5)
    while not rospy.is_shutdown():
        if base_to_fwd_cam_transform is None:
            try:
                transform = tf_buffer.lookup_transform("base_link",  "zephyr_delta_wing_ardupilot_camera__camera_pole__fwd_cam_link", rospy.Time(0))
                # matrix = np.array([[0, -1, 0],
                #                    [1, 0, 0],
                #                    [0, 0, 1],])
                # base_to_fwd_cam_transform = Transformation(Rotation.from_matrix(matrix), np.zeros((3,1))) @ Transformation.from_TransformStamped(transform)
                base_to_fwd_cam_transform = Transformation.from_TransformStamped(transform)
                print(base_to_fwd_cam_transform)
                print("tf found")
            except tf2_ros.LookupException as e:
                print("tf not present")
        if fwd_cam_img is not None:
            cv2.imshow("fwd_cam", fwd_cam_img)
        else:
            pass
            # print(fwd_cam_img)
        if cv2.waitKey(1) == ord('q'):
            break

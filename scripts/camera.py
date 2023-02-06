import rospy
from sensor_msgs.msg import CameraInfo, Image

class ROSCamera(object):
    def __init__(self, camera_topic, camera_info_topic):
        rospy.Subscriber(camera_topic, Image)
        rospy.Subscriber(camera_info_topic, CameraInfo)

    def _image_callback(self, data:Image):
        pass
    def _info_callback(self, data:CameraInfo):
        pass
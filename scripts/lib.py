import rospy
from sensor_msgs.msg import CameraInfo, Image
from collections import deque
import tf2_ros
import bisect
from kudrone_py_utils.transformation import Transformation
from kudrone_py_utils import duration_to_sec
import numpy as np

class ROSCamera(object):
    def __init__(self, img_topic, info_topic, map_frame_id, camera_frame_id, queue_max_len:int, queue_max_duration:rospy.Duration, tf_buffer=None):
        self._camera_info = None

        self.img_queue = deque(maxlen=queue_max_len)
        self.last_image_with_transform_index = 0
        self.queue_max_duration = queue_max_duration

        if tf_buffer is None:
            self._tf_buffer = tf2_ros.Buffer(cache_time=queue_max_duration)
            tf2_ros.TransformListener(self._tf_buffer)
        else:
            self._tf_buffer = tf_buffer

        self.map_frame_id = map_frame_id
        self.camera_frame_id = camera_frame_id

        rospy.Subscriber(img_topic, Image, self._img_callback)
        rospy.Subscriber(info_topic, CameraInfo, self._info_callback)

    def _img_callback(self, data:Image):
        data_to_append = [data, None]
        self.img_queue.append(data_to_append)
        while (data_to_append[0].header.stamp - self.img_queue[0][0].header.stamp)>self.queue_max_duration:
            self.img_queue.popleft() 
            if self.last_image_with_transform_index > 0:
                self.last_image_with_transform_index -= 1
        
    def fill_transforms(self):
        # for msg in self.img_queue:
        #     print(duration_to_sec(msg[0].header.stamp-self.img_queue[0][0].header.stamp), end=" ")
        # print()

        for i in range(self.last_image_with_transform_index, len(self.img_queue)):
            try:
                transform = self._tf_buffer.lookup_transform(self.map_frame_id, self.camera_frame_id, self.img_queue[i][0].header.stamp)
                self.img_queue[i][1] = Transformation.from_TransformStamped(transform)
                self.last_image_with_transform_index += 1
            except tf2_ros.ExtrapolationException as e:
                if "past" in e.args:
                    self.img_queue.popleft()
                    continue
                else:
                    break

    def _info_callback(self, data:CameraInfo):
        self._camera_info = data

    def _tf_trigger_callback(self, data):
        pass
    
    def get_K_matrix(self):
        if self._camera_info is None:
            return None
        else:
            return np.array(self._camera_info.K).reshape(3,3)

    def pop_message(self, has_tf):
        if not has_tf:
            return self.img_queue.pop()
        
        for i in range(len(self.img_queue)-1, 0, -1):
            if self.img_queue[i][1] is not None:
                ret = self.img_queue[i]
                del self.img_queue[i]
                return ret

    def pop_message_left(self, has_tf):
        if not has_tf:
            return self.img_queue.popleft()
        
        for i in range(len(self.img_queue)):
            if self.img_queue[i][1] is not None:
                ret = self.img_queue[i]
                del self.img_queue[i]
                return ret
    
    def num_images_with_tf(self):
        return self.last_image_with_transform_index

    def clear_buffer(self):
        self.img_queue.clear()
        self.last_image_with_transform_index = 0

class ROSMultiCamera():
    def __init__(self, map_frame_id, queue_max_len:int, queue_max_duration:rospy.Duration):
        self.queue_max_duration = queue_max_duration

        self._tf_buffer = tf2_ros.Buffer(cache_time=queue_max_duration)
        tf2_ros.TransformListener(self._tf_buffer)

        self.map_frame_id = map_frame_id

        self.cameras = {str:ROSCamera}

    def add_camera(self, camera_name:str, img_topic:str, info_topic:str, camera_frame_id:str):
        camera = ROSCamera(img_topic, info_topic, self.map_frame_id, camera_frame_id, self.queue_max_len, self.queue_max_duration, tf_buffer=self.tf_buffer)
        self.cameras[camera_name] = camera

    def fill_transforms(self):
        for camera in self.cameras.values():
            camera.fill_transforms()

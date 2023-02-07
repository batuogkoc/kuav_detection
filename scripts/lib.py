import rospy
from sensor_msgs.msg import CameraInfo, Image
from collections import deque
import tf2_ros
import bisect
from kudrone_py_utils.transformation import Transformation
from kudrone_py_utils import duration_to_sec

class ROSCamera(object):
    def __init__(self, img_topic, info_topic, map_frame_id, camera_frame_id, queue_max_len:int, queue_max_duration:rospy.Duration):
        self._camera_info = None

        self.img_queue = deque(maxlen=queue_max_len)
        self.last_image_with_transform_index = 0
        self.queue_max_duration = queue_max_duration

        self._tf_buffer = tf2_ros.Buffer(cache_time=queue_max_duration)
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)

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
        for i in range(self.last_image_with_transform_index, len(self.img_queue)):
            try:
                transform = self._tf_buffer.lookup_transform(self.map_frame_id, self.camera_frame_id, self.img_queue[i][0].header.stamp)
                self.img_queue[i][1] = Transformation.from_TransformStamped(transform)
                self.last_image_with_transform_index += 1
            except tf2_ros.ExtrapolationException as e:
                print(e)

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
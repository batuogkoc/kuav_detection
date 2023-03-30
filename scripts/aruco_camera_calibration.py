from dronekit import connect, VehicleMode, LocationGlobalRelative, LocationGlobal, Command, LocationLocal, Vehicle, Locations
import time
import numpy as np
import math
from pymavlink import mavutil
from pymavlink.dialects.v20.common import MAVLink
import pymap3d as pm
from coordinate_changer import egm96ToEllipsoid, ellipsoidToEgm96
from scipy.spatial.transform import Rotation as R
ATTITUDE_QUATERNION = 31
EXTENDED_SYS_STATE = 245
GPS_GLOBAL_ORIGIN = 49
GLOBAL_POSITION_INT = 33

flight_controller_orientation=None

def request_ekf3_origin(vehicle: Vehicle):
    vehicle.wait_for_armable()

    # print("requesting ekf origin")
    vehicle.message_factory.command_long_send(0,0, mavutil.mavlink.MAV_CMD_REQUEST_MESSAGE, 0, GPS_GLOBAL_ORIGIN, 0, 0, 0, 0, 0, 0)
    ekf_origin = vehicle._master.recv_match(type='GPS_GLOBAL_ORIGIN', blocking=True)
    return (ekf_origin.latitude*1e-7, ekf_origin.longitude*1e-7, ekf_origin.altitude*1e-3)

def request_global_position(vehicle: Vehicle):
    vehicle.wait_for_armable()

    vehicle.message_factory.command_long_send(0,0, mavutil.mavlink.MAV_CMD_REQUEST_MESSAGE, 0, GLOBAL_POSITION_INT, 0, 0, 0, 0, 0, 0)
    global_position = vehicle._master.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
    # print(global_position)
    return (global_position.lat*1e-7, global_position.lon*1e-7, global_position.alt*1e-3)


def ned_to_global(n, e, d, ekf_origin):
    lat, lon, alt = ekf_origin
    return pm.ned2geodetic(n, e, d, lat, lon, alt)

def global_to_ned(lat, lon, alt, ekf_origin):
    lat0, lon0, h0 = ekf_origin
    return pm.geodetic2ned(lat, lon, alt, lat0, lon0, h0)

def global_position_callback(self, name, message):
    global_position_msl = (message.lat*1e-7, message.lon*1e-7, message.alt*1e-3)
    global_position_ellipsoid = egm96ToEllipsoid(*global_position_msl)
    # print(global_position)
    # print(global_to_ned(*global_position_msl, ekf_origin_msl))
    ned_position = global_to_ned(*global_position_ellipsoid, ekf_origin)
    # print(ned_position)

def attitude_callback(self, name, message):
    flight_controller_orientation = R.from_quat((message.q2, message.q3, message.q4, message.q1))
    print(flight_controller_orientation.as_euler("xyz")/math.pi*180)

def wait_for_condition(condition, timeout):
    start_time = time.perf_counter()
    while time.perf_counter()-start_time>timeout or timeout == -1:
        # print("a")
        if condition() is True:
            return True
    return False

#    MAVLink(file).command_long_send(target_system, target_component, command, confirmation, param1, param2, param3, param4, param5, param6, param7)
    # #{'velocity', 'location.global_relative_frame', 'location.global_frame', 'battery', 'airspeed', 'groundspeed', 'last_heartbeat', 'attitude', 'autopilot_version', 'heading', 'location', 'gps_0', 'location.local_frame', 'home_location'}
if __name__ == "__main__":
    vehicle = connect("udpin:127.0.0.1:14550")


    #position callback
    vehicle.wait_for_armable()
    ekf_origin_msl = request_ekf3_origin(vehicle)
    ekf_origin_ellipsoid = egm96ToEllipsoid(*ekf_origin_msl)
    ekf_origin = ekf_origin_ellipsoid
    print("got origin")
    vehicle.add_message_listener("GLOBAL_POSITION_INT", global_position_callback)
    vehicle.message_factory.command_long_send(0,0, mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0, GLOBAL_POSITION_INT, int(1e+6/10), 0, 0, 0, 0, 0)
    print("set position callback")

    vehicle.add_message_listener("ATTITUDE_QUATERNION", attitude_callback)
    vehicle.message_factory.command_long_send(0,0, mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0, ATTITUDE_QUATERNION, int(1e+6/10), 0, 0, 0, 0, 0)
    
    while True:
        pass


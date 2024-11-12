from pathlib import Path
from typing import Generator

from rosbags.rosbag2 import Reader
from rosbags.typesys import get_types_from_msg, get_typestore, Stores
from sensor_msgs.msg import PointCloud2

import pin_slam.utils.point_cloud2 as pc2

import os
import argparse
import numpy as np
import open3d as o3d

from module import ply

gps_msg_text = Path('/opt/ros/humble/share/gps_msgs/msg/GPSFix.msg').read_text()
gps_status_msg_text = Path('/opt/ros/humble/share/gps_msgs/msg/GPSStatus.msg').read_text()
add_types = {}
add_types.update(get_types_from_msg(gps_msg_text, 'gps_msgs/msg/GPSFix'))
add_types.update(get_types_from_msg(gps_status_msg_text, 'gps_msgs/msg/GPSStatus'))

# Create a typestore and get the string class.
typestore = get_typestore(Stores.LATEST)
typestore.register(add_types)

def get_pcd_msgs(msg_reader: Reader, topic_name: str = '/cepton_pcl2') -> Generator[PointCloud2, None, None]:
    if topic_name not in msg_reader.topics.keys():
        while True:
            yield None

    connections_point_cloud = [x for x in msg_reader.connections if x.topic == topic_name]
    for (point_cloud_conn, _, point_cloud_raw) in msg_reader.messages(connections=connections_point_cloud):
        point_cloud_msg = typestore.deserialize_cdr(point_cloud_raw, point_cloud_conn.msgtype)
        yield point_cloud_msg


def rosbag2ply(args):

    os.makedirs(args.output_folder, 0o755, exist_ok=True)
    shift_timestamp = 0
    output_folder_pcd = args.output_folder + "_pcd"

    if args.output_pcd:
        os.makedirs(output_folder_pcd, 0o755, exist_ok=True)

    
    begin_flag = False

    print('Start extraction')
    with Reader(args.input_bag) as msg_reader:
        t = 0
        for point_cloud_msg in get_pcd_msgs(msg_reader, args.topic):
            if point_cloud_msg is None:
                continue

            gen = pc2.read_points(point_cloud_msg)

            array = np.array((gen['x'], gen['y'], gen['z'], gen['intensity'], np.float32(gen['timestamp_s']) + np.float32(gen['timestamp_us']) / 1e-9)).T

            # NOTE: point cloud array: x,y,z,intensity,timestamp,ring,others...
            # could be different for some other rosbags
            # print(array[:, :6])

            timestamps = array[:, 4]
            if not begin_flag:
                shift_timestamp = timestamps[0]
                begin_flag = True

            timestamps_shifted = timestamps - shift_timestamp
            # print(timestamps_shifted)

            field_names = ['x','y','z','intensity','timestamp']
            ply_file_path = os.path.join(args.output_folder, str(t)+".ply")

            if ply.write_ply(ply_file_path, [array[:, :4], timestamps_shifted], field_names):
                print("Export : "+ply_file_path)
            else:
                print('ply.write_ply() failed')

            if args.output_pcd:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(array[:, :3])
                pcd_file_path = os.path.join(output_folder_pcd, str(t)+".pcd")
                o3d.io.write_point_cloud(pcd_file_path, pcd)

            t += 1



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input_bag', help="path to the input rosbag")
    parser.add_argument('-o','--output_folder', help="path for output folder")
    parser.add_argument('-t','--topic', help="name of the point cloud topic used in the rosbag", default="/hesai/pandar_points")
    parser.add_argument('-p','--output_pcd', action='store_true', help='Also output the pcd file')
    args = parser.parse_args()
    print("usage: python3 rosbag2ply.py -i [path to input rosbag] -o [path to point cloud output folder] -t [name of point cloud rostopic]")
    
    rosbag2ply(args)

# Copyright 2008 Willow Garage, Inc.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#    * Neither the name of the Willow Garage, Inc. nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
This file is based on https://github.com/ros2/common_interfaces/blob/4bac182a0a582b5e6b784d9fa9f0dabc1aca4d35/sensor_msgs_py/sensor_msgs_py/point_cloud2.py
All rights reserved to the original authors: Tim Field and Florian Vahl.
"""

import sys
from collections import namedtuple
from typing import Iterable, List, Optional, Tuple

import ctypes
import struct

import numpy as np

try:
    # from rosbags.typesys.types import sensor_msgs__msg__PointCloud2 as PointCloud2
    # from rosbags.typesys.types import sensor_msgs__msg__PointField as PointField
    from sensor_msgs.msg import PointCloud2, PointField
except ImportError as e:
    raise ImportError('rosbags library not installed, run "pip install -U rosbags"') from e


_DATATYPES = {PointField.INT8: np.dtype(np.int8), PointField.UINT8: np.dtype(np.uint8),
              PointField.INT16: np.dtype(np.int16), PointField.UINT16: np.dtype(np.uint16),
              PointField.INT32: np.dtype(np.int32), PointField.UINT32: np.dtype(np.uint32),
              PointField.FLOAT32: np.dtype(np.float32), PointField.FLOAT64: np.dtype(np.float64)}

DUMMY_FIELD_PREFIX = "unnamed_field"


def read_point_cloud(msg: PointCloud2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract poitns and timestamps from a PointCloud2 message.

    :return: Tuple of [points, timestamps]
        points: array of x, y z points, shape: (N, 3)
        timestamps: array of per-pixel timestamps, shape: (N,)
    """
    field_names = ["x", "y", "z"]
    t_field = None
    for field in msg.fields:
        if field.name in ["t", "timestamp", "time", "ts", "timestamps"]:
            t_field = field.name
            field_names.append(t_field)
            break

    points_structured = read_points(msg, field_names=field_names)
    points = np.column_stack(
        [points_structured["x"], points_structured["y"], points_structured["z"]]
    )

    # Remove nan if any
    points = points[~np.any(np.isnan(points), axis=1)]

    if t_field:
        timestamps = points_structured[t_field].astype(np.float64)
        min_timestamp = np.min(timestamps)
        max_timestamp = np.max(timestamps)
        if min_timestamp == max_timestamp:
            timestamps = None
        else:
            timestamps = (timestamps - min_timestamp) / (max_timestamp - min_timestamp) # normalized to 0-1
    else:
        timestamps = None
    return points.astype(np.float64), timestamps


def read_points(
    cloud: PointCloud2,
    field_names: Optional[List[str]] = None,
    uvs: Optional[Iterable] = None,
    reshape_organized_cloud: bool = False,
) -> np.ndarray:
    """
    Read points from a sensor_msgs.PointCloud2 message.
    :param cloud: The point cloud to read from sensor_msgs.PointCloud2.
    :param field_names: The names of fields to read. If None, read all fields.
                        (Type: Iterable, Default: None)
    :param uvs: If specified, then only return the points at the given
        coordinates. (Type: Iterable, Default: None)
    :param reshape_organized_cloud: Returns the array as an 2D organized point cloud if set.
    :return: Structured NumPy array containing all points.
    """
    # Cast bytes to numpy array
    points = np.ndarray(
        shape=(cloud.width * cloud.height,),
        dtype=dtype_from_fields(cloud.fields, point_step=cloud.point_step),
        buffer=cloud.data,
    )

    # Keep only the requested fields
    if field_names is not None:
        assert all(
            field_name in points.dtype.names for field_name in field_names
        ), "Requests field is not in the fields of the PointCloud!"
        # Mask fields
        points = points[list(field_names)]

    # Swap array if byte order does not match
    if bool(sys.byteorder != "little") != bool(cloud.is_bigendian):
        points = points.byteswap(inplace=True)

    # Select points indexed by the uvs field
    if uvs is not None:
        # Don't convert to numpy array if it is already one
        if not isinstance(uvs, np.ndarray):
            uvs = np.fromiter(uvs, int)
        # Index requested points
        points = points[uvs]

    # Cast into 2d array if cloud is 'organized'
    if reshape_organized_cloud and cloud.height > 1:
        points = points.reshape(cloud.width, cloud.height)

    return points


def dtype_from_fields(fields: Iterable[PointField], point_step: Optional[int] = None) -> np.dtype:
    """
    Convert a Iterable of sensor_msgs.msg.PointField messages to a np.dtype.
    :param fields: The point cloud fields.
                   (Type: iterable of sensor_msgs.msg.PointField)
    :param point_step: Point step size in bytes. Calculated from the given fields by default.
                       (Type: optional of integer)
    :returns: NumPy datatype
    """
    # Create a lists containing the names, offsets and datatypes of all fields
    field_names = []
    field_offsets = []
    field_datatypes = []
    for i, field in enumerate(fields):
        # Datatype as numpy datatype
        datatype = _DATATYPES[field.datatype]
        # Name field
        if field.name == "":
            name = f"{DUMMY_FIELD_PREFIX}_{i}"
        else:
            name = field.name
        # Handle fields with count > 1 by creating subfields with a suffix consiting
        # of "_" followed by the subfield counter [0 -> (count - 1)]
        assert field.count > 0, "Can't process fields with count = 0."
        for a in range(field.count):
            # Add suffix if we have multiple subfields
            if field.count > 1:
                subfield_name = f"{name}_{a}"
            else:
                subfield_name = name
            assert subfield_name not in field_names, "Duplicate field names are not allowed!"
            field_names.append(subfield_name)
            # Create new offset that includes subfields
            field_offsets.append(field.offset + a * datatype.itemsize)
            field_datatypes.append(datatype.str)

    # Create dtype
    dtype_dict = {"names": field_names, "formats": field_datatypes, "offsets": field_offsets}
    if point_step is not None:
        dtype_dict["itemsize"] = point_step
    return np.dtype(dtype_dict)


def read_points_list(cloud: PointCloud2, field_names=None, skip_nans=False, uvs=None):
    """
    Read points from a L{sensor_msgs.PointCloud2} message. This function returns a list of namedtuples.
    It operates on top of the read_points method. For more efficient access use read_points directly.

    @param cloud: The point cloud to read from.
    @type  cloud: L{sensor_msgs.PointCloud2}
    @param field_names: The names of fields to read. If None, read all fields. [default: None]
    @type  field_names: iterable
    @param skip_nans: If True, then don't return any point with a NaN value.
    @type  skip_nans: bool [default: False]
    @param uvs: If specified, then only return the points at the given coordinates. [default: empty list]
    @type  uvs: iterable
    @return: List of namedtuples containing the values for each point
    @rtype: list
    """
    if uvs is None:
        uvs = []

    if field_names is None:
        field_names = [f.name for f in cloud.fields]

    Point = namedtuple("Point", field_names)

    return [Point._make(l) for l in read_points(cloud, field_names, skip_nans, uvs)]


def create_cloud(header, fields, points):
    """
    Create a L{sensor_msgs.msg.PointCloud2} message.

    @param header: The point cloud header.
    @type  header: L{std_msgs.msg.Header}
    @param fields: The point cloud fields.
    @type  fields: iterable of L{sensor_msgs.msg.PointField}
    @param points: The point cloud points.
    @type  points: list of iterables, i.e. one iterable for each point, with the
                   elements of each iterable being the values of the fields for
                   that point (in the same order as the fields parameter)
    @return: The point cloud.
    @rtype:  L{sensor_msgs.msg.PointCloud2}
    """

    cloud_struct = struct.Struct(_get_struct_fmt(False, fields))

    buff = ctypes.create_string_buffer(cloud_struct.size * len(points))

    point_step, pack_into = cloud_struct.size, cloud_struct.pack_into
    offset = 0
    for p in points:
        pack_into(buff, offset, *p)
        offset += point_step

    pcd = PointCloud2()
    pcd.header = header
    pcd.height = 1
    pcd.width = len(points)
    pcd.is_dense = False
    pcd.is_bigendian = False
    pcd.fields = fields
    pcd.point_step = cloud_struct.size
    pcd.row_step = cloud_struct.size * len(points)
    pcd.data = buff.raw

    return pcd


def create_cloud_xyz32(header, points):
    """
    Create a L{sensor_msgs.msg.PointCloud2} message with 3 float32 fields (x, y, z).

    @param header: The point cloud header.
    @type  header: L{std_msgs.msg.Header}
    @param points: The point cloud points.
    @type  points: iterable
    @return: The point cloud.
    @rtype:  L{sensor_msgs.msg.PointCloud2}
    """
    x_pcd = PointField()
    x_pcd.name = "x"
    x_pcd.offset = 0
    x_pcd.datatype = PointField.FLOAT32
    x_pcd.count = 1

    y_pcd = PointField()
    y_pcd.name = "y"
    y_pcd.offset = 4
    y_pcd.datatype = PointField.FLOAT32
    y_pcd.count = 1

    z_pcd = PointField()
    z_pcd.name = "z"
    z_pcd.offset = 8
    z_pcd.datatype = PointField.FLOAT32
    z_pcd.count = 1

    fields = [x_pcd,
              y_pcd,
              z_pcd]
    return create_cloud(header, fields, points)


def _get_struct_fmt(is_bigendian, fields, field_names=None):
    fmt = '>' if is_bigendian else '<'

    offset = 0
    for field in (f for f in sorted(fields, key=lambda f: f.offset) if field_names is None or f.name in field_names):
        if offset < field.offset:
            fmt += 'x' * (field.offset - offset)
            offset = field.offset
        if field.datatype not in _DATATYPES:
            print('Skipping unknown PointField datatype [%d]' % field.datatype, file=sys.stderr)
        else:
            datatype_np_format = _DATATYPES[field.datatype]
            fmt += field.count * datatype_np_format.char
            offset += field.count * datatype_np_format.itemsize

    return fmt
#!/usr/bin/env python3
# @file      pin_slam.py
# @author    Yue Pan     [yue.pan@igg.uni-bonn.de]
# Copyright (c) 2024 Yue Pan, all rights reserved

import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # by default 0, change it here if you want to use other GPU 
import sys

import numpy as np
import open3d as o3d
import torch
import dtyper as typer
import wandb
from rich import print
from tqdm import tqdm
from typing import Annotated, Optional, Tuple

from dataset.dataset_indexing import set_dataset_path
from dataset.slam_dataset import SLAMDataset
from dataset.dataloaders import available_dataloaders
from model.decoder import Decoder
from model.neural_points import NeuralPoints
from utils.config import Config
from utils.loop_detector import (
    NeuralPointMapContextManager,
    detect_local_loop,
)
from utils.mapper import Mapper
from utils.mesher import Mesher
from utils.pgo import PoseGraphManager
from utils.tools import (
    freeze_decoders,
    get_time,
    save_implicit_map,
    setup_experiment,
    split_chunks,
    transform_torch,
    remove_gpu_cache
)
from utils.tracker import Tracker
from utils.visualizer import MapVisualizer

'''
    📍PIN-SLAM: LiDAR SLAM Using a Point-Based Implicit Neural Representation for Achieving Global Map Consistency
     Y. Pan et al. from IPB
'''

app = typer.Typer(add_completion=False, rich_markup_mode="rich", context_settings={"help_option_names": ["-h", "--help"]})

_available_dl_help = available_dataloaders()

docstring = f"""
:round_pushpin: PIN-SLAM: a full-fledged implicit neural LiDAR SLAM system \n

[bold green]Examples: [/bold green]

# Process all pointclouds in the given <data-dir> (*.ply, *.pcd, *.bin, etc.) using default config file
$ python pin_slam.py -i <data-dir>:open_file_folder: -vsm

# Process all pointclouds in the given <data-dir> using specific config file (e.g. run_kitti.yaml)
$ python pin_slam.py <path-to-config-file.yaml>:page_facing_up: -i <data-dir>:open_file_folder: -vsm

# Process a given [bold]ROS1/ROS2 [/bold]rosbag file (directory:open_file_folder:, ".bag":page_facing_up:)
$ python pin_slam.py <path-to-config-file.yaml>:page_facing_up: rosbag -i <path-to-my-rosbag>[:open_file_folder:/:page_facing_up:] -dvsm

# Use a more specific dataloader: {", ".join(_available_dl_help)}
# For example, to process KITTI dataset sequence 00:
$ python pin_slam.py ./config/lidar_slam/run_kitti.yaml kitti 00 -i <path-to-kitti-root>:open_file_folder: -dvsm

# For example, to process Replica dataset sequence room0:
$ python pin_slam.py ./config/rgbd_slam/run_replica.yaml replica room0 -i <path-to-replica-root>:open_file_folder: -dvsm
"""

@app.command(help=docstring)
def run_pin_slam(
    config_path: str = typer.Argument('config/lidar_slam/run.yaml', help='Path to *.yaml config file'),
    dataset_name: Optional[str] = typer.Argument(None, help='Name of a specific dataset, example: kitti, mulran, or rosbag (when -d is set)'),
    sequence_name: Optional[str] = typer.Argument(None, help='Name of a specific data sequence or the rostopic for point cloud (when -d is set)'),
    input_path: Optional[str] = typer.Option(None, '--input-path', '-i', help='Path to the point cloud input directory'),
    output_path: Optional[str] = typer.Option(None, '--output-path', '-o', help='Path to the result output directory'),
    frame_range: Optional[Tuple[int, int, int]] = typer.Option(None, '--range', help='Specify the start, end and step of the processed frame, e.g. --range 10 1000 1'),
    seed: int = typer.Option(42, help='Set the random seed'),
    data_loader_on: bool = typer.Option(False, '--data-loader-on', '-d', help='Use specific data loader'),
    visualize: bool = typer.Option(False, '--visualize', '-v', help='Turn on the visualizer'),
    cpu_only: bool = typer.Option(False, '--cpu-only', '-c', help='Run only on CPU'),
    log_on: bool = typer.Option(False, '--log-on', '-l', help='Turn on the logs printing'),
    rerun_on: bool = typer.Option(False, '--rerun-on', '-r', help='Turn on the rerun logging'),
    wandb_on: bool = typer.Option(False, '--wandb-on', '-w', help='Turn on the weight & bias logging'),
    save_map: bool = typer.Option(False, '--save-map', '-s', help='Save the PIN map after SLAM'),
    save_mesh: bool = typer.Option(False, '--save-mesh', '-m', help='Save the reconstructed mesh after SLAM'),
    save_merged_pc: bool = typer.Option(False, '--save-merged-pc', '-p', help='Save the merged point cloud after SLAM'),
    deskew: bool = typer.Option(False, '--deskew', help='Try to deskew the LiDAR scans (this would overwrite the config file deskew parameter)'),
):
    config = Config()
    config.load(config_path)
    config.use_dataloader = data_loader_on
    config.seed = seed
    config.silence = not log_on
    config.wandb_vis_on = wandb_on
    config.rerun_vis_on = rerun_on
    config.o3d_vis_on = visualize
    config.save_map = save_map
    config.save_mesh = save_mesh
    config.save_merged_pc = save_merged_pc
    
    if not config.deskew and deskew:
        config.deskew = True
    
    if frame_range:
        config.begin_frame, config.end_frame, config.step_frame = frame_range
        
    if cpu_only:
        config.device = 'cpu'
        
    if input_path:
        config.pc_path = input_path
        
    if output_path:
        config.output_root = output_path
        
    if dataset_name:
        set_dataset_path(config, dataset_name, sequence_name)
        
    argv = sys.argv
    run_path = setup_experiment(config, argv)
    print("[bold green]PIN-SLAM starts[/bold green]")

    # non-blocking visualizer
    if config.o3d_vis_on:
        o3d_vis = MapVisualizer(config)

    if config.rerun_vis_on:
        import rerun as rr
        rr.init("pin_slam_rerun_viewer", spawn=True)

    # initialize the mlp decoder
    geo_mlp = Decoder(config, config.geo_mlp_hidden_dim, config.geo_mlp_level, 1)
    sem_mlp = Decoder(config, config.sem_mlp_hidden_dim, config.sem_mlp_level, config.sem_class_count + 1) if config.semantic_on else None
    color_mlp = Decoder(config, config.color_mlp_hidden_dim, config.color_mlp_level, config.color_channel) if config.color_on else None

    # initialize the neural points
    neural_points: NeuralPoints = NeuralPoints(config)

    # loop closure detector
    lcd_npmc = NeuralPointMapContextManager(config) # npmc: neural point map context

    mapping_on = True
    # Load the decoder model
    # for the localization with pre-built map mode, set load_model as True and provide the model path in the config file
    if config.load_model:
        loaded_model = torch.load(config.model_path)
        neural_points = loaded_model["neural_points"]
        geo_mlp.load_state_dict(loaded_model["geo_decoder"])
        if 'sem_decoder' in loaded_model.keys():
            sem_mlp.load_state_dict(loaded_model["sem_decoder"])
        if 'color_decoder' in loaded_model.keys():
            color_mlp.load_state_dict(loaded_model["color_decoder"])
        freeze_decoders(geo_mlp, sem_mlp, color_mlp, config)
        
        print("PIN Map loaded")  
        neural_points.recreate_hash(torch.zeros(3).to(config.device), None, True, False)
        mapping_on = False # localization mode
        neural_points.temporal_local_map_on = False # don't use travel distance for filtering
        config.pgo_on = False

    # dataset
    dataset = SLAMDataset(config)

    # odometry tracker
    tracker = Tracker(config, neural_points, geo_mlp, sem_mlp, color_mlp)
    if config.load_model and not mapping_on: 
        tracker.reg_local_map = False

    # mapper
    mapper = Mapper(config, dataset, neural_points, geo_mlp, sem_mlp, color_mlp)

    # mesh reconstructor
    mesher = Mesher(config, neural_points, geo_mlp, sem_mlp, color_mlp)
    cur_mesh = None

    # pose graph manager (for back-end optimization) initialization
    pgm = PoseGraphManager(config) 
    init_pose = dataset.gt_poses[0] if dataset.gt_pose_provided else np.eye(4)  
    pgm.add_pose_prior(0, init_pose, fixed=True)

    last_frame = dataset.total_pc_count-1
    loop_reg_failed_count = 0

    # save merged point cloud map from gt pose as a reference map
    # if config.save_merged_pc and dataset.gt_pose_provided:
    #     print("Load and merge the map point cloud with the reference (GT) poses ... ...")
    #     dataset.write_merged_point_cloud(use_gt_pose=True, out_file_name='merged_gt_pc', 
    #     frame_step=5, merged_downsample=True)
        
    # for each frame
    # frame id as the processed frame, possible skipping done in data loader
    for frame_id in tqdm(range(dataset.total_pc_count)): 

        # I. Load data and preprocessing
        T0 = get_time()

        remove_gpu_cache()

        if config.use_dataloader:
            dataset.read_frame_with_loader(frame_id)
        else:
            dataset.read_frame(frame_id)

        T1 = get_time()
        
        valid_frame = dataset.preprocess_frame()
        if not valid_frame:
            dataset.processed_frame += 1
            continue 

        T2 = get_time()
        
        # II. Odometry
        if frame_id > 0: 
            if config.track_on:
                tracking_result = tracker.tracking(dataset.cur_source_points, dataset.cur_pose_guess_torch, 
                                                   dataset.cur_source_colors, dataset.cur_source_normals,
                                                   vis_result=config.o3d_vis_on and not config.o3d_vis_raw)
                cur_pose_torch, cur_odom_cov, weight_pc_o3d, valid_flag = tracking_result
                dataset.lose_track = not valid_flag
                dataset.update_odom_pose(cur_pose_torch) # update dataset.cur_pose_torch
                
                if not valid_flag and config.o3d_vis_on and o3d_vis.debug_mode > 0:
                    o3d_vis.stop()
                
            else: # incremental mapping with gt pose
                if dataset.gt_pose_provided:
                    dataset.update_odom_pose(dataset.cur_pose_guess_torch) 
                else:
                    sys.exit("You are using the mapping mode, but no pose is provided.")

        travel_dist = dataset.travel_dist[:frame_id+1]
        neural_points.travel_dist = torch.tensor(travel_dist, device=config.device, dtype=config.dtype) # always update this
                                                                                                                                                            
        T3 = get_time()

        # III. Loop detection and pgo
        if config.pgo_on: 
            if config.global_loop_on:
                if config.local_map_context and frame_id >= config.local_map_context_latency: # local map context
                    local_map_frame_id = frame_id-config.local_map_context_latency
                    local_map_pose = torch.tensor(dataset.pgo_poses[local_map_frame_id], device=config.device, dtype=torch.float64)
                    if config.local_map_context_latency > 0:
                        neural_points.reset_local_map(local_map_pose[:3,3], None, local_map_frame_id, config.loop_local_map_by_travel_dist, config.loop_local_map_time_window)
                    context_pc_local = transform_torch(neural_points.local_neural_points.detach(), torch.linalg.inv(local_map_pose)) # transformed back into the local frame
                    neural_points_feature = neural_points.local_geo_features[:-1].detach() if config.loop_with_feature else None
                    lcd_npmc.add_node(local_map_frame_id, context_pc_local, neural_points_feature)
                else: # first frame not yet have local map, use scan context
                    lcd_npmc.add_node(frame_id, dataset.cur_point_cloud_torch)
            pgm.add_frame_node(frame_id, dataset.pgo_poses[frame_id]) # add new node and pose initial guess
            pgm.init_poses = dataset.pgo_poses[:frame_id+1]
            if frame_id > 0:
                cur_edge_cov = cur_odom_cov if config.use_reg_cov_mat else None
                pgm.add_odometry_factor(frame_id, frame_id-1, dataset.last_odom_tran, cov = cur_edge_cov) # T_p<-c
                pgm.estimate_drift(travel_dist, frame_id, correct_ratio=0.01) # estimate the current drift
                if config.pgo_with_pose_prior: # add pose prior
                    pgm.add_pose_prior(frame_id, dataset.pgo_poses[frame_id])
            local_map_context_loop = False
            if frame_id - pgm.last_loop_idx > config.pgo_freq and not dataset.stop_status:
                # detect candidate local loop, find the nearest history pose and activate certain local map
                loop_candidate_mask = ((travel_dist[-1] - travel_dist) > (config.min_loop_travel_dist_ratio*config.local_map_radius)) # should not be too close
                loop_id = None
                if np.any(loop_candidate_mask): # have at least one candidate. firstly try to detect the local loop by checking the distance
                    loop_id, loop_dist, loop_transform = detect_local_loop(dataset.pgo_poses[:frame_id+1], loop_candidate_mask, pgm.drift_radius, frame_id, loop_reg_failed_count, config.local_loop_dist_thre, config.local_loop_dist_thre*3.0, config.silence)
                    if loop_id is None and config.global_loop_on: # global loop detection (large drift)
                        loop_id, loop_cos_dist, loop_transform, local_map_context_loop = lcd_npmc.detect_global_loop(dataset.pgo_poses[:frame_id+1], pgm.drift_radius*config.loop_dist_drift_ratio_thre, loop_candidate_mask, neural_points) # latency has been considered here     
                if loop_id is not None:
                    if config.loop_z_check_on and abs(loop_transform[2,3]) > config.voxel_size_m*4.0: # for multi-floor buildings, z may cause ambiguilties
                        loop_id = None # delta z check failed
                if loop_id is not None: # if a loop is found, we refine loop closure transform initial guess with a scan-to-map registration                    
                    pose_init_torch = torch.tensor((dataset.pgo_poses[loop_id] @ loop_transform), device=config.device, dtype=torch.float64) # T_w<-c = T_w<-l @ T_l<-c 
                    neural_points.recreate_hash(pose_init_torch[:3,3], None, True, True, loop_id) # recreate hash and local map at the loop candidate frame for registration, this is the reason why we'd better to keep the duplicated neural points until the end
                    loop_reg_source_point = dataset.cur_source_points.clone()
                    pose_refine_torch, loop_cov_mat, weight_pcd, reg_valid_flag = tracker.tracking(loop_reg_source_point, pose_init_torch, loop_reg=True, vis_result=config.o3d_vis_on)
                    if config.o3d_vis_on and o3d_vis.debug_mode > 1: # visualize the loop closure and loop registration (when the vis debug mode is on)
                        points_torch_init = transform_torch(loop_reg_source_point, pose_init_torch) # apply transformation
                        points_o3d_init = o3d.geometry.PointCloud()
                        points_o3d_init.points = o3d.utility.Vector3dVector(points_torch_init.detach().cpu().numpy().astype(np.float64))
                        loop_neural_pcd = neural_points.get_neural_points_o3d(query_global=False, color_mode=o3d_vis.neural_points_vis_mode, random_down_ratio=1)
                        o3d_vis.update(points_o3d_init, neural_points=loop_neural_pcd, pause_now=True)
                        o3d_vis.update(weight_pcd, neural_points=loop_neural_pcd, pause_now=True)
                    # only conduct pgo when the loop and loop constraint is correct
                    if reg_valid_flag: # refine succeed
                        pose_refine_np = pose_refine_torch.detach().cpu().numpy()
                        loop_transform = np.linalg.inv(dataset.pgo_poses[loop_id]) @ pose_refine_np # T_l<-c = T_l<-w @ T_w<-c # after refinement
                        cur_edge_cov = loop_cov_mat if config.use_reg_cov_mat else None
                        reg_valid_flag = pgm.add_loop_factor(frame_id, loop_id, loop_transform, cov = cur_edge_cov)
                    if reg_valid_flag:
                        if not config.silence:
                            print("[bold green]Refine loop transformation succeed [/bold green]")
                        pgm.optimize_pose_graph() # conduct pgo
                        cur_loop_vis_id = frame_id-config.local_map_context_latency if local_map_context_loop else frame_id
                        pgm.loop_edges_vis.append(np.array([loop_id, cur_loop_vis_id],dtype=np.uint32)) # only for vis
                        pgm.loop_edges.append(np.array([loop_id, frame_id],dtype=np.uint32))
                        pgm.loop_trans.append(loop_transform)
                        # update the neural points and poses
                        pose_diff_torch = torch.tensor(pgm.get_pose_diff(), device=config.device, dtype=config.dtype)
                        dataset.cur_pose_torch = torch.tensor(pgm.cur_pose, device=config.device, dtype=config.dtype)
                        neural_points.adjust_map(pose_diff_torch) # transform neural points (position and orientation) along with associated frame poses # time consuming part
                        neural_points.recreate_hash(dataset.cur_pose_torch[:3,3], None, (not config.pgo_merge_map), config.rehash_with_time, frame_id) # recreate hash from current time
                        mapper.transform_data_pool(pose_diff_torch) # transform global pool
                        dataset.update_poses_after_pgo(pgm.pgo_poses)
                        pgm.last_loop_idx = frame_id
                        pgm.min_loop_idx = min(pgm.min_loop_idx, loop_id)
                        loop_reg_failed_count = 0
                        if config.o3d_vis_on:
                            o3d_vis.before_pgo = False
                    else:
                        if not config.silence:
                            print("[bold red]Registration failed, reject the loop candidate [/bold red]")
                        neural_points.recreate_hash(dataset.cur_pose_torch[:3,3], None, True, True, frame_id) # if failed, you need to reset the local map back to current frame
                        loop_reg_failed_count += 1
                        if config.o3d_vis_on and o3d_vis.debug_mode > 1:
                            o3d_vis.stop()

        T4 = get_time()
        
        # IV: Mapping and bundle adjustment
        # if lose track, we will not update the map and data pool (don't let the wrong pose to corrupt the map)
        # if the robot stop, also don't process this frame, since there's no new oberservations
        if mapping_on and (frame_id < 5 or (not dataset.lose_track and not dataset.stop_status)):
            mapper.process_frame(dataset.cur_point_cloud_torch, dataset.cur_sem_labels_torch,
                                dataset.cur_pose_torch, frame_id, (config.dynamic_filter_on and frame_id > 0))
        else:
            mapper.determine_used_pose()
            neural_points.reset_local_map(dataset.cur_pose_torch[:3,3], None, frame_id) # not efficient for large map
                                    
        T5 = get_time()

        if mapping_on:
            # for the first frame, we need more iterations to do the initialization (warm-up)
            cur_iter_num = config.iters * config.init_iter_ratio if frame_id == 0 else config.iters
            if dataset.stop_status:
                cur_iter_num = max(1, cur_iter_num-10)
            if frame_id == config.freeze_after_frame: # freeze the decoder after certain frame 
                freeze_decoders(geo_mlp, sem_mlp, color_mlp, config)
                neural_points.compute_feature_principle_components(down_rate = 17) # prime number

            # conduct local bundle adjustment (with lower frequency)
            if config.track_on and config.ba_freq_frame > 0 and (frame_id+1) % config.ba_freq_frame == 0:
                mapper.bundle_adjustment(config.ba_iters, config.ba_frame)
            
            # mapping with fixed poses (every frame)
            if frame_id % config.mapping_freq_frame == 0:
                mapper.mapping(cur_iter_num)
            
        T6 = get_time()
        
        if not config.silence:
            print("time for frame reading          (ms):", (T1-T0)*1e3)
            print("time for frame preprocessing    (ms):", (T2-T1)*1e3)
            if config.track_on:
                print("time for odometry               (ms):", (T3-T2)*1e3)
            if config.pgo_on:
                print("time for loop detection and PGO (ms):", (T4-T3)*1e3)
            print("time for mapping preparation    (ms):", (T5-T4)*1e3)
            print("time for mapping                (ms):", (T6-T5)*1e3)

        # regular saving logs
        if config.log_freq_frame > 0 and (frame_id+1) % config.log_freq_frame == 0:
            dataset.write_results_log()

        # V: Mesh reconstruction and visualization
        cur_mesh = None
        if config.o3d_vis_on: # if visualizer is off, there's no need to reconstruct the mesh

            o3d_vis.cur_frame_id = frame_id # frame id in the data folder
            dataset.update_o3d_map()
            if config.track_on and frame_id > 0 and (not o3d_vis.vis_pc_color) and (weight_pc_o3d is not None): 
                dataset.cur_frame_o3d = weight_pc_o3d

            T7 = get_time()

            if frame_id == last_frame:
                o3d_vis.vis_global = True
                o3d_vis.ego_view = False
                mapper.free_pool()

            neural_pcd = None
            if o3d_vis.render_neural_points or (frame_id == last_frame): # last frame also vis
                neural_pcd = neural_points.get_neural_points_o3d(query_global=o3d_vis.vis_global, color_mode=o3d_vis.neural_points_vis_mode, random_down_ratio=1) # select from geo_feature, ts and certainty

            # reconstruction by marching cubes
            if config.mesh_freq_frame > 0:
                if o3d_vis.render_mesh and (frame_id == 0 or frame_id == last_frame or (frame_id+1) % config.mesh_freq_frame == 0 or pgm.last_loop_idx == frame_id):              
                    # update map bbx
                    global_neural_pcd_down = neural_points.get_neural_points_o3d(query_global=True, random_down_ratio=23) # prime number
                    dataset.map_bbx = global_neural_pcd_down.get_axis_aligned_bounding_box()
                    
                    mesh_path = None # no need to save the mesh
                    if frame_id == last_frame and config.save_mesh: # save the mesh at the last frame
                        mc_cm_str = str(round(o3d_vis.mc_res_m*1e2))
                        mesh_path = os.path.join(run_path, "mesh", 'mesh_frame_' + str(frame_id) + "_" + mc_cm_str + "cm.ply")
                    
                    # figure out how to do it efficiently
                    if not o3d_vis.vis_global: # only build the local mesh
                        # cur_mesh = mesher.recon_aabb_mesh(dataset.cur_bbx, o3d_vis.mc_res_m, mesh_path, True, config.semantic_on, config.color_on, filter_isolated_mesh=True, mesh_min_nn=o3d_vis.mesh_min_nn)
                        chunks_aabb = split_chunks(global_neural_pcd_down, dataset.cur_bbx, o3d_vis.mc_res_m * 100) # reconstruct in chunks
                        cur_mesh = mesher.recon_aabb_collections_mesh(chunks_aabb, o3d_vis.mc_res_m, mesh_path, True, config.semantic_on, config.color_on, filter_isolated_mesh=True, mesh_min_nn=o3d_vis.mesh_min_nn)    
                    else:
                        aabb = global_neural_pcd_down.get_axis_aligned_bounding_box()
                        chunks_aabb = split_chunks(global_neural_pcd_down, aabb, o3d_vis.mc_res_m * 300) # reconstruct in chunks
                        cur_mesh = mesher.recon_aabb_collections_mesh(chunks_aabb, o3d_vis.mc_res_m, mesh_path, False, config.semantic_on, config.color_on, filter_isolated_mesh=True, mesh_min_nn=o3d_vis.mesh_min_nn)    
            cur_sdf_slice = None
            if config.sdfslice_freq_frame > 0:
                if o3d_vis.render_sdf and (frame_id == 0 or frame_id == last_frame or (frame_id + 1) % config.sdfslice_freq_frame == 0):
                    slice_res_m = config.voxel_size_m * 0.2
                    sdf_bound = config.surface_sample_range_m * 4.0
                    query_sdf_locally = True
                    if o3d_vis.vis_global:
                        cur_sdf_slice_h = mesher.generate_bbx_sdf_hor_slice(dataset.map_bbx, dataset.cur_pose_ref[2,3] + o3d_vis.sdf_slice_height, slice_res_m, False, -sdf_bound, sdf_bound) # horizontal slice
                    else:
                        cur_sdf_slice_h = mesher.generate_bbx_sdf_hor_slice(dataset.cur_bbx, dataset.cur_pose_ref[2,3] + o3d_vis.sdf_slice_height, slice_res_m, query_sdf_locally, -sdf_bound, sdf_bound) # horizontal slice (local)
                    if config.vis_sdf_slice_v:
                        cur_sdf_slice_v = mesher.generate_bbx_sdf_ver_slice(dataset.cur_bbx, dataset.cur_pose_ref[0,3], slice_res_m, query_sdf_locally, -sdf_bound, sdf_bound) # vertical slice (local)
                        cur_sdf_slice = cur_sdf_slice_h + cur_sdf_slice_v
                    else:
                        cur_sdf_slice = cur_sdf_slice_h
                                
            pool_pcd = mapper.get_data_pool_o3d(down_rate=23, only_cur_data=o3d_vis.vis_only_cur_samples) if o3d_vis.render_data_pool else None # down rate should be a prime number
            odom_poses, gt_poses, pgo_poses = dataset.get_poses_np_for_vis()
            loop_edges = pgm.loop_edges_vis if config.pgo_on else None
            o3d_vis.update_traj(dataset.cur_pose_ref, odom_poses, gt_poses, pgo_poses, loop_edges)
            o3d_vis.update(dataset.cur_frame_o3d, dataset.cur_pose_ref, cur_sdf_slice, cur_mesh, neural_pcd, pool_pcd)

            if config.rerun_vis_on:
                if neural_pcd is not None:
                    rr.log("world/neural_points", rr.Points3D(neural_pcd.points, colors=neural_pcd.colors, radii=0.05))
                if dataset.cur_frame_o3d is not None:
                    rr.log("world/input_scan", rr.Points3D(dataset.cur_frame_o3d.points, colors=dataset.cur_frame_o3d.colors, radii=0.03))
                if cur_mesh is not None:
                    rr.log("world/mesh_map", rr.Mesh3D(vertex_positions=cur_mesh.vertices, triangle_indices=cur_mesh.triangles, vertex_normals=cur_mesh.vertex_normals, vertex_colors=cur_mesh.vertex_colors))
            
            T8 = get_time()

            if not config.silence:
                print("time for o3d update             (ms):", (T7-T6)*1e3)
                print("time for visualization          (ms):", (T8-T7)*1e3)

        cur_frame_process_time = np.array([T2-T1, T3-T2, T5-T4, T6-T5, T4-T3]) # loop & pgo in the end, visualization and I/O time excluded
        dataset.time_table.append(cur_frame_process_time) # in s

        if config.wandb_vis_on:
            wandb_log_content = {'frame': frame_id, 'timing(s)/preprocess': T2-T1, 'timing(s)/tracking': T3-T2, 'timing(s)/pgo': T4-T3, 'timing(s)/mapping': T6-T4} 
            wandb.log(wandb_log_content)
        
        dataset.processed_frame += 1
    
    # VI. Save results
    pose_eval_results = dataset.write_results()
    if config.pgo_on and pgm.pgo_count>0:
        print("# Loop corrected: ", pgm.pgo_count)
        pgm.write_g2o(os.path.join(run_path, "final_pose_graph.g2o"))
        pgm.write_loops(os.path.join(run_path, "loop_log.txt"))
        if config.o3d_vis_on:
            pgm.plot_loops(os.path.join(run_path, "loop_plot.png"), vis_now=False)  
    
    neural_points.prune_map(config.max_prune_certainty, 0, True) # prune uncertain points for the final output    
    neural_points.recreate_hash(None, None, False, False) # merge the final neural point map 

    neural_pcd = neural_points.get_neural_points_o3d(query_global=True, color_mode = 0)
    if config.save_map:
        o3d.io.write_point_cloud(os.path.join(run_path, "map", "neural_points.ply"), neural_pcd) # write the neural point cloud
    
    output_mc_res_m = config.mc_res_m*0.6
    mc_cm_str = str(round(output_mc_res_m*1e2))
    if config.save_mesh and cur_mesh is None:    
        chunks_aabb = split_chunks(neural_pcd, neural_pcd.get_axis_aligned_bounding_box(), output_mc_res_m * 300) # reconstruct in chunks
        mesh_path = os.path.join(run_path, "mesh", "mesh_" + mc_cm_str + "cm.ply")
        cur_mesh = mesher.recon_aabb_collections_mesh(chunks_aabb, output_mc_res_m, mesh_path, False, config.semantic_on, config.color_on, filter_isolated_mesh=True, mesh_min_nn=config.mesh_min_nn)
    neural_points.clear_temp() # clear temp data for output
    if config.save_map:
        save_implicit_map(run_path, neural_points, geo_mlp, color_mlp, sem_mlp)
        # lcd_npmc.save_context_dict(mapper.used_poses, run_path)
        print("Use 'python vis_pin_map.py {} {} neural_points.ply mesh_out_{}cm.ply' to inspect the map offline.".format(run_path, output_mc_res_m, mc_cm_str))

    if config.save_merged_pc:
        dataset.write_merged_point_cloud() # replay: save merged point cloud map

    if config.o3d_vis_on:
        while True:
            o3d_vis.ego_view = False
            o3d_vis.update(dataset.cur_frame_o3d, dataset.cur_pose_ref, cur_sdf_slice, cur_mesh, neural_pcd, pool_pcd)
            odom_poses, gt_poses, pgo_poses = dataset.get_poses_np_for_vis()
            o3d_vis.update_traj(dataset.cur_pose_ref, odom_poses, gt_poses, pgo_poses, loop_edges)
    


    return pose_eval_results

if __name__ == "__main__":
    app()
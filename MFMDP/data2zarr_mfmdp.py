import argparse
import json
import os
import shutil

import numpy as np
import zarr
from PIL import Image
from tqdm import tqdm
import copy
import open3d as o3d
import pickle

from sklearn.decomposition import PCA

def main():
    parser = argparse.ArgumentParser(
        description="Convert data to zarr format for diffusion policy"
    )
    parser.add_argument(
        "task_name",
        type=str,
        default="Fold_Dress",
        help="The name of the task (e.g., Fold_Dress)",
    )
    parser.add_argument(
        "stage_str",
        type=str,
        default="all",
        help="The index of current stage (e.g., 1)",
    )
    parser.add_argument(
        "train_data_num",
        type=int,
        default=200,
        help="Number of data to process (e.g., 200)",
    )
    parser.add_argument(
        "active_object_n_components",
        type=int,
        help="active_object_n_components",
    )
    parser.add_argument(
        "passive_object_n_components",
        type=int,
        help="passive_object_n_components",
    )
    args = parser.parse_args()
    
    task_name = args.task_name
    stage_str = args.stage_str
    train_data_num = args.train_data_num
    active_object_n_components = args.active_object_n_components
    passive_object_n_components = args.passive_object_n_components
    
    current_abs_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_abs_dir)
    print("Project Root Dir : ", parent_dir)
    
    load_dir = parent_dir + f"/Data/{task_name}/train_data"
    print("Meta Data Load Dir : ", load_dir)
    
    save_dir = f"data/{task_name}_stage_{stage_str}_{train_data_num}.zarr"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    print("Save Dir : ", save_dir)
    
    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")
    
    # ZARR datasets will be created dynamically during the first batch write
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    
    # Batch processing settings
    batch_size = 100
    environment_point_cloud_arrays = []
    active_object_point_cloud_arrays = []
    passive_object_point_cloud_arrays = []
    active_object_point_feature_arrays = []
    passive_object_point_feature_arrays = []
    active_object_point_feature_arrays_topca = []
    passive_object_point_feature_arrays_topca = []

    state_arrays = []
    action_arrays = []
    episode_ends_arrays = []
    total_count = 0
    current_batch = 0

    active_object_pca = PCA(n_components=active_object_n_components) if active_object_n_components > 0 else None
    passive_object_pca = PCA(n_components=passive_object_n_components) if passive_object_n_components > 0 else None

    for current_ep in tqdm(range(train_data_num), desc=f"Processing {train_data_num} MetaData"):
        data = np.load(load_dir + f'/data_{current_ep}.npz', allow_pickle=True)
        meta_data = data[f'stage_{stage_str}']
        data_length = len(meta_data)
        for i in range(data_length-1):
            # environment point cloud
            assert meta_data[i]['env_point_cloud'].shape == (2048, 6)
            environment_point_cloud_arrays.append(meta_data[i]['env_point_cloud'])
            # active_object point cloud
            # assert meta_data[i]['active_object_point_cloud'].shape == (2048, 3)
            active_object_point_cloud_arrays.append(meta_data[i]['active_object_point_cloud'])
            # passive_object point cloud
            # assert meta_data[i]['passive_object_point_cloud'].shape == (2048, 3)
            passive_object_point_cloud_arrays.append(meta_data[i]['passive_object_point_cloud'])
            # active_object_point_feature
            active_object_point_feature_arrays.append(meta_data[i]['active_object_point_feature'])
            # passive_object_point_feature
            passive_object_point_feature_arrays.append(meta_data[i]['passive_object_point_feature'])
            # state and action
            state_arrays.append(meta_data[i]['joint_state'])
            action_arrays.append(meta_data[i+1]['joint_state'])
            total_count += 1
        episode_ends_arrays.append(copy.deepcopy(total_count))
        assert (meta_data[-1]['active_object_point_feature'] == meta_data[-2]['active_object_point_feature']).all()
        active_object_point_feature_arrays_topca.append(meta_data[-1]['active_object_point_feature'])
        passive_object_point_feature_arrays_topca.append(meta_data[-1]['passive_object_point_feature'])

        # Write to ZARR if batch is full or if this is the last episode
        if (current_ep + 1) % batch_size == 0 or (current_ep + 1) == train_data_num:
            active_object_all = np.concatenate(active_object_point_feature_arrays_topca, axis=0)
            passive_object_all = np.concatenate(passive_object_point_feature_arrays_topca, axis=0)
            print("active_object_all shape : ", active_object_all.shape)
            print("passive_object_all shape : ", passive_object_all.shape)

            if active_object_pca is not None:
                active_object_pca.fit(active_object_all)
                active_object_point_feature_arrays = [active_object_pca.transform(feat) for feat in active_object_point_feature_arrays]
            if passive_object_pca is not None:
                passive_object_pca.fit(passive_object_all)
                passive_object_point_feature_arrays = [passive_object_pca.transform(feat) for feat in passive_object_point_feature_arrays]


            # Convert arrays to NumPy
            environment_point_cloud_arrays = np.array(environment_point_cloud_arrays)
            active_object_point_cloud_arrays = np.array(active_object_point_cloud_arrays)
            passive_object_point_cloud_arrays = np.array(passive_object_point_cloud_arrays)
            active_object_point_feature_arrays = np.array(active_object_point_feature_arrays)
            passive_object_point_feature_arrays = np.array(passive_object_point_feature_arrays)
            action_arrays = np.array(action_arrays)
            state_arrays = np.array(state_arrays)
            episode_ends_arrays = np.array(episode_ends_arrays)

            print("environment_point_cloud_arrays shape : ", environment_point_cloud_arrays.shape)
            print("active_object_point_cloud_arrays shape : ", active_object_point_cloud_arrays.shape)
            print("passive_object_point_cloud_arrays shape : ", passive_object_point_cloud_arrays.shape)
            print("active_object_point_feature_arrays shape : ", active_object_point_feature_arrays.shape)
            print("passive_object_point_feature_arrays shape : ", passive_object_point_feature_arrays.shape)
            print("state_arrays shape : ", state_arrays.shape)
            print("action_arrays shape : ", action_arrays.shape)
            
            # Create datasets dynamically during the first write
            if current_batch == 0:
                zarr_data.create_dataset(
                    "environment_point_cloud",
                    shape=(0, *environment_point_cloud_arrays.shape[1:]),
                    chunks=(batch_size, *environment_point_cloud_arrays.shape[1:]),
                    dtype=environment_point_cloud_arrays.dtype,
                    compressor=compressor,
                    overwrite=True,
                )
                zarr_data.create_dataset(
                    "active_object_point_cloud",
                    shape=(0, *active_object_point_cloud_arrays.shape[1:]),
                    chunks=(batch_size, *active_object_point_cloud_arrays.shape[1:]),
                    dtype=active_object_point_cloud_arrays.dtype,
                    compressor=compressor,
                    overwrite=True,
                )
                zarr_data.create_dataset(
                    "passive_object_point_cloud",
                    shape=(0, *passive_object_point_cloud_arrays.shape[1:]),
                    chunks=(batch_size, *passive_object_point_cloud_arrays.shape[1:]),
                    dtype=passive_object_point_cloud_arrays.dtype,
                    compressor=compressor,
                    overwrite=True,
                )
                zarr_data.create_dataset(
                    "active_object_point_feature",
                    shape=(0, *active_object_point_feature_arrays.shape[1:]),
                    chunks=(batch_size, *active_object_point_feature_arrays.shape[1:]),
                    dtype=active_object_point_feature_arrays.dtype,
                    compressor=compressor,
                    overwrite=True,
                )
                zarr_data.create_dataset(
                    "passive_object_point_feature",
                    shape=(0, *passive_object_point_feature_arrays.shape[1:]),
                    chunks=(batch_size, *passive_object_point_feature_arrays.shape[1:]),
                    dtype=passive_object_point_feature_arrays.dtype,
                    compressor=compressor,
                    overwrite=True,
                )
                zarr_data.create_dataset(
                    "state",
                    shape=(0, state_arrays.shape[1]),
                    chunks=(batch_size, state_arrays.shape[1]),
                    dtype="float32",
                    compressor=compressor,
                    overwrite=True,
                )
                zarr_data.create_dataset(
                    "action",
                    shape=(0, action_arrays.shape[1]),
                    chunks=(batch_size, action_arrays.shape[1]),
                    dtype="float32",
                    compressor=compressor,
                    overwrite=True,
                )
                zarr_meta.create_dataset(
                    "episode_ends",
                    shape=(0,),
                    chunks=(batch_size,),
                    dtype="int64",
                    compressor=compressor,
                    overwrite=True,
                )

            # active_object_point_feature_arrays = active_object_pca.transform(active_object_point_feature_arrays)
            print("After PCA, active_object_point_feature_arrays shape : ", active_object_point_feature_arrays.shape)
            print("After PCA, passive_object_point_feature_arrays shape : ", passive_object_point_feature_arrays.shape)

            # Append data to ZARR datasets
            zarr_data["environment_point_cloud"].append(environment_point_cloud_arrays)
            zarr_data["active_object_point_cloud"].append(active_object_point_cloud_arrays)
            zarr_data["passive_object_point_cloud"].append(passive_object_point_cloud_arrays)
            zarr_data["active_object_point_feature"].append(active_object_point_feature_arrays)
            zarr_data["passive_object_point_feature"].append(passive_object_point_feature_arrays)
            zarr_data["state"].append(state_arrays)
            zarr_data["action"].append(action_arrays)
            zarr_meta["episode_ends"].append(episode_ends_arrays)
            
            print(
                f"Batch {current_batch + 1} written with {len(active_object_point_cloud_arrays)} samples."
            )

            # Clear arrays for next batch
            environment_point_cloud_arrays = []
            active_object_point_cloud_arrays = []
            passive_object_point_cloud_arrays = []
            active_object_point_feature_arrays = []
            passive_object_point_feature_arrays = []
            action_arrays = []
            state_arrays = []
            episode_ends_arrays = []
            current_batch += 1

    file_path = save_dir
    if active_object_pca is not None:
        with open(file_path + "/active_object_pca.pkl", "wb") as f:
            pickle.dump(active_object_pca, f)
    if passive_object_pca is not None:
        with open(file_path + "/passive_object_pca.pkl", "wb") as f:
            pickle.dump(passive_object_pca, f)

if __name__ == "__main__":
    main()
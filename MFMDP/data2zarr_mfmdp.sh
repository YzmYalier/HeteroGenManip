# Examples:
# bash data2zarr_mfmdp.sh Hang_Coat 1 100

# 'task_name' e.g. Hang_Coat, Hang_Tops, Wear_Scarf, etc.
# 'stage_index' e.g. 1, 2, 3, etc.
# 'train_data_num' means number of training data, e.g. 100, 200, 300, etc.



task_name=${1}
stage_index=${2}
train_data_num=${3}
active_object_n_components=${4}
passive_object_n_components=${5}

# python_path=~/isaacsim_4.5.0/python.sh

# export ISAAC_PATH=$python_path

python data2zarr_mfmdp.py ${task_name} ${stage_index} ${train_data_num} ${active_object_n_components} ${passive_object_n_components}
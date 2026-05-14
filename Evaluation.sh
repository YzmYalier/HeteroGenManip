# Examples:
# bash Validation.sh Hang_Coat 100 100

# 'task_name' e.g. Hang_Coat, Hang_Tops, Wear_Scarf, etc.
# 'validation_num' The episodes number you need to validate. e.g. 50, 100, etc.
# 'training_data_num' The expert data number used for training policy. e.g. 100, 200, 300, etc.

# when you run this script, you need to input the checkpoint number parameters according to task.
# for example:
# As for Fold Dress which has three stages, 
# you need to input the stage_1_checkpoint_num, stage_2_checkpoint_num, stage_3_checkpoint_num.

#!/bin/bash

# 获取参数
task_name=$1
validation_num=$2
training_data_num=$3
checkpoint_num=$4
stage_str=$5
type=$6

seed_file="Env_Validation/${task_name}_seeds.txt"

mapfile -t seeds < "$seed_file"

isaac_path=~/isaacsim_4.5.0/python.sh

export ISAAC_PATH=$isaac_path

# 创建目录和文件
base_dir="Data/${task_name}_Validation_${type}_${stage_str}_${checkpoint_num}_${training_data_num}"
mkdir -p "${base_dir}/final_state_pic"
mkdir -p "${base_dir}/video"
touch "${base_dir}/validation_log.txt"

# 获取当前数据数量
current_num=$(ls "${base_dir}/final_state_pic" | wc -l)

# 进度条函数（写 stderr）
print_progress() {
    local current=$1
    local total=$2
    local task=$3
    local width=50
    local percent=$((100 * current / total))
    local filled=$((width * current / total))
    local empty=$((width - filled))

    local bar=$(printf "%0.s█" $(seq 1 $filled))
    bar+=$(printf "%0.s " $(seq 1 $empty))

    # 输出任务名和进度条
    printf "\rTask: %-20s |%s| %3d%% (%d/%d)" "$task" "$bar" "$percent" "$current" "$total" >&2
}

# 每个seed最多执行三次
cnt=0

# 数据采集循环
while [ "$current_num" -lt "$validation_num" ]; do

    seed=${seeds[$current_num]}

    # 打印进度条
    print_progress "$current_num" "$validation_num" "$task_name"

    # 执行 isaac 命令（stdout 保留）
    $ISAAC_PATH Env_Validation/${task_name}.py \
        --env_random_flag True \
        --garment_random_flag True \
        --record_video_flag True \
        --validation_flag True \
        --training_data_num "$training_data_num" \
        --checkpoint_num "$checkpoint_num" \
        --stage_str "$stage_str" \
        --policy_name "$type" \
        --seed "$seed" \
        # > /dev/null 2>&1

    # 更新数量
    # current_num=$(ls "${base_dir}/final_state_pic" | wc -l)

    # current_num=$((current_num + 1))

    if [ "$current_num" -eq "$(ls "${base_dir}/final_state_pic" | wc -l)" ]; then
        cnt=$((cnt + 1))
    else
        cnt=0
        current_num=$(ls "${base_dir}/final_state_pic" | wc -l)
    fi

    if [ "$cnt" -ge 3 ]; then
        echo -e "\n\033[31mWarning: Seed $seed may cause issues. Skipping to next seed.\033[0m" >&2
        cnt=0
        current_num=$((current_num + 1))
    fi

    sleep 5
done

# 打印进度条
print_progress "$current_num" "$validation_num" "$task_name"

# 完成后换行
echo >&2

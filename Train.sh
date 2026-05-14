task_name=$1
stage_str=remain
train_data_num=$2

pca_components_0_0_tasks=("Stack_Tops")
pca_components_0_5_tasks=("Hang_Coat" "Hang_Tops" "Store_Tops" "Store_Tops" "Wear_Bowlhat")
pca_components_5_5_tasks=("Store_Mug")

if [[ " ${pca_components_0_0_tasks[@]} " == *" ${task_name} "* ]]; then
    echo "task_name: ${task_name}, active_object_n_components: 0, passive_object_n_components: 0"
    active_object_n_components=0
    passive_object_n_components=0
elif [[ " ${pca_components_0_5_tasks[@]} " == *" ${task_name} "* ]]; then
    echo "task_name: ${task_name}, active_object_n_components: 0, passive_object_n_components: 5"
    active_object_n_components=0
    passive_object_n_components=5
elif [[ " ${pca_components_5_5_tasks[@]} " == *" ${task_name} "* ]]; then
    echo "task_name: ${task_name}, active_object_n_components: 5, passive_object_n_components: 5"
    active_object_n_components=5
    passive_object_n_components=5
fi

cd MFMDP
bash data2zarr_mfmdp.sh ${task_name} ${stage_str} ${train_data_num} ${active_object_n_components} ${passive_object_n_components}
# bash train.sh ${task_name}_stage_${stage_str} ${train_data_num} 42 0 False
cd ..
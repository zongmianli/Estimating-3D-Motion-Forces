#!/bin/bash

# Configure
from_stage=1 # run from stage number
until_stage=7 # run until stage number
visualize=1
max_num_it=20 # maximum number of iterations
fun_tolerance=1e-8 # set function tolerance
print_summary=1 # print Ceres summary
save_results=1

# Execution settings
out_dir=$(pwd)/validation/results/test
experiment=debug
save_name=trial1
action=kv
video_name=kv01_PKFC
var_param_name=seq_len
var_param_value=10
params_path=validation/parameters/params_init.txt
virtual_object=1
data_dir=$(pwd)/data/Parkour-dataset

# Create useful paths
repo_dir=$(pwd)
exec_path="${repo_dir}/estimate.py"
video_path="${data_dir}/videos/${video_name}.mp4"
path_openpose="${data_dir}/Openpose-video/Openpose-video-demo.pkl"
path_contact_states="${data_dir}/contact-recognizer/contact-recognizer-demo.pkl"
path_object_2d_endpoints="${data_dir}/object_2d_endpoints/endpoints/${video_name}_endpoints.txt"
path_hmr="${data_dir}/HMR/HMR-demo.pkl"
person_models_dir="${repo_dir}/person_models"
object_models_dir="${repo_dir}/object_models"
evaluator_dir="${data_dir}/lib"
gt_dir="${data_dir}/gt_motion_forces"

exec_command=" \
    ${exec_path} ${experiment} ${action} ${video_path} \
    --path-person-2d-poses=${path_openpose} --path-contact-states=${path_contact_states} \
    --path-object-2d-keypoints=${path_object_2d_endpoints} --path-init-person-3d-poses=${path_hmr} \
    --person-models-dir=${person_models_dir} --object-models-dir=${object_models_dir} \
    --gt-dir=${gt_dir} --evaluator-dir=${evaluator_dir} --out-dir=${out_dir} \
    --save-name=${save_name} \
    --from-stage=${from_stage} --until-stage=${until_stage} \
    --max-num-it=${max_num_it} --fun-tolerance=${fun_tolerance} \
    --print-summary=${print_summary}"

# Append the parameters to the command line
params=($(cat ${params_path})) # default parameter values
num_params=$(( ${#params[@]} / 2 ))
for i in $(seq 0 $(( $num_params - 1 )))
do
    param_name=${params[$(( $i * 2 ))]}
    param_value=${params[$(( $i * 2 + 1 ))]}
    # Update the value of the variable parameter
    if [ $param_name = $var_param_name ]
    then
        param_value=${var_param_value}
    fi
    exec_command="${exec_command} --${param_name}=${param_value}"
done

# Other options
if [ $virtual_object -eq 1 ]
then
    exec_command="${exec_command} --virtual-object"
fi

if [ $save_results -eq 1 ]
then
    exec_command="${exec_command} --save-results"
fi

if [ $visualize -eq 1 ]
then
    exec_command="${exec_command} --visualize"
fi

#echo python ${exec_command}
python ${exec_command}

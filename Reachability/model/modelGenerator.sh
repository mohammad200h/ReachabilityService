#!/bin/sh

input=$3/model.sdf.erb
output=$3/model_$7.sdf
graspaxisoffset=($8)
IFS=' ' read -r -a array <<< "$8"

echo $input
echo $output

if ($1 );
then
    echo "Generating Model .."
    echo "IFS ${array[@]}"
    echo "grasp_axis_offset $8"
    echo "graspaxisoffset $graspaxisoffset"
    erb path=$3 lib_path=$4 load_ws=$5 load_ws_pcd=$6  control_mode=$7 grasp_axis_offset="$8" $input > $output
fi

if ( $2 );
then
    echo "Launching Model .."
    python launch_hand_model.py
fi

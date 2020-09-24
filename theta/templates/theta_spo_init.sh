#!/usr/bin/env bash

dataset_name=$1
mv Makefile.spo_task Makefile.${dataset_name}
mv spo_task.py ${dataset_name}.py
mv spo_params.py ${dataset_name}_params.py
mv run_spo_task.py run_${dataset_name}.py

sed -i -e "s/spo_task/${dataset_name}/g" Makefile.${dataset_name}
sed -i -e "s/spo_task/${dataset_name}/g" run_${dataset_name}.py
sed -i -e "s/spo_task/${dataset_name}/g" ${dataset_name}_params.py
sed -i -e "s/from spo_params/from ${dataset_name}_params/g" run_${dataset_name}.py


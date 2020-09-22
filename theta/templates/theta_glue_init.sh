#!/usr/bin/env bash

dataset_name=$1
mv Makefile.glue_task Makefile.${dataset_name}
mv glue_task.py ${dataset_name}.py
mv glue_params.py ${dataset_name}_params.py
mv run_glue_task.py run_${dataset_name}.py

sed -i -e "s/glue_task/${dataset_name}/g" Makefile.${dataset_name}
sed -i -e "s/glue_task/${dataset_name}/g" run_${dataset_name}.py
sed -i -e "s/glue_task/${dataset_name}/g" ${dataset_name}_params.py
sed -i -e "s/from glue_params/from ${dataset_name}_params/g" run_${dataset_name}.py


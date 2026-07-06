#!/bin/bash
cd /home/jingliu/workspece/AdaShield
source /home/jingliu/miniconda3/etc/profile.d/conda.sh
conda activate venv_neurostrike
CUDA_VISIBLE_DEVICES=1 bash master_run_all_models.sh --resume >> runs/latest/master.log 2>&1

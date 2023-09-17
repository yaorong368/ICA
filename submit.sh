#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=32g
#SBATCH -p qTRDGPUH
#SBATCH -t 120:00:00
#SBATCH -D /data/users2/yxiao11/model/ICA
#SBATCH -J ica100
#SBATCH -e /data/users2/yxiao11/model/ICA/outputs/error100.err
#SBATCH -o /data/users2/yxiao11/model/ICA/outputs/print100.txt
#SBATCH -A trends108c146
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yxiao11@student.gsu.edu
#SBATCH --oversubscribe
#SBATCH --gres=gpu:1

sleep 5s

source activate p37

python ./script/exp_fmri.py

sleep 10s     
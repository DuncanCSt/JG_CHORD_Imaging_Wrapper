#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=def-acliu
#SBATCH --gpus-per-node=v100l:4
#SBATCH --mem=0M                # memory (per node)
#SBATCH --time=0-01:40            # time (DD-HH:MM)

module --force purge
module use /project/rrg-kmsmith/shared/chord_env/modules/modulefiles/
module load chord/chord_pipeline/2023.06
module load cudacore/.12.2.2

cd ~/scratch/jgoodeve/cuda_dirtymap_josh/JG_Wrapper
python pyscript_3D.py

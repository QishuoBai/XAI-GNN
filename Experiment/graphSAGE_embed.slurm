#!/bin/bash

#SBATCH --job-name=graphSAGE-embed
#SBATCH --partition=64c512g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --output=taskInfos/graphSAGE-embed%j.out
#SBATCH --error=taskInfos/graphSAGE-embed%j.err

python graphSAGE-embed.py
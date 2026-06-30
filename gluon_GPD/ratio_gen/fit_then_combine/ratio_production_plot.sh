#!/bin/bash
#SBATCH -A lgt132
#SBATCH -p batch
#SBATCH -J ratio_fit
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH -t 3:00:00
#SBATCH -o ./logs/ratio_fit_andes.%j.out
#SBATCH -e ./logs/ratio_fit_andes.%j.out
export N_WORKERS="${N_WORKERS:-30}"
module load python/.3.11-anaconda3

python3 ratio_production_plot.py 

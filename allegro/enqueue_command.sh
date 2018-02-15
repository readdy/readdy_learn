#!/usr/bin/env bash
#SBATCH --job-name=rlearn
#SBATCH --workdir=/home/chrisfr/workspace/readdy_learn/allegro
#SBATCH --output=/data/scratch/chrisfr/workspace/data/rlearn/rlearn.%j.%a.out
#SBATCH --partition=small
#SBATCH --exclude=cmp245
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000M
#SBATCH --time=4:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=christoph.froehner@fu-berlin.de

echo "Running enqueue_command .."

if [[ $# -eq 0 ]]; then
  echo "Specify the command to run"
  exit 1
fi

hostname
date
echo "Calling command:"
echo $1

eval $1

echo "Done .."


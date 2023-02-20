#!/bin/bash

#SBATCH --job-name=DeepFocus
#SBATCH -nodes=3
#SBATCH --ntasks3
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=2

nodes =( $( scontrol show hostname $SLURM_NODELIST ) ) # get the list of nodes
nodes_array = ($nodes)
head_node = ${nodes_array[0]} # get the head node
head_node_ip = $( srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address ) # get the ip address of the head node

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

srun torchrun \
--nnodes 3 \
--nproc_per_node 2 \ # corresponds to the number of gpus per node --gpus-per-task 
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $head_node_ip:29500 \
train_multinode.py


#!/bin/bash
# Скрипт запуска MPI+CUDA программы на кластере
# Использование: mpisubmit.pl -p 2 --gpu 2 00:05 ./main_mpi_cuda -- 2400 3600

mpisubmit.pl -p 2 --gpu 2 00:05 ./main_mpi_cuda -- 2400 3600


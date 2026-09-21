#!/usr/bin/env bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate prometheus-vla
exec python ~/testes_g1/verifica_melhor.py "$@"

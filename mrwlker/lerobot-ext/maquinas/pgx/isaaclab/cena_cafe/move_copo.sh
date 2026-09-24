#!/usr/bin/env bash
# Move o copo na cena do cafe. Sem argumentos, mostra onde os dois estao.
#   ~/move_copo.sh 20 2        # 20 cm a frente do robo, 2 cm a esquerda
exec python3 ~/testes_g1/poe_cafe.py copo "$@"

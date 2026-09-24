#!/usr/bin/env bash
# Move o coador na cena do cafe. Sem argumentos, mostra onde os dois estao.
#   ~/move_coador.sh 20 2        # 20 cm a frente do robo, 2 cm a esquerda
exec python3 ~/testes_g1/poe_cafe.py coador "$@"

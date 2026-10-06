# Ψ0, Task 3 ("Pick bottle, turn and pour into cup")

Checkpoint real da Task 3 do [paper](https://arxiv.org/abs/2603.12263) (Fig. 6), avaliado em replay sobre os episódios gravados deles.

- Pesos: `USC-PSI-Lab/psi-model`, `psi0/real-checkpoints/task3/checkpoints/ckpt_40000` (6,25 GB).
- Dados: `USC-PSI-Lab/psi-data`, `real/Pick_bottle_and_turn_and_pour_into_cup.zip` (80 episódios, 30 Hz, câmera da cabeça 480x640).
- Entrada: imagem da cabeça, estado (mãos 14 + braços 14 + torso 4) e a instrução `g1/Pick_bottle_and_turn_and_pour_into_cup`.
- Saída: blocos de 30 ações de 36 dimensões: mãos, braços, torso rpy e altura, vx, vy, vyaw e yaw alvo (para o controlador de pernas AMO).
- Ambiente: o mesmo de `../base/README.md`.

## Servidor

```bash
CUDA_VISIBLE_DEVICES=1 serve_psi0_amo --host 0.0.0.0 --port 8014 --action_exec_horizon 30 \
  --policy psi --rtc --run-dir=$PSI0_CKPT --ckpt-step=40000
```

## Replay

`eval_rtc.py` simula o controlador RTC do servidor oficial passo a passo: replaneja a cada 15 passos com as 6 ações em fila como prefixo. `gravado` usa as ações gravadas como prefixo; `auto` usa a própria previsão. O erro conta só os passos previstos; a referência é repetir a última ação conhecida.

```bash
export PSI0_CKPT=<pasta do checkpoint> PSI0_DATA=<pasta do dataset>
python eval_rtc.py resultados 6 15 gravado 0 20 50 79
python eval_rtc.py resultados 6 15 auto 0 20 50 79
python render_video.py resultados/ep000_d6_s15_gravado.npz resultados/ep000_d6_s15_auto.npz <episode_000000.mp4> saida.mp4
```

| episódio | modo | mãos (rad) | braços (rad) | repetir a última ação |
|---|---|---|---|---|
| 0 | gravado | 0,085 | 0,031 | 0,020 |
| 20 | gravado | 0,076 | 0,027 | 0,023 |
| 50 | gravado | 0,070 | 0,024 | 0,024 |
| 79 | gravado | 0,083 | 0,030 | 0,021 |
| 0, 20, 50, 79 | auto | 0,23 a 0,27 | 0,16 a 0,31 | 0,020 a 0,024 |

Em replay o Ψ0 não bate a referência ingênua: depende do prefixo RTC e, realimentando a própria previsão sobre observações que não reagem a ela, trava numa pose. Só malha fechada decide. Números em `resultados/resultados_rtc.jsonl`.

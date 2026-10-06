# Ψ0, Task 3 ("Pick bottle, turn and pour into cup")

Checkpoint real oficial da Task 3 do paper [Ψ0: An Open Foundation Model Towards Universal Humanoid Loco-Manipulation](https://arxiv.org/abs/2603.12263) (Fig. 6), rodado no servidor de GPU do laboratório e avaliado em replay sobre os episódios gravados deles. **Ainda não rodou no Prometheus** (ver "O que falta para o robô").

## O que é o modelo

- Código: [physical-superintelligence-lab/Psi0](https://github.com/physical-superintelligence-lab/Psi0), commit `2830f93`.
- Pesos: `USC-PSI-Lab/psi-model`, pasta `psi0/real-checkpoints/task3`, `ckpt_40000` (6,25 GB): Qwen3-VL-2B (2,13 B) + action expert MM-DiT de fluxo (498 M).
- Dados: `USC-PSI-Lab/psi-data`, `real/Pick_bottle_and_turn_and_pour_into_cup.zip`: 80 episódios, 77.584 quadros a 30 Hz, câmera egocêntrica da cabeça 480x640.
- Entrada: imagem da cabeça, estado (mãos Dex3 14 + braços 14 + torso rpy e altura 4) e a instrução `g1/Pick_bottle_and_turn_and_pour_into_cup` (é o que o cliente oficial manda).
- Saída: blocos de 30 ações de 36 dimensões: mãos (14), braços (14), torso rpy e altura (4), vx, vy, vyaw e yaw alvo (4, para o controlador de pernas AMO).

## Instalação

```bash
git clone https://github.com/physical-superintelligence-lab/Psi0.git && cd Psi0
uv venv .venv-psi --python 3.11 && source .venv-psi/bin/activate
# o uv.lock do HEAD está quebrado (seção do jsonschema duplicada, psygnal ambíguo); o do commit anterior funciona
git checkout 47993ee -- uv.lock
GIT_LFS_SKIP_SMUDGE=1 uv sync --active --group serve --group viz --group psi
```

## Servidor (como no `scripts/deploy/serve_psi0-rtc.sh` oficial)

```bash
CUDA_VISIBLE_DEVICES=1 serve_psi0_amo --host 0.0.0.0 --port 8014 --action_exec_horizon 30 \
  --policy psi --rtc --run-dir=$PSI0_CKPT --ckpt-step=40000
```

Na A100 cada bloco de 30 ações leva 0,23 a 0,36 s (8 passos de fluxo), com ~8 GB de VRAM.

## Avaliação em replay

`eval_rtc.py` simula passo a passo o controlador RTC do servidor oficial (`RealTimeChunkController`): o primeiro bloco sai sem prefixo, e daí em diante replaneja a cada 15 passos com as 6 ações já em fila como prefixo (atraso d = 6). As observações são as do episódio gravado. Há dois modos:

- `gravado`: o prefixo são as 6 ações gravadas que estariam na fila.
- `auto`: o prefixo é a previsão do próprio modelo, como no robô, mas com observações que não reagem a ela.

O erro só conta os passos previstos de fato (os 6 de prefixo ficam de fora). A referência é "repetir a última ação conhecida".

```bash
export PSI0_CKPT=<pasta com argv.txt, run_config.json e checkpoints/ckpt_40000> PSI0_DATA=<pasta do dataset>
python eval_rtc.py resultados 6 15 gravado 0 20 50 79
python eval_rtc.py resultados 6 15 auto 0 20 50 79
python render_video.py resultados/ep000_d6_s15_gravado.npz resultados/ep000_d6_s15_auto.npz <episode_000000.mp4> psi0_task3_ep000.mp4
```

| episódio | modo | mãos (rad) | braços (rad) | repetir a última ação (mãos+braços) |
|---|---|---|---|---|
| 0 | gravado | 0,085 | 0,031 | 0,020 |
| 20 | gravado | 0,076 | 0,027 | 0,023 |
| 50 | gravado | 0,070 | 0,024 | 0,024 |
| 79 | gravado | 0,083 | 0,030 | 0,021 |
| 0, 20, 50, 79 | auto | 0,23 a 0,27 | 0,16 a 0,31 | 0,020 a 0,024 |

Lido honestamente: em replay o Ψ0 não bate a referência ingênua. Ele depende muito do prefixo RTC (com prefixo zerado o erro vai a 0,22 a 0,28; com o gravado cai para 0,035 a 0,06 no bloco), e realimentando a própria previsão sobre observações gravadas ele trava numa pose. Replay não consegue validar uma política assim; só malha fechada (robô ou simulação da mesma cena) decide se ela faz a tarefa. O mesmo erro sem prefixo (0,18 a 0,23) sai com o carregador de treino oficial deles, então não é pré-processamento nosso. O exemplo oficial `examples/psi0/openloop_eval_simple.py` está desatualizado (a chave `raw_images` não existe mais).

`resultados/*.npz` guarda tudo de cada episódio: ação executada a cada passo, quais passos são prefixo, blocos, ação gravada, estado, timestamp do episódio e log de cada inferência com hora de parede e latência.

## O que falta para o robô

1. **Braço esquerdo:** a Task 3 usa muito o braço e a mão esquerdos (cotovelo esquerdo com desvio de 0,38 rad nos dados), e o nosso G1 está com kp=0 no braço esquerdo.
2. **Pernas:** a tarefa anda e gira (vx até 0,35 m/s, yaw alvo até -1,7 rad). Isso precisa do controlador AMO no robô, que não está instalado.
3. **Câmera e cliente:** falta o `real/teleop/image_server/realsense_server.py` no PC do G1 (câmera da cabeça 480x640) e o cliente `real/deploy/psi-inference_rtc.py`, com o `unitree_sdk2_python` modificado deles.
4. **Cena:** garrafa, copo e bandeja como nos episódios, porque o modelo só viu essa cena.

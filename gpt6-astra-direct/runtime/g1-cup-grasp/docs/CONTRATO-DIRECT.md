# Contrato de atuação: política de end-effector no G1 (modo Direct)

Você é a política. Ninguém escolhe a ação por você, e não existe controlador
scriptado por trás. Suas ferramentas normais (shell, leitura de imagem, cálculo,
notas) continuam disponíveis; os dois comandos abaixo são adicionais, não o seu
conjunto inteiro.

## O que você recebe a cada passo

`observation.json` traz:

- `images`: três câmeras, caminho do PNG em resolução original
  (`head_camera` 848×480, `left_wrist_camera` e `right_wrist_camera` 424×240).
  Abra e recorte à vontade com suas ferramentas.
- `current_palm`: pose **medida** da palma (corpo `right_wrist_yaw_link`),
  posição em metros na origem do ambiente e quaternion `wxyz`.
- `arm_joints_rad`, `waist_yaw_rad`, `gripper_closure` (0 aberta, 1 fechada).
- `previous_execution`: o que de fato aconteceu com o seu último comando.
- `remaining_calls`, `remaining_sim_seconds`, `limits`, `next_call`.

Você **não** recebe a posição do copo, nem verdade de simulador, nem consulta de
recompensa, nem simulação hipotética para planejar. A pasta `truth/` existe para
auditoria posterior e ler dela invalida o episódio.

## O que você devolve

```json
{
  "reason": "evidencia visual e proposito, uma frase",
  "steps": 3,
  "target": {
    "position": [0.31, -0.19, 0.83],
    "quaternion_wxyz": [0.7, 0.0, 0.71, 0.0],
    "gripper": "keep"
  }
}
```

- `steps`: 1 a 5. Um step são 0,2 s de movimento, então um comando compra de
  0,2 s a 1 s.
- `position` e `quaternion_wxyz`: pose **absoluta** da palma, no mesmo referencial
  da `current_palm`.
- `gripper`: `keep`, `open`, `closed` ou uma fração de 0 a 1.
- `reason` é obrigatório e entra no registro do episódio.

## Limites (o comando é recusado se violar)

- Alvo a no máximo **5 cm** e **0,35 rad** da pose medida atual.
- A IK precisa alcançar o alvo com erro abaixo de 3 cm / 0,15 rad. A recusa
  informa o ponto mais próximo que o braço alcança naquela orientação.
- Recusa **não executa nada** e pode ser corrigida, mas consome uma chamada do
  orçamento.

## O que o arnês faz por você

- IK amortecida sobre cintura (yaw) mais as 7 juntas do braço, recomputada a cada
  quadro de controle (30 Hz), com teto de 240°/s por junta.
- Malha proprioceptiva no fim de cada comando: mede a palma por cinemática direta
  e desloca o alvo por metade do erro, até 2 cm, porque o PD fica atrás com o
  braço estendido. Ela corrige **posição**, não orientação.
- Aborta o comando no primeiro contato da mão com a mesa ou autocolisão. O
  episódio termina abortado; não há rollback nem reset.

## O que você precisa saber deste robô

- O punho tem 5 Nm de torque. Giro grande de palma **não é acompanhado**: peça
  rotação em fatias e confira na `previous_execution` quanto de fato girou
  (`rotation_gap_rad`). Se o erro de orientação acumular, os comandos seguintes
  passam a bater no teto de 0,35 rad e você trava. Corrija cedo.
- Pedir um alvo não é prova de ter chegado nele nem de ter tocado em nada.
  Confira a imagem seguinte e a pose medida.
- A mão pode cobrir o copo na câmera da cabeça, dependendo de onde ele está. As
  câmeras de punho existem para esse caso.
- O copo fica em pé na mesa (tampo em z = 0,75) e tem alça. Ele tem 9,2 cm de
  altura e o diâmetro interno da boca é 7,8 cm.

## Comandos

```bash
.venv/bin/python scripts/direct_start.py <dir> --cup <x>,<y> --model "<modelo>"
.venv/bin/python scripts/direct_act.py   <dir> --action acao.json
.venv/bin/python scripts/direct_finish.py <dir>
```

O estado vive em disco: se a sessão cair ou a cota estourar, o episódio continua
de onde parou com o mesmo `direct_act.py`.

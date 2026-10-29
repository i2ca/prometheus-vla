# Ajuste fino do Ψ0 na caneca (simulador MuJoCo)

Esta pasta gera demonstrações sintéticas do G1 pegando a caneca branca (etapa 1) e colocando-a sob o coador de pano (etapa 2), converte para LeRobot, treina o Ψ0 base em cima delas e avalia o checkpoint em malha fechada no mesmo simulador.

As demonstrações são roteirizadas: cinemática inversa no punho direito (e na cintura, quando liberada), sem modelo nenhum. É dado "mockado", gerado pelo próprio simulador, não gravado de teleoperação. Vídeos como `etapa2_v20_multicam.mp4` são renders dessas demos, não inferência de policy.

Tudo roda na cena `../cena/cena_cafe_medidas.xml`, com a mesma montagem do cliente `../base/rodar_mujoco.py` (afastamento da mesa, câmera da cabeça, pose inicial, ganhos do deploy). Demo e avaliação precisam ver o mesmo mundo, por isso o `--afasta` tem de bater entre as duas.

## Ambiente

Python com `mujoco`, `numpy`, `scipy`, `opencv-python`, `pandas`, `pyarrow` e `ffmpeg` no PATH. Renderização sem tela: `MUJOCO_GL=egl`. Treino e servidor exigem GPU e o checkout do Psi0 descrito em `../base/README.md` (o HEAD do repositório do Ψ0 quebra os pesos antigos em silêncio; use o commit de janeiro de 2026 indicado lá).

## 1. Gerar as demonstrações

```bash
cd psi0/finetune
MUJOCO_GL=egl python gera_demos.py saida/demos_v3 --n 135 --semente 0 --cintura            # etapa 1, cintura livre
MUJOCO_GL=egl python gera_demos.py saida/demos_colocar --n 30 --semente 0 --cintura --humano --colocar   # etapa 2
```

Modos, do mais simples ao mais completo:

| flags | o que muda | estado/ação |
|---|---|---|
| (nenhuma) | só braço e mão direitos, cintura parada, mesa a 0,05 m, região x 0,24 a 0,28, y -0,22 a -0,07 | 28 valores |
| `--cintura` | pitch e yaw da cintura entram na IK junto com o braço, mesa a 0,15 m, região x 0,24 a 0,30, y -0,22 a -0,04 | 31 valores |
| `--cintura --humano` | gesto contínuo com velocidade em sino, mão abrindo no caminho, punho quase reto, mesa a 0,10 m | 31 valores |
| `--cintura --humano --colocar` | etapa 2: pega, leva até o disco do coador, apoia, solta, recolhe o braço | 31 valores |

Cada episódio sai em `epNNN/` com `dados.npz` (estado e ação a 30 Hz), `qpos.npz` (configuração completa do simulador, para renderizar depois), `trace.json` e `info.json`. O `info.json` traz as métricas físicas medidas (subida da caneca, dedos em contato, inclinação, interpenetração, mão na mesa, tronco, punho, erro de apoio, toque no coador) e o campo `aceita`, que é o critério físico de aceitação. O `resumo.json` na raiz junta tudo. Episódios reprovados ficam gravados, só não são convertidos.

Posição fixa para depurar: `--fixo` (padrão da cena) ou `--pos X Y`. Tronco máximo ao apoiar: variável `TRONCO_COLOCAR` em graus (padrão 21).

## 2. Renderizar as câmeras

```bash
MUJOCO_GL=egl python renderiza_cameras.py saida/demos_colocar          # head, left_wrist e right_wrist, um processo por câmera
MUJOCO_GL=egl python renderiza_cameras.py saida/demos_colocar/ep000 --cam head
```

Gera `head.mp4`, `left_wrist.mp4` e `right_wrist.mp4` (640x480, h264) em cada episódio a partir do `qpos.npz`. O `gera_demos.py` não renderiza nada, só simula; é isto que torna a geração rápida.

Para revisar visualmente uma leva de demos (uma folha com seis momentos por episódio) existe o `scripts/folhas_demos.py` no pacote `g1-cup-grasp` do robotics-lab. O vídeo com câmera externa e painel de métricas (como `etapa2_v20_multicam.mp4`) sai do `scripts/render_psi0_ft.py` do mesmo pacote, recebendo o mp4 de saída e as pastas `epNNN` a encadear.

## 3. Converter para LeRobot v2.1

```bash
python para_lerobot.py saida/demos_colocar <raiz_dos_datasets>/dataset_colocar [--excluir ep007 ep012] [--incluir ep000 ep001]
```

Só as demos com `aceita=true` entram. Saída no molde do dataset da Task 3 do Ψ0: parquet por episódio com `states` e `action` em float32, vídeos das três câmeras, `meta/info.json`, `tasks.jsonl`, `episodes.jsonl`, `episodes_stats.jsonl` e `stats_psi0.json` (este último é o que o treino lê com `action_norm_type=bounds`). A instrução de linguagem vem do `info.json` das demos.

## 4. Treinar

Na máquina com GPU, com o checkout de janeiro do Psi0 e seu venv:

```bash
PSI0_JAN=<checkout 57b52b1> PSI0_VENV=<venv do Psi0> PSI0_PESOS=<pasta com pre/ e postpre/> DADOS=<raiz_dos_datasets> \
  bash treina.sh dataset_colocar 10000 32        # nome do dataset, passos, lote
```

`treina.sh` reproduz a receita `finetune-real-psi0.sh` do Ψ0 (flow matching, 1000 passos de difusão, bloco de 30 ações, estado e ação preenchidos até 36 dimensões, VLM congelado) passando por `sdpa.py`, que troca flash-attention por sdpa e corrige o collator para três câmeras. Checkpoints a cada 2000 passos (`CKPT=`), validação a cada 1000 (`VALIDA=`), retomada com `RESUME=<pasta do checkpoint>`.

Referência do que já foi medido. Etapa 1 com cintura (135 demos): o passo 4000 acertou 29 de 30 posições novas, contra 11 de 30 do replay cego; com 120 demos sem cintura, 28 de 30 contra 13 de 30; passos acima de 6000 pioraram. Etapa 2 (102 demos de treino e 12 de validação, lote 4, 10000 passos): o passo 10000 completou 10 de 10 colocações, 9 de 10 no critério estrito de centralização.

## 5. Servir e avaliar

Na máquina com GPU, suba o servidor do checkpoint (porta à escolha, aqui 8779):

```bash
CUDA_VISIBLE_DEVICES=<gpu> PSI0_REPO=<repo do Psi0> PSI0_CKPT=<pasta da run> PSI0_PASSO=4000 PSI0_SEM_CORPO=1 \
  PSI0_INSTR='the robot uses its right hand to pick up the white mug kept on the right side of the desk' \
  python ../task3/servidor_task3.py 8779
```

`PSI0_SEM_CORPO=1` diz ao servidor que o checkpoint usa estado de 28 ou 31 valores (sem pernas). A instrução tem de ser a mesma gravada nas demos (`info.json`, campo `instrucao`); a da etapa 2 é "the robot uses its right hand to pick up the white mug and place it under the coffee filter".

No notebook, com túnel SSH para a porta:

```bash
HOST=<alias ssh da máquina com GPU> PORTA=8779 bash avalia_ckpt.sh saida/aval_4000 --n 30 --semente 100 --cintura
```

O script abre o túnel, roda `avalia.py` (N posições da caneca sorteadas com semente diferente das demos, cada uma via `../base/rodar_mujoco.py --ft`) e depois `audita.py`. Um único episódio, à mão:

```bash
MUJOCO_GL=egl python ../base/rodar_mujoco.py --out saida/ep --ft --cintura --executa 15 --blocos 20 --afasta 0.15 \
  --porta 8779 --copo 0.27 -0.15 --yaw-copo 90
```

Critérios. `avalia.py` chama sucesso quando a caneca termina mais de 50 mm acima do tampo, com dois ou mais elos da mão em contato e inclinação máxima abaixo de 15°. `audita.py` é o verificador adversarial: reexamina os quadros e exige ainda polegar, médio e indicador na caneca em mais de 90% do tempo no ar, interpenetração abaixo de 3 mm, mão nunca na mesa e nenhum outro objeto deslocado ("sucesso estrito"). Também imprime quanto o punho acompanha a posição da caneca (`mira`), o que separa uma policy que olha da que repete um movimento fixo.

Contraprova obrigatória: `replay_cego.py` repete às cegas a demo mais central nas mesmas posições de avaliação. Se o checkpoint não for bem melhor que isso, o teste não mostra uso da câmera.

```bash
MUJOCO_GL=egl python replay_cego.py saida/demos_v3 --semente 100 --n 30 --cintura
```

## Armadilhas já encontradas

A câmera da cabeça (FOV horizontal 54°) não vê a caneca em y menor ou igual a -0,22. A Dex3 fecha no plano x-y do punho e a caneca de 7,7 cm só cabe com o polegar aberto. Com cintura parada a mesa tem de ficar a 0,05 m para a mão alcançar. Girar a caneca com a mão quebra o punho em 70 a 90°, por isso a etapa 2 não gira. Ao soltar, abrir com o polegar para fora (`SOLTA`), senão ele arrasta a caneca. Avaliar sempre contra o replay cego numa região de sorteio larga: uma região de 6x7 cm não discriminava nada.

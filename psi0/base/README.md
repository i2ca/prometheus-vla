# Ψ0 base (sem ajuste) no MuJoCo

O modelo base do Ψ0 ([paper](https://arxiv.org/abs/2603.12263), [código](https://github.com/physical-superintelligence-lab/Psi0)) é a soma de duas peças publicadas em `USC-PSI-Lab/psi-model`: o VLM pré-treinado (`psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k`, 4,3 GB) e o action expert pós-treinado no Humanoid Everyday (`psi0/postpre.1by1.pad36.2601131206.ckpt.he30k`, 2 GB). É o ponto de partida dos ajustes por tarefa (a Task 3 em `../task3` nasce dele).

Aqui ele roda em malha fechada no MuJoCo, com o G1 + Dex3 e a cena do café com os objetos nas medidas reais (`../cena`).

## 1. Baixar o modelo

```bash
mkdir -p pesos/pre pesos/postpre && cd pesos
B=https://huggingface.co/USC-PSI-Lab/psi-model/resolve/main/psi0
for f in added_tokens.json chat_template.jinja config.json generation_config.json merges.txt preprocessor_config.json \
         special_tokens_map.json tokenizer.json tokenizer_config.json video_preprocessor_config.json vocab.json model.safetensors; do
  curl -fL $B/pre.fast.1by1.2601091803.ckpt.ego200k.he30k/$f -o pre/$f
done
curl -fL $B/postpre.1by1.pad36.2601131206.ckpt.he30k/action_header.safetensors -o postpre/action_header.safetensors
```

## 2. Ambiente do Ψ0 (máquina com GPU)

```bash
git clone https://github.com/physical-superintelligence-lab/Psi0.git && cd Psi0
uv venv .venv-psi --python 3.11 && source .venv-psi/bin/activate
git checkout 47993ee -- uv.lock   # o uv.lock do HEAD está quebrado (seção duplicada); o do commit anterior funciona
GIT_LFS_SKIP_SMUDGE=1 uv sync --active --group serve --group viz --group psi
```

Sem `flash_attn` instalado o servidor usa `sdpa`.

## 3. Servidor

```bash
CUDA_VISIBLE_DEVICES=1 PSI0_REPO=<clone do Psi0> PSI0_PESOS=<pasta pesos> <clone>/.venv-psi/bin/python servidor_base.py 8777
```

Ele monta o modelo como o início do finetune deles e usa a configuração exata do pós-treino (`scripts/train/psi0/posttrain-he-psi0.sh`): estado cru (mãos Dex3 14 + braços 14), saída de 16 ações relativas normalizadas por q99 (`assets/stats/he_raw_rel_stats_combined_no_static.json`), imagem egocêntrica 240x320. Ao subir, deve imprimir `action header: 181 chaves, faltando 0, sobrando 0` e `pronto: chunk 16 dim 36 norma bounds_q99 normaliza estado False`. Escuta só em 127.0.0.1; ~8 GB de VRAM, ~0,24 s por bloco numa A100.

## 4. Cliente no MuJoCo (qualquer máquina)

Se o servidor estiver em outra máquina: `ssh -f -N -L 8777:127.0.0.1:8777 <servidor>`.

```bash
pip install mujoco==3.13.0 numpy opencv-python
MUJOCO_GL=egl python rodar_mujoco.py --out saida/ep01 --blocos 40 --copo 0.26 -0.20 --afasta 0.15 \
  --instrucao "the robot uses its right hand to pick up the white mug kept on the right side of the desk"
```

- `--afasta 0.15` afasta mesa e objetos 15 cm (a base do G1 é fixa). Deixa a câmera da cabeça parecida com a dos dados do Humanoid Everyday (mãos visíveis, objetos no centro); com 0,20 a caneca sai do alcance do braço com a cintura parada.
- Use instruções no estilo do Humanoid Everyday (frases longas de `Psi0/assets/stats/task_description_dict.json`), por exemplo `"use the right hand to grab the handle of the kettle from the base and place it on the right side"`.
- Saída: `qpos.npz` (qpos por passo, alvos, bloco, subida da caneca, dedos em contato), `trace.json` (estado, ações e latência de cada bloco com horário de parede) e `entradas/` (as imagens exatas enviadas ao modelo).

## O que vimos

Sem ajuste, nenhuma tarefa concluída. A instrução muda o comportamento (o punho difere até 27 cm entre instruções), mas ele não pega nada: com a mesa a 15 cm, na instrução da caneca o dedo médio encosta nela aos 4 s sem fechar a mão, e aos 19 s a mão esquerda derruba a jarra da base. Com a mesa colada no robô as mãos ficam perto do corpo. O próximo passo é o ajuste fino com demonstrações nesta cena.

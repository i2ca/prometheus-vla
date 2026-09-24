# GPT-6 Astra Direct no Unitree G1

O `gpt-6-astra` controla o braço direito do G1 no MuJoCo, uma decisão por vez, para pegar uma caneca e levantá-la.
Ele vê as 3 câmeras (cabeça e dois punhos, RGB-D) e a propriocepção. Não recebe a posição da caneca.
Depois de cada episódio, um crítico (o mesmo modelo, com esforço máximo) lê tudo, inclusive a verdade do
simulador, e reescreve regras gerais que entram no prompt do episódio seguinte.

Nenhum episódio foi aceito até 23/09/2026. Os melhores (`p-29` e `q-36`) cumpriram todos os critérios por vários
passos seguidos e só falharam porque o robô não declarou a tarefa concluída. Os episódios 01 a 07 rodaram sem imagens de verdade
(ver `docs/NOTA-VISAO.md`).

## Onde está cada coisa

| caminho | o que é |
|---|---|
| `runtime/g1-cup-grasp/` | tudo para rodar: cena do G1 com a Dex3 e a caneca (`scene/`), semente (`seed/`), parâmetros e scripts |
| `runtime/g1-cup-grasp/scripts/direct_env.py` | o arnês: executa cada comando, recusa o que bateria no próprio corpo, grava observação e verdade |
| `runtime/g1-cup-grasp/scripts/direct_run_omniroute.py` | o laço do episódio: manda observação ao modelo, recebe a ação, executa |
| `runtime/g1-cup-grasp/scripts/direct_reflect.py` | o crítico: lê o episódio e reescreve as lições |
| `runtime/g1-cup-grasp/scripts/direct_finish.py` | o avaliador (mesmos critérios do controlador scriptado) |
| `runtime/g1-cup-grasp/scripts/sim_finish.py` | aplica o avaliador como se o robô tivesse encerrado em cada passo |
| `runtime/g1-cup-grasp/scripts/make_narration.py`, `make_direct_story.py` | narração em português e vídeo de um episódio |
| `results/<episódio>/x0.40_y-0.20/` | um episódio: `policy-calls/` (cada decisão), `actions/` e `truth/` (verdade), `report.json`, vídeo |
| `results/direct-lessons-visao-real.json` | a memória de lições do crítico (a versão mais recente é a que entra no prompt) |

Log bruto: `raw/` guarda o pedido exato enviado ao modelo e a resposta exata, chamada por chamada (inclusive as
sondagens e prévias de cada decisão e a do crítico), tirados do log do gateway, com o resumo do raciocínio que a
OpenAI devolve. As imagens aparecem só como hash e tamanho (ficam em `obs/` na máquina onde o episódio rodou). O gateway apaga
os logs mais antigos, então do episódio 08 ao c-18 o log bruto já não existe; a partir dos próximos episódios o
runner grava o seu próprio em `raw-runner/`.

Em cada pasta de episódio: `policy-calls/call-NNN.json` tem o que o modelo viu (`what_i_see`), o motivo
(`reason`), a ação, as sondagens de profundidade e prévias, os tokens e os horários; `report.json` é o veredito;
`direct-story.mp4` é o vídeo narrado. Quadros e imagens brutas ficam só na máquina onde rodou.

## Como rodar

Precisa de um gateway OmniRoute com uma conta Codex (rota `cx/gpt-6-astra`). O modelo roda na OpenAI; a máquina
local só simula e desenha.

Requisitos medidos num episódio (arnês completo, sem contar a espera pelo modelo): pico de 2,8 GB de RAM, cerca de
1,5 núcleo de CPU e até 38% de GPU, só para desenhar as câmeras (a física roda na CPU). Qualquer GPU NVIDIA com
driver EGL serve; sem GPU, `MUJOCO_GL=osmesa` funciona, mais devagar. O tempo de um episódio é dominado pelo modelo
(30 a 70 s por decisão com esforço `max`).

Uma DGX Spark é a melhor opção para dois usos: o vídeo narrado em paralelo (`--workers N` usa um processo por
núcleo, cada um com uns 2 a 3 GB; 18 processos ficam na faixa de 40 a 50 GB) e vários episódios ao mesmo tempo.
Num notebook, use `--workers 2` ou `3`.

```bash
cd runtime/g1-cup-grasp
uv sync && uv pip install edge-tts           # MuJoCo 3.13, OpenCV, Pillow; edge-tts só para a voz
export MUJOCO_GL=egl OMNIROUTE_API_KEY=...   # chave de cliente do OmniRoute

# um episódio (termina quando o robô declara a tarefa completa; teto oculto de 200 chamadas)
.venv/bin/python scripts/direct_run_omniroute.py results/meu-ep/x0.40_y-0.20 --cup 0.40,-0.20 \
    --model cx/gpt-6-astra --effort max --lessons results/direct-lessons-visao-real.json

# crítico: lê o episódio e atualiza a memória de lições
.venv/bin/python scripts/direct_reflect.py results/meu-ep/x0.40_y-0.20 --memory results/direct-lessons-visao-real.json

# vídeo narrado (18 processos em paralelo; ~2 min para um episódio de 50 passos)
.venv/bin/python scripts/make_narration.py results/meu-ep/x0.40_y-0.20
.venv/bin/python scripts/make_direct_story.py results/meu-ep/x0.40_y-0.20 --workers 18
```


## Estado em 23/09/2026

Parado pela cota do Codex: o gateway responde 429 ("All codex accounts reached configured quota threshold"),
com reinício por volta de 29/09. O esforço `max` custou cerca de 10 vezes mais tokens que o padrão.

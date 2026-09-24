# Checkpoint: Direct com gpt-6-astra + ciclo de reflexão (23/09/2026)

## ATENÇÃO: episódios 01 a 07 do gpt-6-astra NÃO viram as imagens (descoberto 23/09 ~12:40)
O OmniRoute 3.8.50 tem um "Vision Bridge": modelo fora da lista interna de modelos com visão recebe cada imagem
trocada por uma legenda de 2 ou 3 frases feita pelo `openai/gpt-4o-mini`. O `gpt-6-astra` não estava na lista.
Prova: imagem 848x480 entrava como ~90 tokens, e o próprio modelo respondia "recebi só uma descrição".
Correção: linhas em `model_capabilities` (provider `codex`, `gpt-6-astra*`, modalities_input text+image) no
banco do gateway na spark-aff4 (backup `storage.sqlite.bak-vision-20260923`). Depois disso: 540 tokens e o modelo
aponta a caneca a ~15 px da posição real. O crítico também recebia legenda, mas diagnosticou pelos números da
verdade do simulador (texto), que chegaram certos. O `omniroute-01` (gpt-5.6-sol) viu as imagens de verdade.
Consequência: tudo que foi atribuído à "visão" do Astra nos episódios 01 a 07 é, na verdade, desempenho sobre
legendas. O vídeo `ciclo-aprendizado.mp4` (render de 23/09) narra como se fosse visão e precisa ser refeito.
Episódios com visão real: direct-astra-08-visao-real-base (sem lições) e -09-visao-real-licoes (com lições),
ambos com a sonda de profundidade (probe_depth, D435i simulada).

## Objetivo desta linha
LLM como atuador direto do G1 (pega da caneca, caso x=0,40 y=-0,20), com um crítico que lê o episódio
inteiro, inclusive a verdade do simulador, e reescreve lições genéricas para o prompt do próximo episódio.
Critério: aceito pelo `direct_finish.py`, com os mesmos critérios do controlador scriptado.

## Estado
| episódio | lições | resultado | erro dominante (verdade do simulador) |
|---|---|---|---|
| direct-omniroute-01 | nenhuma, gpt-5.6-sol-high | reprovado, lift 0 | fechou a mão no ar e "segurou" o nada por 6 chamadas |
| direct-astra-01 | nenhuma | reprovado, lift 1,9 cm (tombou) | dedo médio empurrou a caneca a 27° com a mão aberta |
| direct-astra-02-licoes | v1 específicas | interrompido na 3 | só recuava |
| direct-astra-03-licoes-genericas | v1 genéricas | abortado, autocolisão | 7 recuos, polegar no tronco |
| direct-astra-04-licoes-revisadas | v2 | reprovado, desistiu | falso alarme de tombamento (0° real) |
| direct-astra-05 | v3 | reprovado, 39 chamadas | fechou 4x no ar (legenda dizia "caneca entre os dedos") |
| direct-astra-06-sem-limite | v4, sem limite visível | interrompido na 58 | divergiu: mão subindo e se afastando |
| direct-astra-07-profundidade | v4 + probe_depth | desistiu na 10 | só legendas; percebeu que não tinha imagem |
| direct-astra-08-visao-real-base | nenhuma, visão real + profundidade cabeça | abortado, mão na mesa | foi direto, tombou a caneca |
| direct-astra-09-visao-real-licoes | v4 (era das legendas) | desistiu ("inseguro") na 20 | lições antigas deixaram medroso |
| direct-astra-10 / -11 | memória nova a partir do ep08; -11 com profundidade nos punhos | tombou | dedos tocavam a caneca com a palma a 17 cm |
| direct-astra-12-geometria-mao | + pontas dos dedos na observação | **pegou e levantou 9,3 cm, reteve 2 s**; reprovado por 40° | primeira pega real |
| direct-astra-13 e ciclo c-14..c-21 (esforço padrão) | lições evoluindo | 0/8 aceitos; c-17 e c-20 levantaram ~9,8 cm e retiveram, 42° e 29° | pose inicial com dedos colados na caneca |
| ciclo e-22..e-26 (esforço max) | idem | 0 aceitos; 3 abortos por mão na mesa | rotação da mão varria os dedos para baixo |
| p-27 | esforço max + preview_pose | abortado: ombro no tronco (na correção final) | levou à correção 8 |
| p-28 | idem | tombou (93°) | polegar empurrou a caneca na descida |
| **p-29** | idem | reprovado só por 24° na janela final; **pegou limpo (6°), levantou 19 cm, reteve** | **teria sido ACEITO se encerrasse em qualquer chamada entre a 12 e a 18** (sim_finish.py); ficou "verificando" e a caneca girou na mão |
| p-30 | lições "encerrar quando cumprir" e "manter o objetivo na pausa" | reprovado (35°, não reteve) | |
| **p-31** | idem | reprovado só por 23,5° ANTES do fecho; levantou 10,7 cm, reteve, 14,75° na janela, **encerrou certo após 2 s** | falha restante: empurrar a caneca na aproximação |
| p-32 | idem | encerrado pelo operador (19° antes do fecho) | **endireitou a caneca na mão de 28° para 8°** |
| p-33 | + lição "orientação é restrição já na aproximação" | abortado na 1: ombro no tronco | levou ao ensaio dinâmico (10) |
| p-34, p-35 | idem (p-35 com ensaio dinâmico) | encerrados pela regra (≥15° antes do fecho) | dedo médio já encostava na pose inicial |
| **q-36** | idem | encerrado pelo operador (laço de verificação) | **seria ACEITO em 13 pontos seguidos** (amostras 36 a 48): 11,4-12,6 cm, retida, 7,7° antes do fecho, 9-14,5° na janela; não declarou conclusão |
| q-37, q-38 | + relógios gripper_closed_for_s / palm_still_for_s | reprovados | empurrão na aproximação |
| q-39 | idem | **cota do Codex esgotada** na chamada 34 (HTTP 429, reset em ~150 h) | aproximação cuidadosa, 0° até ali |

## Mudanças no arnês e no runner (23/09 tarde), em ordem
1. Vision Bridge corrigido (visão real) a partir do ep08.
2. probe_depth na cabeça (ep07+) e nos punhos (ep11+), D435i simulada: ruído ~z², alcance mínimo 0,195 m em
   848x480 e 0,105 m em 424x240, semente fixa por passo e pixel. Validada contra verdade (mesa <1 mm; caneca na parede).
3. finish_episode só aceita "complete" (desde ep10); sem contagem de chamadas visível; teto oculto 200.
4. hand_geometry na observação (ep12+), desde 23/09 ~13:55 pela MALHA real da mão (antes: extrapolação do elo,
   até 3 cm otimista e pior com rotação).
5. Regra do operador: caneca com mais de 80° encerra o episódio (c-14+).
6. Várias tool calls numa resposta: sondas executadas e ação pedida de novo (e-24+). Crítico e política com
   esforço via output_config.effort (o sufixo -xhigh deixou de funcionar no gateway da Spark).
7. preview_pose (p-27+): simula o comando numa cópia do robô com a dinâmica dele e SEM contato com o ambiente;
   devolve pontas dos dedos no fim e o ponto mais baixo da mão no caminho. Validada: prevê as 3 batidas do e-22..e-24
   e erra ≤2,5 cm (para o lado seguro) em 6 ações reais do ep12. Um primeiro laço tinha bug (usava a pose de outro
   objeto) e dava previsão errada; corrigido antes do p-27.
8. Autocolisão também no ombro (roll/yaw, margem 1,2 cm) e regra "se já está dentro da margem, só passa comando
   que aumenta a folga"; e a malha de correção final não pode aproximar o braço do corpo (p-30+). Validado em 41
   ações reais: 0 divergências, precisão da palma mantida (mediana 2,4 mm).
10. Ensaio dinâmico antes de executar (p-35+): o comando inteiro (movimento, acomodação, correção) roda numa
   cópia do robô com a dinâmica dele, autocontatos ligados e ambiente desligado; se o braço encostar no corpo, o
   comando é recusado com o motivo. Pega o que a checagem cinemática não vê (no p-33 o alvo tinha 2,2 cm de folga e
   a execução real encostou o ombro). 43 ações reais: 0 divergências, 0,87 s por comando.
12. Relógios de propriocepção na observação (q-37+): gripper_closed_for_s e palm_still_for_s.
11. Regra do operador: inclinação ≥15° sem dedo na caneca (critério "antes do fecho", sem volta) encerra (p-33+).
9. `scripts/sim_finish.py <ep> <chamadas>`: aplica o avaliador como se a política tivesse encerrado em cada chamada.

## Decisões
- A tarefa continua "pegue a caneca" (Luiz, 23/09). O system prompt do runner fica como está.
- Lições: genéricas, sem nome de objeto, sem referência ao episódio, sem números; o filtro `LEAK` em
  `direct_reflect.py` barra o resto. O crítico recebe as lições ativas e devolve o conjunto revisado completo;
  a memória é o conjunto mais recente, não uma pilha que só cresce.
- Arnês: pré-checagem cinemática de autocolisão com margem de 5 cm (mão/punho) e 2,5 cm (cotovelo) contra
  o tronco e o braço esquerdo. Validada em 22 ações reais (1 recusa esperada, 0 falsos positivos). A mesa fica
  fora de propósito: ambiente só pelas câmeras.
- OmniRoute: `cx/gpt-6-astra` exige `CODEX_CLIENT_VERSION=0.156.1` em `/app/data/server.env` do container.

## Hipóteses
- Confirmado: o crítico diagnostica certo (verdade do simulador bate com o diagnóstico nas 3 reflexões).
- Confirmado: lição do tipo "pare/recue quando ambíguo" trava a política; toda lição precisa de ação seguinte.
- Aberta: sem profundidade nem calibração de câmera, a política não resolve a altura da pega. A política roda
  com 0 a 26 tokens de raciocínio (sem esforço pedido). Próximos experimentos, separados das lições:
  (a) política em `cx/gpt-6-astra-xhigh`; (b) intrínseca/extrínseca das câmeras no prompt; (c) profundidade.
- Falta: validar lições em outra posição da caneca (generalização) quando um episódio for aceito.

## Retomar (tudo roda na spark-aff4 desde 23/09 ~12h)
Estado em 23/09 ~17:55: **parado por cota**. O OmniRoute responde 429 para `codex/gpt-6-astra`: "All codex
accounts reached configured quota threshold (reset after 150h)", ou seja, volta por volta de 29-30/09. O esforço
max (milhares de tokens de raciocínio por decisão, ~US$ 1,3 por 10 chamadas em preço de tabela) consumiu a cota.
Nenhum episódio aceito. Dois episódios (p-29 e q-36) cumpriram todos os critérios por vários passos seguidos e
só falharam por não declarar a conclusão; o p-31 encerrou certo mas empurrou a caneca antes do fecho.
Falha restante: (1) empurrar a caneca na aproximação (pose inicial com o dedo médio quase encostando);
(2) encerrar quando cumprido. Os relógios (item 12) atacam (2) e só rodaram em q-37..q-39.
```bash
ssh fercout@10.9.8.66                         # senha no cofre: ~/.Luiz/lewis-memory/i2ca-lcad/thinkstation-pgx.md
cat ~/ciclo-previa.log                         # resultado de cada episódio do ciclo p
~/ciclo.sh <prefixo> <inicio> <rodadas> <esforco>   # ciclo automático: episódio -> avaliação -> reflexão -> próximo
~/run-ep.sh <nome> [cup_xy] [memoria|none]    # um episódio + reflexão (EFFORT=max no ambiente para esforço)
# memória de lições da visão real: ~/g1-cup-grasp/results/direct-lessons-visao-real.json
# vídeo de um episódio:  MUJOCO_GL=egl .venv/bin/python scripts/make_direct_story.py results/<ep>/x0.40_y-0.20
```
Próximos passos, em ordem: (1) ler o resultado do ciclo p; (2) se a inclinação continuar sendo a única falha,
olhar se a pega lateral inclina a caneca por geometria (alinhar a abertura com o eixo vertical dela) e deixar o
crítico ver a inclinação em cada passo (já vê); (3) refazer o vídeo do ciclo com a história correta (legendas nos
01 a 07, visão real depois) antes de mostrar ao professor; (4) subir episódios novos na branch.
O OmniRoute roda na Spark; no notebook `127.0.0.1:20128` é o túnel `systemctl --user status omniroute-tunnel`.

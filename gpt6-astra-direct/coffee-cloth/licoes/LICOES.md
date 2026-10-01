# Lições do orquestrador do café

Gerado por `scripts/licoes.py`. Uma solução só entra aqui depois que a
etapa passou de verdade, com os números que sustentam.

## Resolvidos

### ligar_chaleira/prensar/forca_de_contato

Resolvido em 2026-09-21T11:36:40 depois de 0 tentativas falhas.

```json
{
 "parametros": {
  "curso_mm": 30,
  "desloc_y_mm": -6,
  "cintura_deg": 25,
  "recuo_mm": 40
 },
 "evidencia": {
  "temperatura_C": 100.0,
  "evento": "automatic_boil_cutoff",
  "pico_de_forca_N": 2.311
 }
}
```

### pegar_chaleira/planejar_pega/pega_sem_oposicao

Resolvido em 2026-09-21T15:55:25 depois de 20 tentativas falhas.

```json
{
 "parametros": {
  "peso_orientacao": 8,
  "folga_alvo_mm": 8,
  "forca_fechamento_N": 20,
  "peso_oposicao": 250,
  "sementes": 60,
  "mao": "direita",
  "folga_conexao_mm": 6.5,
  "cintura_deg": 25
 },
 "evidencia": {
  "nota": "etapa vencida; a falha foi depois, em aproximacao",
  "falha_seguinte": "nenhum candidato com aproximacao valida",
  "medido_da_falha_seguinte": {
   "ultimo_passo": {
    "distance_m": 0.0,
    "error_m": 2.220157710297933e-08,
    "orientation_error_deg": 0.0017812235411134325,
    "collision_free": false
   }
  }
 }
}
```

### pegar_chaleira/aproximacao/aproximacao_invalida

Resolvido em 2026-09-21T17:53:13 depois de 7 tentativas falhas.

```json
{
 "parametros": {
  "peso_oposicao": 220,
  "peso_orientacao": 25,
  "folga_conexao_mm": 6.5,
  "cintura_deg": 25,
  "sementes": 300,
  "forca_fechamento_N": 20,
  "mao": "esquerda",
  "folga_alvo_mm": 8
 },
 "evidencia": {
  "nota": "etapa vencida; a falha foi depois, em levantar",
  "falha_seguinte": "{'reason': 'hand clearance', 'hot_m': 0.005589943983081533, 'table_m': 0.09939641239394015}",
  "medido_da_falha_seguinte": {
   "folga_minima_ao_metal_quente_mm": 5.5899,
   "portao_mm": 6.0,
   "passos_abaixo_do_portao": 26
  }
 }
}
```

### pegar_chaleira/levantar/penetracao

Resolvido em 2026-09-21T20:02:04 depois de 3 tentativas falhas.

```json
{
 "parametros": {
  "portao_de_penetracao_mm": 1.0,
  "portao_de_auto_contato_mm": 0.5,
  "forca_fechamento_N": 8,
  "mao": "esquerda"
 },
 "evidencia": {
  "resultado": "subida de 81,3 mm, pega estavel",
  "medido": {
   "subida_mm": 81.317,
   "penetracao_alca_mm": 0.58,
   "folga_metal_quente_mm": 6.85,
   "rotacao_na_mao_deg": 1.5,
   "deriva_na_sustentacao_mm": 1.99,
   "angulo_na_sustentacao_deg": 1.72,
   "auto_contato_max_mm": 0.285
  },
  "causa_das_3_falhas_anteriores": "portao de penetracao em 0,5 mm, que e o ponto medio do solimp dos geoms da mao e da alca; a penetracao ficava travada em 0,5002/0,5003/0,5011 mm com forca de 20, 14 e 8 N, ou seja nao respondia a forca nenhuma porque era o proprio portao",
  "segunda_causa": "auto-contato ombro-tronco era booleano; um raspao de 7,7 micrometros entre left_shoulder_roll_link e torso_link reprovava uma corrida que ja tinha levantado 30,7 mm. A chaleira parada penetra 227 micrometros, entao 7,7 estava trinta vezes abaixo do piso de ruido",
  "verificacao_visual": "results-claude/lift-regate-13c-video/attempt-six-cameras.mp4, chaleira no ar com sombra na mesa em hold_handle",
  "run": "results-claude/lift-regate-13c"
 }
}
```

### pegar_chaleira/planejar_pega/rendimento_do_planejador

Resolvido em 2026-09-21T19:56:01 depois de 0 tentativas falhas.

```json
{
 "parametros": {
  "cintura_deg": 12,
  "peso_oposicao": 0,
  "semente": 1,
  "folga_conexao_mm": 7.0,
  "mao": "esquerda"
 },
 "evidencia": {
  "taxa_medida_48_sementes": {
   "padrao": "3/48",
   "cintura_25": "2/48",
   "oposicao_250": "2/48"
  },
  "conclusao": "a taxa e 4-6% em qualquer configuracao; a diferenca nao existe com n=48",
  "semente_1_e_deterministica": "sementes 0 e 1 nao levam ruido, mas a pose resultante depende do hot_clearance usado: com 7,0 mm ela passa (pontas 0,14/0,88/0,85, cossenos -0,995/-0,558); com 6,5 mm ela REPROVA (pontas 0,76/6,74/3,77, cosseno +0,62, folga com mao aberta 5,33 mm). Nao trate a semente 1 como garantia: ela e reproduzivel para um hot_clearance fixo, nao entre valores.",
  "semente_1_padrao": {
   "pontas_mm": [
    0.14,
    0.88,
    0.85
   ],
   "cossenos_oposicao": [
    -0.995,
    -0.558
   ],
   "folga_quente_mm": 7.0
  },
  "cintura_25_quebra": {
   "cossenos_oposicao": [
    -0.644,
    0.991
   ],
   "leitura": "o segundo cosseno vira +0,991: os dois dedos empurram para o mesmo lado, nao e pega"
  },
  "cintura_25_mais_oposicao_250": {
   "pontas_mm": [
    1.48,
    14.87,
    13.97
   ],
   "leitura": "duas pontas a 14 mm da alca, fora de contato"
  },
  "medido_em": "results-claude/kgr-aberto-60 (hot_clearance 6,5 mm) contra results/taxa48-* (7,0 mm)"
 }
}
```

### pegar_chaleira/aproximacao/abertura_da_mao

Resolvido em 2026-09-21T20:20:48 depois de 0 tentativas falhas.

```json
{
 "parametros": {
  "abertura": "1.0 ou 0.0, nunca entre 0.2 e 0.8",
  "folga_conexao_mm": 3.5,
  "cintura_deg": 12,
  "mao": "esquerda"
 },
 "evidencia": {
  "geometria_medida_no_candidato_12_de_kettle-grasp-2": {
   "abertura_1.0": {
    "folga_quente_mm": 9.63,
    "contatos_negativos": 0,
    "folga_alca_mm": 0.576
   },
   "abertura_0.9": {
    "folga_quente_mm": 3.54,
    "contatos_negativos": 0
   },
   "abertura_0.8_ate_0.2": {
    "folga_quente_mm": "de -1.9 a -5.7",
    "contatos_negativos": "3 a 4",
    "leitura": "o polegar varre POR DENTRO do corpo quente da chaleira; nenhuma abertura intermediaria e viavel"
   },
   "abertura_0.1": {
    "folga_quente_mm": 0.603
   },
   "abertura_0.0": {
    "folga_quente_mm": 3.787,
    "contatos_negativos": 0
   }
  },
  "por_que_abertura_0_falhava": "a conexao exigia folga_conexao_mm 6.5 e a pose de mao aberta so tem 3.787 mm; baixar folga_conexao para 3.5 aprova a conexao (41 passos)",
  "por_que_abertura_1_falha_depois": "sem curso de fechamento as pontas param a 0.58 mm da alca e as forcas ficam em 0 -> lost three-finger handle grip",
  "proximo_obstaculo": "com abertura 0 e folga 3.5 a conexao passa, mas o executor do lift reprova: 5.217 mm sustentados por 52 ms contra portao de 6 mm / 50 ms, e isso acontece na fase preshape, com o braco ainda parado em cima da chaleira depois de apertar o botao. O planejador de conexao abre os dedos na pose inicial antes de mover o braco; o recuo deveria vir antes da abertura",
  "runs": [
   "results-claude/conn-abertura-1",
   "results-claude/lift-abertura-1",
   "results-claude/conn-ab0-folga35",
   "results-claude/lift-ab0-folga35"
  ]
 }
}
```

### pegar_chaleira/planejar_pega/folga_com_a_mao_aberta

Resolvido em 2026-09-21T20:33:01 depois de 0 tentativas falhas.

```json
{
 "parametros": {
  "usar": "plan_kettle_grasp_v3 com --open-check-factor igual a abertura",
  "abertura": 0,
  "cintura_deg": 12,
  "mao": "esquerda"
 },
 "evidencia": {
  "problema": "o planejador media a folga ao metal quente so na pose de pega, com a mao fechada, mas quem viaja ate la e a mao aberta",
  "medido": {
   "candidato_12": {
    "fechado_mm": 9.63,
    "aberto_mm": 3.79
   },
   "aberturas_intermediarias": "de 0.2 a 0.8 o polegar passa POR DENTRO do corpo quente, ate -5.7 mm"
  },
  "efeito": "candidatos inalcancaveis eram aprovados e a falha so aparecia na conexao, como reverse approach invalid na distancia zero",
  "depois_da_correcao": "4 de 60 candidatos passam, com folga de mao aberta entre 14,4 e 21,4 mm",
  "regra": "folga_alvo_mm precisa superar o portao do executor em cerca de 6 mm, porque o fechamento consome de 2,8 a 5,7 mm",
  "run": "results-claude/kgr-aberto-60"
 }
}
```

### servir_cafe/despejar/portao_de_penetracao

Resolvido em 2026-09-21T20:57:42 depois de 0 tentativas falhas.

```json
{
 "parametros": {
  "usar": "execute_coffee_pour_v2 e execute_coffee_place_v2",
  "duracao_s": 20,
  "max_segundos": 320,
  "recuo_x": 0,
  "recuo_levantamento": 0,
  "taxa_retorno": 0.4
 },
 "evidencia": {
  "problema": "a v1 tem hot_gap<.006 e max_penetration>.0005 fixos no codigo; o despejo reprovava em t=0, na fase approach_filter, com hot_gap 8,90 mm (otimo) e penetracao 0,58 mm",
  "resultado_com_a_v2": {
   "despejado_ml": 244.5,
   "na_xicara_ml": 199.8,
   "retido_no_pano_ml": 5.0,
   "derramado_ml": 0.0,
   "penetracao_mm": 0.598,
   "folga_quente_mm": 8.895,
   "rotacao_na_mao_deg": 1.21
  },
  "devolver_chaleira": {
   "inclinacao_deg": 0.21,
   "forca_de_apoio_N": 11.8,
   "erro_de_centro_mm": 0.13,
   "passos_estaveis": 1000,
   "avisos": 0
  },
  "verificacao_visual": "results-claude/pour-v2-video/attempt-six-cameras.mp4; no quadro final o HUD le xicara 200, filtro 0, derrame 0.0 e a camera da cabeca mostra liquido escuro na xicara",
  "cadeia_completa_validada": "lift-regate-13c -> pour-smoke-plan -> pour-v2 -> place-v2",
  "runs": [
   "results-claude/pour-v2",
   "results-claude/place-v2"
  ]
 }
}
```

### pegar_chaleira/planejar_pega/pega_no_referencial_do_objeto

Resolvido em 2026-09-22T14:45:13 depois de 0 tentativas falhas.

```json
{
 "parametros": {
  "usar": "plan_kettle_grasp_v3 --seed results-claude/pega-referencia-validada --fixed-hand",
  "sementes": 6
 },
 "evidencia": {
  "problema": "a referencia padrao (results/loadable-refined-002) vem de outra geometria de alca (handle-layout-search-003/layout-05); a palma ia ao lugar mas os dedos eram reinventados a cada semente e 39-40 de 60 candidatos reprovavam so na oposicao",
  "solucao": "pega validada (agent-cafe-012/kettle-grasp-13 c74) guardada no referencial da chaleira, dedos travados, termo das pontas desligado; so o braco e resolvido",
  "invariancia": "nas cenas original, +5 cm em x e -5 cm em y a semente 1 cai na mesma geometria de contato ate o terceiro decimal: pontas 0,44/1,60/1,56 mm, oposicao -0,499/-0,514, palma a 0,0 mm",
  "taxa": {
   "original": "4/6 (antes 1/60)",
   "x+5": "6/6 (antes 1/60)",
   "y-5": "3/6 (antes 0/60)",
   "giro30": "0/6, palma a 6,3 mm: problema de alcance, nao de pega"
  },
  "cadeia_x+5": "levantou 81,2 mm, 199,8 ml na xicara, 0 derramado, devolveu",
  "runs": [
   "results-claude/fix-x+5",
   "results-claude/movida-x+5"
  ]
 }
}
```

### servir_cafe/despejar/apoio_no_coador_ao_endireitar

Resolvido em 2026-09-22T15:08:30 depois de 0 tentativas falhas.

```json
{
 "parametros": {
  "return_lift_m": 0.03,
  "return_offset_x": 0,
  "return_rate": 0.4
 },
 "evidencia": {
  "problema": "com return_lift 0 a chaleira endireitava rente ao coador; com ela 5 cm deslocada em y ela se apoiou no coador em return_upright, depois de despejar 244,5 ml",
  "resultado": "com 3 cm de subida as tres cenas (original, x+5, y-5) passam: 199,8 ml na xicara, 0 derramado",
  "runs": [
   "results-claude/movida-y-5/pour-rl3",
   "results-claude/regr-rl3-original",
   "results-claude/regr-rl3-x+5"
  ]
 }
}
```

## Ainda abertos

### pegar_chaleira/?/folga_metal_quente

1 tentativas, nenhuma resolveu. Últimas:

- `{}` -> null

### pegar_chaleira/?/pega_sem_oposicao

2 tentativas, nenhuma resolveu. Últimas:

- `{}` -> null
- `{}` -> null

### pegar_chaleira/levantar/folga_metal_quente

11 tentativas, nenhuma resolveu. Últimas:

- `{"mao": "esquerda", "forca_fechamento_N": 30.0}` -> {"folga_minima_ao_metal_quente_mm": 5.6355, "portao_mm": 6.0, "passos_abaixo_do_portao": 26}
- `{"mao": "esquerda", "folga_alvo_mm": 8.0, "sementes": 300, "peso_oposicao": 220.0, "cintura_deg": 25.0, "abertura": 0.0, "forca_fechamento_N": 20.0, "folga_conexao_mm": 6.5, "peso_orientacao": 25.0}` -> {"folga_minima_ao_metal_quente_mm": 5.5899, "portao_mm": 6.0, "passos_abaixo_do_portao": 26}
- `{"mao": "esquerda", "folga_alvo_mm": 7.0, "sementes": 20, "peso_oposicao": 0.0, "cintura_deg": 12.0, "abertura": 0.0, "forca_fechamento_N": 8.0, "folga_conexao_mm": 7.0, "peso_orientacao": 30.0}` -> {"folga_minima_ao_metal_quente_mm": 5.3506, "portao_mm": 6.0, "passos_abaixo_do_portao": 26}
- `{"mao": "esquerda", "folga_alvo_mm": 8.8, "sementes": 60, "peso_oposicao": 0.0, "cintura_deg": 12.0, "abertura": 0.0, "forca_fechamento_N": 8.0, "folga_conexao_mm": 3.5, "peso_orientacao": 30.0}` -> {"folga_minima_ao_metal_quente_mm": 5.2233, "portao_mm": 6.0, "passos_abaixo_do_portao": 26}
- `{"mao": "esquerda", "folga_alvo_mm": 12.0, "sementes": 100, "peso_oposicao": 0.0, "cintura_deg": 12.0, "abertura": 0.0, "forca_fechamento_N": 8.0, "folga_conexao_mm": 3.5, "peso_orientacao": 30.0}` -> {"folga_minima_ao_metal_quente_mm": 5.66, "portao_mm": 6.0, "passos_abaixo_do_portao": 26}

### ligar_chaleira/prensar/tempo_esgotado

2 tentativas, nenhuma resolveu. Últimas:

- `{"curso_mm": 30.0, "desloc_y_mm": -6.0}` -> {"pico_de_forca_N": 2.6721977279348508}
- `{"curso_mm": 30.0, "desloc_y_mm": -6.0}` -> {"pico_de_forca_N": 2.6721977279348508}

### pegar_chaleira/levantar/outro

1 tentativas, nenhuma resolveu. Últimas:

- `{"candidato": 15, "folga_fechada_estatica_mm": 10.97, "folga_conexao_mm": 9.0, "forca_fechamento_N": [16, 8], "alvo_contato_N": [8, 10]}` -> {"folga_executada_mm": [5.429, 5.32], "onde": "ultimo estado da trajetoria, fase close_handle, geom de left_hand_thumb_2_link", "caminho_planejado_minimo_mm": 14.616, "leitura": "a

### pegar_chaleira/planejar_pega/outro

1 tentativas, nenhuma resolveu. Últimas:

- `{"sintetizador": "scripts/synth_wrap_grasp.py", "poses": 21600, "folga_termica_mm": 3}` -> {"resultado": "0 pegas com os tres dedos na barra", "geometria": "barra 16,6 x 22 mm, vao ate o corpo quente 25-27 mm; falange da Dex3 17,7 mm", "causa": "o polegar da Dex3 sai per

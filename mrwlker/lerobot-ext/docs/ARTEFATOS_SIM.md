# Os artefatos da câmera da cabeça no dataset de simulação

> Medido em 09/09/2026 sobre `sim_copo_mujoco_2026-09-03` (275 episódios, os que
> entraram no co-treino) contra `white_cup_on_dripper_2026-08-11` (o real).
> Conclusão curta: **não é o codec, é o renderizador — e o pior dos três
> problemas não é a sombra, é o brilho.**

---

## 1. O que NÃO é

**Não é a reencodagem de av1 para h264.** Os dois datasets saem com o mesmo
encoder e os mesmos parâmetros — h264, `yuv420p`, `crf 30`, `g 2` — e as taxas de
bits são comparáveis (885 kbit/s no sim, 775 no real). Renderizando a cena de
novo hoje, direto do MuJoCo para PNG sem passar por vídeo nenhum, **o artefato
aparece igual**. Sai do renderizador.

**Não é resolução de shadow map.** O `scene_43dof.xml` usa os defaults
(`shadowsize` 4096, `offsamples` 4, conferidos com `mj_loadXML`). Subir para
16384 dá uma imagem **idêntica** — testado, byte a byte no olho. Aumentar o
shadow map não é o conserto.

---

## 2. Os três artefatos, em ordem de dano

### 2.1 O brilho — este é o grave

Média do quadro inteiro, 8 quadros amostrados por dataset:

| dataset | brilho médio | variação entre quadros |
|---|---:|---:|
| `sim_copo_mujoco_2026-09-03` (275 eps) | **199,3** | ± 4,7 |
| `sim_coador_pega_2026-09-04` | 179,9 | ± 45,8 |
| `white_cup_on_dripper_2026-08-11` (real) | **116,6** | ± 4,5 |

O sim é **1,7× mais claro** que o real, e quase sem variação entre episódios —
uma única condição de iluminação, repetida 275 vezes.

Agora cruze isso com a augmentation dos configs de co-treino
(`fastwamdepth_cotreino_sim_real.yaml`, `pi05_cotreino_sim_real.yaml`):

```yaml
ColorJitter: brightness: [0.8, 1.2]
```

- sim × [0,8 , 1,2] → **[159 , 239]**
- real × [0,8 , 1,2] → **[93 , 140]**

**As duas faixas não se tocam.** Com esta augmentation, a primeira convolução da
rede separa sim de real sem esforço, e o co-treino vira dois treinos paralelos
que compartilham pesos. Não é sutil: é 19 níveis de folga entre os intervalos.

E lembre do que está documentado no `pi05_cotreino_sim_real.yaml`: quatro das
cinco transformações declaradas (MotionBlur, GaussianNoise, RandomShadow,
GammaCorrection) **não rodam** sobre uint8 — têm guarda `is_floating_point()` e
devolvem a imagem intacta, em silêncio. Ou seja, hoje o único mecanismo que
poderia aproximar as duas distribuições é justamente o ColorJitter, e ele está
calibrado estreito demais.

### 2.2 A profundidade — o sim não tem buraco nenhum

Um quadro de cada, mesmo índice:

| | mínimo | zeros (inválidos) | níveis distintos |
|---|---:|---:|---:|
| sim | 176 | **0,0 %** | 550 |
| real | 0 | **11,1 %** | 553 |

A D435i devolve 11% do quadro sem medida — reflexo, oclusão, superfície escura.
O MuJoCo devolve profundidade perfeita em 100% dos pixels, sempre. Duas
consequências, e a segunda é pior:

1. dá para distinguir sim de real contando pixels zero;
2. uma política treinada sobretudo em sim **nunca vê um buraco de profundidade**,
   e no robô real 11% do mapa é exatamente isso.

### 2.3 A sombra — feia, e a menos importante

Sombra binária, sem penumbra, com borda em escada de ~1 cm sobre o tampo. Vem da
única luz da cena (`scene_43dof.xml:35`):

```xml
<light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
```

Uma direcional, dura, a pino, com `castshadow` ligado. No real não existe nada
parecido: a iluminação do laboratório é difusa e a mesa não tem sombra recortada.

Testei três variantes renderizando a mesma pose:

| variante | resultado |
|---|---|
| `shadowsize=16384` | idêntico — não é resolução |
| luz única com `castshadow="false"` | sem sombra, cena chapada |
| **3 luzes em ângulos, 2 sem sombra, 1 fraca com sombra** | sombra suave, sem escada |

Patch da terceira, para `scene_43dof.xml:35`:

```xml
<light pos="0.6 0.6 2.0"  dir="-0.2 -0.2 -1" directional="true" castshadow="false" diffuse="0.35 0.35 0.35"/>
<light pos="-0.4 -0.8 2.0" dir="0.1 0.3 -1"  directional="true" castshadow="false" diffuse="0.3 0.3 0.3"/>
<light pos="0.8 -0.5 1.8" dir="-0.2 0.2 -1"  directional="true" castshadow="true"  diffuse="0.25 0.25 0.25"/>
```

---

## 3. O que fazer, em ordem de retorno

1. **Sortear a iluminação por episódio** no `gerar_dataset_mujoco.py` — direção,
   intensidade e cor da(s) luz(es), mais o `rgba` do tampo. Isso ataca 2.1 e 2.3
   de uma vez, e é o que faz o co-treino ter sentido: o objetivo não é o sim
   ficar bonito, é a distribuição do sim CONTER a do real. Mirar num brilho médio
   em torno de 115–120 com variação larga entre episódios.
2. **Alargar o ColorJitter** dos configs de co-treino (`brightness: [0.5, 1.3]`)
   e consertar a guarda `is_floating_point()` para as outras quatro
   transformações rodarem. Uma coisa não substitui a outra: augmentation não
   inventa a textura do pano da mesa.
3. **Furar a profundidade do sim** — zerar pixels por um modelo simples de falha
   da D435i (bordas de silhueta, superfícies muito inclinadas, e um percentual
   aleatório) até bater a ordem de 11%.
4. **Regravar os 275.** Nenhum dos três conserta o que já está no disco. A
   gravação leva o que levou da última vez, e o dataset atual serve de linha de
   base para medir se valeu.

> Antes de regravar, decida também o marcador de alvo. Os 275 episódios têm um
> **X preto pintado na mesa** no ponto de destino (era o `marca_alvo` da época);
> o XML de hoje já troca o X pelo coador texturizado. Um alvo desenhado é uma
> pista que o real não tem — mas sem alvo nenhum o dataset fica ambíguo, porque
> o destino é sorteado por episódio. O coador resolve os dois lados.

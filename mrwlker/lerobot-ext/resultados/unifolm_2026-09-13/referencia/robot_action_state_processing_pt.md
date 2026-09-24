# Especificação de Processamento de Ação, Estado e Estatísticas do Robô

[Chinês](robot_action_state_processing.md) | [Inglês](robot_action_state_processing_en.md) | **Português**

> Tradução para leitura da spec publicada pela Unitree em `unitreerobotics/unifolm-wla`
> (versão de 13/09/2026). Fórmulas, código e nomes de campo foram mantidos como no
> original. Em caso de dúvida, vale o texto em inglês.

## Resumo

Esta especificação mapeia dados de diferentes corpos de robô (*embodiments*) para um espaço de ação unificado e um espaço de estado unificado:

- cada ação futura é representada por um vetor de 54 dimensões;
- cada estado atual é representado por um vetor de 60 dimensões;
- máscaras booleanas identificam os módulos que de fato estão disponíveis;
- ações de efetuador e de pose da base são representadas como transformações SE(3) relativas ao estado atual;
- todas as demais ações mantêm a semântica de controle definida pelo dataset de origem;
- ações de pose relativa usam, por padrão, normalização Z-score global;
- ações e estados comuns usam, por padrão, normalização pelos percentis 1 e 99;
- estatísticas de tarefas diferentes do mesmo corpo de robô são mescladas com peso igual por tarefa; depois disso, as estatísticas dos efetuadores esquerdo e direito podem, opcionalmente, ser mescladas.

Os termos **deve**, **deveria** e **não deve** expressam requisitos para reproduzir este pipeline de processamento de dados.

---

## 1. Notação, Tipos de Dados e Convenções

### 1.1 Notação de tempo e de dimensões

| Símbolo | Significado |
|---|---|
| $t$ | Instante da observação atual e instante de referência do chunk de ação |
| `k` | Índice do passo de predição dentro do chunk de ação; `k = 0, 1, ..., H - 1` |
| $H$ | Comprimento do chunk de ação |
| $f_s$ | Taxa de quadros de origem |
| $f_t$ | Taxa de quadros alvo |
| $D_a=54$ | Dimensão da ação unificada |
| $D_s=60$ | Dimensão do estado unificado |
| `a[t,k]` | A `k`-ésima ação futura referenciada ao instante `t` |
| `s[t]` | Estado no instante atual |
| `m_a` | Máscara de validade da ação, 54 dimensões |
| `m_s` | Máscara de validade do estado, 60 dimensões |

A configuração padrão é:

```yaml
target_fps: 30
chunk_size: 30
norm_type: minmax_q
rel_norm_type: zscore
state_norm_type: minmax_q
gripper_norm_type: minmax_q
```

Portanto, o chunk de ação padrão contém 30 pontos de ação e cobre aproximadamente o intervalo `[0, 29/30]` segundos.

### 1.2 Tipos numéricos

Ações unificadas, estados unificados, offsets e escalas deveriam ser armazenados em `float32`. As estatísticas podem ser calculadas e mescladas em `float64` para reduzir erro numérico e depois convertidas para `float32` para uso no pipeline de dados.

Máscaras são arrays booleanos:

```math
\mathbf m^a\in\{0,1\}^{54},
\qquad
\mathbf m^s\in\{0,1\}^{60}.
```

### 1.3 Referenciais e unidades

#### 1.3.1 Definição do referencial do robô

O referencial da base do robô é um referencial dextrógiro (regra da mão direita) denotado por $B$:

| Eixo | Sentido positivo |
|---|---|
| $x$ | para a frente |
| $y$ | para a esquerda |
| $z$ | para cima |

Os eixos satisfazem:

```math
\mathbf e_x\times\mathbf e_y=\mathbf e_z.
```

As poses absolutas dos efetuadores esquerdo e direito são denotadas por $T^B_{E_L}$ e $T^B_{E_R}$, respectivamente, e ambas são expressas no referencial da base $B$. Assim, os dois braços usam a mesma convenção de coordenadas de pose: $x$ aponta para a frente, $y$ para a esquerda e $z$ para cima. O braço direito não deve usar um sistema de coordenadas espelhado; o sinal de uma mesma componente deve ter o mesmo significado físico nos dois braços.

A orientação local instantânea de um efetuador é representada pela matriz de rotação $R^B_E$ da sua pose. A convenção acima define o referencial em que as poses absolutas dos efetuadores são expressas; ela não exige que os eixos locais do efetuador, que se movem, permaneçam paralelos aos eixos da base.

#### 1.3.2 Unidades e convenções numéricas

O pipeline de processamento não faz conversão automática de unidades. Todos os datasets que serão combinados devem antes ser convertidos para as seguintes convenções:

- translação: metros;
- ângulo de rotação: radianos;
- velocidade linear: metros por segundo;
- velocidade angular: radianos por segundo;
- ordem do quaternion: `xyzw`;
- ordem dos ângulos de Euler: `xyz`;
- a ordem das juntas esquerda/direita e o sentido de abrir/fechar da garra devem ser consistentes em todo o dataset.

Qualquer entrada que não satisfaça essas convenções deve ser convertida antes de calcular estatísticas ou montar os vetores unificados.

---

## 2. Procedimento de Processamento de Ponta a Ponta

Para cada instante atual $t$, monte a entrada do modelo e o alvo de ação na seguinte ordem:

1. ler o estado atual;
2. converter o estado atual para a representação unificada de 60 dimensões;
3. normalizar o estado módulo a módulo;
4. ler uma janela de ações futuras;
5. calcular as ações relativas de efetuador e de pose da base;
6. normalizar cada módulo de ação de forma independente;
7. opcionalmente, binarizar as ações de garra;
8. reamostrar as sequências de ação da taxa de quadros de origem para a taxa alvo;
9. escrever cada módulo de ação no vetor de ação unificado de 54 dimensões;
10. gerar as máscaras de validade de ação e de estado;
11. emitir o offset e a escala de ação, com 54 dimensões, necessários para desnormalizar.

O caminho do estado é:

```math
\text{raw state}
\longrightarrow
\text{state-format conversion}
\longrightarrow
\text{state normalization}
\longrightarrow
\mathbf s_t.
```

(estado bruto → conversão de formato do estado → normalização do estado → $\mathbf s_t$)

O caminho da ação é:

```math
(\text{current state},\text{future actions})
\longrightarrow
\text{relative actions}
\longrightarrow
\text{action normalization}
\longrightarrow
\text{resampling}
\longrightarrow
\mathbf A_t.
```

((estado atual, ações futuras) → ações relativas → normalização da ação → reamostragem → $\mathbf A_t$)

O chunk de ação unificado é:

```math
\mathbf A_t=
\begin{bmatrix}
\mathbf a_{t,0}^{\mathsf T}\\
\mathbf a_{t,1}^{\mathsf T}\\
\vdots\\
\mathbf a_{t,H-1}^{\mathsf T}
\end{bmatrix}
\in\mathbb R^{H\times54}.
```

---

## 3. Representação Unificada da Ação

### 3.1 Layout da ação de 54 dimensões

| Fatia | Dim. | Módulo | Semântica | Representação canônica |
|---|---:|---|---|---|
| `[0:6]` | 6 | Efetuador esquerdo | Pose futura relativa ao estado atual do efetuador esquerdo | xyz relativo + vetor de rotação |
| `[6:7]` | 1 | Garra esquerda | Comando futuro da garra | Escalar bruto; opcionalmente binarizado após a normalização |
| `[7:13]` | 6 | Mão hábil esquerda | Comando futuro de controle da mão | As 6 primeiras componentes brutas da ação |
| `[13:19]` | 6 | Efetuador direito | Pose futura relativa ao estado atual do efetuador direito | xyz relativo + vetor de rotação |
| `[19:20]` | 1 | Garra direita | Comando futuro da garra | Escalar bruto; opcionalmente binarizado após a normalização |
| `[20:26]` | 6 | Mão hábil direita | Comando futuro de controle da mão | As 6 primeiras componentes brutas da ação |
| `[26:29]` | 3 | Cintura | Ação futura da cintura | As 3 primeiras componentes da ação |
| `[29:32]` | 3 | Tronco | Ação futura do tronco | As 3 primeiras componentes da ação |
| `[32:34]` | 2 | Translação da base | Comando futuro de velocidade linear da base | `vx, vy` |
| `[34:35]` | 1 | Rotação da base | Comando futuro de velocidade de guinada (yaw) da base | `omega_z` |
| `[35:41]` | 6 | Pose da base | Pose futura relativa ao estado atual da base | xyz relativo + vetor de rotação |
| `[41:42]` | 1 | Altura | Comando futuro de altura | Escalar |
| `[42:48]` | 6 | Perna esquerda | Ação futura das juntas da perna esquerda | As 6 primeiras componentes da ação |
| `[48:54]` | 6 | Perna direita | Ação futura das juntas da perna direita | As 6 primeiras componentes da ação |

### 3.2 Componentes da ação de efetuador

As ações dos efetuadores esquerdo e direito usam a seguinte ordem de seis dimensões:

```math
\mathbf a^{ee}_{t,k}
=
[\Delta x,\Delta y,\Delta z,\phi_x,\phi_y,\phi_z]^{\mathsf T}.
```

As três primeiras componentes são a translação relativa expressa no referencial atual do efetuador. As três últimas formam o vetor de rotação relativo.

Um vetor de rotação é definido como:

```math
\boldsymbol\phi=\theta\mathbf u,
```

onde $\mathbf u$ é o eixo de rotação unitário e $\theta$ é o ângulo de rotação. Portanto:

```math
\|\boldsymbol\phi\|_2=\theta.
```

### 3.3 Ações que não são pose

Os módulos de ação a seguir são lidos diretamente da sequência de ações futuras e não passam por conversão para pose relativa SE(3):

- garras;
- mãos hábeis esquerda e direita;
- cintura;
- tronco;
- velocidade da base;
- altura;
- pernas esquerda e direita.

Esses campos podem representar alvos absolutos, comandos de velocidade ou incrementos pré-calculados, conforme o dataset de origem. Todos os datasets mapeados para a mesma fatia unificada devem usar a mesma semântica física de controle.

### 3.4 Regras de preenchimento da ação

Inicialize:

```math
\mathbf a_{t,k}=\mathbf 0\in\mathbb R^{54},
\qquad
\mathbf m^a=\mathbf 0\in\{0,1\}^{54}.
```

Para cada módulo disponível no robô atual:

1. escreva o valor do módulo na sua fatia fixa;
2. coloque 1 na fatia correspondente da máscara.

Módulos indisponíveis permanecem em zero e mantêm máscara zero.

Se um módulo de ação de entrada for mais largo que a fatia de destino, apenas as componentes iniciais são mantidas. Se for mais estreito que a fatia de destino, o schema de entrada é inválido: chunks de ação **não** são completados com zeros automaticamente. Todo módulo de ação habilitado deve, portanto, fornecer pelo menos o número exigido de dimensões.

---

## 4. Representação Unificada do Estado

### 4.1 Layout do estado de 60 dimensões

| Fatia | Dim. | Módulo | Semântica | Representação canônica |
|---|---:|---|---|---|
| `[0:9]` | 9 | Efetuador esquerdo | Pose absoluta atual | xyz + rotação-6D |
| `[9:10]` | 1 | Garra esquerda | Estado atual da garra | Escalar |
| `[10:16]` | 6 | Mão hábil esquerda | Estado atual da mão | 6 dimensões |
| `[16:25]` | 9 | Efetuador direito | Pose absoluta atual | xyz + rotação-6D |
| `[25:26]` | 1 | Garra direita | Estado atual da garra | Escalar |
| `[26:32]` | 6 | Mão hábil direita | Estado atual da mão | 6 dimensões |
| `[32:35]` | 3 | Cintura | Estado atual das juntas da cintura | As 3 primeiras componentes |
| `[35:38]` | 3 | Tronco | Estado atual das juntas do tronco | As 3 primeiras componentes |
| `[38:40]` | 2 | Velocidade de translação da base | `vx, vy` atuais | 2 dimensões |
| `[40:41]` | 1 | Velocidade angular da base | `omega_z` atual | Slot reservado |
| `[41:47]` | 6 | Estado inercial da base | Direção da gravidade e velocidade angular no referencial do corpo | `gx, gy, gz, wx normalizado, wy normalizado, wz normalizado` |
| `[47:48]` | 1 | Altura | Altura atual | Slot reservado |
| `[48:54]` | 6 | Perna esquerda | Estado atual das juntas da perna esquerda | As 6 primeiras componentes |
| `[54:60]` | 6 | Perna direita | Estado atual das juntas da perna direita | As 6 primeiras componentes |

O estado sempre representa a observação absoluta atual. Ele não é tornado relativo a si mesmo.

### 4.2 Rotação-6D para o estado do efetuador

Seja a matriz de rotação:

```math
R=
\begin{bmatrix}
R_{00}&R_{01}&R_{02}\\
R_{10}&R_{11}&R_{12}\\
R_{20}&R_{21}&R_{22}
\end{bmatrix}.
```

Esta especificação monta a rotação-6D a partir das duas primeiras colunas de $R$:

```math
\rho_6(R)=
[R_{00},R_{10},R_{20},R_{01},R_{11},R_{21}]^{\mathsf T}.
```

O estado absoluto do efetuador é, portanto:

```math
\mathbf s^{ee}_t=
[x,y,z,\rho_6(R_t)^{\mathsf T}]^{\mathsf T}
\in\mathbb R^9.
```

Para reconstruir uma matriz de rotação a partir de dois vetores 3D quaisquer $\mathbf a_1$ e $\mathbf a_2$, aplique a ortogonalização de Gram-Schmidt:

```math
\mathbf b_1=
\frac{\mathbf a_1}{\|\mathbf a_1\|_2},
```

```math
\widetilde{\mathbf b}_2
=
\mathbf a_2-(\mathbf b_1^{\mathsf T}\mathbf a_2)\mathbf b_1,
```

```math
\mathbf b_2=
\frac{\widetilde{\mathbf b}_2}{\|\widetilde{\mathbf b}_2\|_2},
\qquad
\mathbf b_3=\mathbf b_1\times\mathbf b_2,
```

```math
R=[\mathbf b_1,\mathbf b_2,\mathbf b_3].
```

Todos os componentes de geração de dados, treino e implantação devem usar a mesma convenção das duas primeiras colunas.

### 4.3 Estado de direção da gravidade e velocidade angular da base

A fatia de estado `[41:47]` não contém uma pose absoluta da base. Ela contém a direção da gravidade e a velocidade angular no referencial do corpo $B$:

```math
\boxed{
\mathbf s_t^{base}=
[
g_x^B,\,
g_y^B,\,
g_z^B,\,
\hat\omega_x^B,\,
\hat\omega_y^B,\,
\hat\omega_z^B
]^{\mathsf T}
}.
```

Aqui:

- `g_B = [gx, gy, gz]` é a direção unitária da gravidade expressa no referencial do corpo $B$;
- `omega_B = [wx, wy, wz]` é a velocidade angular bruta nos três eixos, no referencial do corpo $B$;
- `omega_hat_B` é a velocidade angular normalizada nos três eixos.

#### 4.3.1 Cálculo da direção da gravidade

Seja $R_{WB}$ a matriz que leva vetores do referencial do corpo $B$ para o referencial do mundo. Defina a direção unitária da gravidade no referencial do mundo como:

```math
\mathbf g^W=[0,0,-1]^{\mathsf T}.
```

A direção da gravidade expressa no referencial do corpo $B$ é:

```math
\mathbf g^B=R_{WB}^{\mathsf T}\mathbf g^W.
```

Para eliminar erro numérico do quaternion ou da matriz de rotação de entrada, normalize mais uma vez antes de escrever no estado:

```math
\mathbf g^B
\leftarrow
\frac{\mathbf g^B}{\max(\|\mathbf g^B\|_2,\epsilon)},
```

com valor recomendado $\epsilon=10^{-8}$. O vetor resultante deveria satisfazer:

```math
\|\mathbf g^B\|_2=1.
```

A direção da gravidade codifica rolagem (roll) e arfagem (pitch) em relação à gravidade. Ela não contém nem a posição no mundo nem a guinada (yaw) absoluta em torno do eixo da gravidade.

#### 4.3.2 Normalização da velocidade angular

A velocidade angular bruta no referencial do corpo é:

```math
\boldsymbol\omega^B=
[\omega_x^B,\omega_y^B,\omega_z^B]^{\mathsf T}.
```

Normalize componente a componente usando as estatísticas de estado da velocidade angular:

```math
\widehat{\boldsymbol\omega}^B
=
\frac{\boldsymbol\omega^B-\mathbf o_{\omega}}
{\mathbf c_{\omega}}.
```

O padrão é `state_norm_type=minmax_q`:

```math
\mathbf o_{\omega}
=
\frac{Q_{0.01}(\boldsymbol\omega^B)+Q_{0.99}(\boldsymbol\omega^B)}{2},
```

```math
\mathbf c_{\omega}
=
\frac{Q_{0.99}(\boldsymbol\omega^B)-Q_{0.01}(\boldsymbol\omega^B)}{2}.
```

Se a entrada já fornece a velocidade angular normalizada, escreva-a diretamente em `[44:47]` e não normalize de novo.

#### 4.3.3 Mapeamento dos slots

```math
[g_x^B,g_y^B,g_z^B]
\longrightarrow[41:44],
```

```math
[\hat\omega_x^B,\hat\omega_y^B,\hat\omega_z^B]
\longrightarrow[44:47].
```

Este bloco de estado não contém nem a posição xyz da base nem um vetor de rotação. A ação da base na fatia de ação `[35:41]` não muda e continua representando o xyz relativo da pose da base mais o vetor de rotação.

### 4.4 Preenchimento e máscara do estado

Inicialize:

```math
\mathbf s_t=\mathbf 0\in\mathbb R^{60},
\qquad
\mathbf m^s=\mathbf 0\in\{0,1\}^{60}.
```

Escreva cada módulo de estado disponível na sua fatia fixa e coloque 1 na máscara correspondente. Módulos indisponíveis permanecem em zero. Se um módulo de estado for mais estreito que a fatia de destino, complete o final com zeros e ainda assim marque a fatia inteira como válida. Se for mais largo, mantenha só as componentes iniciais que cabem. As dimensões do estado deveriam, portanto, ser validadas quando um dataset é integrado.

> **Atenção à assimetria:** no estado, módulo curto é completado com zeros; na ação (§3.4), módulo curto é erro de schema.

---

## 5. Conversão de Formato de Pose

Todo cálculo de pose relativa deve primeiro converter a pose de entrada numa transformação homogênea SE(3):

```math
T=
\begin{bmatrix}
R&\mathbf p\\
\mathbf 0^{\mathsf T}&1
\end{bmatrix}.
```

Aqui $\mathbf p=[x,y,z]^{\mathsf T}$.

### 5.1 xyz + RPY

A ordem de entrada é:

```math
[x,y,z,r,p,y].
```

O último $y$ denota a guinada (yaw); abaixo ela é escrita como $\psi$ para evitar ambiguidade. Defina:

```math
R_x(r)=
\begin{bmatrix}
1&0&0\\
0&\cos r&-\sin r\\
0&\sin r&\cos r
\end{bmatrix},
```

```math
R_y(p)=
\begin{bmatrix}
\cos p&0&\sin p\\
0&1&0\\
-\sin p&0&\cos p
\end{bmatrix},
```

```math
R_z(\psi)=
\begin{bmatrix}
\cos\psi&-\sin\psi&0\\
\sin\psi&\cos\psi&0\\
0&0&1
\end{bmatrix}.
```

Use a convenção de ângulos de Euler `xyz` em eixos fixos:

```math
R=R_z(\psi)R_y(p)R_x(r).
```

Todos os ângulos em radianos.

### 5.2 xyz + quaternion

A ordem de entrada é:

```math
[x,y,z,q_x,q_y,q_z,q_w].
```

Primeiro normalize o quaternion:

```math
\bar{\mathbf q}=
\frac{\mathbf q}{\|\mathbf q\|_2}.
```

Com o quaternion normalizado $(q_x,q_y,q_z,q_w)$, calcule:

```math
R=
\begin{bmatrix}
1-2(q_y^2+q_z^2) & 2(q_xq_y-q_zq_w) & 2(q_xq_z+q_yq_w)\\
2(q_xq_y+q_zq_w) & 1-2(q_x^2+q_z^2) & 2(q_yq_z-q_xq_w)\\
2(q_xq_z-q_yq_w) & 2(q_yq_z+q_xq_w) & 1-2(q_x^2+q_y^2)
\end{bmatrix}.
```

### 5.3 xyz + vetor de rotação

A ordem de entrada é:

```math
[x,y,z,\phi_x,\phi_y,\phi_z].
```

Seja:

```math
\boldsymbol\phi=[\phi_x,\phi_y,\phi_z]^{\mathsf T},
\qquad
\theta=\|\boldsymbol\phi\|_2.
```

Para $\theta>0$, seja $\mathbf u=\boldsymbol\phi/\theta$ e use a fórmula de Rodrigues:

```math
R=
I+\sin\theta[\mathbf u]_{\times}
+(1-\cos\theta)[\mathbf u]_{\times}^2.
```

Para $\theta$ muito pequeno, use uma expansão de ângulo pequeno numericamente estável ou uma implementação robusta de SO(3).

---

## 6. Definição e Cálculo da Ação Relativa

### 6.1 Pose relativa do efetuador e da base

Seja a pose do estado atual:

```math
T_t=
\begin{bmatrix}
R_t&\mathbf p_t\\
\mathbf 0^{\mathsf T}&1
\end{bmatrix},
```

e seja o `k`-ésimo alvo de ação futura:

```math
T_{t+k}=
\begin{bmatrix}
R_{t+k}&\mathbf p_{t+k}\\
\mathbf 0^{\mathsf T}&1
\end{bmatrix}.
```

A inversa da pose atual é:

```math
T_t^{-1}=
\begin{bmatrix}
R_t^{\mathsf T}&-R_t^{\mathsf T}\mathbf p_t\\
\mathbf 0^{\mathsf T}&1
\end{bmatrix}.
```

Defina a ação relativa como:

```math
T^{rel}_{t,k}=T_t^{-1}T_{t+k}.
```

Expandindo o produto:

```math
R^{rel}_{t,k}=R_t^{\mathsf T}R_{t+k},
```

```math
\mathbf p^{rel}_{t,k}
=R_t^{\mathsf T}(\mathbf p_{t+k}-\mathbf p_t).
```

Assim, a translação relativa é expressa no referencial atual do efetuador ou da base, e não como uma diferença direta de posição no referencial do mundo.

Converta a matriz de rotação relativa num vetor de rotação:

```math
\boldsymbol\phi^{rel}_{t,k}
=\mathrm{Log}(R^{rel}_{t,k})^{\vee}.
```

A ação relativa final é:

```math
\mathbf a^{rel}_{t,k}
=
\begin{bmatrix}
\mathbf p^{rel}_{t,k}\\
\boldsymbol\phi^{rel}_{t,k}
\end{bmatrix}
\in\mathbb R^6.
```

### 6.2 Pseudocódigo da ação relativa

```text
function relative_pose(current_pose, future_poses, pose_format):
    T_current = pose_to_SE3(current_pose, pose_format)
    T_current_inverse = inverse_SE3(T_current)

    result = []
    for future_pose in future_poses:
        T_future = pose_to_SE3(future_pose, pose_format)
        T_relative = T_current_inverse @ T_future

        relative_xyz = T_relative[0:3, 3]
        relative_rotvec = SO3_log(T_relative[0:3, 0:3])
        result.append(concat(relative_xyz, relative_rotvec))

    return result
```

---

## 7. Estatísticas Globais da Ação Relativa

Só se usam estatísticas globais. Não se mantêm estatísticas separadas para cada posição dentro do chunk de ação.

### 7.1 Cálculo das estatísticas globais

Para cada estado atual válido do dataset, calcule o chunk completo de ações relativas futuras conforme definido na Seção 6:

```math
\mathbf A^{(n)}
=
\begin{bmatrix}
\mathbf a^{rel}_{n,0}\\
\mathbf a^{rel}_{n,1}\\
\vdots\\
\mathbf a^{rel}_{n,H-1}
\end{bmatrix}
\in\mathbb R^{H\times d},
```

onde $n=1,\ldots,N$ indexa as amostras, $H$ é o comprimento do chunk e $d=6$ para uma pose relativa de efetuador ou de base.

Achate as dimensões de amostra e de horizonte numa única matriz:

```math
X_{rel}
=
\mathrm{reshape}
\left(
\{\mathbf A^{(n)}\}_{n=1}^{N},
(NH,d)
\right).
```

Cada ponto de ação relativa de cada chunk é tratado como uma amostra estatística independente, qualquer que seja a sua posição no horizonte.

Para a componente $j$, a média global é:

```math
\mu^{global}_j
=
\frac{1}{NH}
\sum_{n=1}^{N}
\sum_{k=0}^{H-1}
A^{(n)}_{k,j}.
```

O desvio-padrão populacional global é:

```math
\sigma^{global}_j
=
\sqrt{
\frac{1}{NH}
\sum_{n=1}^{N}
\sum_{k=0}^{H-1}
\left(A^{(n)}_{k,j}-\mu^{global}_j\right)^2
}.
```

Os extremos globais são:

```math
x^{global}_{min,j}=\min_{n,k}A^{(n)}_{k,j},
```

```math
x^{global}_{max,j}=\max_{n,k}A^{(n)}_{k,j}.
```

Os percentis globais 1 e 99 são:

```math
Q^{global}_{0.01,j}
=
Q_{0.01}\left(\{A^{(n)}_{k,j}\}_{n,k}\right),
```

```math
Q^{global}_{0.99,j}
=
Q_{0.99}\left(\{A^{(n)}_{k,j}\}_{n,k}\right).
```

Toda estatística global tem formato `(d,)`.

Poses relativas usam, por padrão, normalização Z-score:

```math
\widehat{\mathbf a}^{rel}_{n,k}
=
\frac{\mathbf a^{rel}_{n,k}-\boldsymbol\mu^{global}}
{\boldsymbol\sigma^{global}}.
```

A mesma média e o mesmo desvio-padrão globais são aplicados a todas as posições do horizonte no chunk.

### 7.2 Formato das estatísticas

Cada módulo de ação relativa só precisa das seguintes estatísticas globais:

```json
{
  "relative_action_key": {
    "global_max": [0.0],
    "global_min": [0.0],
    "global_q01": [0.0],
    "global_q99": [0.0],
    "global_mean": [0.0],
    "global_std": [0.0]
  }
}
```

Para uma pose relativa de seis dimensões, todo array tem comprimento 6 e usa a seguinte ordem:

```math
[\Delta x,\Delta y,\Delta z,
\phi_x,\phi_y,\phi_z].
```

### 7.3 Pseudocódigo de geração das estatísticas

```text
function collect_global_relative_statistics(samples):
    all_relative_steps = []

    for sample in samples:
        relative_chunk = relative_pose(
            sample.current_pose,
            sample.future_pose_chunk,
            sample.pose_format,
        )

        for relative_action in relative_chunk:
            all_relative_steps.append(relative_action)

    X = stack(all_relative_steps)  # formato: (número_total_de_passos, action_dim)

    return {
        "global_max": max(X, axis=0),
        "global_min": min(X, axis=0),
        "global_q01": quantile(X, 0.01, axis=0),
        "global_q99": quantile(X, 0.99, axis=0),
        "global_mean": mean(X, axis=0),
        "global_std": population_std(X, axis=0)
    }
```

---

## 8. Estatísticas Comuns de Estado e Ação

Para os campos que não passam por conversão online para pose relativa, empilhe todos os registros de baixa dimensão em:

```math
X=
\begin{bmatrix}
\mathbf x_1^{\mathsf T}\\
\mathbf x_2^{\mathsf T}\\
\vdots\\
\mathbf x_M^{\mathsf T}
\end{bmatrix}
\in\mathbb R^{M\times d}.
```

Calcule componente a componente:

```math
\boldsymbol\mu
=
\frac{1}{M}\sum_{i=1}^{M}\mathbf x_i,
```

```math
\boldsymbol\sigma
=
\sqrt{
\frac{1}{M}
\sum_{i=1}^{M}
(\mathbf x_i-\boldsymbol\mu)^2
},
```

e:

```math
\mathbf x_{min},\quad
\mathbf x_{max},\quad
Q_{0.01}(X),\quad
Q_{0.99}(X).
```

O desvio-padrão é o populacional, com divisor $M$.

Todo campo do arquivo de estatísticas comuns deve conter no mínimo:

```json
{
  "feature_key": {
    "mean": [0.0],
    "std": [1.0],
    "min": [-1.0],
    "max": [1.0],
    "q01": [-0.9],
    "q99": [0.9]
  }
}
```

Um campo opcional `count` pode ser guardado para uma futura mesclagem ponderada pelo número de amostras.

---

## 9. Definições de Normalização

### 9.1 Forma afim unificada

Todas as grandezas contínuas usam:

```math
\widehat{\mathbf x}
=
\frac{\mathbf x-\mathbf o}{\mathbf c},
```

onde a divisão é componente a componente. A desnormalização é:

```math
\mathbf x
=
\widehat{\mathbf x}\odot\mathbf c+
\mathbf o.
```

Os valores normalizados **não** são cortados (*clipping*). Valores fora da faixa estatística podem, portanto, ser menores que $-1$ ou maiores que $1$.

Cada componente da escala é protegida assim:

```math
c_j=
\begin{cases}
1,&c_j<10^{-6},\\
c_j,&\text{otherwise}.
\end{cases}
```

(ou seja, $c_j = 1$ se $c_j < 10^{-6}$; caso contrário, $c_j$ fica como está)

Se um módulo não tem estatísticas, use a normalização identidade:

```math
\mathbf o=\mathbf 0,
\qquad
\mathbf c=\mathbf 1.
```

### 9.2 Normalização min-max por quantis: `minmax_q`

As estatísticas são escolhidas nesta ordem de prioridade:

1. `global_q01/global_q99`;
2. `q01/q99`;
3. `min/max`;
4. normalização identidade.

Sejam os limites inferior e superior $\mathbf l$ e $\mathbf h$. Então:

```math
\mathbf o=
\frac{\mathbf l+\mathbf h}{2},
```

```math
\mathbf c=
\frac{\mathbf h-\mathbf l}{2}.
```

Assim:

```math
\mathbf l\mapsto-1,
\qquad
\mathbf h\mapsto1.
```

### 9.3 Normalização Z-score: `zscore`

As estatísticas são escolhidas nesta ordem de prioridade:

1. `global_mean/global_std`;
2. `mean/std`;
3. normalização identidade.

Os parâmetros são:

```math
\mathbf o=\boldsymbol\mu,
\qquad
\mathbf c=\boldsymbol\sigma.
```

### 9.4 Normalização min-max por extremos: `minmax`

As estatísticas são escolhidas nesta ordem de prioridade:

1. `global_min/global_max`;
2. `min/max`;
3. normalização identidade.

Os parâmetros são:

```math
\mathbf o=
\frac{\mathbf x_{min}+\mathbf x_{max}}{2},
```

```math
\mathbf c=
\frac{\mathbf x_{max}-\mathbf x_{min}}{2}.
```

---

## 10. Normalização Usada em Cada Módulo de Ação

A configuração padrão de normalização da ação é:

```yaml
norm_type: minmax_q
rel_norm_type: zscore
gripper_norm_type: minmax_q
```

| Módulo de ação | Fatia unificada | Fonte das estatísticas | Tipo de normalização | Estatísticas preferidas |
|---|---|---|---|---|
| Pose relativa do efetuador (braço único) | `[0:6]` | Estatísticas de ação relativa | `rel_norm_type` | `global_mean/global_std` |
| Pose relativa do efetuador esquerdo | `[0:6]` | Estatísticas de ação relativa | `rel_norm_type` | `global_mean/global_std` |
| Pose relativa do efetuador direito | `[13:19]` | Estatísticas de ação relativa | `rel_norm_type` | `global_mean/global_std` |
| Pose relativa da base | `[35:41]` | Estatísticas de ação relativa | `rel_norm_type` | `global_mean/global_std` |
| Garra única/esquerda/direita | `[6:7]`, `[19:20]` | Estatísticas comuns de ação | `gripper_norm_type` | `q01/q99` |
| Mão hábil esquerda/direita | `[7:13]`, `[20:26]` | Estatísticas comuns de ação | `gripper_norm_type` | `q01/q99` |
| Cintura | `[26:29]` | Estatísticas comuns de ação | `norm_type` | `q01/q99` |
| Tronco | `[29:32]` | Estatísticas comuns de ação | `norm_type` | `q01/q99` |
| Base `vx, vy` | `[32:34]` | Estatísticas comuns do `base_command` completo | `norm_type` | `q01/q99` das componentes de origem mapeadas |
| Base `omega_z` | `[34:35]` | Estatísticas comuns do `base_command` completo | `norm_type` | Componente mapeada por `vyaw` ou `vw` |
| Comando de altura | `[41:42]` | Estatísticas comuns do `base_command` completo | `norm_type` | Componente mapeada por `height` |
| Perna esquerda/direita | `[42:54]` | Estatísticas comuns de ação | `norm_type` | `q01/q99` |

Detalhes importantes:

1. as ações de mão hábil usam o tipo de normalização da garra, e não o tipo de normalização comum da ação;
2. poses relativas de efetuador e de base devem usar estatísticas de ação relativa, e não estatísticas de pose absoluta;
3. as estatísticas são geradas para o `base_command` de origem completo; depois, as componentes relevantes de offset e escala são selecionadas conforme o mapa de dimensões;
4. módulos ausentes do espaço de ação unificado usam offset 0 e escala 1.

### 10.1 Mapeamento das estatísticas do `base_command`

Suponha que o comando de base de origem seja:

```math
\mathbf b=[b_0,b_1,\ldots,b_{d-1}]^{\mathsf T},
```

com o mapa de dimensões:

```yaml
base_command_dims:
  vx: 0
  vy: 1
  vw: 2
  height: 3
```

Os parâmetros de normalização da ação unificada são mapeados assim:

```math
o^a_{32}=o^b_0,
\qquad
c^a_{32}=c^b_0,
```

```math
o^a_{33}=o^b_1,
\qquad
c^a_{33}=c^b_1,
```

```math
o^a_{34}=o^b_2,
\qquad
c^a_{34}=c^b_2,
```

```math
o^a_{41}=o^b_3,
\qquad
c^a_{41}=c^b_3.
```

Um campo `vyaw` é tratado da mesma forma que `vw`.

---

## 11. Normalização Usada em Cada Módulo de Estado

O tipo padrão de normalização do estado é:

```yaml
state_norm_type: minmax_q
```

| Módulo de estado | Fatia unificada | Normalizado? | Fonte das estatísticas | Observações |
|---|---|---:|---|---|
| xyz do efetuador esquerdo | `[0:3]` | Sim | 3 primeiras componentes das estatísticas do estado absoluto do efetuador esquerdo | Usa `state_norm_type` |
| Rotação-6D do efetuador esquerdo | `[3:9]` | Não | Nenhuma | Preserva a representação geométrica |
| Garra esquerda | `[9:10]` | Sim | Estatísticas do estado da garra esquerda | Usa `state_norm_type` |
| Mão hábil esquerda | `[10:16]` | Sim | Estatísticas do estado da mão esquerda | Usa `state_norm_type` |
| xyz do efetuador direito | `[16:19]` | Sim | 3 primeiras componentes das estatísticas do estado absoluto do efetuador direito | Usa `state_norm_type` |
| Rotação-6D do efetuador direito | `[19:25]` | Não | Nenhuma | Preserva a representação geométrica |
| Garra direita | `[25:26]` | Sim | Estatísticas do estado da garra direita | Usa `state_norm_type` |
| Mão hábil direita | `[26:32]` | Sim | Estatísticas do estado da mão direita | Usa `state_norm_type` |
| Cintura | `[32:35]` | Sim | Estatísticas do estado da cintura | Usa no máximo as 3 primeiras componentes |
| Tronco | `[35:38]` | Sim | Estatísticas do estado do tronco | Usa no máximo as 3 primeiras componentes |
| Direção da gravidade no corpo | `[41:44]` | Só normalização para norma unitária | Nenhuma estatística de dataset | Deve ter norma unitária |
| Velocidade angular do corpo | `[44:47]` | Sim | Estatísticas de estado da velocidade angular nos três eixos | Guarda a velocidade angular normalizada |
| Perna esquerda | `[48:54]` | Sim | Estatísticas do estado da perna esquerda | Usa no máximo as 6 primeiras componentes |
| Perna direita | `[54:60]` | Sim | Estatísticas do estado da perna direita | Usa no máximo as 6 primeiras componentes |
| Slots escalares não usados de velocidade angular e altura da base | `[40:41]`, `[47:48]` | Não | Nenhuma | Valor 0 e máscara 0 |

### 11.1 Por que só o xyz é normalizado no estado de pose do efetuador

Uma pose absoluta de entrada pode usar:

- xyz + RPY: 6 dimensões;
- xyz + quaternion: 7 dimensões;
- xyz + vetor de rotação: 6 dimensões.

O estado unificado do efetuador usa xyz + rotação-6D, que tem 9 dimensões. Aplicar as estatísticas brutas de rotação diretamente na rotação-6D não bateria nem em dimensão nem em geometria. Por isso, só as três primeiras componentes de posição reaproveitam as estatísticas brutas da pose:

```math
\widehat{\mathbf p}_t
=
\frac{\mathbf p_t-\mathbf o_{xyz}}
{\mathbf c_{xyz}},
```

enquanto a rotação-6D permanece inalterada.

O estado da base não usa essa regra de normalização de pose. A fatia `[41:47]` é montada a partir da direção da gravidade e da velocidade angular normalizada, conforme a Seção 4.3.

---

## 12. Mescla de Estatísticas Entre Tarefas do Mesmo Corpo de Robô

### 12.1 Escopo da mescla

As estatísticas só são mescladas dentro do **mesmo corpo de robô**. Trate o dicionário de estatísticas de cada tarefa como um grupo independente e mescle todas as tarefas daquele corpo numa única operação.

Seja o conjunto de tarefas:

```math
\mathcal T=\{\tau_1,\tau_2,\ldots,\tau_M\}.
```

Tarefas só podem entrar na mesma mescla se tiverem:

- o mesmo corpo de robô;
- as mesmas definições de módulos de ação e de estado;
- os mesmos referenciais, sentidos de eixo e unidades;
- o mesmo significado físico para campos de mesmo nome;
- as mesmas dimensões de ação e a mesma ordem de componentes.

Consequentemente:

- toda tarefa tem o mesmo peso;
- toda tarefa contribui com um grupo de estatísticas;
- o número de amostras brutas de uma tarefa não altera seu peso na mescla;
- estatísticas de corpos de robô diferentes não devem ser mescladas;
- corpos diferentes devem manter normalizadores separados, mesmo que usem as mesmas fatias unificadas.

### 12.2 Mescla da média e a hipótese de pesos iguais

Peso igual por grupo não supõe que tarefas diferentes tenham movimentos parecidos. Supõe que cada tarefa é escolhida com probabilidade aproximadamente igual durante o treino. O procedimento de amostragem equivalente é:

1. escolher uma tarefa de forma uniforme;
2. amostrar uma trajetória ou exemplo de treino dessa tarefa.

Se a tarefa $i$ tem distribuição de ações $P_i(\mathbf x)$, a mistura alvo do treino é:

```math
P_{train}(\mathbf x)
=
\frac{1}{M}
\sum_{i=1}^{M}P_i(\mathbf x).
```

Com amostragem balanceada por tarefa, o peso igual por tarefa corresponde à distribuição que o modelo vê, mesmo quando as tarefas têm números diferentes de trajetórias brutas.

Se, em vez disso, o treino amostra uniformemente entre todos os quadros brutos, o peso igual por grupo só aproxima o agrupamento por amostra quando as tarefas contribuem com números parecidos de trajetórias válidas, comprimentos de trajetória ou pontos de ação. A hipótese relevante é sobre contagem efetiva de amostras ou probabilidade de escolha da tarefa, não sobre a semelhança do conteúdo dos movimentos.

Seja $\boldsymbol\mu_i$ a média do grupo $i$. Sob a hipótese de balanceamento por tarefa, a média mesclada é:

```math
\boldsymbol\mu
=
\frac{1}{M}
\sum_{i=1}^{M}\boldsymbol\mu_i.
```

### 12.3 Mescla do desvio-padrão

Seja $\boldsymbol\sigma_i$ o desvio-padrão populacional do grupo $i$. A variância mesclada é:

```math
\boldsymbol\sigma^2
=
\frac{1}{M}
\sum_{i=1}^{M}\boldsymbol\sigma_i^2
+
\frac{1}{M}
\sum_{i=1}^{M}\boldsymbol\mu_i^2
-
\boldsymbol\mu^2.
```

De forma equivalente:

```math
\boldsymbol\sigma^2
=
\underbrace{
\frac{1}{M}\sum_{i=1}^{M}\boldsymbol\sigma_i^2
}_{\text{mean within-group variance}}
+
\underbrace{
\frac{1}{M}\sum_{i=1}^{M}
(\boldsymbol\mu_i-\boldsymbol\mu)^2
}_{\text{between-group variance of means}}.
```

(o primeiro termo é a variância média dentro dos grupos; o segundo é a variância das médias entre os grupos)

Por fim, calcule componente a componente:

```math
\boldsymbol\sigma
=
\sqrt{\max(\boldsymbol\sigma^2,0)}.
```

Se um desvio-padrão não tem média correspondente, use como alternativa:

```math
\boldsymbol\sigma
=
\sqrt{
\frac{1}{M}
\sum_{i=1}^{M}\boldsymbol\sigma_i^2
}.
```

### 12.4 Mescla dos extremos

```math
\mathbf x_{min}
=
\min_i\mathbf x_{min}^{(i)},
```

```math
\mathbf x_{max}
=
\max_i\mathbf x_{max}^{(i)}.
```

Todas as comparações são componente a componente.

### 12.5 Mescla dos quantis

Use um envelope conservador:

```math
Q_{0.01}^{merged}
=
\min_i Q_{0.01}^{(i)},
```

```math
Q_{0.99}^{merged}
=
\max_i Q_{0.99}^{(i)}.
```

Para outros campos de quantil, use a mediana componente a componente entre os grupos.

Esses valores não são os quantis verdadeiros da distribuição misturada das amostras brutas. Em geral, não é possível recuperar os quantis exatos da mistura a partir de alguns quantis por grupo. O cálculo exato exige reler as amostras brutas ou guardar um resumo de distribuição mesclável, como um histograma ou um t-digest.

### 12.6 Mescla da contagem

Se as estatísticas comuns incluem `count`, mescle assim:

```math
N=\sum_{i=1}^{M}n_i.
```

A média e o desvio-padrão mesclados continuam, por padrão, balanceados por tarefa e não usam `count` como peso. No esquema padrão, `count` é apenas metadado.

### 12.7 Compatibilidade de formato

Só arrays de mesmo formato podem ser mesclados para o mesmo campo e a mesma estatística:

- estatísticas de ação relativa devem ter a mesma dimensão de ação;
- estatísticas comuns só são mescladas quando o formato bate com o da primeira estatística válida;
- se menos de dois arrays de estatísticas comuns forem compatíveis em formato, mantenha a primeira estatística válida;
- estatísticas de imagem não participam da mescla de estatísticas de ação/estado de baixa dimensão.

### 12.8 Peso igual por grupo × peso por amostra

Peso igual por grupo é o adequado quando o objetivo é balancear as tarefas. Se as estatísticas precisam representar a distribuição empírica de cada amostra bruta, use ponderação pela contagem de amostras.

Seja $n_i$ o número de amostras do grupo $i$:

```math
N=\sum_{i=1}^{M}n_i,
```

```math
\boldsymbol\mu_{weighted}
=
\frac{1}{N}
\sum_{i=1}^{M}n_i\boldsymbol\mu_i,
```

```math
\boldsymbol\sigma^2_{weighted}
=
\frac{1}{N}
\sum_{i=1}^{M}
 n_i\left(
 \boldsymbol\sigma_i^2+
 \boldsymbol\mu_i^2
 \right)
-
\boldsymbol\mu_{weighted}^2.
```

Essa fórmula ponderada descreve uma política alternativa de estatísticas e não é usada no procedimento de mescla padrão.

---

## 13. Mescla das Estatísticas dos Efetuadores Esquerdo e Direito

### 13.1 Método de mescla

Quando as estatísticas compartilhadas esquerda/direita estão habilitadas, trate os efetuadores esquerdo e direito como dois grupos de mesmo peso.

Sejam os vetores de média esquerdo e direito:

```math
\boldsymbol{\mu}_L,
\qquad
\boldsymbol{\mu}_R.
```

A média mesclada é:

```math
\boldsymbol{\mu}_{LR}
=
\frac{\boldsymbol{\mu}_L+\boldsymbol{\mu}_R}{2}.
```

A variância mesclada é:

```math
\boldsymbol{\sigma}_{LR}^2
=
\frac{\boldsymbol{\sigma}_L^2+
      \boldsymbol{\sigma}_R^2}{2}
+
\frac{\boldsymbol{\mu}_L^2+
      \boldsymbol{\mu}_R^2}{2}
-
\boldsymbol{\mu}_{LR}^2.
```

A faixa de quantis mesclada é:

```math
Q_{0.01}^{LR}
=
\min(Q_{0.01}^{L},Q_{0.01}^{R}),
```

```math
Q_{0.99}^{LR}
=
\max(Q_{0.99}^{L},Q_{0.99}^{R}).
```

Após a mescla, os efetuadores esquerdo e direito devem usar offsets e escalas idênticos.

### 13.2 Ordem da mescla

Use a seguinte ordem:

1. calcular as estatísticas esquerda e direita de cada tarefa;
2. mesclar todas as tarefas do mesmo corpo de robô;
3. mesclar as estatísticas esquerda e direita já mescladas entre tarefas;
4. atribuir as mesmas estatísticas finais às chaves esquerda e direita.

```math
\text{per-task statistics}
\longrightarrow
\text{same-embodiment cross-task merge}
\longrightarrow
\text{left/right merge}
\longrightarrow
\text{normalizer}.
```

(estatísticas por tarefa → mescla entre tarefas do mesmo corpo → mescla esquerda/direita → normalizador)

### 13.3 Requisitos de consistência de coordenadas

A mescla esquerda/direita não espelha eixos, não troca componentes e não inverte sinais automaticamente. Antes de mesclar, garanta que:

- os eixos xyz da translação relativa têm o mesmo significado nos dois lados;
- os vetores de rotação esquerdo e direito usam a mesma convenção dextrógira;
- componentes de mesmo índice representam a mesma direção física;
- as unidades são idênticas.

Se as coordenadas do braço direito precisarem ser espelhadas para um referencial canônico do braço esquerdo, defina antes uma transformação fixa:

```math
\widetilde{\mathbf a}_R=M\mathbf a_R,
```

e calcule as estatísticas do braço direito a partir de $\widetilde{\mathbf a}_R$. A matriz $M$ deve incluir todas as permutações de eixo e trocas de sinal necessárias, tanto para as componentes de translação quanto para as de vetor de rotação.

### 13.4 Finalidade

Um normalizador compartilhado esquerda/direita:

1. elimina diferenças de escala causadas por quantidades desiguais de dados do braço esquerdo e do direito;
2. mantém o espaço numérico consistente sob troca esquerda/direita ou aumento de dados por simetria;
3. permite que uma cabeça de ação compartilhada aprenda um prior unificado de movimento bimanual.

---

## 14. Binarização da Garra

A binarização da garra é aplicada após a normalização. Seja a sequência normalizada da garra:

```math
\widehat{g}_0,\widehat{g}_1,\ldots,
\widehat{g}_{H-1}.
```

Os estados definidos são:

```math
\widehat{g}_k>0.9
\quad\Longrightarrow\quad
b_k=1,
```

```math
\widehat{g}_k<-0.9
\quad\Longrightarrow\quad
b_k=0.
```

Valores em `[-0.9, 0.9]` são estados intermediários. Percorra a sequência de trás para a frente, a partir do fim, e preencha cada valor intermediário com o estado definido futuro mais próximo.

Inicialize a classe do final como:

```math
b_{H-1}^{init}
=
\begin{cases}
1,&\widehat{g}_{H-1}>0,\\
0,&\widehat{g}_{H-1}\le 0.
\end{cases}
```

Pseudocódigo:

```text
carry = 1 if normalized_gripper[-1] > 0 else 0

for k from H-1 down to 0:
    if normalized_gripper[k] > 0.9:
        carry = 1
    else if normalized_gripper[k] < -0.9:
        carry = 0

    binary_gripper[k] = carry
```

A saída pertence a `{0, 1}` e deixa de ser um valor normalizado contínuo e simétrico comum.

---

## 15. Reamostragem da Sequência de Ações

### 15.1 Reamostragem linear

Para um chunk de origem com $N_s$ pontos, defina os timestamps de origem como:

```math
t_i^{src}=
\frac{i}{f_s},
\qquad i=0,\ldots,N_s-1.
```

Defina os timestamps alvo como:

```math
t_j^{tgt}=
\frac{j}{f_t},
\qquad j=0,\ldots,f_t-1.
```

Interpole cada componente da ação de forma independente, com interpolação linear por partes. Se um timestamp alvo cair fora da faixa de origem, use o valor da extremidade mais próxima em vez de extrapolar linearmente.

O modo linear padrão, de um segundo, lê $N_s=f_s$ pontos no intervalo:

```math
\left[0,\frac{f_s-1}{f_s}\right].
```

### 15.2 Reamostragem por B-spline

A reamostragem suavizada lê dois segundos de contexto, incluindo a âncora em $t=0$:

```math
N_s=2f_s+1.
```

Os timestamps de origem são:

```math
t_i^{src}=\frac{i}{f_s},
\qquad i=0,\ldots,2f_s.
```

O alvo continua sendo a janela do primeiro segundo:

```math
t_j^{tgt}=\frac{j}{f_t},
\qquad j=0,\ldots,f_t-1.
```

Use uma B-spline cúbica uniforme, de grau 3, com:

```math
K=\max\left(4,\left\lfloor\frac{N_s}{2}\right\rfloor+1\right)
```

funções de base. Sejam $B_{src}$ e $B_{tgt}$ as matrizes de base avaliadas nos timestamps de origem e alvo, e seja:

```math
\lambda=10^{-9}.
```

Pré-calcule a matriz de reamostragem:

```math
W=
B_{tgt}
\left(B_{src}^{\mathsf T}B_{src}+\lambda I\right)^{-1}
B_{src}^{\mathsf T}.
```

Para cada componente da ação:

```math
A_{tgt}=WA_{src}.
```

Reproduzir bit a bit exige definições idênticas de nós da B-spline uniforme, condições de contorno, ordem das funções de base e precisão de ponto flutuante.

### 15.3 Ordem de execução

A ordem de processamento da ação é fixa:

```math
\text{relative-pose conversion}
\longrightarrow
\text{normalization}
\longrightarrow
\text{optional gripper binarization}
\longrightarrow
\text{resampling}
\longrightarrow
\text{unified-slot mapping}.
```

(conversão para pose relativa → normalização → binarização opcional da garra → reamostragem → mapeamento nas fatias unificadas)

Só o quadro de estado atual é usado; os estados não são reamostrados no tempo.

---

## 16. Checklist de Reprodução

Uma implementação conforme deve satisfazer todos os itens abaixo:

1. ações têm exatamente 54 dimensões e estados têm exatamente 60 dimensões;
2. o estado do efetuador usa xyz absoluto + rotação-6D;
3. a rotação-6D usa as duas primeiras colunas da matriz de rotação;
4. as ações de efetuador e de pose da base usam $T_t^{-1}T_{t+k}$;
5. poses relativas são representadas como xyz + vetor de rotação;
6. as estatísticas de ação relativa são globais, obtidas achatando as dimensões de amostra e de horizonte;
7. os normalizadores de ação relativa usam as estatísticas `global_*`;
8. poses relativas usam, por padrão, normalização Z-score;
9. as demais ações e estados usam, por padrão, normalização min-max por q01/q99;
10. no estado de pose do efetuador, só o xyz é normalizado; a rotação-6D fica inalterada;
11. a fatia de estado da base `[41:47]` contém a direção da gravidade no referencial do corpo e a velocidade angular normalizada nos três eixos;
12. as ações de mão hábil usam o tipo de normalização da garra;
13. as estatísticas só são mescladas entre tarefas diferentes do mesmo corpo de robô, com peso igual por tarefa;
14. os q01/q99 mesclados usam um envelope, e não os quantis verdadeiros da distribuição misturada;
15. as estatísticas mescladas esquerda/direita devem produzir offsets e escalas idênticos para os dois braços;
16. a semântica de coordenadas esquerda/direita deve estar alinhada antes de mesclar as estatísticas;
17. slots indisponíveis mantêm valor 0, offset 0 e escala 1, e são excluídos pelas suas máscaras.

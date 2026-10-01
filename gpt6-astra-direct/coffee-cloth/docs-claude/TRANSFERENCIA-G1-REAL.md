# O que transfere da simulação do café para o G1 real

Levantamento feito na madrugada de 21/09/2026 a partir dos transcripts e das notas do vault
(`Prometheus VLA/04 - Pesquisa/2026-09-02 Três câmeras no G1, Isaac com punhos e café.md`),
do estado do `coffee-cloth` e do `prometheus-vla`.

## O que existe hoje na simulação

Cadeia em MuJoCo com pelve fixa, mão direita dosando pó com colher e mão esquerda operando a
chaleira. Componentes já aprovados, alguns em ramos separados:

| etapa | estado | número que sustenta |
|---|---|---|
| dosagem 20 g de pó | aceita | 19,8513 g no filtro, 0 derramado, 21 ciclos |
| soltar a colher | aceita (portões corrigidos por mim) | hold_drift 0,0 por 2 s, mão 9 cm acima |
| alcançar o botão | aceita | erro de ponta 0,056 mm |
| conexão da mão ao botão | aceita | recuo de 40 mm válido, 5° de orientação |
| **prensar o botão** | **travado** | balancim chega a 0,116 rad, precisa 0,14 |
| ferver 800 ml | aprovado em ramo separado | 25 → 99,9996 °C, 362 486 J elétricos |
| pegar a alça e levantar | aprovado em ramo separado | subida 79,8 mm, drift 0,15 mm em 2 s |
| servir 245 ml | aprovado em ramo separado | 244,6188 ml, 0 derramado |
| devolver a jarra | aprovado em ramo separado | base 13,16 N contra 13,10 N de peso |

Os ramos separados nunca rodaram na mesma cadeia com os 20 g de pó: foram validados com 0,44 g.

## O que transfere, o que não transfere

**Transfere.** A sequência de estados da tarefa e os critérios de aceite de cada etapa. As poses
de aproximação e os ângulos de despejo (inclinação de 85° na jarra, alavanca de 18 mm no botão).
A descoberta de que dosar 20 g exige ~21 ciclos de colher com a Dex3, o que dá o orçamento de
tempo da tarefa real. E a auditoria de limiares: o projeto tinha portões abaixo do ruído do
próprio simulador, e no robô real o equivalente seria confiar em sensor abaixo da resolução dele.

**Não transfere.** Toda a física reduzida: água sem CFD, pó como massa agregada com ângulo de
repouso, calor por balanço de energia, pano como flex2D sem permeabilidade. A pelve fixa some no
robô real, onde o equilíbrio de corpo inteiro entra. E, principalmente, **o controlador usa TF do
simulador para saber onde os objetos estão**: no robô real isso não existe, tem que vir de
percepção.

## O caminho de verdade, segundo o que o lab já levantou

A nota de 02/09 é clara sobre as opções. O `unitree_sim_isaaclab` já traz o G1 29 DoF com Dex3
e câmeras de punho configuradas, falando DDS igual ao robô real, o que dá um caminho de
sim-to-real bem mais curto que o MuJoCo caseiro. A referência mais próxima da meta é o UniDex
(mar/2026): "Make Coffee" = pegar chaleira e despejar no dripper, com **50 demos de teleop por
tarefa** sobre pré-treino de 9 M frames de vídeo humano, 81% de progresso médio. Ou seja, o
estado da arte chega lá por demonstração, não por planejamento analítico.

Isso importa para a decisão: o `coffee-cloth` é um planejador com IK analítica e portões, ótimo
para entender a tarefa e medir o que ela exige, mas não é o artefato que vai rodar no G1. O que
roda no G1 é política treinada, e o pipeline de teleop por Quest 3 já existe no lab (a ponte ZMQ
hoje só publica `head_camera`, falta ligar as duas câmeras dos punhos, que é pré-requisito para
dataset com visão de punho).

## Bloqueio de hardware conhecido

O braço esquerdo do G1 está com kp=0 (registro do redesenho de 01/09). A cadeia do café usa
**as duas mãos**: direita para a colher, esquerda para botão e chaleira. Sem o braço esquerdo
funcionando, a tarefa completa não roda no robô real, independentemente de software.

## Ordem que eu proporia

1. Fechar a cadeia em simulação até café pronto, que é o que está em andamento, para ter a
   especificação completa da tarefa e os critérios de sucesso medidos.
2. Portar a cena para o `unitree_sim_isaaclab`, que já fala o mesmo DDS do robô.
3. Resolver o braço esquerdo e ligar as câmeras dos punhos na ponte de teleop.
4. Coletar demos por teleop das subtarefas na ordem de dificuldade medida na simulação.
5. Só então treinar política, usando os critérios de aceite da simulação como métrica de eval.

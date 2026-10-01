# Como um humano faz café no coador de pano: critérios para o robô (16/09/2026)

Fontes lidas (texto) e ouvidas (legendas automáticas baixadas com yt-dlp; vídeo não assistido):
- Coffee&Joy, "Como usar o coador de pano": https://coffeeandjoy.com.br/coador-de-pano
- Unique Cafés, "Como fazer café no coador de pano": https://uniquecafes.com.br/como-fazer-cafe-no-coador-de-pano/
- Canal do Cafezeiro (barista), "COADOR DE PANO - COMO USAR DO JEITO CERTO": https://www.youtube.com/watch?v=OLKT-fjb8bg (legenda pt)
- Receita caseira, "COMO FAZER CAFÉ COM COADOR DE PANO, SIMPLES RÁPIDO E SABOROSO": https://www.youtube.com/watch?v=V1F6Vm589hE (legenda pt)
- Lojas (medidas de suportes): Facile nº 3 (19 cm, base 13, boca 8), Pressca mini (15,5 cm), Decorfy/Grão Cafés (mini), "suporte serve
  para canecas com até 12 cm de altura" (Camicado).

## Sequência que aparece em todas as fontes
1. Coador encaixado no suporte; xícara embaixo. Antes de tudo, escaldar: água quente pelo pano (e pela xícara), descartar essa
   água ("saturar o pano para ele não roubar o óleo/sabor do café; aquece a xícara").
2. Pó no coador com a superfície nivelada. Doses: barista 15 g para uma xícara; Coffee&Joy 20 g / 200 ml (2 a 3 colheres de sopa
   cheias); 10 g por 100 ml é a regra geral; receita caseira 4 colheres de sopa para 2 L (café fraco/doce).
3. Água "levantando fervura, espera ~1 min" (90 a 96 °C). Pré-infusão: "coloca um pouquinho de água, cobriu o pó, começou a
   filtrar, espera uns 30 segundos".
4. Despejo: "movimento circular, sem jogar direto na parede do pano, o círculo é bem central, jato não muito forte (senão a água
   corre por fora do pano sem extrair)"; "mais ou menos um dedo de água por volta", repetido até a medida da xícara; "aos poucos,
   bem devagar".
5. Esperar filtrar por completo; servir em seguida. Depois: borra no lixo/composteira, coador lavado só com água e guardado
   em pote com água na geladeira.

## O que isso vira em critério de aceite no simulador
- Etapa 1 (copo sob o filtro): copo em pé, centrado no eixo da ponta do pano (< 1,5 cm), solto, sem tocar o pano, sem derrubar
  o suporte. [78 cumpre; robustez a posições iniciais em teste]
- Etapa 2 (pó): dose entre 10 e 20 g; pó cai DENTRO do pano, superfície aproximadamente nivelada (o robô pode dar uma sacudida
  leve ou simplesmente despejar no centro); nada de pó fora do pano; scoop devolvido ao lugar.
- Etapa 3 (água): antes, escaldar o pano e descartar na xícara/descartar a água; água só depois do "fervido" (botão da base +
  espera declarada); pré-infusão: água só até cobrir o pó, pausa de 30 s; despejo em fio central, movimento circular de raio
  pequeno (não na parede do pano), jato fraco, em voltas de "um dedo" (cerca de 30 a 50 ml por volta), até 150 a 200 ml; sem
  derrame fora do pano; esperar drenar antes de mexer no copo.
- Geral: o bico da chaleira nunca toca o pano; a xícara não sai do lugar durante o despejo.

## O que falta saber do laboratório
Posição da haste do suporte real, altura livre para a xícara (os suportes vendidos aceitam canecas de até 12 cm), onde o scoop
fica guardado e quantos gramas ele mede.

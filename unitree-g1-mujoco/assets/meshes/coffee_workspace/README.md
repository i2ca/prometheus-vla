# Objetos da estação de café

Convertidos dos arquivos `chaleira.glb`, `coador.glb`, `pote.glb`, `tampa.glb`
e `scoop.glb` fornecidos em `/home/felipe/Downloads/assetsWorkspace`.
Os OBJ conservam UVs e os PNG contêm as texturas embutidas originais.
Materiais e posições são definidos em `assets/coffee_workspace.xml`.
Todos os assets necessários estão neste repositório; Downloads não é usado
durante a simulação.

As transformações do GLB foram aplicadas às malhas, convertendo Y para Z
como eixo vertical, centralizando X/Y e colocando a base em Z=0.
Escala uniforme aproximada, usando a maior dimensão de cada objeto:

| Objeto | Maior dimensão | Posição X/Y (m) |
| --- | --- | --- |
| Chaleira | 0,22 m | 0,32 / 0,28 |
| Coador | 0,16 m | 0,34 / -0,22 |
| Pote | 0,16 m | 0,29 / -0,34 |
| Tampa | 0,14 m | 0,48 / -0,23 |
| Scoop | 0,14 m | 0,275 / 0,02 |

Todos são dinâmicos: juntas livres, gravidade e atrito permitem empurrar e
agarrar os objetos por contato. Nascem em Z=0,752 m, 2 mm acima do tampo,
para assentar sem penetração inicial. As colisões são caixas aproximadas dos
limites das malhas; aberturas e alças não têm colisão detalhada. As caixas
evitam tremulação causada pela superfície irregular dos scans a 250 Hz.

Massas estimadas: chaleira 600 g, coador 150 g, pote 250 g, tampa 60 g e
scoop 30 g. Atrito deslizante 1,5; amortecimento das juntas 0,01.
Dimensões e massas são aproximações, não medições dos objetos reais.
O pote e a tampa mantêm suas proporções originais.

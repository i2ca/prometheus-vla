# Nota sobre os episódios direct-astra-01 a 07

Nesses episódios o gpt-6-astra não recebeu as imagens das câmeras. O gateway usado (OmniRoute 3.8.50) tem um
"Vision Bridge" que troca cada imagem por uma legenda de 2 ou 3 frases feita pelo gpt-4o-mini quando o modelo de
destino não está na lista interna de modelos com visão, e o gpt-6-astra não estava. O problema foi achado em
23/09/2026 (imagem de 848x480 entrava como ~90 tokens e o próprio modelo respondia que só recebeu descrição) e
corrigido registrando a capacidade de visão do modelo no gateway. A partir do direct-astra-08 a política recebe
as imagens de verdade.

O que continua válido nos episódios 01 a 07: a verdade do simulador, os diagnósticos do crítico (feitos pelos
números do simulador) e as mudanças no arnês. O que não vale: qualquer conclusão sobre a percepção visual do
modelo. O episódio direct-omniroute-01 (gpt-5.6-sol) recebeu as imagens normalmente.

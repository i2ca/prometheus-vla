# Café brasileiro com coador de pano — Unitree G1

Objetivo ativo solicitado por Luiz em 2026-09-16: iterar até reproduzir o preparo na simulação. Responsável: gpt-6-astra. Não executa hardware.

## Referência humana

- [Unique Cafés, passo a passo](https://uniquecafes.com.br/como-fazer-cafe-no-coador-de-pano/): escaldar o tecido, adicionar café, molhar e esperar aproximadamente 30 segundos; completar a água com movimentos circulares na região central.
- [Coffee&Joy](https://coffeeandjoy.com.br/coador-de-pano): exemplo de20g de pó e200ml de água para preparo individual; pré-infusão e despejo lento. A bebida recolhida será menor que a água adicionada por retenção no pó/pano.
- [Pressca](https://www.pressca.com.br/produtos/mini-coador-de-cafe/): exemplo de coador de algodão em suporte metálico, boca8cm, altura total15,5cm; água em fio com movimento circular.
- [Vídeo humano localizado](https://www.youtube.com/watch?v=yCM0jZnagak): “Como Fazer Café no Coador de Pano”. Localizado pela busca; vídeo não assistido integralmente nesta sessão, não usado para alegar trajetórias articulares medidas.

Há variantes regionais, inclusive misturar pó e água antes de filtrar. Para este experimento adotamos pó dentro do coador apoiado em suporte. As medidas são um protocolo de teste, não uma receita brasileira única.

## Etapas e critérios

1. Layout/alcance: utensílios acessíveis, sem colisão com tronco/mesa.
2. Pega e inclinação: recipiente livre, segurado por contato; alcançar bico sobre boca do filtro, inclinar e retornar sem queda.
3. Água: medir volume que chega ao filtro, derrame, transbordamento e massa conservada; distinguir modelo reduzido de fluido validado.
4. Montagem/escaldamento: posicionar filtro e recipiente, enxaguar e descartar água de enxágue.
5. Dosagem: transferir20g de pó ao filtro, sem confundí-lo com decoração da cena.
6. Extração: pré-infusão30s, despejo circular até200ml de água total, drenagem.
7. Serviço: retirar/posicionar a xícara sem derrame, registrar vídeo e métricas.
8. Robustez: repetir variando poses e parâmetros com falhas preservadas.

## Limites atuais

Ainda não há café preparado. Primeira etapa é diagnóstico de alcance/pega, aproveitando a tentativa31 do projeto irmão. O copo existente serve apenas de recipiente de ensaio; não equivale a uma chaleira segurada pela alça. A dinâmica térmica e a extração química ainda não estão implementadas. Nenhuma política neural foi treinada.

Todas as tentativas terão diretórios novos em results/, parâmetros, métricas, hashes e modelo executor. Nunca sobrescrever tentativas. Continuar por CHECKPOINT.md.

## Rodando fora da máquina original

Os XMLs de `scene/` usam caminhos absolutos com o prefixo `/opt/i2ca/robotics-lab` (o original foi anonimizado). Pra rodar, aponte o prefixo pro seu checkout:

    sudo mkdir -p /opt/i2ca && sudo ln -s <este-repo>/gpt6-astra-direct /opt/i2ca/robotics-lab

e renomeie, ou troque o prefixo com `sed -i 's#/opt/i2ca/robotics-lab#<seu-caminho>#g' scene/*.xml`. Os `coffee-cloth/` e `g1-cup-grasp/` desse prefixo correspondem a `coffee-cloth/` e `runtime/g1-cup-grasp/` aqui. Os resultados (`results/`, `results-claude/`, ~52 GB) não subiram.

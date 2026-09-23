# GPT-6 Astra Direct no Unitree G1

Este subprojeto executa o `gpt-6-astra` como **politica visuomotora no loop** do
MuJoCo. Em cada chamada, o modelo recebe tres imagens RGB e a pose medida da
palma. Ele devolve a proxima pose cartesiana curta e o comando da garra. A
cinematica inversa e o controlador PD apenas convertem esse comando em juntas e
torques do G1.

Isso preserva a separacao do experimento original GPT-as-Policy:

- o modelo nao recebe a posicao verdadeira do objeto;
- o modelo decide cada movimento curto do efetuador, em vez de escolher uma
  primitiva pronta;
- colisoes, gravidade, massa, contatos e limites de torque sao calculados pelo
  MuJoCo;
- observacoes, chamadas do modelo, acoes aceitas/rejeitadas e estado fisico sao
  gravados sem sobrescrever tentativas anteriores.

## Estado desta branch

A primeira entrega contem o cliente oficial da Responses API e o contrato
auditavel da policy. O ambiente MuJoCo reproduzivel e os artefatos do cafe sao
adicionados em commits separados na mesma branch.

## Instalacao

```bash
cd gpt6-astra-direct
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
export OPENAI_API_KEY='...'
```

`OPENAI_BASE_URL` pode apontar para um gateway compativel com a API da OpenAI.
Sem essa variavel, o SDK usa a API oficial.

## Contrato

Uma acao move a palma no maximo 5 cm e 0,35 rad a partir da pose **medida**.
`steps` fica entre 1 e 5, com 0,2 s por passo. O modelo pode chamar:

- `actuate`: pose absoluta da palma, orientacao `wxyz` e abertura da garra;
- `finish_episode`: encerra quando a tarefa terminou ou nao pode continuar.

Veja [docs/CONTRATO-DIRECT.md](docs/CONTRATO-DIRECT.md) para a fronteira completa
entre a policy e o controlador.

## Execucao

Depois que o ambiente estiver instalado:

```bash
astra-direct \
  --episode results/direct-$(date +%Y%m%d-%H%M%S) \
  --cup 0.30,-0.20 \
  --model gpt-6-astra \
  --reasoning-effort xhigh
```

Cada episodio ganha um diretorio novo. O runner se recusa a reutilizar uma
pasta existente.


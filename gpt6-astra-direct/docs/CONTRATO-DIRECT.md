# Contrato da policy Direct

## Responsabilidade do GPT-6 Astra

O modelo ve, a cada ciclo:

1. RGB da camera da cabeca;
2. RGB das cameras dos dois pulsos;
3. pose cartesiana medida da palma ativa;
4. juntas do braco, cintura e fracao de fechamento da garra;
5. resultado observavel da acao anterior;
6. limites restantes de chamadas e tempo simulado.

O modelo escolhe diretamente a proxima pose absoluta da palma e a garra. A
acao e curta e replanejada depois de uma nova observacao. O campo `reason`
registra apenas a evidencia visual e o proposito imediato; nao solicita nem
armazena raciocinio oculto.

## Informacao proibida para a policy

O runner nao envia arquivos de `truth/`, pose real do objeto, contatos internos,
forcas, estado do simulador ou metricas futuras. Esses dados ficam disponiveis
somente para auditoria posterior.

## Responsabilidade deterministica do ambiente

O ambiente:

- valida deslocamento, rotacao, formato e orcamento;
- resolve a cinematica inversa;
- interpola o alvo e aplica PD em torque com saturacao do atuador;
- integra a fisica no MuJoCo;
- interrompe em contato perigoso da mao com mesa, tronco ou outro braco;
- renderiza a observacao seguinte;
- salva estado, acao, imagens, video e dados privilegiados de avaliacao.

IK e PD nao escolhem a tarefa nem geram trajetoria semantica. Eles sao a camada
de execucao de baixo nivel entre a acao cartesiana do modelo e os motores.

## Limites por chamada

| Campo | Limite |
|---|---:|
| translacao da palma | 0,05 m |
| rotacao da palma | 0,35 rad |
| `steps` | 1 a 5 |
| duracao de um `step` | 0,2 s |
| velocidade articular | 240 graus/s |

Uma rejeicao nao altera a fisica. Ela e registrada e consome uma chamada da
policy, permitindo que o modelo corrija o comando com a mesma observacao.

## Persistencia e autoria

Cada episodio e criado com `exist_ok=False`. O manifesto registra
`policy_model=gpt-6-astra`, effort, identificadores das respostas e uso de
tokens. Nenhuma tentativa e sobrescrita.


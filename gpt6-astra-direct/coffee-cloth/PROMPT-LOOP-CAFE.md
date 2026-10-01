Trabalhe em iterações até o robô G1 preparar um café coado em pano inteiro em simulação, com os objetos reais do LCAD. A cada iteração escolha o gargalo mais importante, meça antes de mexer, faça UMA mudança, rode, audite com números e com os quadros do vídeo, registre, e siga. Não pare para me perguntar: decida com os dados, registre a decisão e continue. Pare só se um resultado exigir uma medida física que só eu tenho, e nesse caso continue por outro caminho enquanto isso.

CONTEXTO E CAMINHOS
Pacote principal: ~/I2CA/robotics-lab/g1-cup-grasp (controlador run_attempt.py, kinematics.py, vision.py, g1_sim.py, recorder.py, audit_attempt.py, make_dataset.py; venv em .venv; MUJOCO_GL=egl obrigatório em tudo que renderiza).
Projeto do café: ~/I2CA/robotics-lab/coffee-cloth (cenas em scene/, objetos convertidos em assets/mesh/, desenhos técnicos em assets/desenhos/, medidas em MEDIDAS-REAIS.md e assets/dimensions.json, procedimento humano em research/PROCEDIMENTO-HUMANO.md, histórico em CHECKPOINT.md).
Leia antes de agir: CHECKPOINT.md (do fim para o começo), MEDIDAS-REAIS.md, research/PROCEDIMENTO-HUMANO.md e a seção mais recente de g1-cup-grasp/RELATORIO-PARA-O-ORQUESTRADOR.md.

ESTADO QUANDO ESTE LOOP COMEÇOU (16/09/2026, noite)
Funciona: pegar a caneca da mesa e colocá-la sob o filtro, com o coador que eu tinha estimado (tentativa 78, pacote em coffee-cloth/results/etapa1-copo-sob-filtro-tentativa78; 22 s, 2,8 m/s² de pico, 6 mm de erro, zero contato da mão com o coador). Robustez medida: 10 de 20 posições iniciais (results/sweep-place-02).
Não funciona: a mesma tarefa com o coador das medidas reais (SUP-001: haste 205 mm, base 80 mm, filtro 70 mm). A base tem o diâmetro da xícara e a haste sai da borda. O transporte entrega o copo a 2 mm do alvo, mas descer os últimos 4,4 cm perde 2,5 a 3 cm porque o braço estendido afunda sob carga, e isso basta para a xícara ficar meio fora e tombar. Tentativas 88 a 100, todas registradas.
Impossível com esta mão: tirar a tampa do pote pela torre central (20 mm de diâmetro por 8 de altura). Varri 100 poses de aproximação e todas penetram a tampa entre 13 e 35 mm. Trate o pote como já aberto, ou proponha e implemente outra forma de servir o pó.

PRIORIDADE 1, O CONTROLADOR
Implemente compensação de gravidade e de carga no espaço da tarefa: em vez do PD por junta com qfrc_bias, calcule o torque necessário para sustentar a carga na mão usando a jacobiana transposta, ou implemente controle de impedância cartesiana. Sem isso, toda fase de descida ou de despejo com peso vai errar centímetros, e as etapas seguintes (jarra com água, 1 kg mais líquido) são piores que esta. Valide com a etapa 1 no coador real: alvo é erro abaixo de 15 mm, copo em pé (menos de 3°), zero contato da mão com o coador, pico de aceleração da palma abaixo de 3 m/s². Depois revalide a tentativa 78 e a varredura de 20 posições, que não podem piorar.

ETAPAS DO CAFÉ E CRITÉRIOS (derivados de research/PROCEDIMENTO-HUMANO.md)
1. Xícara sob o filtro: copo em pé, eixo a menos de 30 mm do eixo da ponta do pano, solto, sem tocar o pano, suporte não se move.
2. Pó no filtro: 10 a 20 g dentro do pano, superfície aproximadamente nivelada, nada de pó fora, utensílio devolvido. Se o scoop for inalcançável na mesa (o cabo fica a 3 cm e nenhuma pinça alcançou), estude despejar direto do pote pelas alças (raio de semicírculo 15 mm, projeção de 18 mm) ou reposicione o scoop apoiado na borda do pote e registre a decisão.
3. Água quente: o robô aperta o botão da base de carregamento e espera um tempo declarado (sem termodinâmica), depois pega a jarra pela alça. A jarra tem 232 mm de altura, 110 de base, 90 de boca e 1 kg vazia; hoje está a 58 cm do robô, fora do alcance, então reposicione a cena e prove o alcance offline antes de mover o braço.
4. Escaldar: despejar água pelo pano, descartar essa água, antes do pó.
5. Pré-infusão: água só até cobrir o pó, pausa de 30 s.
6. Despejo: fio central, círculo pequeno de raio de 1 a 2 cm, sem tocar a parede do pano, cerca de 30 a 50 ml por volta, até 150 a 200 ml; sem derrame fora do pano; o bico nunca toca o tecido; a xícara não sai do lugar.
7. Serviço: esperar drenar e entregar a xícara.
Para água use o modelo reduzido conservativo que o Astra escreveu (coffee-cloth/scripts/liquid.py, water_trial.py) e declare em todo relatório que é volume conservado, não fluido.

REGRAS DE TRABALHO (vêm de erros que já custaram caro)
Uma mudança por tentativa, em diretório novo que nunca se sobrescreve, com o campo reason dizendo o que mudou e por quê, a partir da observação da tentativa anterior.
Meça antes de varrer parâmetros: quando algo não pega ou colide, calcule a geometria offline (mj_geomDistance entre os geoms da mão e do objeto, mapas de alcance por IK) em vez de tentar parâmetros no escuro. Poses de teste que já começam com penetração produzem resultados falsos.
O que se vê tem que ser o que colide: visual e colisão do mesmo objeto, e desconfie do CoACD (já criou peça fantasma no meio do vão do coador).
Percepção só por câmera (RGB com os marcadores), física só para verificar, e nunca misture.
Auditoria antes de declarar pronto: rode audit_attempt.py, olhe as folhas de contato das seis câmeras E extraia quadros do vídeo a cada 1 ou 2 s para olhar de verdade. Responda com número: o que colide com o quê, o objeto escorrega ou tomba, há tranco (aceleração e jerk da palma), a câmera da cabeça vê a tarefa, o movimento é humano, quanto tempo cada fase leva e quanto tempo o robô fica parado.
Não afrouxe critério para passar. Se um critério estiver errado, mostre o número que justifica mudá-lo e registre.
Todo vídeo sai com a faixa de identificação (recorder.py já faz; para vídeos antigos use add_title_band.py).
Registre no fim de cada iteração: uma entrada em coffee-cloth/CHECKPOINT.md com o que mediu, o que mudou e o que ficou aberto. Quando fechar uma etapa, monte um pacote reprodutível (vídeo com faixa, parameters.json, cena, scripts com SHA256, auditoria, README com as medidas e as ressalvas) em coffee-cloth/results/.

ARMADILHAS JÁ CONHECIDAS
MUJOCO_GL=egl precisa estar no ambiente dos processos em segundo plano, senão a varredura inteira falha no fim.
Não rode simulações com renderização em paralelo com uma varredura: os contextos de OpenGL brigam e tudo falha.
A correção proprioceptiva em bloco empurra o objeto quando ele já encosta; a correção a cada quadro confunde atraso do movimento com erro estacionário e faz a palma recuar. Use compensação de verdade, não gambiarra de alvo.
O offset do objeto na mão (carry_offset) muda quando a força de fecho muda; remeça sempre que mexer no fecho.
Os modelos 3D não têm as proporções reais: confira sempre contra assets/desenhos e use a escala anisotrópica do convert_glb.py.
Quando a alça da caneca aponta para a haste do suporte, ela bate; a orientação da alça na colocação depende do yaw inicial e dos giros do tronco.

FIM DE CADA ITERAÇÃO
Escreva no máximo dez linhas: o que mediu, o que mudou, o número que melhorou ou piorou, e qual é o próximo gargalo. Se a etapa fechou, diga qual e mostre o caminho do pacote. Continue na próxima iteração sem esperar resposta.

# Retomada do Claude (22/09/2026, ~19h): cadeia do café inteira, do zero, com postura corrigida

## Onde parou
Cadeia completa passa do zero numa execução só: 199,8 ml na xícara, zero derramado, 5 ml retidos no pano.
Vídeo: `results-claude/cadeia-natural5/video/cafe-completo.mp4`. Repositório: github.com/lewislf/robotics-lab
(raiz em `~/I2CA/robotics-lab`, results fora do git).

Etapas (`scripts/run_cadeia_completa.py`):
0. `relaxar_braco --side both --waist-to 0 0 0`: braços caídos e tronco no centro depois da colher.
1-3. alcance, conexão e prensa do botão com cintura ≤ 6° e 0,35 rad, `--free-orientation --human-weights
   --fix-idle-arm --elbow-lateral-m 0.10`; candidatos testados do menor ao maior esforço (prensa SEM `--boil`).
3b. `relaxar_braco --side both --waist-to 0 0 0 --until-boil`: braços caídos, tronco reto, espera ferver.
4. `plan_kettle_grasp_v3 --seed results-claude/pega-referencia-validada --fixed-hand --human-weights
   --elbow-comfort-deg 50 --fix-idle-arm`: pega no referencial da chaleira, cotovelo dobrado.
5-6. `plan_kettle_connection_v2` e `execute_kettle_lift_v2` com `--human-weights --fix-idle-arm`.
7-9. `plan_kettle_pour_v2`, `execute_coffee_pour_v2 --return-lift 0.03`, `execute_coffee_place_v2`, todos `--fix-idle-arm`.

## Decisões confirmadas (com números no caderno `licoes/`)
- Pega guardada no referencial da chaleira: taxa de ~2% para 50-100% com a chaleira deslocada 5 cm (x+5 e y-5 passam a cadeia).
- Pega envolvendo a barra da alça NÃO existe para a Dex3 nesta chaleira (0 de 21.600 poses): polegar perpendicular
  à palma e vão de 25 mm. A mão acima do ombro no despejo é geometria; só muda com outra chaleira ou coador mais baixo.
- Cotovelo parado era a trava de 75° da junta + IK de menor mudança; `--elbow-comfort-deg 50` resolve.
- Pesos por junta só funcionam via `x_scale` no least_squares (`postura_humana.py`).
- Portões de penetração e auto-contato a 0,5 mm mediam o `solimp`, não dano (1 mm e profundidade).

## Abertos
- Chaleira girada 30°: braço não alcança a pega de referência (falta biblioteca de pegas ou reposicionar o corpo).
- Prensa: punho ainda torce ~51° na ida ao botão (antes 16°); cintura e úmero resolvidos.
- Tudo usa estado privilegiado do MuJoCo; percepção pelas câmeras não existe ainda.
- Linha do astra (gpt-6-astra) intacta: nenhum script original foi alterado, só cópias `_v2`/`_v3` e scripts novos.

## Como rodar
```bash
cd ~/I2CA/robotics-lab/coffee-cloth
../g1-cup-grasp/.venv/bin/python scripts/run_cadeia_completa.py results-claude/<nova-pasta>
../g1-cup-grasp/.venv/bin/python scripts/medir_postura.py --fases <pasta-de-etapa>
```
Um processo MuJoCo por vez nesta máquina (14 GB, 1,6 GB cada).

---
# Retomada do Claude (16/09/2026, ~13h30): etapa 1 do café, copo sob o filtro

Leia antes: `results/etapa1-copo-sob-filtro-tentativa50/AUDITORIA-MINUCIOSA.md` (o que estava errado na 50) e a seção 25 do
`../g1-cup-grasp/RELATORIO-PARA-O-ORQUESTRADOR.md`.

## Onde parou
- (SUPERADO: ver seção "16/09 ~15h" do CHECKPOINT.md; etapa 1 aceita na tentativa 68.) Correção em curso: "colocar é a pega ao contrário". `run_attempt.py` ganhou a fase `turn_to_place` (gira o tronco para o
  suporte com o copo na mão, palma fixa no mundo pela IK) e `place_approach_dir: "palm"` (o copo entra à frente da palma).
  Cena `scene/setup-luiz-013.xml`: coador em (0,36, -0,16), haste para +x; copo parte de (0,26, 0,08).
- Tentativa 58 (`../g1-cup-grasp/results/attempt-58`, parâmetros em `experiments/attempt-58.parameters.json`):
  **copo a 2,3 mm do alvo em xy** (era 2,7 cm), sem autocolisão, sem mão na mesa, mas:
  1. o copo termina inclinado 25° e com o fundo a 1,5 cm acima da base (z 0,769): ficou apoiado torto, provavelmente na
     borda do disco da base ou no cone; o indicador ainda toca o coador durante release (54 quadros) e recuo. Ver
     `trajectory.json` (env_contacts por geom e posição) nas fases set_down/release.
  2. no giro inicial (turn, +42°) a mão em repouso ainda varre o coador (80 quadros punho x coador, 34 indicador):
     o coador em (0,36, -0,16) continua no raio da mão. Opções: coador mais longe da pelve (mas x <= 0,36 pelo alcance),
     ou pose de foto com a mão mais alta/recolhida, ou girar menos (copo mais perto do centro).
  3. slide com erro de IK 28,8 mm e tranco 7 m/s²; set_down 25,9 mm. Checar se a haste/base bloqueia ou se é limite de junta.
- Próximos passos exatos: (a) ler contatos de set_down/release da 58 por geom; (b) resolver o apoio torto (folga de descida,
  soltar mais devagar, dedos abrindo na ordem indicador/médio antes do polegar); (c) tirar o coador do raio do giro; (d) só
  então avaliar com o critério de 1,5 cm e a auditoria completa (folhas agora cobrem o vídeo todo).
- Não feito nesta etapa: pó no filtro, botão da chaleira, despejo. Chaleira em (0,58, -0,36) está fora do alcance do braço.

## Como rodar
```bash
cd ~/I2CA/robotics-lab/g1-cup-grasp
.venv/bin/python scripts/run_attempt.py --parameters experiments/attempt-58.parameters.json --run-dir results/attempt-59
.venv/bin/python scripts/audit_attempt.py results/attempt-59
```
Uma mudança por tentativa, `reason` escrito, nunca sobrescrever. Cena nova = `coffee-cloth/scripts/build_setup_scene.py --out scene/setup-luiz-014.xml`
(o construtor renderiza a foto inicial e só aceita a cena se a calibração passar).

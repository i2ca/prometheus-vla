# Overlays visuais do surrogate de água

Worker: modelo REQUISITADO `claude/claude-fable-5` via Claude Code + OmniRoute,
orquestrador gpt-6-astra. Evidência de backend: nenhuma verificável em runtime;
o harness se auto-reporta como Claude Fable 5. Não há como o script confirmar a
identidade do modelo, e isso está registrado no report.

Data: 2026-09-16. MuJoCo 3.13.0, MUJOCO_GL=egl, cena `scene/coffee-006.xml`.

## API

`water_visuals.add_water_visuals(renderer, sim, water, latest) -> None`

- `renderer`: `mujoco.Renderer`; chamar DEPOIS de `renderer.update_scene(...)`
  e ANTES de `renderer.render()`. Como `update_scene` reconstrói a MjvScene a
  cada frame, os geoms transientes não vazam entre frames (verificado).
- `sim`: qualquer objeto com `.m` (MjModel) e `.d` (MjData); o `G1Sim` do
  trial serve direto.
- `water`: instância de `liquid.WaterTransfer`, fonte da verdade dos volumes.
- `latest`: último dict retornado por `water.step()`; aceita `{}` antes do
  primeiro step.

Integração no `water_trial.py` (não aplicada por mim, arquivo é do
orquestrador): inserir `add_water_visuals(renderer, sim, water, latest)` entre
`renderer.update_scene(d, camera='coffee_closeup')` e `renderer.render()`.

## O que é desenhado

Só geoms transientes na `MjvScene` (`mjv_initGeom` + `mjv_connector`), sem
nenhuma edição no modelo físico, volumes ou passo de simulação.

1. Água no receiver: cilindro semitransparente quando `receiver_ml > 0.05`.
   Altura = `receiver_ml * 1e-6 / (π * water.radius²)`, limitada à altura
   interna do copo (`water.top - water.bottom`, dimensões que a cena
   compartilha com o cilindro do surrogate). Raio 0.0365 m, 0.5 mm aquém da
   face interna da parede (0.037) pra evitar z-fighting. Posição e orientação
   seguem o frame do body `receiver`.
2. Jato: cápsula quando `latest['flow_ml_s'] > 0.01`, vertical a partir de
   `latest['lip']` (coerente com o jato vertical do surrogate). Destino:
   plano da boca do `filter_ring` se `latest['capture']`; senão o tampo da
   mesa, ou o chão se o pé do jato cair fora da projeção do tampo. Raio entre
   1.2 e 4.0 mm crescendo com `sqrt(flow / max_flow)`: indicação visual da
   vazão do surrogate, não hidrodinâmica medida.

Guard de capacidade: cada inserção checa `ngeom < maxgeom` e desiste em
silêncio se a cena estiver cheia (maxgeom da cena atual é 10000, uso é +2).

## O que foi omitido, e por quê

- Líquido no copo fonte inclinado: o recorte correto de um volume parcial
  contra um cilindro tombado é uma superfície cortada por plano; a MjvScene só
  tem primitivas inteiras, então qualquer cilindro ou elipsoide vazaria pelas
  paredes ou pela boca do copo. Conforme o combinado, omitido em vez de
  falsificado.
- Água retida no pano (`filter_ml`): fora do escopo pedido.
- Sem partículas, gotas ou espalhamento: o surrogate não modela isso e o
  overlay não finge que modela.

## Limitações conhecidas

- A superfície da água no receiver acompanha o eixo do copo, não a gravidade;
  se o receiver tombar, o nível desenhado não se inclina. No trial o receiver
  fica em pé (deslocamento aceito ≤ 3 mm), então isso não aparece.
- Na câmera `coffee_closeup` o preenchimento fica oculto pelas paredes opacas
  da caneca cerâmica; ele existe e está posicionado certo (checks numéricos),
  como água numa caneca opaca real. O jato é visível.
- O jato perdido termina no tampo ou no chão sem respingo nem poça; o volume
  derramado só existe como número (`spilled_ml`).
- A conversão volume→altura usa as dimensões do cilindro do surrogate
  (`radius=0.037`, `bottom=0.006`, `top=0.092`), que batem com a geometria do
  receiver na cena 006; `receiver_capacity_ml=350` do surrogate cabe (altura
  0.0814 ≤ 0.086).

## Verificação

`scripts/check_water_visuals.py`, standalone (carrega a cena com
`mj_forward`, não roda o trial nem replay). 13 checks, todos PASS, saída
salva em `results/claude-visuals-001/` (`report.json`, `full_capture.png`,
`miss_table.png`), criada com `exist_ok=False`:

- volume zero e fluxo zero (e `latest={}`) não adicionam geom nenhum e o
  render é finito;
- receiver cheio (350 ml) + fluxo capturado adiciona exatamente 2 geoms, com
  centro do preenchimento em (0.38, 0.07, 0.8047) e topo local 0.0874 dentro
  do copo, e ponto médio do jato em (0.39, 0.07, 0.96);
- jato perdido sobre a mesa termina em z=0.75, fora da mesa termina em z=0;
- dois frames consecutivos mantêm `ngeom` idêntico (sem vazamento);
- com `ngeom == maxgeom` a inserção não estoura a cena.

Arquivos deste worker: `scripts/water_visuals.py`,
`scripts/check_water_visuals.py`, este documento. Nada mais foi tocado.

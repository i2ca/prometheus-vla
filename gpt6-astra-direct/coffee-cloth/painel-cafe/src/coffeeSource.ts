/* Fonte de dados do painel do café.

O app original lia tópicos ROS 2 por rosbridge, porque no Isaac Sim a cena roda
continuamente. Aqui cada etapa é um processo MuJoCo que grava `report.json`, e o
orquestrador registra tudo em `progress.json`. Este módulo lê esse arquivo e
devolve os mesmos formatos que os componentes já esperam. */
import { type ChatMessage, type LogEntry, type RobotAction, type MetricsData } from './components/theme';

export interface Episodio {
  model: string;
  goal: string;
  fonte_atual: string;
  coffee_completed: boolean;
  justificativa_final?: string;
  parada?: string;
  turnos: Turno[];
}
interface Turno {
  turno: number;
  texto: string;
  chamadas: string[];
  resultados?: Record<string, any>[];
}

/* Qual mão executa cada ferramenta. A direita dosou o pó e largou a colher; a
   esquerda opera botão e chaleira. */
const MAO: Record<string, 'G1_ESQUERDA' | 'G1_DIREITA'> = {
  ligar_chaleira: 'G1_ESQUERDA',
  pegar_chaleira: 'G1_ESQUERDA',
  servir_cafe: 'G1_ESQUERDA',
  devolver_chaleira: 'G1_ESQUERDA',
};

export const ROTULO: Record<string, string> = {
  observar_cena: 'observar cena',
  ligar_chaleira: 'ligar chaleira',
  pegar_chaleira: 'pegar chaleira',
  servir_cafe: 'servir café',
  devolver_chaleira: 'devolver chaleira',
  concluir: 'concluir',
};

export interface EstadoDerivado {
  messages: ChatMessage[];
  logs: LogEntry[];
  actions: RobotAction[];
  actionResults: RobotAction[];
  metrics: MetricsData | null;
  episodio: Episodio;
  observacao: Record<string, any> | null;
  frameUrl: string | null;
}

export function derivar(ep: Episodio, episodioId: string): EstadoDerivado {
  const messages: ChatMessage[] = [];
  const logs: LogEntry[] = [];
  const actions: RobotAction[] = [];
  const actionResults: RobotAction[] = [];
  let seq = 1;
  const base = new Date();

  const robots: MetricsData['robots'] = {
    G1_ESQUERDA: { state: 'INIT', phase: '', action: '', target: '',
                   busy_pct: 0, idle_pct: 100, tasks_completed: 0, tasks_failed: 0 },
    G1_DIREITA: { state: 'COLHER_LARGADA', phase: 'dosagem concluída', action: '', target: '',
                  busy_pct: 0, idle_pct: 100, tasks_completed: 1, tasks_failed: 0 },
  };

  let observacao: Record<string, any> | null = null as Record<string, any> | null;
  let idxObservacao = -1;

  messages.push({ id: 'goal', role: 'user', text: ep.goal, ts: base });

  ep.turnos.forEach((t) => {
    const ts = new Date(base.getTime() + t.turno * 1000);
    if (t.texto) {
      messages.push({ id: `t${t.turno}`, role: 'vla', text: t.texto, ts,
                      senderName: 'Orquestrador', emoji: '🤖' });
    }
    (t.chamadas || []).forEach((nome) => {
      actions.push({ id: seq++, raw: JSON.stringify({ action: nome }), ts });
      logs.push({ id: seq++, level: 20, name: 'orquestrador',
                  msg: `turno ${t.turno}: chamando ${ROTULO[nome] || nome}`, ts });
      const mao = MAO[nome];
      if (mao) { robots[mao].state = nome.toUpperCase(); robots[mao].action = nome; }
    });
    (t.resultados || []).forEach((r) => {
      Object.entries(r).forEach(([nome, v]: [string, any]) => {
        actionResults.push({ id: seq++, raw: JSON.stringify(v), ts });
        const falhou = v && (v.sucesso === false || v.erro);
        logs.push({ id: seq++, level: falhou ? 40 : 20, name: nome,
                    msg: falhou ? `falhou: ${v.motivo || v.erro}` : JSON.stringify(v), ts });
        messages.push({ id: `r${seq}`, role: 'system',
                        text: `${ROTULO[nome] || nome} → ${falhou ? '❌ ' + (v.motivo || v.erro) : '✅ ' + JSON.stringify(v)}`, ts });
        const mao = MAO[nome];
        if (mao) {
          if (falhou) robots[mao].tasks_failed += 1; else robots[mao].tasks_completed += 1;
          robots[mao].phase = falhou ? 'falha' : 'concluída';
        }
        if (nome === 'observar_cena' && v && v.temperatura_C !== undefined) {
          observacao = v; idxObservacao = t.turno - 1;
        }
      });
    });
  });

  const ativo = !ep.justificativa_final && !ep.parada;
  Object.values(robots).forEach((r) => {
    r.busy_pct = ativo && r.action ? 100 : 0;
    r.idle_pct = 100 - r.busy_pct;
    r.target = observacao ? (r === robots.G1_ESQUERDA ? 'chaleira' : 'colher') : '';
  });

  const metrics: MetricsData = {
    timestamp: Date.now() / 1000,
    robots,
    tower_height: observacao ? Number((observacao as any).agua_na_xicara_ml || 0) : 0,
    center_occupied_by: null,
  };

  return {
    messages, logs, actions, actionResults, metrics, episodio: ep, observacao,
    frameUrl: idxObservacao >= 0
      ? `/results/${episodioId}/obs-${idxObservacao}/view.png?t=${Date.now()}` : null,
  };
}

export async function buscar(episodioId: string): Promise<Episodio> {
  const r = await fetch(`/results/${episodioId}/progress.json?t=${Date.now()}`);
  if (!r.ok) throw new Error(`progress.json: ${r.status}`);
  return r.json();
}

/* ── Estrutura real da tarefa ────────────────────────────────────────────
   O grafo do repo original tinha três agentes e três braços. A tarefa do café
   tem, por baixo de cada ferramenta, uma cadeia de planejamento e execução,
   modelos físicos próprios e uma camada de portões que pode vetar qualquer
   etapa. O que segue descreve essa estrutura para o grafo desenhar. */

export interface Etapa { id: string; rotulo: string; script: string; nota?: string }
export interface Ferramenta {
  nome: string; rotulo: string; mao: 'esquerda' | 'direita' | 'nenhuma';
  etapas: Etapa[]; requer?: string;
}

export const FERRAMENTAS: Ferramenta[] = [
  { nome: 'observar_cena', rotulo: 'observar cena', mao: 'nenhuma', etapas: [
      { id: 'obs', rotulo: 'render + estado', script: 'observe_state', nota: 'câmera da cabeça' } ] },
  { nome: 'ligar_chaleira', rotulo: 'ligar chaleira', mao: 'esquerda',
    requer: 'chaleira na base', etapas: [
      { id: 'hr', rotulo: 'alcançar botão', script: 'plan_heater_reach_v2', nota: '15 orientações' },
      { id: 'hc', rotulo: 'conectar ao botão', script: 'plan_heater_connection_v2', nota: 'recuo 40 mm' },
      { id: 'hp', rotulo: 'prensar e ferver', script: 'execute_heater_press_v2', nota: 'curso 30 mm, alvo −6 mm' } ] },
  { nome: 'pegar_chaleira', rotulo: 'pegar chaleira', mao: 'esquerda',
    requer: 'água fervida', etapas: [
      { id: 'kg', rotulo: 'pega na alça', script: 'plan_kettle_grasp_v2', nota: 'oposição dos dedos' },
      { id: 'kc', rotulo: 'aproximação', script: 'plan_kettle_connection_v2', nota: 'abertura 0' },
      { id: 'kl', rotulo: 'levantar', script: 'execute_kettle_lift_v2', nota: 'alocação de forças 20 N' } ] },
  { nome: 'servir_cafe', rotulo: 'servir café', mao: 'esquerda',
    requer: 'chaleira na mão', etapas: [
      { id: 'pp', rotulo: 'planejar despejo', script: 'plan_kettle_pour', nota: 'inclinação e bico' },
      { id: 'po', rotulo: 'despejar e drenar', script: 'execute_coffee_pour', nota: '245 ml, exige quente' } ] },
  { nome: 'devolver_chaleira', rotulo: 'devolver chaleira', mao: 'esquerda',
    requer: 'despejo feito', etapas: [
      { id: 'kp', rotulo: 'apoiar e soltar', script: 'execute_coffee_place', nota: 'transferência de peso' } ] },
];

export const MODELOS = [
  { id: 'po', rotulo: 'Pó de café', script: 'coffee_grounds.py',
    detalhe: 'densidade 0,35 g/ml · repouso 28° · eficiência de coleta 0,55' },
  { id: 'liquido', rotulo: 'Água', script: 'kettle_liquid.py',
    detalhe: 'capacidade por orientação · jato · captura · drenagem · transbordo' },
  { id: 'calor', rotulo: 'Aquecimento', script: 'kettle_heater.py',
    detalhe: 'balanço de energia · corte automático na fervura · intertravamento' },
  { id: 'contato', rotulo: 'Contato e preensão', script: 'contact_wrench_control.py',
    detalhe: 'pirâmides de atrito · limites de motor · sem força externa no objeto' },
  { id: 'brew', rotulo: 'Estado do preparo', script: 'brew_state.py',
    detalhe: 'massa de pó, água e temperatura acopladas ao MuJoCo a cada passo' },
];

/* Portões presentes no código, com o número de scripts que os aplicam.
   Levantado por varredura, não escrito de memória. */
export const PORTOES = [
  { id: 'g_anat', rotulo: 'anatomia do cotovelo', escopo: '34 scripts', limite: '5 a 145°, tolerância 2°' },
  { id: 'g_derrame', rotulo: 'derramamento', escopo: '33 scripts', limite: '0,5 ml / 0,05 g' },
  { id: 'g_pen', rotulo: 'penetração', escopo: '21 scripts', limite: 'corrigido para 1 mm' },
  { id: 'g_mesa', rotulo: 'folga mão/mesa', escopo: '17 scripts', limite: '8 a 20 mm' },
  { id: 'g_punho', rotulo: 'limites de punho', escopo: '14 scripts', limite: '62 / 47 / 32°' },
  { id: 'g_proib', rotulo: 'contato proibido', escopo: '12 scripts', limite: 'qualquer penetração com o robô' },
  { id: 'g_num', rotulo: 'aviso numérico', escopo: '11 scripts', limite: 'solver sem warnings' },
  { id: 'g_deriva', rotulo: 'deriva de props', escopo: '7 scripts', limite: '5 mm por run' },
  { id: 'g_quente', rotulo: 'folga do metal quente', escopo: '6 scripts', limite: '6 mm' },
];

export const OBJETOS = [
  { id: 'o_pote', rotulo: 'Pote de pó', chave: 'pote' },
  { id: 'o_colher', rotulo: 'Colher', chave: 'scoop' },
  { id: 'o_coador', rotulo: 'Coador de pano', chave: 'coador', nota: 'flex 2D, 65 vértices' },
  { id: 'o_xicara', rotulo: 'Xícara', chave: 'copo' },
  { id: 'o_chaleira', rotulo: 'Chaleira', chave: 'chaleira' },
  { id: 'o_base', rotulo: 'Base elétrica', chave: 'base_eletrica', nota: 'balancim 0,3 N·m/rad' },
];

export type StatusEtapa = 'ok' | 'falha' | 'pendente';

/* Descobre o que cada etapa fez, lendo os nomes de pasta que o orquestrador
   criou e o motivo de falha que ele registrou. */
export function statusDasEtapas(ep: Episodio): Record<string, { status: StatusEtapa; motivo?: string }> {
  const out: Record<string, { status: StatusEtapa; motivo?: string }> = {};
  const chamadasFeitas = new Set<string>();
  const falhas: Record<string, string> = {};
  ep.turnos.forEach((t) => {
    (t.chamadas || []).forEach((c) => chamadasFeitas.add(c));
    (t.resultados || []).forEach((r) => Object.entries(r).forEach(([nome, v]: [string, any]) => {
      if (v && v.sucesso === false) falhas[nome] = String(v.motivo || v.erro || '');
    }));
  });
  FERRAMENTAS.forEach((f) => {
    const chamada = chamadasFeitas.has(f.nome);
    const motivo = falhas[f.nome];
    f.etapas.forEach((e, i) => {
      if (!chamada) { out[e.id] = { status: 'pendente' }; return; }
      if (!motivo) { out[e.id] = { status: 'ok' }; return; }
      // a etapa que falhou é identificada pelo texto do motivo
      const ehPlanejamento = /nenhuma|valida|sem candidato/i.test(motivo);
      const idxFalha = ehPlanejamento ? 0 : f.etapas.length - 1;
      out[e.id] = i < idxFalha ? { status: 'ok' }
                : i === idxFalha ? { status: 'falha', motivo }
                : { status: 'pendente' };
    });
  });
  return out;
}

/* Qual portão barrou, a partir do texto do motivo. */
export function portaoAtingido(motivo: string): string | null {
  const m = motivo.toLowerCase();
  if (m.includes('hand clearance') || m.includes('hot')) return 'g_quente';
  if (m.includes('forbidden')) return 'g_proib';
  if (m.includes('penetration') || m.includes('penetra')) return 'g_pen';
  if (m.includes('elbow')) return 'g_anat';
  if (m.includes('wrist')) return 'g_punho';
  if (m.includes('spill') || m.includes('derram')) return 'g_derrame';
  if (m.includes('drift') || m.includes('prop')) return 'g_deriva';
  if (m.includes('table')) return 'g_mesa';
  return null;
}

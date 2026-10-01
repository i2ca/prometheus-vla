/* Grafo da tarefa do café.

Releitura, não adaptação de rótulos. O grafo do repo original tinha três
agentes e três braços porque a tarefa era empilhar blocos. O café tem seis
camadas: intenção, orquestrador, ferramentas, a cadeia de planejamento e
execução dentro de cada ferramenta, os atuadores e objetos com estado, os
modelos físicos reduzidos e os portões que podem vetar qualquer etapa. */
import { useMemo } from 'react';
import {
  ReactFlow, Background, Controls, MiniMap, Handle, Position,
  type Node, type Edge, MarkerType, BackgroundVariant,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { Compass, Cpu, Wrench, Hand, Package, Waves, ShieldAlert, Coffee } from 'lucide-react';
import { type MetricsData } from './theme';
import {
  FERRAMENTAS, MODELOS, PORTOES, OBJETOS,
  statusDasEtapas, portaoAtingido, type Episodio, type StatusEtapa,
} from '../coffeeSource';

const COR: Record<StatusEtapa, string> = { ok: '#22c55e', falha: '#ef4444', pendente: '#475569' };
const caixa = (cor: string, largura: number): React.CSSProperties => ({
  background: 'linear-gradient(135deg, rgba(20,28,42,.95), rgba(12,18,28,.95))',
  border: `1px solid ${cor}`, borderRadius: 12, padding: '10px 13px', width: largura,
  color: '#e2e8f0', fontFamily: 'system-ui, sans-serif',
  boxShadow: '0 8px 28px rgba(0,0,0,.45)',
});
const rotulo: React.CSSProperties = {
  fontSize: 9, textTransform: 'uppercase', letterSpacing: '.07em', color: '#94a3b8', fontWeight: 700,
};
const alfinete = { background: '#475569', width: 6, height: 6, border: 'none' };

const NoIntencao = ({ data }: any) => (
  <div style={caixa('rgba(56,189,248,.45)', 300)}>
    <Handle type="source" position={Position.Bottom} style={alfinete} />
    <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 5 }}>
      <Compass size={14} color="#38bdf8" /><span style={rotulo}>Objetivo do operador</span>
    </div>
    <div style={{ fontSize: 12.5, lineHeight: 1.45 }}>{data.goal}</div>
  </div>
);

const NoOrquestrador = ({ data }: any) => (
  <div style={caixa('rgba(167,139,250,.5)', 260)}>
    <Handle type="target" position={Position.Top} style={alfinete} />
    <Handle type="source" position={Position.Bottom} style={alfinete} />
    <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 5 }}>
      <Cpu size={14} color="#a78bfa" /><span style={rotulo}>Orquestrador</span>
    </div>
    <div style={{ fontSize: 12.5, fontWeight: 600 }}>{data.model}</div>
    <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>
      {data.turnos} turnos · escolhe ferramenta por function calling
    </div>
    <div style={{ fontSize: 10.5, color: '#64748b', marginTop: 3 }}>
      vê a câmera da cabeça, não a pose dos objetos
    </div>
  </div>
);

const NoFerramenta = ({ data }: any) => (
  <div style={caixa(data.chamada ? 'rgba(56,189,248,.5)' : 'rgba(71,85,105,.5)', 210)}>
    <Handle type="target" position={Position.Top} style={alfinete} />
    <Handle type="source" position={Position.Bottom} style={alfinete} />
    <div style={{ display: 'flex', gap: 7, alignItems: 'center', marginBottom: 4 }}>
      <Wrench size={13} color={data.chamada ? '#38bdf8' : '#64748b'} />
      <span style={rotulo}>ferramenta</span>
      {data.vezes > 0 && (
        <span style={{ marginLeft: 'auto', fontSize: 10, color: '#94a3b8',
                       fontFamily: 'monospace' }}>{data.vezes}×</span>
      )}
    </div>
    <div style={{ fontSize: 13, fontWeight: 700 }}>{data.rotulo}</div>
    {data.requer && (
      <div style={{ fontSize: 10.5, color: '#64748b', marginTop: 3 }}>exige {data.requer}</div>
    )}
    <div style={{ fontSize: 10.5, color: '#94a3b8', marginTop: 3 }}>
      mão {data.mao}
    </div>
  </div>
);

const NoEtapa = ({ data }: any) => (
  <div style={{ ...caixa(COR[data.status as StatusEtapa] + '88', 196), padding: '8px 11px' }}>
    <Handle type="target" position={Position.Top} style={alfinete} />
    <Handle type="source" position={Position.Bottom} style={alfinete} />
    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
      <span style={{ width: 7, height: 7, borderRadius: 99, background: COR[data.status as StatusEtapa] }} />
      <span style={{ fontSize: 12, fontWeight: 600 }}>{data.rotulo}</span>
    </div>
    <div style={{ fontSize: 10, color: '#64748b', fontFamily: 'monospace', marginTop: 3 }}>
      {data.script}
    </div>
    {data.nota && <div style={{ fontSize: 10.5, color: '#94a3b8', marginTop: 2 }}>{data.nota}</div>}
    {data.motivo && (
      <div style={{ fontSize: 10.5, color: '#fca5a5', marginTop: 5, lineHeight: 1.4 }}>{data.motivo}</div>
    )}
  </div>
);

const NoAtuador = ({ data }: any) => (
  <div style={caixa(data.ativo ? 'rgba(34,197,94,.5)' : 'rgba(71,85,105,.45)', 200)}>
    <Handle type="target" position={Position.Top} style={alfinete} />
    <Handle type="source" position={Position.Bottom} style={alfinete} />
    <div style={{ display: 'flex', gap: 7, alignItems: 'center', marginBottom: 4 }}>
      <Hand size={13} color={data.ativo ? '#22c55e' : '#64748b'} /><span style={rotulo}>atuador</span>
    </div>
    <div style={{ fontSize: 12.5, fontWeight: 700 }}>{data.nome}</div>
    <div style={{ fontSize: 10.5, color: '#94a3b8', marginTop: 3 }}>{data.papel}</div>
    {data.contagem && (
      <div style={{ fontSize: 10.5, color: '#64748b', marginTop: 3, fontFamily: 'monospace' }}>
        {data.contagem}
      </div>
    )}
  </div>
);

const NoObjeto = ({ data }: any) => (
  <div style={{ ...caixa('rgba(148,163,184,.3)', 176), padding: '8px 11px' }}>
    <Handle type="target" position={Position.Top} style={alfinete} />
    <Handle type="source" position={Position.Bottom} style={alfinete} />
    <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
      <Package size={12} color="#94a3b8" />
      <span style={{ fontSize: 11.5, fontWeight: 600 }}>{data.rotulo}</span>
    </div>
    {data.estado && (
      <div style={{ fontSize: 11, color: '#38bdf8', fontFamily: 'monospace', marginTop: 3 }}>
        {data.estado}
      </div>
    )}
    {data.nota && <div style={{ fontSize: 10, color: '#64748b', marginTop: 2 }}>{data.nota}</div>}
  </div>
);

const NoModelo = ({ data }: any) => (
  <div style={{ ...caixa('rgba(6,182,212,.35)', 232), padding: '8px 11px' }}>
    <Handle type="target" position={Position.Left} style={alfinete} />
    <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
      <Waves size={12} color="#06b6d4" />
      <span style={{ fontSize: 11.5, fontWeight: 600 }}>{data.rotulo}</span>
    </div>
    <div style={{ fontSize: 9.5, color: '#64748b', fontFamily: 'monospace', marginTop: 2 }}>
      {data.script}
    </div>
    <div style={{ fontSize: 10.5, color: '#94a3b8', marginTop: 3, lineHeight: 1.4 }}>{data.detalhe}</div>
  </div>
);

const NoPortao = ({ data }: any) => (
  <div style={{ ...caixa(data.atingido ? 'rgba(239,68,68,.65)' : 'rgba(217,119,6,.32)', 222),
                padding: '7px 10px' }}>
    <Handle type="target" position={Position.Right} style={alfinete} />
    <Handle type="source" position={Position.Left} style={alfinete} />
    <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
      <ShieldAlert size={12} color={data.atingido ? '#ef4444' : '#d97706'} />
      <span style={{ fontSize: 11.5, fontWeight: 600 }}>{data.rotulo}</span>
      {data.atingido && (
        <span style={{ marginLeft: 'auto', fontSize: 9, color: '#ef4444', fontWeight: 700 }}>BARROU</span>
      )}
    </div>
    <div style={{ fontSize: 10.5, color: '#94a3b8', marginTop: 2 }}>{data.limite}</div>
    <div style={{ fontSize: 9.5, color: '#64748b', fontFamily: 'monospace' }}>{data.escopo}</div>
  </div>
);

const NoResultado = ({ data }: any) => (
  <div style={caixa(data.pronto ? 'rgba(34,197,94,.6)' : 'rgba(239,68,68,.5)', 300)}>
    <Handle type="target" position={Position.Top} style={alfinete} />
    <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 6 }}>
      <Coffee size={14} color={data.pronto ? '#22c55e' : '#ef4444'} />
      <span style={rotulo}>resultado medido</span>
    </div>
    <div style={{ display: 'flex', gap: 14 }}>
      <div>
        <div style={{ fontSize: 9, color: '#94a3b8' }}>CAFÉ NA XÍCARA</div>
        <div style={{ fontSize: 22, fontFamily: 'monospace',
                      color: data.pronto ? '#22c55e' : '#ef4444' }}>{data.ml} ml</div>
      </div>
      <div>
        <div style={{ fontSize: 9, color: '#94a3b8' }}>PÓ NO FILTRO</div>
        <div style={{ fontSize: 22, fontFamily: 'monospace' }}>{data.po} g</div>
      </div>
    </div>
    {data.justificativa && (
      <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 7, lineHeight: 1.5 }}>
        {data.justificativa}
      </div>
    )}
  </div>
);

const TIPOS = {
  intencao: NoIntencao, orquestrador: NoOrquestrador, ferramenta: NoFerramenta,
  etapa: NoEtapa, atuador: NoAtuador, objeto: NoObjeto, modelo: NoModelo,
  portao: NoPortao, resultado: NoResultado,
};

interface Props {
  episodio: Episodio | null;
  metrics: MetricsData | null;
  observacao: Record<string, any> | null;
}

export default function CoffeeWorkflowGraph({ episodio, metrics, observacao }: Props) {
  const { nodes, edges } = useMemo(() => {
    const ns: Node[] = []; const es: Edge[] = [];
    const ep = episodio;
    const status = ep ? statusDasEtapas(ep) : {};
    const vezes: Record<string, number> = {};
    const falhaPorFerramenta: Record<string, string> = {};
    ep?.turnos.forEach((t) => {
      (t.chamadas || []).forEach((c) => { vezes[c] = (vezes[c] || 0) + 1; });
      (t.resultados || []).forEach((r) => Object.entries(r).forEach(([n, v]: [string, any]) => {
        if (v && v.sucesso === false) falhaPorFerramenta[n] = String(v.motivo || v.erro || '');
      }));
    });
    const portoesBarrados = new Set(
      Object.values(falhaPorFerramenta).map(portaoAtingido).filter(Boolean) as string[]);

    const linha = (id: string, alvo: string, cor = 'rgba(148,163,184,.28)', animado = false): Edge => ({
      id: `${id}->${alvo}`, source: id, target: alvo, type: 'smoothstep', animated: animado,
      style: { stroke: cor, strokeWidth: animado ? 2 : 1 },
      markerEnd: { type: MarkerType.ArrowClosed, color: cor, width: 12, height: 12 },
    });

    // nível 0 e 1
    ns.push({ id: 'intencao', type: 'intencao', position: { x: 640, y: 0 },
              data: { goal: ep?.goal || 'Preparar café coado em pano' } });
    ns.push({ id: 'orq', type: 'orquestrador', position: { x: 660, y: 130 },
              data: { model: ep?.model || '—', turnos: ep?.turnos.length || 0 } });
    es.push(linha('intencao', 'orq', 'rgba(56,189,248,.45)'));

    // nível 2 e 3: ferramentas e suas etapas
    const largura = 250;
    FERRAMENTAS.forEach((f, i) => {
      const x = i * largura;
      const idF = `f_${f.nome}`;
      ns.push({ id: idF, type: 'ferramenta', position: { x, y: 290 },
                data: { rotulo: f.rotulo, mao: f.mao, requer: f.requer,
                        chamada: (vezes[f.nome] || 0) > 0, vezes: vezes[f.nome] || 0 } });
      es.push(linha('orq', idF, (vezes[f.nome] || 0) > 0 ? 'rgba(56,189,248,.5)' : 'rgba(148,163,184,.2)',
                    (vezes[f.nome] || 0) > 0));
      let anterior = idF;
      f.etapas.forEach((e, j) => {
        const st = status[e.id]?.status || 'pendente';
        const idE = `e_${e.id}`;
        ns.push({ id: idE, type: 'etapa', position: { x: x + 10, y: 425 + j * 108 },
                  data: { rotulo: e.rotulo, script: e.script, nota: e.nota, status: st,
                          motivo: status[e.id]?.motivo } });
        es.push(linha(anterior, idE, COR[st] + '66', st === 'ok'));
        anterior = idE;
      });
      // liga a última etapa ao atuador
      const idAt = f.mao === 'direita' ? 'at_dir' : f.mao === 'esquerda' ? 'at_esq' : null;
      if (idAt) es.push(linha(anterior, idAt, 'rgba(148,163,184,.25)'));
    });

    // nível 4: atuadores
    const esqOk = metrics?.robots?.G1_ESQUERDA?.tasks_completed || 0;
    const esqBad = metrics?.robots?.G1_ESQUERDA?.tasks_failed || 0;
    ns.push({ id: 'at_esq', type: 'atuador', position: { x: 330, y: 790 },
              data: { nome: 'Mão esquerda · Dex3', papel: 'botão, alça da chaleira, despejo',
                      ativo: esqOk > 0, contagem: `${esqOk} ok · ${esqBad} falhas` } });
    ns.push({ id: 'at_dir', type: 'atuador', position: { x: 560, y: 790 },
              data: { nome: 'Mão direita · Dex3', papel: 'colher, dosagem de 19,85 g',
                      ativo: true, contagem: '21 ciclos concluídos' } });
    ns.push({ id: 'at_tronco', type: 'atuador', position: { x: 790, y: 790 },
              data: { nome: 'Cintura e tronco', papel: 'alcance; pelve fixa nesta cena',
                      ativo: esqOk > 0, contagem: 'roll/pitch ±25°' } });
    es.push(linha('at_esq', 'o_chaleira', 'rgba(148,163,184,.25)'));
    es.push(linha('at_dir', 'o_colher', 'rgba(148,163,184,.25)'));

    // nível 5: objetos com estado
    OBJETOS.forEach((o, i) => {
      let estado = '';
      if (observacao) {
        if (o.chave === 'coador') estado = `${observacao.po_no_filtro_g} g de pó`;
        if (o.chave === 'chaleira') estado = `${observacao.agua_na_chaleira_ml} ml · ${observacao.temperatura_C} °C`;
        if (o.chave === 'copo') estado = `${observacao.agua_na_xicara_ml} ml`;
        if (o.chave === 'base_eletrica') estado = observacao.aquecedor_ligado ? 'ligado' : (observacao.ultimo_evento_aquecedor || '');
        if (o.chave === 'scoop') estado = `${observacao.po_na_colher_g} g`;
      }
      ns.push({ id: o.id, type: 'objeto', position: { x: i * 200, y: 940 },
                data: { rotulo: o.rotulo, estado, nota: (o as any).nota } });
    });
    es.push(linha('o_chaleira', 'resultado', 'rgba(34,197,94,.3)'));
    es.push(linha('o_coador', 'resultado', 'rgba(34,197,94,.3)'));
    es.push(linha('o_xicara', 'resultado', 'rgba(34,197,94,.3)'));

    // lateral esquerda: portões
    PORTOES.forEach((g, i) => {
      ns.push({ id: g.id, type: 'portao', position: { x: -330, y: 300 + i * 92 },
                data: { rotulo: g.rotulo, limite: g.limite, escopo: g.escopo,
                        atingido: portoesBarrados.has(g.id) } });
    });
    // o portão que barrou aponta para a etapa que morreu
    Object.entries(falhaPorFerramenta).forEach(([nome, motivo]) => {
      const g = portaoAtingido(motivo);
      const f = FERRAMENTAS.find((x) => x.nome === nome);
      if (g && f) {
        const alvo = `e_${f.etapas[f.etapas.length - 1].id}`;
        es.push({ id: `${g}->${alvo}`, source: g, target: alvo, type: 'smoothstep', animated: true,
                  label: 'barrou', labelStyle: { fill: '#ef4444', fontSize: 10, fontWeight: 700 },
                  labelBgStyle: { fill: '#0d111a' },
                  style: { stroke: '#ef4444', strokeWidth: 2, strokeDasharray: '4 3' },
                  markerEnd: { type: MarkerType.ArrowClosed, color: '#ef4444' } });
      }
    });

    // lateral direita: modelos físicos reduzidos
    MODELOS.forEach((mo, i) => {
      ns.push({ id: mo.id, type: 'modelo', position: { x: 1560, y: 300 + i * 118 },
                data: { rotulo: mo.rotulo, script: mo.script, detalhe: mo.detalhe } });
    });

    // resultado
    ns.push({ id: 'resultado', type: 'resultado', position: { x: 620, y: 1090 },
              data: { ml: observacao?.agua_na_xicara_ml ?? 0,
                      po: observacao?.po_no_filtro_g ?? 0,
                      pronto: !!ep?.coffee_completed,
                      justificativa: ep?.justificativa_final } });

    return { nodes: ns, edges: es };
  }, [episodio, metrics, observacao]);

  return (
    <div style={{ flex: 1, minHeight: 0 }}>
      <ReactFlow nodes={nodes} edges={edges} nodeTypes={TIPOS as any}
                 fitView fitViewOptions={{ padding: 0.12 }} minZoom={0.15} maxZoom={1.6}
                 proOptions={{ hideAttribution: true }}>
        <Background variant={BackgroundVariant.Dots} gap={22} size={1} color="rgba(148,163,184,.13)" />
        <Controls showInteractive={false} />
        <MiniMap pannable zoomable maskColor="rgba(8,10,15,.85)"
                 style={{ background: '#0d111a', border: '1px solid rgba(255,255,255,.08)' }}
                 nodeColor={(n) => (n.type === 'portao' ? '#d97706'
                                  : n.type === 'modelo' ? '#06b6d4'
                                  : n.type === 'etapa' ? '#38bdf8' : '#475569')} />
      </ReactFlow>
    </div>
  );
}

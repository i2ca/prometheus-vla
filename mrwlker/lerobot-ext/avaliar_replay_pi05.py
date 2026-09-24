#!/usr/bin/env python
"""
Replay offline do π0.5 — erro em RADIANOS nos episódios gravados
=================================================================
Pega um checkpoint `pi05depth`, roda os episódios gravados através dele e
compara, a cada `--passo` quadros, o chunk previsto com o que o teleoperador
executou. Sem robô, sem simulador, sem servidor.

POR QUE ISTO E NÃO O `val_loss`

  * o `val_loss` é a loss de flow matching no espaço NORMALIZADO, e cada dataset
    normaliza com as suas próprias estatísticas. Duas corridas com datasets
    diferentes (ex.: solo × co-treino completo) têm `val_loss` em réguas
    diferentes. Radiano é a mesma régua para todo mundo;
  * um erro médio bonito espalhado por 29 juntas esconde uma mão que nunca
    fecha. Aqui o erro sai por grupo de juntas, e a abertura da mão direita tem
    métrica própria;
  * sempre ao lado de uma LINHA DE BASE: "ficar parado" (repetir o estado atual
    no chunk inteiro). Um modelo que não bate o robô parado não aprendeu nada,
    por menor que seja o número.

O TESTE DE LINGUAGEM (`--tarefas-teste`)

  Na mesma observação, troca só o texto da tarefa e mede quanto a ação muda.
  Para saber se "mudou" significa algo, compara com a diferença entre duas
  amostragens com o MESMO texto e ruído inicial diferente. Se trocar o texto
  muda tanto quanto trocar o ruído, o modelo está ignorando o texto.

Réplica do que o trainer faz na validação (`policies/pi0_depth/run_train.py`):
mesmo `LeRobotDataset` (`return_uint8=True`, `delta_timestamps`, `tolerance_s`),
pré e pós-processador carregados do próprio checkpoint.

USO (na athena, com o env `prometheus-vla`)

    python avaliar_replay_pi05.py \\
        --checkpoint /data/mrwlker/train_output/pi05_cotreino_completo/best_val_checkpoint/pretrained_model \\
        --episodios 22,23,24,290,295,301,1,27 --passo 25 \\
        --tarefas-teste "toasted bread" "pour water" \\
        --saida /data/mrwlker/avaliacao/pi05_completo

    Sai `<saida>.json` (tudo) e `<saida>.png` (trajetórias do 1º episódio).
    Dataset e raiz vêm do `train_config.json` do checkpoint (o `val_dataset`);
    `--root` e `--repo-id` sobrescrevem.
"""

import argparse
import json
import sys
import time
from pathlib import Path

GRUPOS = {
    "braço esq": list(range(0, 7)),
    "braço dir": list(range(7, 14)),
    "cintura": [14],
    "mão esq": list(range(15, 22)),
    "mão dir": list(range(22, 29)),
}
JUNTAS_GRAFICO = {7: "R shoulder pitch", 10: "R elbow", 14: "waist yaw"}
MAO_DIR = list(range(22, 29))


def carrega_politica(checkpoint: Path, device):
    import torch
    from safetensors.torch import load_file
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from policies.pi0_depth.modeling_pi05 import PI05DEPTHPolicy

    config = PreTrainedConfig.from_pretrained(checkpoint)
    if config.type != "pi05depth":
        raise SystemExit(f"❌ checkpoint é '{config.type}', este script é para pi05depth")
    politica = PI05DEPTHPolicy(config)
    faltando, sobrando = politica.load_state_dict(load_file(checkpoint / "model.safetensors"), strict=False)
    print(f"    pesos: {len(faltando)} ausentes, {len(sobrando)} inesperados")
    if faltando:
        print("    ⚠️  ausentes:", faltando[:5])
    politica.eval().to(device)
    pre, pos = make_pre_post_processors(policy_cfg=politica.config, pretrained_path=str(checkpoint))
    torch.backends.cuda.matmul.allow_tf32 = True
    return politica, pre, pos


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help=".../pretrained_model")
    p.add_argument("--episodios", required=True, help="lista separada por vírgula")
    p.add_argument("--passo", type=int, default=25, help="de quantos em quantos quadros pedir um chunk")
    p.add_argument("--lote", type=int, default=8)
    p.add_argument("--root", default=None)
    p.add_argument("--repo-id", default=None)
    p.add_argument("--tarefas-teste", nargs="*", default=[],
                   help="textos alternativos para o teste de linguagem")
    p.add_argument("--episodios-linguagem", default=None,
                   help="onde rodar o teste de linguagem (padrão: os 3 primeiros de --episodios)")
    p.add_argument("--saida", required=True, help="prefixo de saída (.json e .png)")
    args = p.parse_args()

    import numpy as np
    import torch
    from torch.utils.data import default_collate

    sys.path.insert(0, str(Path(__file__).parent))
    import policies  # noqa: F401  registra pi05depth
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    ckpt = Path(args.checkpoint)
    treino = json.loads((ckpt / "train_config.json").read_text())
    ds_cfg = treino.get("val_dataset") or treino["dataset"]
    root = args.root or ds_cfg["root"]
    repo_id = args.repo_id or ds_cfg["repo_id"]
    episodios = [int(e) for e in args.episodios.split(",")]
    ep_ling = ([int(e) for e in args.episodios_linguagem.split(",")] if args.episodios_linguagem
               else episodios[:3])
    saida = Path(args.saida)
    saida.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    t0 = time.perf_counter()
    print(f">>> {ckpt}")
    politica, pre, pos = carrega_politica(ckpt, device)
    print(f"    carregado em {time.perf_counter() - t0:.0f} s | VRAM {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    meta = LeRobotDatasetMetadata(repo_id, root=root)
    ds = LeRobotDataset(
        repo_id, root=root, episodes=episodios,
        delta_timestamps=resolve_delta_timestamps(politica.config, meta),
        image_transforms=None, return_uint8=True,
        tolerance_s=treino.get("tolerance_s", 1e-4),
    )
    faixa = {int(e): (int(a), int(b)) for e, a, b in zip(
        meta.episodes["episode_index"], meta.episodes["dataset_from_index"], meta.episodes["dataset_to_index"])}
    rel = getattr(ds, "absolute_to_relative_idx", None)

    def item(abs_idx: int) -> dict:
        return ds[rel[abs_idx] if rel is not None else abs_idx]

    def infere(amostras: list[dict], tarefa: str | None = None, semente: int = 0):
        lote = default_collate(amostras)
        if tarefa is not None:
            lote["task"] = [tarefa] * len(amostras)
        real = lote["action"].float().clone()
        pad = lote["action_is_pad"].clone()
        estado = lote["observation.state"].float().clone()
        torch.manual_seed(semente)
        with torch.no_grad():
            prev = politica.predict_action_chunk(pre(lote))
            prev = pos(prev)
        return prev.float().cpu(), real, pad, estado

    resultados = {"checkpoint": str(ckpt), "root": root, "passo": args.passo,
                  "chunk": int(politica.config.chunk_size), "episodios": {}}
    trilha = None

    for ep in episodios:
        a, b = faixa[ep]
        idxs = list(range(a, b, args.passo))
        tarefa_ep = item(a)["task"]
        soma = {g: {"modelo": 0.0, "modelo_10": 0.0, "parado": 0.0, "parado_10": 0.0} for g in GRUPOS}
        n_val = n_val10 = 0
        mao_prev, mao_real = [], []
        t_ep = time.perf_counter()
        prim = {"quadro": [], "prev": [], "real": []}
        for i in range(0, len(idxs), args.lote):
            bloco = idxs[i:i + args.lote]
            prev, real, pad, estado = infere([item(k) for k in bloco])
            valido = (~pad).float().unsqueeze(-1)                     # B, H, 1
            parado = estado.unsqueeze(1).expand_as(real)
            e_mod = (prev - real).abs() * valido
            e_par = (parado - real).abs() * valido
            for g, dims in GRUPOS.items():
                soma[g]["modelo"] += e_mod[..., dims].mean(-1).sum().item()
                soma[g]["parado"] += e_par[..., dims].mean(-1).sum().item()
                soma[g]["modelo_10"] += e_mod[:, :10, dims].mean(-1).sum().item()
                soma[g]["parado_10"] += e_par[:, :10, dims].mean(-1).sum().item()
            n_val += valido.sum().item()
            n_val10 += valido[:, :10].sum().item()
            mao_prev += prev[..., MAO_DIR].mean(dim=(1, 2)).tolist()
            mao_real += real[..., MAO_DIR].mean(dim=(1, 2)).tolist()
            if trilha is None:
                for j, k in enumerate(bloco):
                    prim["quadro"].append(k - a)
                    prim["prev"].append(prev[j, :args.passo].numpy())
                    prim["real"].append(real[j, :args.passo].numpy())
        erro = {g: {m: v / (n_val10 if m.endswith("_10") else n_val) for m, v in s.items()} for g, s in soma.items()}
        corr = float(np.corrcoef(mao_prev, mao_real)[0, 1]) if np.std(mao_real) > 1e-6 and np.std(mao_prev) > 1e-6 else None
        resultados["episodios"][ep] = {
            "tarefa": tarefa_ep, "quadros": b - a, "chunks": len(idxs), "erro_rad": erro,
            "mao_dir": {"corr_prev_real": corr,
                        "real_min_max": [float(min(mao_real)), float(max(mao_real))],
                        "prev_min_max": [float(min(mao_prev)), float(max(mao_prev))]},
        }
        if trilha is None:
            trilha = {"episodio": ep, **prim}
        linha = "  ".join(f"{g} {erro[g]['modelo']:.3f}/{erro[g]['parado']:.3f}" for g in GRUPOS)
        print(f"  ep {ep:4d} ({len(idxs):3d} chunks, {time.perf_counter() - t_ep:4.0f} s)  modelo/parado  {linha}"
              f"  | corr mão dir {corr if corr is None else round(corr, 2)}")

    if args.tarefas_teste:
        print(">>> teste de linguagem")
        soma_ruido = {"braços": 0.0, "mão dir": 0.0}
        soma_alt = {t: {"braços": 0.0, "mão dir": 0.0} for t in args.tarefas_teste}
        n = 0
        for ep in ep_ling:
            a, b = faixa[ep]
            idxs = list(range(a, b, args.passo))
            for i in range(0, len(idxs), args.lote):
                amostras = [item(k) for k in idxs[i:i + args.lote]]
                base, *_ = infere(amostras, semente=0)
                ruido, *_ = infere(amostras, semente=1)
                for nome, dims in (("braços", list(range(0, 15))), ("mão dir", MAO_DIR)):
                    soma_ruido[nome] += (ruido - base)[..., dims].abs().mean(dim=(1, 2)).sum().item()
                for t in args.tarefas_teste:
                    alt, *_ = infere(amostras, tarefa=t, semente=0)
                    for nome, dims in (("braços", list(range(0, 15))), ("mão dir", MAO_DIR)):
                        soma_alt[t][nome] += (alt - base)[..., dims].abs().mean(dim=(1, 2)).sum().item()
                n += len(amostras)
        ling = {"episodios": ep_ling, "amostras": n,
                "ruido_mesmo_texto": {k: v / n for k, v in soma_ruido.items()},
                "texto_trocado": {t: {k: v / n for k, v in d.items()} for t, d in soma_alt.items()}}
        resultados["linguagem"] = ling
        print(f"    mesmo texto, ruído diferente: braços {ling['ruido_mesmo_texto']['braços']:.4f} rad"
              f" | mão dir {ling['ruido_mesmo_texto']['mão dir']:.4f} rad")
        for t, d in ling["texto_trocado"].items():
            print(f"    texto → {t!r:24s} braços {d['braços']:.4f} rad | mão dir {d['mão dir']:.4f} rad")

    saida.with_suffix(".json").write_text(json.dumps(resultados, indent=2, ensure_ascii=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, eixos = plt.subplots(len(JUNTAS_GRAFICO) + 1, 1, figsize=(12, 9), sharex=True)
    series = list(JUNTAS_GRAFICO.items()) + [(None, "mão dir (média 22-28)")]
    for eixo, (junta, nome) in zip(eixos, series):
        for q, pv, rl in zip(trilha["quadro"], trilha["prev"], trilha["real"]):
            t = np.arange(q, q + len(pv))
            ypv = pv[:, MAO_DIR].mean(-1) if junta is None else pv[:, junta]
            yrl = rl[:, MAO_DIR].mean(-1) if junta is None else rl[:, junta]
            eixo.plot(t, yrl, color="black", lw=1.6)
            eixo.plot(t, ypv, color="tab:red", lw=1.2)
        eixo.set_ylabel(nome, fontsize=9)
        eixo.grid(alpha=0.3)
    eixos[0].plot([], [], color="black", label="teleoperador")
    eixos[0].plot([], [], color="tab:red",
                  label=f"π0.5 (primeiros {min(args.passo, resultados['chunk'])} passos de cada chunk)")
    eixos[0].legend(loc="upper right", fontsize=8)
    eixos[-1].set_xlabel("quadro")
    fig.suptitle(f"episódio {trilha['episodio']} — {ckpt.parent.parent.name}", fontsize=11)
    fig.tight_layout()
    fig.savefig(saida.with_suffix(".png"), dpi=110)
    print(f">>> {saida.with_suffix('.json')}  {saida.with_suffix('.png')}  ({(time.perf_counter() - t0) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

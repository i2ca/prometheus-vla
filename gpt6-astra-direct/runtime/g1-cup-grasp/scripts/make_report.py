"""Gera learning-history.json e learning-report.md a partir de results/attempt-NN.

Le parameters.json, physics-report.json, evaluation.json, provenance.json e timeline.json de cada
tentativa. Nao roda simulacao nem altera as tentativas.
Uso: .venv/bin/python scripts/make_report.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(path):
    return json.loads(path.read_text()) if path.exists() else {}


def evaluator_line(rows):
    hashes = sorted({r["evaluator_sha256"] for r in rows if r.get("evaluator_sha256")})
    if len(hashes) == 1:
        return f"Avaliador congelado: SHA-256 `{hashes[0]}`, a mesma versao em todas as {len(rows)} tentativas."
    vers = sorted({str(r.get("evaluator_version")) for r in rows if r.get("score") is not None})
    return ("AVISO: as notas NAO sao comparaveis entre todas as tentativas. Versoes do avaliador presentes: " + ", ".join(vers) +
            " (coluna da nota). As tentativas 01 a 14 usaram o copo antigo e so podem ser avaliadas pela v3 (a v5 usa o modelo "
            "do copo atual); 15 e 16 usam a v5. Compare notas so dentro da mesma versao e da mesma cena.")


def main():
    rows = []
    for run in sorted((ROOT / "results").glob("attempt-[0-9][0-9]")):
        params, phys, ev, prov, tl = (load(run / n) for n in ("parameters.json", "physics-report.json", "evaluation.json", "provenance.json", "timeline.json"))
        rows.append({
            "attempt": run.name, "reason": params.get("reason", ""), "model": prov.get("model", ""),
            "parameters": {k: v for k, v in params.items() if k != "reason"},
            "score": ev.get("score"), "evaluator_version": ev.get("evaluator_version", 3 if ev else None), "rise_ratio": ev.get("rise_ratio"), "pixel_agreement": ev.get("pixel_agreement"),
            "cup_left_table_line": ev.get("cup_left_table_line"), "evaluator_sha256": ev.get("evaluator_sha256"),
            "status": phys.get("status"), "retained_for_2s": phys.get("retained_for_2s"), "lift_m": phys.get("lift_m"),
            "fingers_at_hold": phys.get("fingers_at_hold"), "hand_table_frames": phys.get("hand_table_frames"),
            "cup_tilt_deg_before_close": phys.get("cup_tilt_deg_before_close"), "cup_tilt_deg_at_hold": phys.get("cup_tilt_deg_at_hold"),
            "upright_grasp": phys.get("upright_grasp"), "accepted": phys.get("accepted"), "error": phys.get("error"),
            "max_substep_table_penetration_m": phys.get("max_substep_table_penetration_m"), "warnings": phys.get("warnings"),
            "cup_estimate_error_xy_m": load(run / "cup-detection.json").get("estimate_error_xy_m"),
            "video_seconds": phys.get("video_seconds") or (tl.get("frames", 0) / tl.get("fps", 30) if tl else None),
        })
    (ROOT / "learning-history.json").write_text(json.dumps({"task": "G1 + Dex3: pega do copo guiada por camera (MuJoCo)", "attempts": rows}, indent=2, ensure_ascii=False) + "\n")

    best = max((r for r in rows if r["score"] is not None), key=lambda r: r["score"], default=None)
    out = ["# Tentativas: G1 pega o copo guiado pela camera", "",
           f"{len(rows)} tentativas completas em MuJoCo, gravadas pela camera da cabeca e por uma vista externa. "
           "Entre uma tentativa e outra eu olhei a trajetoria, os contatos e a imagem final e mudei um parametro, "
           "registrando o motivo. " + (f"A melhor sob a metrica de imagem congelada e a {best['attempt']}." if best else ""), "",
           "| Tentativa | Mudanca baseada na observacao anterior | Nota camera /100 | Retido 2 s | Copo em pe (inclinacao antes do fecho / na espera) | Subida real | Mao na mesa (quadros) | Video |",
           "| --- | --- | ---: | --- | --- | ---: | ---: | --- |"]
    for r in rows:
        lift = f"{r['lift_m']*100:.1f} cm" if r["lift_m"] is not None else "-"
        vid = f"[{r['video_seconds']:.1f} s](results/{r['attempt']}/grasp-run.mp4)" if r["video_seconds"] else "-"
        if r["status"] == "failed":
            up = f"falhou: {r['error']}"
        elif r["cup_tilt_deg_before_close"] is None:
            up = "nao medido (tentativa anterior a medicao; reexecucao da 05 deu 82 / 86 graus)"
        else:
            up = f"{'sim' if r['upright_grasp'] else 'nao'} ({r['cup_tilt_deg_before_close']:.0f} / {r['cup_tilt_deg_at_hold']:.0f} graus)"
        out.append(f"| [{r['attempt'][-2:]}](results/{r['attempt']}/) | {r['reason']} | {r['score']} (v{r['evaluator_version']}) | {'sim' if r['retained_for_2s'] else 'nao'} | {up} | {lift} | {r['hand_table_frames']} | {vid} |")
    out += ["", "## O que e medido", "",
            "A nota vem de `scripts/evaluate_grasp.py`, que le so as duas imagens RGB da camera da cabeca (inicial e final), "
            "a calibracao pelos quatro marcadores e a altura de levantamento pedida. Formula: "
            "`100 x (0,50 x subida + 0,30 x concordancia + 0,20 x copo visivel)`. E uma heuristica de geometria na imagem, "
            "nao um juiz de qualidade da pega. Subir mais que o pedido ou o copo escorregar na mao reduz a nota. "
            "Versoes: v1 (subida, concordancia, visivel) nao via copo tombado; v2 acrescentou o termo de copo em pe pela razao "
            "da caixa envolvente; v3 une os componentes brancos perto do pixel previsto antes de medir, porque um dedo na frente "
            "do copo dividia a silhueta. As avaliacoes v1 e v2 ficam guardadas em cada tentativa como evaluation-v1.json e "
            "evaluation-v2.json; a tabela usa a v3 em todas. Limite conhecido da v3: o termo de copo em pe reconhece o copo caido "
            "de lado, mas nao o copo tombado com o fundo virado para a camera (a tentativa 02 recebe 0,96 nesse termo com o copo "
            "deitado). Por isso o aceite de uma tentativa vem da inclinacao medida pelo simulador (coluna 'Copo em pe', limite de "
            "15 graus antes do fecho e na espera), que e informacao privilegiada e esta declarada como tal.", "",
            "`physics-report.json` de cada tentativa le o estado do simulador (altura real do copo, dedos em contato, "
            "contato mao-mesa em cada subpasso, avisos do MuJoCo) apenas para verificar. O controle usa somente a estimativa "
            "RGB do copo, a calibracao e a cinematica do braco. O copo e colocado uma vez na mesa e nunca mais tocado pelo script.", "",
            evaluator_line(rows), "",
            "## Modelos", ""]
    for r in rows:
        out.append(f"- {r['attempt']}: {r['model']}")
    out += ["", "Controlador scriptado com IK; nenhuma politica neural foi treinada. Os parametros iniciais vieram do teste "
            "com pose conhecida aprovado pelo orquestrador (`results/free-cup-20260915-172322-156952/`)."]
    (ROOT / "learning-report.md").write_text("\n".join(out) + "\n")
    print(f"{len(rows)} tentativas -> learning-history.json, learning-report.md")


if __name__ == "__main__":
    main()

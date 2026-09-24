"""Copia o log bruto (pedido e resposta exatos) de cada chamada de um episodio, guardado pelo OmniRoute.

Uso (na maquina do gateway):
  python3 scripts/export_raw_logs.py results/<ep>/x0.40_y-0.20 [--data ~/omniroute-data]

O gateway grava cada chamada em <data>/call_logs/<dia>/<arquivo>.json (requestBody + responseBody). Aqui
achamos as do episodio pelo id da resposta (o mesmo que o runner registrou em policy-calls/ e reflection.json)
dentro da janela de horario do episodio, e gravamos em <ep>/raw/. Imagens em base64 viram {sha256, bytes}: elas ja
estao no episodio (obs/) e deixariam o git enorme. So entram chamadas deste episodio; o gateway tambem atende outras
pessoas.
"""
import argparse, datetime as dt, hashlib, json, sqlite3
from pathlib import Path


def _omit(b64text):
    """O gateway ja corta o base64 no proprio log; guardamos so a identificacao."""
    return {"omitted": "imagem", "chars_no_log": len(b64text), "sha256_do_texto": hashlib.sha256(b64text.encode()).hexdigest(),
            "cortada_pelo_gateway": len(b64text) % 4 != 0}


def strip_images(x):
    if isinstance(x, dict):
        if x.get("type") == "base64" and isinstance(x.get("data"), str) and len(x["data"]) > 500:
            return {**{k: v for k, v in x.items() if k != "data"}, "data": _omit(x["data"])}
        for key in ("url", "image_url"):
            v = x.get(key)
            if isinstance(v, str) and v.startswith("data:") and len(v) > 500:
                return {**x, key: _omit(v.split(",", 1)[1])}
        return {k: strip_images(v) for k, v in x.items()}
    if isinstance(x, list):
        return [strip_images(v) for v in x]
    if isinstance(x, str) and x.startswith("data:image") and len(x) > 500:
        return _omit(x.split(",", 1)[1])
    return x


def norm(rid):
    """O runner guarda 'chatcmpl-resp_...'; o gateway grava o id da OpenAI, 'resp_...'."""
    return (rid or "").removeprefix("chatcmpl-")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode_dir")
    ap.add_argument("--data", default=str(Path.home() / "omniroute-data"))
    a = ap.parse_args()
    ep, data = Path(a.episode_dir), Path(a.data).expanduser()
    calls = [json.loads(p.read_text()) for p in sorted((ep / "policy-calls").glob("call-*.json"))]
    wanted = {}   # id da resposta -> nome do arquivo de saida
    for c in calls:
        ids = [pr.get("response_id") for pr in c.get("depth_probes", []) if pr.get("response_id")]
        ids.append(c.get("response_id"))
        for k, rid in enumerate(dict.fromkeys(i for i in ids if i)):   # sem repetir, na ordem
            wanted.setdefault(norm(rid), f"call-{c['call']:03d}-r{k + 1}")
    refl = ep / "reflection.json"
    if refl.exists():
        rid = json.loads(refl.read_text()).get("response_id")
        if rid:
            wanted[norm(rid)] = "critico"
    t0 = min(dt.datetime.fromisoformat(c["sent_at"]) for c in calls if c.get("sent_at")) - dt.timedelta(minutes=2)
    t1 = max(dt.datetime.fromisoformat(c["received_at"]) for c in calls if c.get("received_at")) + dt.timedelta(hours=2)
    db = sqlite3.connect(f"file:{data / 'storage.sqlite'}?mode=ro", uri=True)
    rows = db.execute("select artifact_relpath from call_logs where model like '%astra%' and timestamp between ? and ?",
                      (t0.strftime("%Y-%m-%dT%H:%M:%S"), t1.strftime("%Y-%m-%dT%H:%M:%S"))).fetchall()
    out_dir = ep / "raw"; out_dir.mkdir(exist_ok=True)
    found = 0
    for (rel,) in rows:
        if not rel or not (data / "call_logs" / rel).exists():
            continue
        d = json.loads((data / "call_logs" / rel).read_text())
        body = d.get("responseBody")
        if isinstance(body, str):   # algumas respostas ficam gravadas como texto
            try:
                body = json.loads(body)
            except ValueError:
                body = {}
        rid = norm((body or {}).get("id") if isinstance(body, dict) else None)
        if rid in wanted:
            (out_dir / f"{wanted.pop(rid)}.json").write_text(json.dumps(strip_images(d), indent=1, ensure_ascii=False) + "\n")
            found += 1
    print(f"{found} chamadas exportadas para {out_dir}; sem log no gateway: {sorted(wanted.values())[:10]}"
          + (" ..." if len(wanted) > 10 else ""))


if __name__ == "__main__":
    main()

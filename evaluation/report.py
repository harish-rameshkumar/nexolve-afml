"""Static, standalone HTML + SVG and CSV exports; no plotting dependencies."""
import argparse
import csv
from html import escape
from pathlib import Path
from evaluation.common import read_json, require

COLORS = {"T": "#2463a3", "D": "#ad4b20", "P": "#247459"}

def chart(runs, metric, title, ylabel, caption, xlabel="Cumulative unique assistant-response tokens"):
    series = []
    for run in runs:
        rows = run["curve"]
        if metric == "strict_score":
            points = [(r["unique_response_tokens"], r[metric]) for r in rows]
        elif metric == "response_nll":
            if not all(metric in r for r in rows): continue
            points = [(r["unique_response_tokens"], r[metric]["nll"]) for r in rows]
        elif metric == "lambda_mean":
            if not all(r.get("lambda") for r in rows): continue
            points = []
            for r in rows:
                values = [v for k,v in r["lambda"].items() if k.endswith("/mean") and "/head" not in k]
                require(bool(values), "Missing per-layer lambda means")
                points.append((r["unique_response_tokens"], sum(values)/len(values)))
        else:
            if not all(metric in r for r in rows): continue
            points = [(r["unique_response_tokens"], r[metric]) for r in rows]
        series.append((run["arm"], run["pretrain_seed"], points))
    if not series: return ""
    w, h, left, top, right, bottom = 860, 390, 85, 45, 130, 65
    xmax = max(x for _, _, ps in series for x, _ in ps)
    ys = [y for _, _, ps in series for _, y in ps]
    lo, hi = (0, 1) if metric == "strict_score" else (min(ys), max(ys))
    if hi == lo: lo -= .05; hi += .05
    def xcoord(x): return left + x / xmax * (w-left-right)
    def ycoord(y): return top + (hi-y) / (hi-lo) * (h-top-bottom)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" aria-label="{escape(title)}">',
        f'<rect width="{w}" height="{h}" fill="white"/>',
        f'<text x="{left}" y="23" font-size="17">{escape(title)}</text>']
    for i in range(5):
        y = lo + (hi-lo)*i/4; yp = ycoord(y)
        parts.append(f'<line x1="{left}" y1="{yp}" x2="{w-right}" y2="{yp}" stroke="#ddd"/>')
        parts.append(f'<text x="{left-10}" y="{yp+4}" text-anchor="end" font-size="12">{y:.3f}</text>')
    for i in range(5):
        x = xmax*i/4; xp = xcoord(x)
        parts.append(f'<text x="{xp}" y="{h-bottom+22}" text-anchor="middle" font-size="12">{x:g}</text>')
    for i, (arm, seed, points) in enumerate(series):
        coords = " ".join(f"{xcoord(x):.2f},{ycoord(y):.2f}" for x,y in points)
        dash = "" if seed == min(s for a,s,_ in series if a == arm) else 'stroke-dasharray="6 4"'
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{COLORS[arm]}" stroke-width="2" {dash}/>')
        parts.append(f'<text x="{w-right+12}" y="{top+18*i}" fill="{COLORS[arm]}" font-size="12">{arm} seed {seed}</text>')
    parts.extend([f'<text x="{(left+w-right)/2}" y="{h-15}" text-anchor="middle" font-size="13">{escape(xlabel)}</text>',
        f'<text transform="translate(19,{(top+h-bottom)/2}) rotate(-90)" text-anchor="middle" font-size="13">{escape(ylabel)}</text>', '</svg>'])
    return '<figure>' + "".join(parts) + '<figcaption>' + escape(caption) + '</figcaption></figure>'

def render(analysis, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    require(bool(analysis["runs"]), "No runs to report")
    caption = f'Source: evaluation {analysis["evaluation_id"][:12]}; token range 0–{analysis["protocol"]["budgets"][-1]}. Individual seeds; no smoothing.'
    figures = [
        chart(analysis["runs"], "strict_score", "Strict instruction-following learning curves", "Strict prompt pass fraction", caption),
        chart(analysis["runs"], "response_nll", "Held-out assistant-response NLL", "NLL (nats / response token)", caption),
        chart(analysis["runs"], "retention_delta_nll", "Pretraining retention change from own parent", "NLL change (nats / token)", caption),
        chart(analysis["runs"], "lambda_mean", "D fixed-probe lambda movement", "Mean sigmoid(lambda), layers averaged", caption + " Fixed input probe; descriptive only.")]
    headers = ["Arm", "Seed", "Normalized AUC", "Final strict score", "Threshold tokens"]
    rows = []
    for run in analysis["runs"]:
        t = run["threshold"]
        attainment = str(t["observed_tokens"]) if t["reached"] else f'> {t["greater_than_tokens"]} (censored)'
        rows.append([run["arm"], str(run["pretrain_seed"]), f'{run["normalized_auc"]:.5f}',
            f'{run["final_strict_score"]:.5f}', attainment])
    table = '<table><tr>' + "".join(f"<th>{escape(h)}</th>" for h in headers) + "</tr>"
    table += "".join("<tr>" + "".join(f"<td>{escape(v)}</td>" for v in row) + "</tr>" for row in rows) + "</table>"
    contrasts = "".join(f'<li>{escape(k)}: mean paired AUC difference {v["mean"]:+.5f}; seed SD {v["seed_sd"]:.5f}; '
        f'n={v["n_pretrain_seeds"]}. Individual differences: {escape(str(v["differences"]))}.</li>'
        for k,v in analysis["paired_auc"].items())
    limitations = "".join(f"<li>{escape(s)}</li>" for s in analysis["limitations"])
    html = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
    html += '<title>WS4 evaluation report</title><style>body{font:16px system-ui;margin:36px auto;max-width:1000px;padding:0 20px;color:#202020}h1{font-size:24px}h2{font-size:20px}figure{margin:26px 0}svg{width:100%}figcaption{font-size:12px;color:#555}table{border-collapse:collapse;width:100%}td,th{padding:9px;text-align:left;border-bottom:1px solid #ddd}li{margin:9px 0}.label{font-weight:700;padding:14px;background:#eee}</style>'
    html += f'<h1>WS4 evaluation and analysis</h1><p class="label">{escape(analysis["label"])}</p>'
    if "SYNTHETIC" in analysis["label"]:
        html += '<p>These are artificial pipeline checks, not measured instruction-learning results.</p>'
    html += "".join(figures) + '<h2>Per-seed endpoints</h2>' + table
    html += '<h2>Paired AUC contrasts</h2><ul>' + contrasts + '</ul><p>No confirmatory significance or equivalence claim. Seed counts are small.</p>'
    html += '<h2>Interpretation limits</h2><ul>' + limitations + '</ul>'
    html += '<p>Threshold crossing is first observed attainment; see analysis.json for intervals and later regression. Primary AUC uses a linear token axis.</p>'
    html += f'<p>Evaluation identity: {escape(analysis["evaluation_id"])}</p></html>'
    (out / "report.html").write_text(html, encoding="utf-8")
    for name, figure in zip(("strict_curves", "response_nll", "retention", "lambda"), figures):
        if figure:
            svg = figure[figure.index("<svg"):figure.index("</svg>")+6]
            (out / f"{name}.svg").write_text(svg, encoding="utf-8")
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(headers); writer.writerows(rows)
    curve_rows, lambda_rows, family_rows = [], [], []
    for run in analysis["runs"]:
        for r in run["curve"]:
            identity = {"arm": run["arm"], "pretrain_seed": run["pretrain_seed"], "unique_response_tokens": r["unique_response_tokens"]}
            curve_rows.append({**identity, "strict_score": r["strict_score"], "prompt_count": r["prompt_count"],
                "response_nll": r.get("response_nll", {}).get("nll"),
                "response_target_tokens": r.get("response_nll", {}).get("target_tokens"),
                "retention_delta_nll": r.get("retention_delta_nll")})
            lambda_rows.extend({**identity, "statistic": k, "value": v} for k,v in r.get("lambda", {}).items())
            family_rows.extend({**identity, "family": k, **v} for k,v in r.get("families", {}).items())
    for filename, records in (("curve_metrics.csv", curve_rows), ("lambda.csv", lambda_rows), ("family_scores.csv", family_rows)):
        if records:
            with (out / filename).open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    return out / "report.html"

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    print(render(read_json(a.analysis), a.out))

if __name__ == "__main__":
    main()

"""Report received pilot logs and rerun validation; never invent strict scores or AUC."""
import argparse
import csv
from html import escape
from pathlib import Path
from evaluation.common import read_json,require
from evaluation.report import COLORS
from evaluation.ws3 import read_jsonl

def plot_curves(curves, caption, out):
    import io,textwrap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8.6,4.8),dpi=150)
    for run in curves:
        rows=run["curve"]
        ax.plot([r["unique_response_tokens"]/1000 for r in rows],
            [r["response_nll"]["nll"] for r in rows],color=COLORS[run["arm"]],
            linestyle="-" if run["pretrain_seed"]==0 else "--",linewidth=1.6,
            label=f'{run["arm"]} seed {run["pretrain_seed"]}')
    ax.set(title="Logged held-out response-token NLL during SFT",
        xlabel="Attempted supervised tokens (thousands, including END)",ylabel="NLL (nats / supervised token)")
    ax.grid(alpha=.2);ax.spines[["top","right"]].set_visible(False);ax.legend(frameon=False,fontsize=8)
    fig.subplots_adjust(bottom=.25)
    fig.text(.02,.02,textwrap.fill(caption,130),fontsize=7,va="bottom")
    fig.savefig(out/"logged_response_nll.svg");fig.savefig(out/"logged_response_nll.png")
    stream=io.StringIO();fig.savefig(stream,format="svg");plt.close(fig)
    svg=stream.getvalue();svg=svg[svg.index("<svg"):]
    return "<figure>"+svg+"<figcaption>"+escape(caption)+"</figcaption></figure>"

def render(result,manifest,out,pilot_strict=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    m=read_json(manifest);base=Path(manifest).resolve().parent
    require(result["complete"],"Evaluation is incomplete")
    curves=[]
    for run in m["runs"]:
        points={}
        for row in read_jsonl(base/run["metrics"]["path"]):
            if row["type"] in ("val","val_final"):
                points[row["tokens_seen"]]=row["val_nll"]
        curves.append({"arm":run["run"][0],"pretrain_seed":int(run["run"][-1]),
            "curve":[{"unique_response_tokens":x,"response_nll":{"nll":y}} for x,y in sorted(points.items())]})
    caption="Source: supplied WS3 CUDA FP16 logs, 0–100,000 attempted supervised tokens including END. Individual seeds; response NLL, not strict instruction-following."
    figure=plot_curves(curves,caption,out)
    summary=[]
    for r in result["rows"]:
        summary.append({"run":r["run"],"parent_response_nll_fp32":r["parent_response_nll"]["nll"],
            "sft_response_nll_fp32":r["sft_response_nll"]["nll"],"logged_final_nll_fp16":r["logged_cuda_fp16_final_nll"],
            "response_target_tokens":r["sft_response_nll"]["target_tokens"],
            "attempted_response_tokens":r["token_accounting"]["attempted_response_tokens"],
            "successful_update_response_tokens":r["token_accounting"]["successful_update_response_tokens"],
            "retention_delta_nll":r["retention"]["delta_nll"],"retention_target_tokens":r["retention"]["target_tokens"],
            "retention_coverage":r["retention"]["coverage"]})
    with (out/"pilot_validation_summary.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    table="<table><tr><th>Run</th><th>Parent NLL</th><th>SFT NLL</th><th>Successful-update tokens</th><th>Retention Δ NLL</th></tr>"
    for r in summary:
        table+=f'<tr><td>{r["run"]}</td><td>{r["parent_response_nll_fp32"]:.6f}</td><td>{r["sft_response_nll_fp32"]:.6f}</td><td>{r["successful_update_response_tokens"]:,}</td><td>{r["retention_delta_nll"]:+.6f}</td></tr>'
    table+="</table>"
    coverage=sorted({r["retention"]["coverage"] for r in result["rows"]})
    tokens=sorted({r["retention"]["target_tokens"] for r in result["rows"]})
    html='<!doctype html><html lang="en"><meta charset="utf-8"><title>WS3 pilot validation</title><style>body{font:16px system-ui;max-width:1050px;margin:32px auto;padding:0 20px;color:#222}h1{font-size:24px}h2{font-size:20px}svg{width:100%}figure{margin:24px 0}figcaption{font-size:12px;color:#555}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:10px;border-bottom:1px solid #ddd}</style>'
    html+='<h1>WS3 pilot validation and WS4 integration</h1><p><b>Pilot evidence only. The primary strict learning-curve evaluation has not been run.</b></p>'
    html+='<p>The repository labels the SFT dataset as a pilot. No agreed strict benchmark, frozen prompt set or threshold was found in the supplied materials. These results verify response prediction and checkpoint integration; they do not establish instruction-sample efficiency.</p>'
    if pilot_strict is not None:
        require(pilot_strict["complete"] and not pilot_strict["candidate_main_evaluated"], "Invalid development pilot results")
        html+='<h2>Exploratory instruction-compliance development check</h2><p>24 local development prompts, four in each of six categories. These are not main-benchmark results; the separate 300-prompt candidate set remains unevaluated.</p>'
        html+='<table><tr><th>Run</th><th>Strict passes</th><th>Prompt count</th><th>Output-cap hits</th></tr>'
        for row in pilot_strict["rows"]:
            html+=f'<tr><td>{row["run"]}</td><td>{round(row["strict_score"]*row["prompt_count"])}</td><td>{row["prompt_count"]}</td><td>{row["cap_hits"]}</td></tr>'
        html+='</table>'
        if max(row["strict_score"] for row in pilot_strict["rows"]) < .1:
            html+='<p><b>Near-floor development scores:</b> this small local set provides little resolution for comparing the models. It does not establish equality or an architecture ranking. Revisit development feasibility before freezing the main suite and threshold.</p>'
        html+='<p>Greedy decoding, END=5 stopping, 32-token output cap, no response normalization. The check measures constrained output compliance, not semantic helpfulness. Main thresholds and an intermediate-checkpoint schedule remain unapproved.</p>'
    html+=figure+'<h2>Re-evaluated response prediction</h2>'+table
    html+=f'<p>Response NLL uses all {result["validation_examples"]} held-out examples and the original labels, including END=5, with CPU FP32 execution. The supplied logs used CUDA FP16. Small numerical differences are expected.</p>'
    html+=f'<p>Retention coverage: {escape(str(coverage))}; {escape(str(tokens))} target tokens per model. A positive change means worse pretraining NLL. A subset check is preliminary, not the full retention endpoint.</p>'
    html+='<p>The 100,000-token budget counts attempted supervised exposures, including skipped updates. Successful-update counts are shown separately. Two seeds provide limited evidence; no architecture ranking, significance or equivalence claim is made.</p>'
    html+=f'<p>Distinct normalized prompt groups shared by training and validation: {len({r["hash"] for r in result["data_audit"]["overlaps"] if r["kind"]=="prompt"})}. This does not rule out near duplicates or web contamination.</p>'
    html+='<p>Remaining primary requirement: one approved strict suite and intermediate evaluations or checkpoints for all arms and paired seeds. The two T_s0 recovery snapshots do not supply that grid. Pretraining need not be repeated.</p>'
    html+=f'<p>Evaluation identity: {escape(result["evaluation_id"])}</p></html>'
    (out/"report.html").write_text(html,encoding="utf-8")
    return out/"report.html"

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results",required=True);p.add_argument("--manifest",required=True);p.add_argument("--out",required=True)
    p.add_argument("--pilot-strict")
    a=p.parse_args(argv);print(render(read_json(a.results),a.manifest,a.out,read_json(a.pilot_strict) if a.pilot_strict else None))

if __name__=="__main__":main()

"""Fingerprint the received WS3/WS2 artifacts without modifying checkpoints."""
import argparse
import os
from pathlib import Path
from evaluation.common import atomic_json, sha256_file, require

def prepare(root, output):
    root=Path(root).resolve();output=Path(output).resolve()
    def asset(path):
        require(path.is_file(),f"Missing handoff file: {path}")
        return {"path":os.path.relpath(path,output.parent),"sha256":sha256_file(path),"bytes":path.stat().st_size}
    assets={name:asset(root/path) for name,path in {
        "tokenizer":"sft/tokenizer.json","training":"sft/pilot_train.jsonl",
        "validation":"sft/pilot_val.jsonl","pretraining_validation":"pretrain/val.bin",
        "pretraining_meta":"pretrain/meta.json"}.items()}
    runs=[]
    for arm in "TDP":
        for seed in (0,1):
            name=f"{arm}_s{seed}"; r={"run":name}
            for field,path in {"checkpoint":f"{name}/final.pt","parent":f"parent_{name}/final.pt",
                "run_config":f"{name}/run_config.json","metrics":f"{name}/metrics.jsonl",
                "done":f"{name}/done.json","parent_run_config":f"parent_{name}/run_config.json"}.items():
                r[field]=asset(root/path)
            runs.append(r)
    manifest={"schema_version":1,"purpose":"ws3_final_validation","assets":assets,"runs":runs,
        "integrity_note":"Local SHA-256 fingerprints; pretrained hashes also checked against WS3 configs. No external SFT final-file checksums were supplied.",
        "source_repo_commit":"015aae2","strict_benchmark":None}
    atomic_json(output,manifest);return manifest

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--handoff-root",required=True);p.add_argument("--out",required=True)
    a=p.parse_args(argv);prepare(a.handoff_root,a.out);print(a.out)

if __name__=="__main__":main()

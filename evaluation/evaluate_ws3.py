"""Evaluate existing six WS3 final checkpoints without inventing a learning curve.

No strict suite -> no strict scores, AUC or threshold claims. A retention subset is
explicitly labeled; pass --retention-windows 0 for the full frozen split.
"""
import argparse
from pathlib import Path
import platform
import time
from evaluation.common import atomic_json, canonical_hash, finite, read_json, require, resolve, sha256_file, verify_file
from evaluation.config import validate_tokenizer
from evaluation.data import leakage_audit
from evaluation.ws3 import audit_training_log, decoded_examples, load_ws3_checkpoint, read_jsonl, validate_tokenized

def evaluate(manifest_path, out, device="cpu", batch_size=4, retention_windows=32):
    import torch
    from tokenizers import Tokenizer
    from evaluation.metrics import response_nll
    from pretraining.dataset import PackedTokens
    from pretraining.evaluate import val_loss
    from models import lambda_log_dict
    from evaluation.run import source_hash
    m = read_json(manifest_path); base = Path(manifest_path).resolve().parent
    require(m.get("schema_version") == 1 and m.get("purpose") == "ws3_final_validation", "Wrong manifest type")
    expected = {f"{a}_s{s}" for a in "TDP" for s in (0,1)}
    require({r["run"] for r in m["runs"]} == expected and len(m["runs"]) == 6, "Need six distinct final runs")
    paths = {}
    for name in ("tokenizer", "training", "validation", "pretraining_validation", "pretraining_meta"):
        a = m["assets"][name]; paths[name] = resolve(base, a["path"]); verify_file(paths[name], a["sha256"])
    meta = read_json(paths["pretraining_meta"])
    require(meta["dtype"] == "uint16", "Wrong pretraining validation dtype")
    for filename, name in (("tokenizer.json","tokenizer"),("val.bin","pretraining_validation")):
        require(meta["files"][filename]["sha256"] == m["assets"][name]["sha256"], "WS2 data identity mismatch")
    for record in m["runs"]:
        for name in ("checkpoint","parent","run_config","metrics","done","parent_run_config"):
            asset=record[name];verify_file(resolve(base,asset["path"]),asset["sha256"])
    tok = Tokenizer.from_file(str(paths["tokenizer"]))
    training = read_jsonl(paths["training"]); validation = read_jsonl(paths["validation"])
    encoded_train = validate_tokenized(training, meta["vocab_size"], 1024)
    encoded_val = validate_tokenized(validation, meta["vocab_size"], 1024)
    audit = leakage_audit({"sft_train":decoded_examples(encoded_train,tok), "sft_validation":decoded_examples(encoded_val,tok)})
    # Preserve and report overlap findings; do not erase examples or pretend a clean audit.
    provenance = {"manifest_sha256":sha256_file(manifest_path), "code_sha256":source_hash(),
        "device":device, "torch":torch.__version__, "python":platform.python_version(),
        "precision":"float32", "amp":"none", "batch_size":batch_size, "retention_windows_limit":retention_windows,
        "strict_evaluation":"not run: no agreed benchmark found in supplied materials",
        "run_status":"pilot evidence: sft/prepare_data.py explicitly labels the dataset as pilot"}
    identity = canonical_hash(provenance); out = Path(out); out.mkdir(parents=True,exist_ok=True)
    output_path = out/"validation_results.json"
    result = {"schema_version":1, "evaluation_id":identity, "label":"WS3 PILOT VALIDATION - NOT THE PRIMARY STRICT LEARNING CURVE",
        "provenance":provenance, "data_audit":audit, "validation_examples":len(encoded_val),
        "strict_scores":None, "learning_curve_auc":None, "complete":False, "rows":[]}
    if output_path.exists():
        saved=read_json(output_path)
        require(saved["evaluation_id"] == identity, "Output from a different configuration; use a new output folder")
        result=saved
    torch.set_num_threads(min(torch.get_num_threads(),4))
    torch.use_deterministic_algorithms(True)
    shared_recipe = None; common_model_cfg = None; shared_pretraining = None; shared_pretraining_data = None
    for r in sorted(m["runs"],key=lambda r:r["run"]):
        cached = any(row["run"] == r["run"] for row in result["rows"])
        print("Verifying "+r["run"],flush=True)
        local={}
        for name in ("checkpoint","parent","run_config","metrics","done","parent_run_config"):
            a=r[name];local[name]=resolve(base,a["path"]);verify_file(local[name],a["sha256"])
        config=read_json(local["run_config"]); done=read_json(local["done"]); logs=read_jsonl(local["metrics"])
        require(r["run"] == f'{config["arm"]}_s{config["seed"]}', "Run name/config mismatch")
        require(config["train_sha256"] == m["assets"]["training"]["sha256"] and
            config["validation_sha256"] == m["assets"]["validation"]["sha256"],"SFT split mismatch")
        recipe={k:v for k,v in config.items() if k not in ("arm","seed","pretrained_sha256")}
        if shared_recipe is None: shared_recipe=recipe
        require(recipe == shared_recipe,"SFT recipes differ")
        counters=audit_training_log(config,logs,training)
        final_log=[x for x in logs if x["type"]=="val_final"][-1]
        require(done["final_val_nll"] == final_log["val_nll"] and done["tokens_seen"]==config["token_budget"],
            "Final log/completion mismatch")
        parent_cfg=read_json(local["parent_run_config"])
        require(parent_cfg["pretrain"]["arm"]==config["arm"] and parent_cfg["pretrain"]["seed"]==config["seed"],
            "Parent run config identity mismatch")
        require(parent_cfg["data_meta"]["files"]["tokenizer.json"]["sha256"]==m["assets"]["tokenizer"]["sha256"],
            "Parent tokenizer mismatch")
        pretrain_recipe=parent_cfg["pretrain"]
        comparable_pretrain={k:v for k,v in pretrain_recipe.items() if k not in
            ("arm","seed","data_dir","out_dir","device","compile","micro_batch","grad_accum")}
        comparable_pretrain["effective_batch"]=pretrain_recipe["micro_batch"]*pretrain_recipe["grad_accum"]
        data_hash=canonical_hash(parent_cfg["data_meta"])
        if shared_pretraining is None:shared_pretraining,shared_pretraining_data=comparable_pretrain,data_hash
        require(shared_pretraining==comparable_pretrain and shared_pretraining_data==data_hash,
            "Pretraining recipe/data mismatch across parents")
        if cached:continue
        start=time.perf_counter()
        parent, ck=__import__("models",fromlist=["load_checkpoint"]).load_checkpoint(local["parent"],device=device)
        require(ck["step"]==parent_cfg["pretrain"]["max_steps"] and ck["config"]==parent_cfg["model"],
            "Parent checkpoint/run config mismatch")
        validate_tokenizer(tok,parent.cfg.vocab_size)
        require(parent_cfg["data_meta"]["files"]["val.bin"]["sha256"]==m["assets"]["pretraining_validation"]["sha256"], "Parent validation data mismatch")
        cfg=parent.cfg.to_dict();comparable={k:v for k,v in cfg.items() if k!="ffn_hidden_p"}
        if common_model_cfg is None:common_model_cfg=comparable
        require(comparable==common_model_cfg,"Architecture config differs across arms/seeds")
        val=PackedTokens(str(paths["pretraining_validation"]),parent.cfg.context_len)
        require(val.n_windows>0,"Empty frozen pretraining validation")
        n=val.n_windows if retention_windows==0 else min(retention_windows,val.n_windows)
        baseline_ret=val_loss(parent,val,device,batch_size=1,amp="none",max_windows=n)
        baseline_response=response_nll(parent,encoded_val,batch_size=batch_size)
        del parent,ck
        model,info=load_ws3_checkpoint(local["checkpoint"],local["parent"],config,device=device)
        require(info["parameter_count"]==parent_cfg["n_params"],"Parent/SFT parameter-count mismatch")
        require(info["step"]==done["step"]==final_log["step"],"Checkpoint/log step mismatch")
        require(info["logged_final_val_nll"]==done["final_val_nll"],"Checkpoint/log stored NLL mismatch")
        final_response=response_nll(model,encoded_val,batch_size=batch_size)
        final_ret=val_loss(model,val,device,batch_size=1,amp="none",max_windows=n)
        finite(baseline_ret,"baseline retention");finite(final_ret,"final retention")
        first=encoded_val[0]
        probe=torch.tensor([first["ids"][:first["assistant_start"]]],dtype=torch.long,device=device)
        diagnostics=lambda_log_dict(model,probe) if config["arm"]=="D" else None
        row={"run":r["run"],"checkpoint":info,"checkpoint_sha256":r["checkpoint"]["sha256"],
            "parent_sha256":r["parent"]["sha256"],"token_accounting":counters,
            "parent_response_nll":baseline_response,"sft_response_nll":final_response,
            "logged_cuda_fp16_final_nll":done["final_val_nll"],
            "cpu_fp32_minus_logged_nll":final_response["nll"]-done["final_val_nll"],
            "retention":{"parent_nll":baseline_ret,"sft_nll":final_ret,"delta_nll":final_ret-baseline_ret,
                "windows":n,"target_tokens":n*val.seq_len,"total_complete_windows":val.n_windows,
                "coverage":"full" if n==val.n_windows else "fixed-prefix subset; preliminary"},
            "lambda_fixed_prompt":diagnostics,"lambda_probe_hash":canonical_hash(probe.cpu().tolist()),
            "elapsed_seconds":time.perf_counter()-start}
        result["rows"].append(row);atomic_json(output_path,result)
        print(r["run"]+" response NLL "+str(final_response["nll"]),flush=True)
        del model
    result["complete"]=len(result["rows"])==6
    atomic_json(output_path,result)
    return result

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",required=True);p.add_argument("--out",required=True)
    p.add_argument("--device",default="cpu");p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--retention-windows",type=int,default=32)
    a=p.parse_args(argv)
    require(a.batch_size>0 and a.retention_windows>=0,"Invalid evaluation limits")
    evaluate(a.manifest,a.out,a.device,a.batch_size,a.retention_windows)

if __name__=="__main__":main()

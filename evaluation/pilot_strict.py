"""Exploratory strict compliance on a separately frozen development subset.

The 300-prompt candidate set is intentionally not evaluated by this command.
No main-experiment AUC or threshold is computed from final checkpoints.
"""
import argparse
from pathlib import Path
import time
from evaluation.common import atomic_json,canonical_hash,read_json,require,resolve,sha256_file,verify_file
from evaluation.checks import CHECKER_VERSION,score_response
from evaluation.config import prompt_ids,validate_tokenizer
from evaluation.generate import greedy_generate,decode_generated
from evaluation.run import source_hash
from evaluation.scoring import summarize_predictions
from evaluation.ws3 import load_ws3_checkpoint,read_jsonl,validate_tokenized,decoded_examples

def evaluate(manifest_path,benchmark_root,out,device="cpu"):
    import torch
    from tokenizers import Tokenizer
    torch.set_num_threads(min(torch.get_num_threads(),2))
    torch.use_deterministic_algorithms(True)
    m=read_json(manifest_path);base=Path(manifest_path).resolve().parent
    benchmark_root=Path(benchmark_root);bench=read_json(benchmark_root/"manifest.json")
    suite=read_json(benchmark_root/"pilot_subset.json")
    require(canonical_hash(suite)==bench["pilot_hash"],"Pilot subset changed after its manifest")
    require(len(suite)==bench["pilot_subset_count"],"Wrong pilot prompt count")
    require(not bench["candidate_main_evaluated"],"Use a distinct protocol for a main evaluation")
    require(not bench["near_duplicate_flags"],"Review flagged near duplicates before this pilot")
    t=m["assets"]["tokenizer"];tok_path=resolve(base,t["path"]);verify_file(tok_path,t["sha256"])
    tok=Tokenizer.from_file(str(tok_path))
    for field in ("training","validation"):
        asset=m["assets"][field];path=resolve(base,asset["path"]);verify_file(path,asset["sha256"])
        decoded=decoded_examples(validate_tokenized(read_jsonl(path),tok.get_vocab_size(),1024),tok)
        name="sft_train" if field=="training" else "sft_validation"
        require(canonical_hash(decoded)==bench["audit_input_hashes"][name],"Benchmark audit used different SFT data")
    decoding=bench["pilot_decoding"]
    require(decoding["template"]=="nexolve_ws3_v1" and decoding["stop_ids"]==[5] and decoding["greedy"],
        "Unsupported WS3 pilot decoding")
    encoded={r["id"]:prompt_ids(tok,r["prompt"],"nexolve_ws3_v1") for r in suite}
    provenance={"manifest_sha256":sha256_file(manifest_path),"benchmark_manifest_sha256":sha256_file(benchmark_root/"manifest.json"),
        "pilot_subset_hash":bench["pilot_hash"],"code_sha256":source_hash(),"checker_version":CHECKER_VERSION,
        "device":device,"execution_dtype":"float32","amp":"none","torch":torch.__version__,"decoding":decoding}
    identity=canonical_hash(provenance);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    result_path=out/"pilot_strict_results.json"
    result={"schema_version":1,"evaluation_id":identity,"label":"EXPLORATORY DEVELOPMENT COMPLIANCE - NOT THE MAIN BENCHMARK",
        "provenance":provenance,"candidate_main_evaluated":False,"primary_auc":None,"threshold":None,
        "complete":False,"rows":[]}
    if result_path.exists():
        prior=read_json(result_path)
        require(prior["evaluation_id"]==identity,"Pilot output belongs to a different protocol")
        result=prior
    for run in sorted(m["runs"],key=lambda r:r["run"]):
        for key in ("checkpoint","parent","run_config"):
            asset=run[key];verify_file(resolve(base,asset["path"]),asset["sha256"])
        if any(r["run"]==run["run"] for r in result["rows"]):continue
        print("Pilot strict: "+run["run"],flush=True)
        config=read_json(resolve(base,run["run_config"]["path"]))
        model,info=load_ws3_checkpoint(resolve(base,run["checkpoint"]["path"]),resolve(base,run["parent"]["path"]),config,device=device)
        validate_tokenizer(tok,model.cfg.vocab_size)
        require(all(len(ids)+decoding["generation_cap"]<=model.cfg.context_len for ids in encoded.values()),
            "Pilot prompt plus cap exceeds model context")
        path=out/(run["run"]+".predictions.json")
        predictions=[]
        if path.exists():
            saved=read_json(path)
            require(saved["evaluation_id"]==identity,"Prediction cache identity mismatch")
            predictions=saved["predictions"]
            require(len(predictions)<=len(suite),"Too many cached predictions")
            for prior,row in zip(predictions,suite):
                require(prior["id"]==row["id"] and prior["family"]==row["family"],"Cached prompt mismatch")
                require(prior["text"]==decode_generated(tok,prior,[5]),"Cached decode mismatch")
                score=score_response(prior["text"],row["checks"])
                require(all(prior[k]==v for k,v in score.items()),"Cached score mismatch")
        started=time.perf_counter()
        for row in suite[len(predictions):]:
            generated=greedy_generate(model,encoded[row["id"]],decoding["generation_cap"],[5],amp="none")
            text=decode_generated(tok,generated,[5])
            predictions.append({"id":row["id"],"family":row["family"],"text":text,**generated,**score_response(text,row["checks"])})
            atomic_json(path,{"evaluation_id":identity,"checkpoint_sha256":run["checkpoint"]["sha256"],"predictions":predictions})
        result["rows"].append({"run":run["run"],"tokens_seen":info["tokens_seen"],
            **summarize_predictions(predictions),"cap_hits":sum(p["cap_hit"] for p in predictions),
            "generated_tokens":sum(p["generated_token_count"] for p in predictions),
            "checkpoint_sha256":run["checkpoint"]["sha256"],"seconds_this_invocation":time.perf_counter()-started})
        atomic_json(result_path,result)
        print(run["run"]+" development strict score "+str(result["rows"][-1]["strict_score"]),flush=True)
        del model
    result["complete"]=len(result["rows"])==6
    atomic_json(result_path,result);return result

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",required=True);p.add_argument("--benchmark-root",required=True);p.add_argument("--out",required=True)
    a=p.parse_args(argv);evaluate(a.manifest,a.benchmark_root,a.out)

if __name__=="__main__":main()

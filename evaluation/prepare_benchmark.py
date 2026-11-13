"""Build outcome-blind WS4 candidate/development constraints from the original role brief."""
import argparse
from pathlib import Path
from evaluation.common import atomic_json,canonical_hash,read_json,require
from evaluation.data import leakage_audit,normalized
from evaluation.ws3 import decoded_examples,read_jsonl,validate_tokenized

WORDS=["amber","cedar","river","stone","maple","cloud","silver","green","quiet","bright"]
def example(category,index,split):
    tag=f"{'DEV' if split=='development' else 'TEST'}-{index:03d}"
    dev=split=="development"
    if category=="exact":
        value=f"{'sample' if dev else 'signal'}{index:03d}"
        prompt=(f"Instruction {tag}: Reply with exactly {value}. Nothing else." if dev else
                f"For task {tag}, the entire response must be {value}. Include no other characters.")
        check={"kind":"exact","value":value};positive=value;negative=[value+"!"," "+value,value.upper()]
    elif category=="choice":
        choices=["red","blue"] if index%2 else ["yes","no"]
        prompt=(f"Instruction {tag}: Answer with only {choices[0]} or {choices[1]}. Choose one." if dev else
                f"Task {tag} accepts two possible replies: {choices[0]} and {choices[1]}. Output exactly one of those replies.")
        check={"kind":"choice","value":choices};positive=choices[0];negative=[choices[0]+" "+choices[1],choices[0]+".","unknown"]
    elif category=="lines":
        n=2+index%2
        prompt=(f"Instruction {tag}: Write exactly {n} nonempty lines. Use LF line breaks with no final newline or carriage return." if dev else
                f"For {tag}, return {n} lines of text. Every line must be nonempty. Separate lines by LF only; do not finish with a line break.")
        check={"kind":"lines","value":n};positive="\n".join(WORDS[:n]);negative=["one",positive+"\n",positive.replace("\n","\r\n")]
    elif category=="words":
        n=1+index%5
        prompt=(f"Instruction {tag}: Write exactly {n} ASCII-letter words, separated by single spaces. No other characters." if dev else
                f"For task {tag}, use {n} words made only of A-Z or a-z. Put one space between adjacent words and nothing before or after them.")
        check={"kind":"words","value":n};positive=" ".join(WORDS[:n]);negative=[positive+" extra",positive+"."," "+positive]
    elif category=="prefix":
        value=(["Note:","Answer:","Result:"] if dev else ["Summary:","Output:","Response:"])[index%3]
        prompt=(f"Instruction {tag}: Start your response with {value} exactly. You may add text after it." if dev else
                f"Task {tag} requires the exact prefix {value} at the very beginning of the response. Any text after that prefix is allowed.")
        check={"kind":"prefix","value":value};positive=value+" ready";negative=[" "+value,value.lower(),"Ready."]
    else:
        key=(["name","color","count"] if dev else ["label","status","item"])[index%3]
        prompt=(f'Instruction {tag}: Return only a valid JSON object with exactly the key "{key}". Do not repeat keys or add code fences.' if dev else
                f'For task {tag}, produce one JSON object whose only key is "{key}". Include no duplicate keys, Markdown fences or surrounding prose.')
        check={"kind":"json_keys","value":[key]};positive='{"'+key+'":"ok"}';negative=['{}','{"'+key+'":NaN}',positive+" extra"]
    return {"id":f"{split}_{category}_{index:03d}","family":category,"prompt":prompt,
        "checks":[check],"passing_response":positive,"failing_responses":negative}

def build(out,training_text=None,validation_text=None):
    out=Path(out);categories=["exact","choice","lines","words","prefix","json"]
    candidate=[example(c,i,"candidate") for c in categories for i in range(1,51)]
    development=[example(c,i,"development") for c in categories for i in range(1,11)]
    pilot=[r for c in categories for r in [x for x in development if x["family"]==c][:4]]
    from evaluation.checks import score_response
    for row in candidate+development:
        require(score_response(row["passing_response"],row["checks"])["strict_pass"],"Invalid positive control")
        require(all(not score_response(x,row["checks"])["strict_pass"] for x in row["failing_responses"]),"Invalid negative control")
    sets={"candidate":candidate,"development":development}
    if training_text is not None:sets["sft_train"]=training_text
    if validation_text is not None:sets["sft_validation"]=validation_text
    audit=leakage_audit(sets)
    problematic=[r for r in audit["overlaps"] if set(r["sets"]) & {"candidate","development"}]
    require(not problematic,"Evaluation/development prompts overlap other supplied splits")
    # Token-set similarity is a flag for review, not a proof about template contamination.
    training=[set(normalized(r["prompt"]).split()) for r in (training_text or [])]
    near=[]
    for row in candidate+development:
        words=set(normalized(row["prompt"]).split())
        similarity=max((len(words & t)/len(words | t) for t in training),default=0)
        if similarity>=.8:near.append({"id":row["id"],"max_token_set_jaccard":similarity})
    manifest={"schema_version":1,"scope":"WS4 local constrained-output compliance; no semantic-helpfulness claim",
        "status":"candidate main set; NOT team-approved or main-experiment-frozen",
        "source":"Original WS4 brief: build held-out set and deterministic scorer. Earlier 300-prompt design was a proposal.",
        "candidate_count":len(candidate),"development_count":len(development),"pilot_subset_count":len(pilot),
        "categories":categories,"candidate_hash":canonical_hash(candidate),"development_hash":canonical_hash(development),
        "pilot_hash":canonical_hash(pilot),"audit_input_hashes":{"sft_train":canonical_hash(training_text or []),"sft_validation":canonical_hash(validation_text or [])},"audit":audit,"near_duplicate_flags":near,
        "near_duplicate_method":"Word-set Jaccard >= 0.8 against SFT training; manual template review remains necessary.",
        "strict_outcomes_used_to_select_prompts":False,"main_score_threshold":None,
        "pilot_decoding":{"template":"nexolve_ws3_v1","stop_ids":[5],"generation_cap":32,"greedy":True,
            "normalization":"none; preserve whitespace and nonterminating special tokens"},
        "candidate_main_evaluated":False}
    atomic_json(out/"candidate.json",candidate);atomic_json(out/"development.json",development)
    atomic_json(out/"pilot_subset.json",pilot);atomic_json(out/"manifest.json",manifest)
    return manifest

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--out",required=True)
    p.add_argument("--training");p.add_argument("--validation");p.add_argument("--tokenizer")
    a=p.parse_args(argv);train=val=None
    if a.training or a.validation:
        require(a.training and a.validation and a.tokenizer,"Need both splits and tokenizer for audit")
        from tokenizers import Tokenizer
        tok=Tokenizer.from_file(a.tokenizer)
        train=decoded_examples(validate_tokenized(read_jsonl(a.training),tok.get_vocab_size(),1024),tok)
        val=decoded_examples(validate_tokenized(read_jsonl(a.validation),tok.get_vocab_size(),1024),tok)
    print(build(a.out,train,val)["status"])

if __name__=="__main__":main()

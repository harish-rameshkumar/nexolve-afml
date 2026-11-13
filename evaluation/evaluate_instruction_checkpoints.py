"""Run the final applied-token checkpoint grid using a frozen protocol."""
from pathlib import Path
from collections import Counter
from statistics import mean
import argparse, hashlib, importlib.util, json, time

ROOT=Path(__file__).resolve().parents[1]
import torch
from tokenizers import Tokenizer
from configs.model_config import ModelConfig
from models import build_model
from evaluation.analysis import normalized_auc,paired_summary
from evaluation.config import prompt_ids
from evaluation.generate import decode_generated,greedy_generate
from evaluation.common import atomic_json

def read(p):
    return json.loads(p.read_bytes())

def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def require(value,message):
    if not value:
        raise ValueError(message)

def rotate(attention,tensor,positions):
    a,b=tensor.chunk(2,dim=-1)
    cos=attention.rope.cos[positions].unsqueeze(-2)
    sin=attention.rope.sin[positions].unsqueeze(-2)
    return torch.cat((a*cos-b*sin,b*cos+a*sin),dim=-1)

@torch.inference_mode()
def generate_batch(model,prefixes,cap,stop):
    """Cache the projected keys and values for causal greedy decoding."""
    import torch.nn.functional as F
    size=len(prefixes)
    lengths=torch.tensor([len(p) for p in prefixes])
    width=int(lengths.max())
    capacity=width+cap
    inputs=torch.full((size,width),1,dtype=torch.long)
    for i,ids in enumerate(prefixes):
        inputs[i,:len(ids)]=torch.tensor(ids)
    rows=torch.arange(size)
    positions=torch.arange(width).expand(size,-1)
    result=[[] for _ in prefixes]
    reasons=['cap']*size
    alive=torch.ones(size,dtype=torch.bool)
    cache=[]
    for step in range(cap):
        x=model.embed(inputs)
        for layer,block in enumerate(model.blocks):
            attn=block.attn
            z=block.attn_norm(x)
            h,d=attn.n_heads,attn.head_dim
            diff=model.arm=='D'
            qh=h*2 if diff else h
            q=rotate(attn,attn.q_proj(z).reshape(size,-1,qh,d),positions).transpose(1,2)
            k=rotate(attn,attn.k_proj(z).reshape(size,-1,h,d),positions).transpose(1,2)
            v=attn.v_proj(z).reshape(size,-1,h,d).transpose(1,2)
            if diff:
                k=k.repeat_interleave(2,dim=1)
                v=v.repeat_interleave(2,dim=1)
            if step==0:
                saved_k=k.new_zeros(size,qh,capacity,d)
                saved_v=v.new_zeros(size,qh,capacity,d)
                saved_k[:,:,:width]=k
                saved_v[:,:,:width]=v
                cache.append((saved_k,saved_v))
                o=F.scaled_dot_product_attention(q,k,v,is_causal=True)
            else:
                saved_k,saved_v=cache[layer]
                saved_k[rows,:,positions[:,0]]=k[:,:,0]
                saved_v[rows,:,positions[:,0]]=v[:,:,0]
                used=int(positions.max())+1
                mask=torch.arange(used)[None,None,None,:]<=positions[:,None,:,None]
                o=F.scaled_dot_product_attention(q,saved_k[:,:,:used],saved_v[:,:,:used],attn_mask=mask)
            o=o.transpose(1,2)
            if diff:
                lam=torch.sigmoid(attn.lambda_proj(z)).unsqueeze(-1)
                o=o[:,:,0::2]-lam*o[:,:,1::2]
            x=x+attn.o_proj(o.reshape(size,-1,h*d))
            x=x+block.ffn(block.ffn_norm(x))
        last=x[rows,lengths-1] if step==0 else x[:,0]
        logits=model.lm_head(model.final_norm(last)).float()
        require(torch.isfinite(logits).all(),'Nonfinite logits')
        tokens=logits.argmax(dim=-1)
        for i in alive.nonzero().flatten().tolist():
            token=int(tokens[i])
            result[i].append(token)
            if token in stop:
                reasons[i]='stop'
                alive[i]=False
        if not alive.any():
            break
        inputs=tokens[:,None]
        positions=(lengths+step)[:,None]
    return [dict(generated_ids=ids,stop_reason=r,generated_token_count=len(ids),cap_hit=r=='cap')
            for ids,r in zip(result,reasons)]

def load_target(record,name,budget,asset_root=ROOT):
    base=asset_root/'work/ws3-final'
    run=record['runs'][name]
    parent=record['pretrained_parents'][name]
    row=next((r for r in run['snapshots'] if r['requested_budget']==budget),None)
    path=(base/name/f'sft_{budget}.pt') if budget else (asset_root/'work/handoff'/f'parent_{name}'/'final.pt')
    expected=row['sha256'] if budget else parent['sha256']
    require(sha(path)==expected,f'{name} at {budget}: hash mismatch')
    state=torch.load(path,map_location='cpu',weights_only=True)
    require(state['arm']==run['arm'] and state['config']==parent['model_config'],f'{name}: model identity mismatch')
    extra=state['extra']
    if budget:
        require(extra['pretraining_seed']==run['pretraining_seed'] and extra['sft_seed']==run['sft_seed'],'Seed mismatch')
        require(extra['requested_budget']==extra['applied_tokens']==row['actual_applied_response_tokens']==budget,'Applied count mismatch')
        require(extra['attempted_tokens']>=budget and state['step']==row['optimizer_step_attempts'],'Counter mismatch')
        require(extra['successful_updates']==row['successful_optimizer_updates'] and extra['skipped_updates']==row['skipped_optimizer_updates'],'Update count mismatch')
        require(extra['sft_config']==run['training_config'] and extra['sft_config']['pretrained_sha256']==parent['sha256'],'Training provenance mismatch')
    else:
        require(extra['seed']==parent['seed'],'Parent seed mismatch')
    model=build_model(state['arm'],ModelConfig.from_dict(state['config']))
    model.load_state_dict(state['model'],strict=True)
    return model.eval(),expected,(extra['applied_tokens'] if budget else 0)

def read_benchmark(directory,protocol):
    for name,expected in protocol['asset_hashes'].items():
        require(sha(directory/name)==expected,f'Frozen asset changed: {name}')
    spec=importlib.util.spec_from_file_location('frozen_scorer',directory/'scorer.py')
    scorer=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    prompts=read(directory/'prompts.json')
    require(len({p['id'] for p in prompts})==len(prompts),'Repeated prompt IDs')
    for p in prompts:
        require(scorer.score(p['positive_control'],p['check'])['strict_pass'],'Passing control failed')
        require(not scorer.score(p['negative_control'],p['check'])['strict_pass'],'Failing control passed')
    return prompts,scorer

def score_model(model,prompts,tok,scorer,protocol):
    started=time.perf_counter()
    results=[]
    batch_size=protocol['batch_size']
    for offset in range(0,len(prompts),batch_size):
        group=prompts[offset:offset+batch_size]
        ids=[prompt_ids(tok,p['prompt'],protocol['prompt_template']) for p in group]
        cap,stop=protocol['max_new_tokens'],protocol['stop_token_ids']
        require(all(len(x)+cap<=model.cfg.context_len for x in ids),'Prompt exceeds context')
        generations=generate_batch(model,ids,cap,stop)
        for row,generation in zip(group,generations):
            response=decode_generated(tok,generation,stop)
            results.append(dict(id=row['id'],family=row['family'],response=response,**generation,**scorer.score(response,row['check'])))
    counts=Counter(r['family'] for r in results)
    family={f:sum(r['strict_pass'] for r in results if r['family']==f)/n for f,n in counts.items()}
    return {'strict_score':mean(r['strict_pass'] for r in results),'cap_hits':sum(r['cap_hit'] for r in results),
            'family_scores':family,'seconds':time.perf_counter()-started,'predictions':results}

def analyze(rows,grid):
    curves={}
    for arm in 'TDP':
        for seed in (0,1):
            name=f'{arm}_s{seed}'
            points=[(r['applied_response_tokens'],r['strict_score']) for r in rows if r['run']==name]
            auc=normalized_auc(points,grid)
            curves[name]={'points':points,'auc_normalized':auc,'auc_raw':auc*grid[-1]}
    paired={f'D-{a}':paired_summary([curves[f'D_s{s}']['auc_normalized']-curves[f'{a}_s{s}']['auc_normalized'] for s in (0,1)]) for a in 'TP'}
    return {'curves':curves,'paired_auc':paired}

def main():
    cli=argparse.ArgumentParser()
    cli.add_argument('--protocol',type=Path,required=True)
    cli.add_argument('--out',type=Path,required=True)
    cli.add_argument('--asset-root', type=Path, required=True)
    cli.add_argument('--smoke',action='store_true')
    args=cli.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    path=args.asset_root/'work/ws3-final/training_record.json'
    record=read(path)
    tok_path=args.asset_root/'work/handoff/sft/tokenizer.json'
    require(sha(tok_path)==record['shared_assets']['tokenizer']['sha256'],'Tokenizer hash mismatch')
    tok=Tokenizer.from_file(str(tok_path))
    protocol=read(args.protocol)
    require(protocol['purpose']==('development' if args.smoke else 'descriptive_final_grid'),'Wrong evaluation protocol')
    prompts,scorer=read_benchmark(args.protocol.parent,protocol)
    grid=protocol['applied_response_token_budgets']
    keys=[('T_s0',0),('T_s0',100000)] if args.smoke else [(f'{a}_s{s}',b) for a in 'TDP' for s in (0,1) for b in grid]
    identity={'protocol_sha256':sha(args.protocol),'training_record_sha256':sha(path),
              'evaluator_sha256':sha(Path(__file__)),'torch':torch.__version__,'device':'cpu','mode':'FP32'}
    output=read(args.out) if args.out.exists() else {'identity':identity,'rows':[],'complete':False}
    require(output['identity']==identity,'Existing results belong to a different protocol')
    for name,budget in keys:
        model,checkpoint,actual=load_target(record,name,budget,args.asset_root)
        if any(r['run']==name and r['applied_response_tokens']==actual and r['checkpoint_sha256']==checkpoint for r in output['rows']):
            continue
        if not any(r['run'][0]==name[0] for r in output['rows']):
            ids=[prompt_ids(tok,p['prompt'],protocol['prompt_template']) for p in prompts[:3]]
            cap=protocol['max_new_tokens']
            stop=protocol['stop_token_ids']
            require(generate_batch(model,ids,cap,stop)==[greedy_generate(model,x,cap,stop) for x in ids],'Batch and native greedy outputs disagree in smoke check')
        scored=score_model(model,prompts,tok,scorer,protocol)
        if name.startswith('D'):
            from models.diagnostics import lambda_summary
            probe=prompt_ids(tok,protocol['lambda_probe'],protocol['prompt_template'])
            scored['lambda_probe']={k:v.tolist() for k,v in lambda_summary(model,torch.tensor([probe])).items()}
        output['rows'].append(dict(run=name,applied_response_tokens=actual,checkpoint_sha256=checkpoint,**scored))
        atomic_json(args.out,output)
        print(name,actual,scored['strict_score'],scored['cap_hits'],round(scored['seconds'],1),flush=True)
    output['complete']=True
    if not args.smoke:
        output['analysis']=analyze(output['rows'],grid)
    atomic_json(args.out,output)

if __name__=='__main__':
    main()

"""Standard-library tests; optional model tests skip explicitly without torch."""
import copy
import importlib.util
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from evaluation.analysis import analyze, normalized_auc, paired_summary, select_loss_matched, threshold_attainment
from evaluation.checks import check_response, score_response
from evaluation.common import atomic_json, canonical_hash, read_json, sha256_file, verify_file
from evaluation.config import prompt_ids, reference_ids, validate_protocol, validate_tokenizer
from evaluation.data import leakage_audit, load_assets
from evaluation.demo import demo
from evaluation.manifest import validate_manifest
from evaluation.report import render
from evaluation.run import preflight, validate_parent_contract
from evaluation.scoring import summarize_predictions

ROOT=Path(__file__).resolve().parents[1]
FIX=ROOT/"tests/fixtures/ws4"
def protocol(): return read_json(ROOT/"evaluation/protocol.example.json")

class Tok:
    def encode(self,text,add_special_tokens=False): return SimpleNamespace(ids=[ord(c)+8 for c in text])
    def get_vocab_size(self): return 512
    def token_to_id(self,token):
        from evaluation.config import SPECIAL_TOKENS
        return SPECIAL_TOKENS.index(token)

def make_manifest(folder,p=None):
    p=p or protocol(); folder=Path(folder)
    suite,refs=load_assets(FIX/"suite.json",FIX/"references.json")
    atomic_json(folder/"suite.json",suite); atomic_json(folder/"refs.json",refs)
    atomic_json(folder/"tokenizer.json",{"fake":"preflight only"})
    (folder/"val.bin").write_bytes(bytes(1026))
    assets={}
    for name,filename in (("suite","suite.json"),("references","refs.json"),("tokenizer","tokenizer.json"),("val_tokens","val.bin")):
        assets[name]={"path":filename,"sha256":sha256_file(folder/filename)}
    meta={"dtype":"uint16","vocab_size":512,"files":{}}
    for filename,name in (("tokenizer.json","tokenizer"),("val.bin","val_tokens")):
        meta["files"][filename]={"sha256":assets[name]["sha256"],"bytes":(folder/filename).stat().st_size}
    atomic_json(folder/"meta.json",meta)
    assets["val_meta"]={"path":"meta.json","sha256":sha256_file(folder/"meta.json")}
    template=canonical_hash(p["template"]); stream=canonical_hash(["synthetic","ordered"])
    rows=[]
    for arm in ("T","D","P"):
        for seed in p["paired_seeds"]:
            parent=None
            for b in p["budgets"]:
                filename=f"{arm}_s{seed}_b{b}.pt"
                (folder/filename).write_bytes(f"fake checkpoint {filename}".encode())
                ch=sha256_file(folder/filename)
                if b==0: parent=ch
                rows.append({"arm":arm,"pretrain_seed":seed,"sft_seed":seed,"step":int(b>0),
                    "phase":"pretrain" if b==0 else "sft","checkpoint_path":filename,"checkpoint_sha256":ch,
                    "parent_pretrain_sha256":parent,"target_response_tokens":b,"unique_response_tokens":b,
                    "processed_response_tokens":b,"chat_template_sha256":template,"sft_stream_sha256":stream})
    m={"schema_version":1,"experiment":p["experiment"],"unique_count_includes_eos":p["unique_count_includes_eos"],**assets,"chat_template_sha256":template,
        "sft_stream_sha256":stream,"checkpoints":rows}
    atomic_json(folder/"manifest.json",m); atomic_json(folder/"protocol.json",p)
    return m

class CheckerTests(unittest.TestCase):
    def test_fixture_controls(self):
        suite,_=load_assets(FIX/"suite.json",FIX/"references.json")
        for row in suite:
            with self.subTest(id=row["id"]):
                self.assertTrue(score_response(row["passing_response"],row["checks"])["strict_pass"])
                for text in row["failing_responses"]:
                    self.assertFalse(score_response(text,row["checks"])["strict_pass"])
    def test_json_adversarial(self):
        c={"kind":"json_keys","value":["a"]}
        for text in ('{"a":NaN}','{"a":Infinity}','{"a":1,"a":2}','{"a":{"b":1,"b":2}}',
                     '{"a":1} extra','{"a":1,"b":2}','["a"]'):
            with self.subTest(text=text): self.assertFalse(check_response(text,c))
        self.assertTrue(check_response(' {"a":1} ',c))
    def test_case_requires_letters(self):
        self.assertFalse(check_response("123",{"kind":"lowercase"}))
        self.assertFalse(check_response("",{"kind":"uppercase"}))
    def test_empty_response_fails(self):
        self.assertFalse(score_response("",[{"kind":"forbids","value":"bad"}])["strict_pass"])
    def test_all_constraints_required(self):
        r=score_response("River",[{"kind":"contains","value":"River"},{"kind":"lowercase"}])
        self.assertEqual(r["constraint_passes"],[True,False]); self.assertFalse(r["strict_pass"])
    def test_invalid_checkers(self):
        for c in ({"kind":"unknown"},{"kind":"lines","value":True},{"kind":"contains","value":""},
                  {"kind":"lowercase","value":"x"}):
            with self.subTest(check=c),self.assertRaises(ValueError): check_response("text",c)
    def test_aggregation_denominators(self):
        rows=[{"id":"1","family":"a","strict_pass":True,"constraint_passes":[True]},
              {"id":"2","family":"b","strict_pass":False,"constraint_passes":[True,False,False]}]
        s=summarize_predictions(rows)
        self.assertEqual(s["strict_score"],.5); self.assertEqual(s["constraint_score"],.5)
        with self.assertRaises(ValueError): summarize_predictions(rows+rows)

class ProtocolTests(unittest.TestCase):
    def test_freeze_requires_ws3(self):
        p=protocol(); p["status"]="frozen"
        with self.assertRaises(ValueError): validate_protocol(p)
    def test_invalid_values(self):
        for k,v in (("budgets",[0,1,1]),("threshold",float("nan")),("paired_seeds",[0,0]),
                    ("generation_cap",True),("amp","fp64")):
            p=protocol(); p[k]=v
            with self.subTest(field=k),self.assertRaises(ValueError): validate_protocol(p)
    def test_golden_formatter(self):
        self.assertEqual(prompt_ids(Tok(),"a"),[3,105,5,4])
        ids,labels=reference_ids(Tok(),{"prompt":"a","response":"b"},True)
        self.assertEqual(ids,[3,105,5,4,106,0]); self.assertEqual(labels,[-100]*4+[106,0])
        self.assertEqual(reference_ids(Tok(),{"prompt":"a","response":"b"},False)[0][-1],106)
    def test_tokenizer_validation(self):
        validate_tokenizer(Tok(),512)
        with self.assertRaises(ValueError): validate_tokenizer(Tok(),128)
    def test_control_injection(self):
        class Bad(Tok):
            def encode(self,*a,**kw): return SimpleNamespace(ids=[3])
        with self.assertRaises(ValueError): prompt_ids(Bad(),"text")

class ManifestTests(unittest.TestCase):
    def test_normalized_leakage(self):
        self.assertTrue(leakage_audit({"train":[{"prompt":"HELLO  world"}],
            "eval":[{"prompt":" hello world "}]})["overlaps"])
    def test_disjoint(self):
        self.assertFalse(leakage_audit({"a":[{"prompt":"a"}],"b":[{"prompt":"b"}]})["overlaps"])
    def test_complete_preflight(self):
        with tempfile.TemporaryDirectory() as d:
            m=make_manifest(d); validate_manifest(m,protocol())
            result=preflight(Path(d)/"protocol.json",Path(d)/"manifest.json")
            self.assertEqual(len(result[1]["checkpoints"]),18)
    def test_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            make_manifest(d); (Path(d)/"val.bin").write_bytes(b"bad")
            with self.assertRaisesRegex(ValueError,"SHA-256 mismatch"):
                preflight(Path(d)/"protocol.json",Path(d)/"manifest.json")
    def test_bad_manifest_variants(self):
        with tempfile.TemporaryDirectory() as d:
            m=make_manifest(d)
            for variant in ("missing","duplicate","overshoot","repeats","parent","template","seed"):
                c=copy.deepcopy(m)
                if variant=="missing": c["checkpoints"].pop()
                if variant=="duplicate": c["checkpoints"].append(c["checkpoints"][0])
                if variant=="overshoot": c["checkpoints"][1]["unique_response_tokens"]+=1
                if variant=="repeats": c["checkpoints"][1]["processed_response_tokens"]+=1
                if variant=="parent": c["checkpoints"][1]["parent_pretrain_sha256"]="f"*64
                if variant=="template": c["checkpoints"][1]["chat_template_sha256"]="f"*64
                if variant=="seed": c["checkpoints"][1]["sft_seed"]=99
                with self.subTest(variant=variant),self.assertRaises(ValueError): validate_manifest(c,protocol())
    def test_ws2_recipe_tuple_list_roundtrip(self):
        recipe={"arm":"T","seed":0,"max_steps":4600,"betas":[.9,.95],"lr":.002}
        ck={"config":{"vocab_size":16384},"step":4600,"extra":{"pretrain":{**recipe,"betas":(.9,.95)}}}
        record={"arm":"T","pretrain_seed":0}
        meta={"files":{"tokenizer.json":{"sha256":"a"*64},"val.bin":{"sha256":"b"*64}}}
        manifest={"tokenizer":{"sha256":"a"*64},"val_tokens":{"sha256":"b"*64}}
        config={"pretrain":recipe,"model":ck["config"],"data_meta":meta}
        comparable,_=validate_parent_contract(record,ck,config,manifest,"primary")
        self.assertEqual(comparable["lr"],.002)
        ck["step"]=4500
        with self.assertRaises(ValueError): validate_parent_contract(record,ck,config,manifest,"primary")
        ck["step"]=4600; ck["extra"]["pretrain"]["lr"]=.001
        with self.assertRaises(ValueError): validate_parent_contract(record,ck,config,manifest,"primary")

    def test_atomic_json(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"result.json"; atomic_json(path,{"a":1}); atomic_json(path,{"a":2})
            verify_file(path,sha256_file(path))
            self.assertEqual(read_json(path),{"a":2})
            with self.assertRaises(ValueError): atomic_json(path,{"a":float("nan")})
            self.assertEqual(read_json(path),{"a":2})
            self.assertFalse(list(Path(d).glob("*.tmp")))

class AnalysisTests(unittest.TestCase):
    def test_auc_uneven(self):
        self.assertAlmostEqual(normalized_auc([(0,.4),(3,.4),(10,.4)]),.4)
        self.assertAlmostEqual(normalized_auc([(0,0),(3,.3),(10,1)]),.5)
    def test_auc_rejects_bad_grid(self):
        for ps in ([(1,0),(2,1)],[(0,0),(0,1)],[(0,0),(1,float("nan"))],[(0,0),(1,2)]):
            with self.subTest(points=ps),self.assertRaises(ValueError): normalized_auc(ps)
        with self.assertRaises(ValueError): normalized_auc([(0,0),(10,1)],[0,5,10])
    def test_threshold_censoring_regression(self):
        r=threshold_attainment([(0,0),(10,.6),(20,.2)],.5)
        self.assertEqual(r["crossing_interval"],[0,10]); self.assertFalse(r["remains_above"])
        r=threshold_attainment([(0,0),(10,.2)],.5)
        self.assertIsNone(r["observed_tokens"]); self.assertEqual(r["greater_than_tokens"],10)
    def test_threshold_baseline(self):
        self.assertEqual(threshold_attainment([(0,.5),(10,.6)],.5)["crossing_interval"],[0,0])
    def test_small_seed_resolution(self):
        self.assertEqual(paired_summary([.1,.2])["descriptive_two_sided_sign_flip_p"],.5)
        self.assertEqual(paired_summary([.1,.2,.3])["descriptive_two_sided_sign_flip_p"],.25)
        self.assertEqual(paired_summary([0,0])["descriptive_two_sided_sign_flip_p"],1)
    def test_loss_matching_available_and_unavailable(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"weights.pt"; path.write_bytes(b"fixture")
            cs=[{"arm":a,"pretrain_seed":0,"step":10,"loaded_val_nll":4.,"path":str(path)} for a in ("T","D","P")]
            self.assertTrue(select_loss_matched(cs,0,4,.01)["available"])
            self.assertFalse(select_loss_matched(cs,0,3,.01)["available"])
            self.assertFalse(select_loss_matched(cs[:2],0,4,.01)["available"])
    def test_replay_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            r=demo(d); a=read_json(Path(d)/"analysis.json")
            self.assertEqual(len(a["runs"]),6); self.assertEqual(a["paired_auc"]["D-T"]["mean"],0)
            self.assertIn("SYNTHETIC REPLAY",(Path(d)/"report.html").read_text())
            self.assertTrue((Path(d)/"strict_curves.svg").exists())
            self.assertFalse((Path(d)/"response_nll.svg").exists())
            self.assertTrue((Path(d)/"summary.csv").exists()); self.assertEqual(r,demo(d))
            for variant in ("missing","duplicate","incomplete"):
                c=copy.deepcopy(r)
                if variant=="missing": c["rows"].pop()
                if variant=="duplicate": c["rows"].append(c["rows"][0])
                if variant=="incomplete": c["complete"]=False
                with self.subTest(variant=variant),self.assertRaises(ValueError): analyze(c)

TORCH=importlib.util.find_spec("torch") is not None
TOKENIZERS=importlib.util.find_spec("tokenizers") is not None

@unittest.skipUnless(TORCH,"torch unavailable: model execution not verified")
class ModelTests(unittest.TestCase):
    def scripted(self,tokens):
        import torch
        class Scripted(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.w=torch.nn.Parameter(torch.zeros(1)); self.calls=0
                self.cfg=SimpleNamespace(context_len=32,vocab_size=16)
            def forward(self,x):
                logits=torch.full((*x.shape,16),-20.,device=x.device)
                logits[:,-1,tokens[min(self.calls,len(tokens)-1)]]=20.
                self.calls+=1
                return logits
        return Scripted()
    def test_greedy_eos_and_mode(self):
        from evaluation.generate import greedy_generate
        m=self.scripted([9,0]); r=greedy_generate(m,[3,8,5,4],5)
        self.assertEqual(r["generated_ids"],[9,0]); self.assertEqual(r["stop_reason"],"stop")
        self.assertTrue(m.training); self.assertIsNone(m.w.grad)
    def test_first_eos_cap_and_context(self):
        from evaluation.generate import greedy_generate
        self.assertEqual(greedy_generate(self.scripted([0]),[3],3)["generated_ids"],[0])
        self.assertTrue(greedy_generate(self.scripted([9]),[3],3)["cap_hit"])
        with self.assertRaises(ValueError): greedy_generate(self.scripted([9]),[3]*31,3)
    def test_response_nll_hand_computed(self):
        import torch
        from evaluation.metrics import response_nll
        class Constant(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.w=torch.nn.Parameter(torch.zeros(1))
                self.cfg=SimpleNamespace(context_len=32,vocab_size=3)
            def forward(self,x):
                return torch.tensor([0.,math.log(2),math.log(3)],device=x.device).expand(*x.shape,3)+self.w*0
        m=Constant()
        r=response_nll(m,[{"id":"a","ids":[0,1],"labels":[-100,1]},
            {"id":"b","ids":[0,2,2,2],"labels":[-100,2,2,2]}])
        self.assertAlmostEqual(r["nll"],(-math.log(2/6)-3*math.log(3/6))/4,places=6)
        self.assertEqual(r["target_tokens"],4); self.assertTrue(m.training); self.assertIsNone(m.w.grad)
        with self.assertRaises(ValueError):
            response_nll(m,[{"id":"bad","ids":[0,1],"labels":[-100,-100]}])
    def test_decode_preserves_specials(self):
        from evaluation.generate import decode_generated
        class Decoder:
            def decode(self,ids,skip_special_tokens): return str((ids,skip_special_tokens))
        self.assertEqual(decode_generated(Decoder(),{"generated_ids":[3,9,0],"stop_reason":"stop"},[0]),"([3, 9], False)")
    @unittest.skipUnless(TOKENIZERS,"tokenizers unavailable")
    def test_native_runner_resume(self):
        import array
        import torch
        from configs import default_config
        from models import build_model,save_checkpoint
        from pretraining.tokenizer import train_tokenizer
        from evaluation.run import evaluate
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            d=Path(directory); p=protocol(); p["budgets"]=[0,4]; p["generation_cap"]=2
            m=make_manifest(d,p)
            atomic_json(d/"suite.json",read_json(FIX/"suite.json")[:2])
            atomic_json(d/"refs.json",read_json(FIX/"references.json")[:1])
            texts=[r["prompt"] for r in read_json(FIX/"suite.json")]
            texts += [r["response"] for r in read_json(FIX/"references.json")]
            tok=train_tokenizer(texts,300); tok.save(str(d/"tokenizer.json"))
            cfg=default_config(vocab_size=300,context_len=512,d_model=16,n_layers=1,n_heads=2,ffn_hidden=48,ffn_hidden_p=54)
            (d/"val.bin").write_bytes(array.array("H",([8,9]*257)[:513]).tobytes())
            for name in ("tokenizer","suite","references","val_tokens"):
                m[name]["sha256"]=sha256_file(d/m[name]["path"])
            meta=read_json(d/"meta.json"); meta["vocab_size"]=300
            for filename,name in (("tokenizer.json","tokenizer"),("val.bin","val_tokens")):
                meta["files"][filename]={"sha256":m[name]["sha256"],"bytes":(d/filename).stat().st_size}
            atomic_json(d/"meta.json",meta); m["val_meta"]["sha256"]=sha256_file(d/"meta.json")
            parents={}
            for r in m["checkpoints"]:
                torch.manual_seed(r["pretrain_seed"]); model=build_model(r["arm"],cfg)
                key=(r["arm"],r["pretrain_seed"]); extra={"seed":r["pretrain_seed"]}
                if r["phase"]=="sft":
                    for field in ("sft_seed","unique_response_tokens","processed_response_tokens",
                                  "chat_template_sha256","sft_stream_sha256"): extra[field]=r[field]
                    extra["unique_count_includes_eos"]=p["unique_count_includes_eos"]
                    extra["parent_pretrain_sha256"]=parents[key]; extra["tokenizer_sha256"]=m["tokenizer"]["sha256"]
                save_checkpoint(d/r["checkpoint_path"],model,step=r["step"],extra=extra)
                r["checkpoint_sha256"]=sha256_file(d/r["checkpoint_path"])
                if r["phase"]=="pretrain": parents[key]=r["checkpoint_sha256"]
                r["parent_pretrain_sha256"]=parents[key]
            atomic_json(d/"manifest.json",m)
            result=evaluate(d/"protocol.json",d/"manifest.json",d/"results",allow_provisional=True)
            self.assertTrue(result["complete"]); self.assertEqual(len(result["rows"]),12)
            render(analyze(result),d/"report")
            again=evaluate(d/"protocol.json",d/"manifest.json",d/"results",allow_provisional=True)
            self.assertEqual([r["strict_score"] for r in result["rows"]],[r["strict_score"] for r in again["rows"]])

if __name__=="__main__": unittest.main()

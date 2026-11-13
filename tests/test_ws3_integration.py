"""Actual-format regression checks independent of the production handoff files."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from evaluation.config import prompt_ids,reference_ids
from evaluation.ws3 import audit_training_log,validate_tokenized
from tests.test_ws4 import Tok

class FormatTests(unittest.TestCase):
    def test_actual_ws3_formatter(self):
        self.assertEqual(prompt_ids(Tok(),"a","nexolve_ws3_v1"),[3,105,4])
        ids,labels=reference_ids(Tok(),{"prompt":"a","response":"b"},True,"nexolve_ws3_v1")
        self.assertEqual(ids,[3,105,4,106,5]);self.assertEqual(labels,[-100,-100,-100,106,5])
    def test_tokenized_masks(self):
        row={"input_ids":[3,8,4,9,5],"labels":[-100,-100,-100,9,5]}
        self.assertEqual(validate_tokenized([row],16,8)[0]["assistant_start"],3)
        for bad in ([3,8,4,9,0],[3,8,4,9,16]):
            with self.assertRaises(ValueError):validate_tokenized([{**row,"input_ids":bad}],16,8)
    def test_mask_hole_rejected(self):
        with self.assertRaises(ValueError):
            validate_tokenized([{"input_ids":[3,8,4,9,5],"labels":[-100,-100,-100,-100,5]}],16,8)
    def test_attempts_vs_successful_updates(self):
        examples=[{"labels":[-100,-100,8,5]},{"labels":[-100,-100,9,5]}]
        logs=[{"type":"train_skipped_overflow","step":1,"tokens_seen":2,"batch_supervised_tokens":2},
            {"type":"train","step":2,"tokens_seen":3,"batch_supervised_tokens":1}]
        r=audit_training_log({"seed":0,"token_budget":3},logs,examples)
        self.assertEqual(r["attempted_response_tokens"],3)
        self.assertEqual(r["successful_update_response_tokens"],1)
    def test_incorrect_counter_fails(self):
        with self.assertRaises(ValueError):
            audit_training_log({"seed":0,"token_budget":2},[{"type":"train","step":1,"tokens_seen":3,"batch_supervised_tokens":2}],
                [{"labels":[-100,8,5]}])

class BenchmarkTests(unittest.TestCase):
    def test_added_constraint_semantics(self):
        from evaluation.checks import score_response
        cases=[("yes",{"kind":"choice","value":["yes","no"]},True),
            ("yes.",{"kind":"choice","value":["yes","no"]},False),
            ("red blue",{"kind":"words","value":2},True),
            ("red  blue",{"kind":"words","value":2},False),
            ("red blue.",{"kind":"words","value":2},False),
            ("Note: ready",{"kind":"prefix","value":"Note:"},True),
            (" Note: ready",{"kind":"prefix","value":"Note:"},False)]
        for text,check,expected in cases:
            with self.subTest(text=text):self.assertEqual(score_response(text,[check])["strict_pass"],expected)
    def test_candidate_is_separate_from_pilot(self):
        from evaluation.prepare_benchmark import build
        from evaluation.common import read_json
        with tempfile.TemporaryDirectory() as directory:
            m=build(directory)
            self.assertEqual((m["candidate_count"],m["development_count"],m["pilot_subset_count"]),(300,60,24))
            self.assertFalse(m["candidate_main_evaluated"])
            candidate=read_json(Path(directory)/"candidate.json")
            dev=read_json(Path(directory)/"development.json")
            self.assertFalse({r["prompt"] for r in candidate}&{r["prompt"] for r in dev})

@unittest.skipUnless(importlib.util.find_spec("torch"),"torch unavailable")
class RuntimeTests(unittest.TestCase):
    def test_right_padding_preserves_real_token_nll(self):
        import torch
        from configs import default_config
        from models import build_model
        from evaluation.metrics import response_nll
        torch.set_num_threads(1);torch.manual_seed(4)
        cfg=default_config(vocab_size=32,context_len=16,d_model=16,n_layers=1,n_heads=2,ffn_hidden=32)
        rows=[{"id":"a","ids":[3,8,4,9,5],"labels":[-100,-100,-100,9,5]},
              {"id":"b","ids":[3,10,11,4,12,13,5],"labels":[-100,-100,-100,-100,12,13,5]}]
        for arm in "TDP":
            with self.subTest(arm=arm):
                m=build_model(arm,cfg)
                single=response_nll(m,rows,batch_size=1);batch=response_nll(m,rows,batch_size=2)
                self.assertAlmostEqual(single["nll"],batch["nll"],places=6)
                self.assertEqual(single["target_tokens"],5)
    def test_final_and_recovery_loading(self):
        import torch
        from configs import default_config
        from models import build_model,save_checkpoint
        from evaluation.common import sha256_file
        from evaluation.ws3 import load_ws3_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            d=Path(directory);cfg=default_config(vocab_size=32,context_len=16,d_model=16,n_layers=1,n_heads=2,ffn_hidden=32)
            m=build_model("T",cfg);save_checkpoint(d/"parent.pt",m,step=2,extra={"seed":0})
            config={"arm":"T","seed":0,"token_budget":10,"pretrained_sha256":sha256_file(d/"parent.pt")}
            save_checkpoint(d/"final.pt",m,step=1,extra={"seed":0,"tokens_seen":10,"sft_config":config,"final_val_nll":3.})
            final,info=load_ws3_checkpoint(d/"final.pt",d/"parent.pt",config)
            self.assertEqual(info["tokens_seen"],10)
            torch.save({"config":config,"model":m.state_dict(),"step":1,"tokens_seen":5},d/"resume.pt")
            resumed,info=load_ws3_checkpoint(d/"resume.pt",d/"parent.pt",config,"resume")
            self.assertEqual(info["tokens_seen"],5)
            config["pretrained_sha256"]="f"*64
            with self.assertRaises(ValueError):load_ws3_checkpoint(d/"final.pt",d/"parent.pt",config)

if __name__=="__main__":unittest.main()

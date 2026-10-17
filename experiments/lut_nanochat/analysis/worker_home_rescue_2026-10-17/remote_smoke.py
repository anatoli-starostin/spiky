import sys, os, json, torch
BR=os.path.expanduser('~/projects/spiky/experiments/ffn_replacement')
sys.path.insert(0, BR+'/tools')
os.environ.setdefault('NANOCHAT_ROOT', os.path.expanduser('~/projects/nanochat'))
sys.path.insert(0, os.environ['NANOCHAT_ROOT'])
from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from model_build import build_model
from fixed_eval import evaluate_bpb_fixed
run=sys.argv[1]
cfg=json.load(open(os.path.join(BR,'runs_corrected',run,'config.json')))
DEV='cuda'
print('CONFIG', {k:cfg.get(k) for k in ('ffn_type','ffn_lut_kind','lut_n_heads','lut_n_anchor_pairs','lut_tables_per_head','lut_inner_in_dim','lut_inner_out_dim','device_batch_size','seq_len')})
tok=RustBPETokenizer.from_directory(os.path.join(get_base_dir(),'tokenizer')); V=tok.get_vocab_size(); tb=get_token_bytes(device=DEV)
m=build_model(cfg,V,DEV); tp=sum(p.numel() for p in m.parameters()); print('PARAMS', tp)
bs=cfg.get('device_batch_size',48)
opt=torch.optim.AdamW(m.parameters(),lr=1e-4)
loader=tokenizing_distributed_data_loader_bos_bestfit(tok, bs, cfg['seq_len'], split='train', device=DEV)
x,y=next(iter(loader)); m.train(); loss=m(x,y); loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
torch.cuda.synchronize(); print('TRAIN_STEP_OK loss', round(float(loss),4), 'peak_GiB', round(torch.cuda.max_memory_allocated()/2**30,1))
m.eval()
with torch.no_grad():
    bpb=evaluate_bpb_fixed(m,tok,tb,cfg['seq_len'],DEV,eval_batch_size=48,eval_steps=2,skip_rows=12)
print('FIXED_EVAL_OK bpb(2-step smoke)', round(bpb,4))
print('SMOKE_DONE')

l="nanochat/lut_ffn.py"; s=open(l).read()
if "proj_bf16" not in s:
    s=s.replace("    proj_fp8_muon: bool = False", "    proj_bf16: bool = False\n    proj_fp8_muon: bool = False", 1)
    s=s.replace("        else:\n            self.mhl = ProjectionMHL(cart, d_model=d_model).float()  # fp32 island",
                "        elif cfg.proj_bf16:\n            self.mhl = ProjectionMHL(cart, d_model=d_model, projection_dtype=torch.bfloat16).float()\n        else:\n            self.mhl = ProjectionMHL(cart, d_model=d_model).float()  # fp32 island", 1)
    assert "proj_bf16: bool" in s and "projection_dtype=torch.bfloat16" in s
    open(l,"w").write(s); print("LUTFFN_PATCHED")
else: print("LUTFFN_ALREADY")
b="scripts/base_train.py"; t=open(b).read()
if "lut_ffn_proj_bf16" not in t:
    t=t.replace("parser.add_argument(\"--lut-ffn-proj-fp8-muon\"",
                "parser.add_argument(\"--lut-ffn-proj-bf16\", action=\"store_true\", help=\"[lut_nanochat] bf16 projection GEMMs\")\nparser.add_argument(\"--lut-ffn-proj-fp8-muon\"", 1)
    t=t.replace("proj_fp8_muon=args.lut_ffn_proj_fp8_muon,", "proj_fp8_muon=args.lut_ffn_proj_fp8_muon, proj_bf16=args.lut_ffn_proj_bf16,", 1)
    assert "lut_ffn_proj_bf16" in t
    open(b,"w").write(t); print("BASETRAIN_PATCHED")
else: print("BASETRAIN_ALREADY")

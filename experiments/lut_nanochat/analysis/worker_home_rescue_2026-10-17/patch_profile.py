p="scripts/base_train.py"
s=open(p).read()
if "profile_steps" in s:
    print("ALREADY_PATCHED")
else:
    anchor="    last_step = step == num_iterations"
    block="\n".join(["",
      "    if getattr(args, \"profile_steps\", 0) and step == args.profile_steps:",
      "        from torch.profiler import profile as _TP, ProfilerActivity as _PA",
      "        globals()[\"_PROF\"] = _TP(activities=[_PA.CPU, _PA.CUDA], record_shapes=False)",
      "        _PROF.__enter__()",
      "    if getattr(args, \"profile_steps\", 0) and step == args.profile_steps + 2:",
      "        _PROF.__exit__(None, None, None)",
      "        print0(_PROF.key_averages().table(sort_by=\"self_cuda_time_total\", row_limit=35, max_name_column_width=70))",
      "        break",
      ""])
    i=s.index(anchor); eol=s.index("\n", i)
    s=s[:eol+1]+block+s[eol+1:]
    s=s.replace("parser.add_argument(\"--num-iterations\"", "parser.add_argument(\"--profile-steps\", type=int, default=0)\nparser.add_argument(\"--num-iterations\"", 1)
    open(p,"w").write(s)
    print("PATCHED_OK")

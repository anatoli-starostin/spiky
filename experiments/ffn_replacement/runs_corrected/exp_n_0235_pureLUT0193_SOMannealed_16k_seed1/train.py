import sys
# LOCAL STUB (uncommitted): this SOM arm was OFFLOADED to worker VM 89.169.110.97 (task
# 0daac89e). It must NOT train locally. The local sweep orchestrator runs this file when it
# reaches this arm; exiting 0 immediately makes it a no-op so the orchestrator moves on without
# consuming the local GPU. The REAL train.py (committed at 6c26db9a) runs on the VM.
print("SKIPPED LOCALLY — exp_n_0235 SOM arm offloaded to worker VM 89.169.110.97 (task 0daac89e)")
sys.exit(0)

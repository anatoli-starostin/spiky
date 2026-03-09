#pragma once

#ifdef ENABLE_PROFILING
    #ifndef NO_CUDA
    #define LUTORCH_MANAGER_ANCHOR_PAIRS_FORWARD_NA1_PROFILER_OP 0
    #define N_LUTORCH_PROFILER_OPS 1
    #else
    #define N_LUTORCH_PROFILER_OPS 0
    #endif
#endif

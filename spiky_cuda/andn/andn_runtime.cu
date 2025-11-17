#include <algorithm>
#include "andn_runtime.cuh"

namespace {
#include "aux/andn_runtime_kernels_logic.cu"
}
namespace py = pybind11;

ANDN_RUNTIME_CONTEXT_CLASS::ANDN_RUNTIME_CONTEXT_CLASS(
    uint8_t *andn_data,
    int device,
    uint32_t n_inputs,
    uint32_t n_outputs,
    uint32_t n_detectors,
    uint32_t max_inputs_per_detector,
    uint32_t forward_group_size,
    uint32_t backward_group_size,
    uint32_t max_forward_groups_per_neuron,
    uint32_t max_backward_groups_per_neuron,
    #ifdef INTEGERS_INSTEAD_OF_FLOATS
    uint64_t n_weights,
    double int_rescaler,
    #endif
    #ifdef ENABLE_PROFILING
    SimpleProfiler& profiler,
    #endif
    BaseSynapseMeta *base_synapse_metas,
    IndexedSynapsesInfo *input_neuron_synapses_infos,
    int32_t *detectors,
    IndexedSynapsesInfo *output_neuron_synapses_infos,
    NeuronDataId_t first_synapse_id
) :
    andn_data(andn_data),
    device(device),
    n_inputs(n_inputs),
    n_outputs(n_outputs),
    n_detectors(n_detectors),
    max_inputs_per_detector(max_inputs_per_detector),
    forward_group_size(forward_group_size),
    backward_group_size(backward_group_size),
    batch_size(0),
    #ifdef ENABLE_PROFILING
    profiler(profiler),
    #endif
    base_synapse_metas(base_synapse_metas),
    input_neuron_synapses_infos(input_neuron_synapses_infos),
    detectors(detectors),
    initial_winning_stat(nullptr),
    min_winning_stat(0),
    firing_buffer(nullptr),
    max_forward_groups_per_neuron(max_forward_groups_per_neuron),
    max_backward_groups_per_neuron(max_backward_groups_per_neuron),
    #ifdef INTEGERS_INSTEAD_OF_FLOATS
    n_weights(n_weights),
    int_rescaler(int_rescaler),
    #endif
    before_detectors_gradients(nullptr),
    output_neuron_synapses_infos(output_neuron_synapses_infos),
    first_synapse_id(first_synapse_id)
{
    __TRACE__("ANDN_RUNTIME_CONTEXT_CLASS constructor\n");

    if(n_outputs > 0) {
        if(device == -1) {
            first_synapse_meta_lr = base_synapse_metas->lr;
        } else {
            #ifndef NO_CUDA
            c10::cuda::CUDAGuard guard(device);
            cuMemcpyDtoH(
                &first_synapse_meta_lr,
                (CUdeviceptr) &(base_synapse_metas->lr),
                sizeof(REAL_DT)
            );
            #endif
        }
    } else {
        first_synapse_meta_lr = 0.0;
    }
}


ANDN_RUNTIME_CONTEXT_CLASS::~ANDN_RUNTIME_CONTEXT_CLASS() {
    __TRACE__("ANDN_RUNTIME_CONTEXT_CLASS destructor\n");
    if (device == -1) {
        if(initial_winning_stat != nullptr) {
            PyMem_Free(initial_winning_stat);
        }
        if(before_detectors_gradients != nullptr) {
            PyMem_Free(before_detectors_gradients);
        }
    } else {
        #ifndef NO_CUDA
        c10::cuda::CUDAGuard guard(device);
        if(initial_winning_stat != nullptr) {
            cudaFree(initial_winning_stat);
        }
        if(before_detectors_gradients != nullptr) {
            cudaFree(before_detectors_gradients);
        }
        #endif
    }
    if(this->firing_buffer != nullptr) {
        delete this->firing_buffer;
    }
}

void ANDN_RUNTIME_CONTEXT_CLASS::_ensure_firing_buffer_size(uint64_t max_groups_to_fire) {
    if((this->firing_buffer == nullptr) || (this->firing_buffer->get_max_firings() < max_groups_to_fire * this->batch_size)) {
        if(this->firing_buffer != nullptr) {
            delete this->firing_buffer;
        }
        this->firing_buffer = new FiringBuffer(max_groups_to_fire, batch_size, device);
    }
}

void ANDN_RUNTIME_CONTEXT_CLASS::forward(
    EXTERNAL_REAL_DT *weights,
    uint32_t batch_size,
    EXTERNAL_REAL_DT *input,
    int32_t *target_input_winner_ids,
    int32_t *target_input_prewinner_ids,
    int32_t *target_input_winning_stat,
    EXTERNAL_REAL_DT *target_output
) {
    __TRACE__("ANDN_RUNTIME_CONTEXT_CLASS::forward, n_detectors %d, n_outputs %d, batch_size %d\n", n_detectors, this->n_outputs, batch_size);
    if(batch_size != this->batch_size) {
        if(device == -1) {
            if(n_detectors > 0) {
                uint64_t memsize = this->n_inputs * batch_size * sizeof(int32_t);
                if(initial_winning_stat != nullptr) {
                    initial_winning_stat = (int32_t *) PyMem_Realloc(initial_winning_stat, memsize);
                } else {
                    initial_winning_stat = (int32_t *) PyMem_Malloc(memsize);
                }
                memset(initial_winning_stat, 0, memsize);
            }
            if(before_detectors_gradients != nullptr) {
                PyMem_Free(before_detectors_gradients);
                before_detectors_gradients = nullptr;
            }
        } else {
            #ifndef NO_CUDA
            c10::cuda::CUDAGuard guard(device);
            if(n_detectors > 0) {
                uint64_t memsize = this->n_inputs * batch_size * sizeof(int32_t);
                if (initial_winning_stat != nullptr) {
                    cudaFree(initial_winning_stat);
                }
                cudaMalloc(&initial_winning_stat, memsize);
                cudaMemset(initial_winning_stat, 0, memsize);
            }
            if(before_detectors_gradients != nullptr) {
                cudaFree(before_detectors_gradients);
                before_detectors_gradients = nullptr;
            }
            #endif
        }

        if(n_detectors > 0) {
            dim3 numBlocks((this->n_detectors + ANDN_RUNTIME_KERNELS_TPB - 1) / ANDN_RUNTIME_KERNELS_TPB, batch_size);
            GRID_CALL_NO_SHARED_MEM(
                numBlocks, prepare_winning_stat, ANDN_RUNTIME_KERNELS_TPB,
                this->n_inputs, this->detectors, this->n_detectors,
                this->max_inputs_per_detector, this->initial_winning_stat
            );

            if(this->min_winning_stat == 0) {
                int32_t *t_buf;
                if(device == -1) {
                    t_buf = (int32_t *) PyMem_Malloc(sizeof(int32_t));
                    *t_buf = 0;
                } else {
                    #ifndef NO_CUDA
                    c10::cuda::CUDAGuard guard(device);
                    cudaMalloc(&t_buf, sizeof(int32_t));
                    cudaMemset(t_buf, 0, sizeof(int32_t));
                    #endif
                }

                dim3 numBlocks((this->n_inputs + ANDN_RUNTIME_KERNELS_TPB - 1) / ANDN_RUNTIME_KERNELS_TPB, 1);
                GRID_CALL_SHARED_MEM(
                    numBlocks, find_min_int, ANDN_RUNTIME_KERNELS_TPB, ANDN_RUNTIME_KERNELS_TPB * sizeof(int32_t),
                    this->initial_winning_stat, this->n_inputs, 0,
                    t_buf, device
                );

                if(device == -1) {
                    this->min_winning_stat = *t_buf;
                    PyMem_Free(t_buf);
                } else {
                    #ifndef NO_CUDA
                    c10::cuda::CUDAGuard guard(device);
                    cudaMemcpy(&this->min_winning_stat, t_buf, sizeof(int32_t), cudaMemcpyDeviceToHost);
                    cudaFree(t_buf);
                    #endif
                }
            }
        }
        this->batch_size = batch_size;
    }

    if(this->n_outputs > 0) {
        uint64_t memsize = this->n_outputs * batch_size * sizeof(EXTERNAL_REAL_DT);
        if(device == -1) {
            memset(target_output, 0, memsize);
        } else {
            #ifndef NO_CUDA
            c10::cuda::CUDAGuard guard(device);
            cudaMemset(target_output, 0, memsize);
            #endif
        }
        _ensure_firing_buffer_size(
            static_cast<uint64_t>((n_detectors > 0) ? n_detectors : n_inputs) * this->max_forward_groups_per_neuron
        );
    }

    if(n_detectors > 0) {
        PROF_START(ANDN_RUNTIME_FIRE_DETECTORS_PROFILER_OP);
        if (device == -1) {
            memcpy(
                target_input_winning_stat,
                this->initial_winning_stat,
                this->n_inputs * batch_size * sizeof(int32_t)
            );
        } else {
            #ifndef NO_CUDA
            c10::cuda::CUDAGuard guard(device);
            cuMemcpyDtoD(
                (CUdeviceptr) target_input_winning_stat,
                (CUdeviceptr) this->initial_winning_stat,
                this->n_inputs * batch_size * sizeof(int32_t)
            );
            #endif
        }
        if(this->n_outputs > 0) {
            firing_buffer->clear();
        }
        dim3 numBlocks((this->n_detectors + ANDN_RUNTIME_KERNELS_TPB - 1) / ANDN_RUNTIME_KERNELS_TPB, this->batch_size);
        GRID_CALL_SHARED_MEM(
            numBlocks, fire_detectors, ANDN_RUNTIME_KERNELS_TPB, ANDN_RUNTIME_KERNELS_TPB * sizeof(uint32_t),
            input, this->n_inputs,
            this->detectors, this->n_detectors, this->max_inputs_per_detector,
            target_input_winner_ids, target_input_prewinner_ids, target_input_winning_stat,
            reinterpret_cast<NoDelaysIndexedSynapsesInfo *>(input_neuron_synapses_infos),
            (this->n_outputs > 0) ? firing_buffer->firings_ptr() : nullptr,
            (this->n_outputs > 0) ? firing_buffer->counter_ptr() : nullptr,
            this->forward_group_size,
            this->andn_data,
            device
        );
        if(this->n_outputs > 0) {
            firing_buffer->update_counter();
        }
        PROF_END(ANDN_RUNTIME_FIRE_DETECTORS_PROFILER_OP);
    } else {
        PROF_START(ANDN_RUNTIME_FIRE_INPUTS_PROFILER_OP);
        firing_buffer->clear();
        dim3 numBlocks((this->n_inputs + ANDN_RUNTIME_KERNELS_TPB - 1) / ANDN_RUNTIME_KERNELS_TPB, this->batch_size);
        GRID_CALL_SHARED_MEM(
            numBlocks, fire_inputs, ANDN_RUNTIME_KERNELS_TPB, ANDN_RUNTIME_KERNELS_TPB * sizeof(uint32_t),
            input, this->n_inputs,
            reinterpret_cast<NoDelaysIndexedSynapsesInfo *>(input_neuron_synapses_infos),
            firing_buffer->firings_ptr(),
            firing_buffer->counter_ptr(),
            this->forward_group_size,
            this->andn_data,
            true,
            device
        );
        firing_buffer->update_counter();
        PROF_END(ANDN_RUNTIME_FIRE_INPUTS_PROFILER_OP);
    }
    if(n_outputs > 0) {
        PROF_START(ANDN_RUNTIME_FILL_OUTPUTS_PROFILER_OP);
        uint64_t n_firings = firing_buffer->number_of_firings();
        dim3 numBlocks((n_firings + ANDN_RUNTIME_KERNELS_TPB - 1) / ANDN_RUNTIME_KERNELS_TPB, 1);
        GRID_CALL_NO_SHARED_MEM(
            numBlocks, fill_outputs, ANDN_RUNTIME_KERNELS_TPB,
            weights, this->first_synapse_id,
            firing_buffer->firings_ptr(),
            n_firings,
            target_output,
            this->n_inputs,
            this->n_outputs,
            this->andn_data
            #ifdef INTEGERS_INSTEAD_OF_FLOATS
            , this->int_rescaler
            #else
            , 0.0
            #endif
        );
        PROF_END(ANDN_RUNTIME_FILL_OUTPUTS_PROFILER_OP);

        #ifdef INTEGERS_INSTEAD_OF_FLOATS
        PROF_START(ANDN_RUNTIME_CONVERT_OUTPUTS_PROFILER_OP);
        numBlocks = dim3((n_outputs + ANDN_RUNTIME_KERNELS_TPB - 1) / ANDN_RUNTIME_KERNELS_TPB, this->batch_size);
        GRID_CALL_NO_SHARED_MEM(
            numBlocks, convert_integers_to_floats, ANDN_RUNTIME_KERNELS_TPB,
            target_output,
            this->n_outputs,
            this->int_rescaler
        );
        PROF_END(ANDN_RUNTIME_CONVERT_OUTPUTS_PROFILER_OP);
        #endif
    }
}

void ANDN_RUNTIME_CONTEXT_CLASS::backward_backprop(
    EXTERNAL_REAL_DT *weights,
    uint32_t batch_size,
    // external gradients
    EXTERNAL_REAL_DT *output_gradients,
    // data from forward pass
    EXTERNAL_REAL_DT *input,
    int32_t *input_winner_ids,
    int32_t *input_prewinner_ids,
    int32_t *input_winning_stat,
    // gradients that we need to calculate
    EXTERNAL_REAL_DT *target_input_gradients,
    EXTERNAL_REAL_DT *target_weights_gradients
) {
    __TRACE__("ANDN_RUNTIME_CONTEXT_CLASS::backward\n");
    if(this->batch_size != batch_size) {
        throw py::value_error("batch_size on backward pass doesn't match the current context batch_size");
    }

    PROF_START(ANDN_RUNTIME_BACKWARD_GRAD_PROFILER_OP);
    uint64_t memsize = this->n_inputs * batch_size * sizeof(SUMMATION32_DT);
    if(before_detectors_gradients == nullptr) {
        if(device == -1) {
            before_detectors_gradients = (SUMMATION32_DT *) PyMem_Malloc(memsize);
        } else {
            #ifndef NO_CUDA
            c10::cuda::CUDAGuard guard(device);
            cudaMalloc(&before_detectors_gradients, memsize);
            #endif
        }
    }

    if(device == -1) {
        memset(before_detectors_gradients, 0, memsize);
    } else {
        #ifndef NO_CUDA
        c10::cuda::CUDAGuard guard(device);
        cudaMemset(before_detectors_gradients, 0, memsize);
        #endif
    }

    if(n_outputs > 0) {
        // 1. gather gradients for winners (both dy/dx and dy/dw)

        if(n_detectors > 0) {
            _ensure_firing_buffer_size(
                static_cast<uint64_t>(n_detectors) * this->max_forward_groups_per_neuron
            );
            firing_buffer->clear();
            dim3 numBlocks((this->n_detectors + ANDN_RUNTIME_KERNELS_TPB - 1) / ANDN_RUNTIME_KERNELS_TPB, this->batch_size);
            GRID_CALL_SHARED_MEM(
                numBlocks, fire_detectors_by_input_ids, ANDN_RUNTIME_KERNELS_TPB, ANDN_RUNTIME_KERNELS_TPB * sizeof(uint32_t),
                input, this->n_inputs,
                this->n_detectors, input_winner_ids, input_winning_stat, 0,
                reinterpret_cast<NoDelaysIndexedSynapsesInfo *>(input_neuron_synapses_infos),
                firing_buffer->firings_ptr(),
                firing_buffer->counter_ptr(),
                this->forward_group_size,
                this->andn_data,
                device
            );
            firing_buffer->update_counter();
        } else {
            _ensure_firing_buffer_size(
                static_cast<uint64_t>(n_inputs) * this->max_forward_groups_per_neuron
            );

            firing_buffer->clear();
            dim3 numBlocks((this->n_inputs + ANDN_RUNTIME_KERNELS_TPB - 1) / ANDN_RUNTIME_KERNELS_TPB, this->batch_size);
            GRID_CALL_SHARED_MEM(
                numBlocks, fire_inputs, ANDN_RUNTIME_KERNELS_TPB, ANDN_RUNTIME_KERNELS_TPB * sizeof(uint32_t),
                input, this->n_inputs,
                reinterpret_cast<NoDelaysIndexedSynapsesInfo *>(input_neuron_synapses_infos),
                firing_buffer->firings_ptr(),
                firing_buffer->counter_ptr(),
                this->forward_group_size,
                this->andn_data,
                false,
                device
            );
            firing_buffer->update_counter();
        }

        uint64_t n_firings = firing_buffer->number_of_firings();
        dim3 numBlocks((n_firings + ANDN_RUNTIME_KERNELS_TPB - 1) / ANDN_RUNTIME_KERNELS_TPB, 1);
        GRID_CALL_NO_SHARED_MEM(
            numBlocks, gather_gradients, ANDN_RUNTIME_KERNELS_TPB,
            weights, this->first_synapse_id,
            firing_buffer->firings_ptr(),
            n_firings,
            output_gradients,
            (n_detectors > 0) ? before_detectors_gradients : reinterpret_cast<SUMMATION32_DT *>(target_input_gradients),
            target_weights_gradients,
            this->n_inputs,
            this->n_outputs,
            this->andn_data
            #ifdef INTEGERS_INSTEAD_OF_FLOATS
            , this->int_rescaler
            #else
            , 0.0
            #endif
        );

        if(n_detectors > 0) {
            // 2. gather gradients for prewinners (only dy/dx)
            firing_buffer->clear();
            dim3 numBlocks((this->n_detectors + ANDN_RUNTIME_KERNELS_TPB - 1) / ANDN_RUNTIME_KERNELS_TPB, this->batch_size);
            GRID_CALL_SHARED_MEM(
                numBlocks, fire_detectors_by_input_ids, ANDN_RUNTIME_KERNELS_TPB, ANDN_RUNTIME_KERNELS_TPB * sizeof(uint32_t),
                input, this->n_inputs,
                this->n_detectors, input_prewinner_ids, input_winning_stat, -1,
                reinterpret_cast<NoDelaysIndexedSynapsesInfo *>(input_neuron_synapses_infos),
                firing_buffer->firings_ptr(),
                firing_buffer->counter_ptr(),
                this->forward_group_size,
                this->andn_data,
                device
            );
            firing_buffer->update_counter();

            uint64_t n_firings = firing_buffer->number_of_firings();
            numBlocks = dim3((n_firings + ANDN_RUNTIME_KERNELS_TPB - 1) / ANDN_RUNTIME_KERNELS_TPB, 1);
            GRID_CALL_NO_SHARED_MEM(
                numBlocks, gather_gradients, ANDN_RUNTIME_KERNELS_TPB,
                weights, this->first_synapse_id,
                firing_buffer->firings_ptr(),
                n_firings,
                output_gradients,
                before_detectors_gradients,
                nullptr,
                this->n_inputs,
                this->n_outputs,
                this->andn_data
                #ifdef INTEGERS_INSTEAD_OF_FLOATS
                , this->int_rescaler
                #else
                , 0.0
                #endif
            );

            // 3. propagate through detectors

            numBlocks = dim3((this->n_detectors + ANDN_RUNTIME_KERNELS_TPB - 1) / ANDN_RUNTIME_KERNELS_TPB, this->batch_size);
            GRID_CALL_NO_SHARED_MEM(
                numBlocks, propagate_through_detectors, ANDN_RUNTIME_KERNELS_TPB,
                input, input_winner_ids, input_prewinner_ids, input_winning_stat,
                this->n_detectors,
                before_detectors_gradients,
                target_input_gradients,
                this->n_inputs
                #ifdef INTEGERS_INSTEAD_OF_FLOATS
                , this->int_rescaler
                #else
                , 0.0
                #endif
            );
        }
    } else {
        #ifdef INTEGERS_INSTEAD_OF_FLOATS
        dim3 numBlocks((this->n_inputs + ANDN_RUNTIME_KERNELS_TPB - 1) / ANDN_RUNTIME_KERNELS_TPB, this->batch_size);
        GRID_CALL_NO_SHARED_MEM(
            numBlocks, copy_floats_to_integers, ANDN_RUNTIME_KERNELS_TPB,
            output_gradients,
            before_detectors_gradients,
            this->n_inputs,
            this->int_rescaler
        );
        numBlocks = dim3((this->n_detectors + ANDN_RUNTIME_KERNELS_TPB - 1) / ANDN_RUNTIME_KERNELS_TPB, this->batch_size);
        GRID_CALL_NO_SHARED_MEM(
            numBlocks, propagate_through_detectors, ANDN_RUNTIME_KERNELS_TPB,
            input, input_winner_ids, input_prewinner_ids, input_winning_stat,
            this->n_detectors,
            before_detectors_gradients,
            target_input_gradients,
            this->n_inputs,
            this->int_rescaler
        );
        #else
        dim3 numBlocks((this->n_detectors + ANDN_RUNTIME_KERNELS_TPB - 1) / ANDN_RUNTIME_KERNELS_TPB, this->batch_size);
        GRID_CALL_NO_SHARED_MEM(
            numBlocks, propagate_through_detectors, ANDN_RUNTIME_KERNELS_TPB,
            input, input_winner_ids, input_prewinner_ids, input_winning_stat,
            this->n_detectors,
            output_gradients,
            target_input_gradients,
            this->n_inputs,
            0.0
        );
        #endif
    }
    PROF_END(ANDN_RUNTIME_BACKWARD_GRAD_PROFILER_OP);

    #ifdef INTEGERS_INSTEAD_OF_FLOATS
    PROF_START(ANDN_RUNTIME_CONVERT_OUTPUTS_PROFILER_OP);
    if(n_outputs > 0) {
        dim3 numBlocks((this->n_weights + ANDN_RUNTIME_KERNELS_TPB - 1) / ANDN_RUNTIME_KERNELS_TPB, 1);
        GRID_CALL_NO_SHARED_MEM(
            numBlocks, convert_integers_to_floats, ANDN_RUNTIME_KERNELS_TPB,
            target_weights_gradients,
            this->n_weights,
            this->int_rescaler
        );
    }
    dim3 numBlocks((this->n_inputs + ANDN_RUNTIME_KERNELS_TPB - 1) / ANDN_RUNTIME_KERNELS_TPB, this->batch_size);
    GRID_CALL_NO_SHARED_MEM(
        numBlocks, convert_integers_to_floats, ANDN_RUNTIME_KERNELS_TPB,
        target_input_gradients,
        this->n_inputs,
        this->int_rescaler
    );
    PROF_END(ANDN_RUNTIME_CONVERT_OUTPUTS_PROFILER_OP);
    #endif
}

void ANDN_RUNTIME_CONTEXT_CLASS::backward_hebb(
    EXTERNAL_REAL_DT *weights,
    uint32_t batch_size,
    double anti_hebb_coeff,
    // data from forward pass
    EXTERNAL_REAL_DT *input,
    int32_t *input_winner_ids,
    int32_t *input_prewinner_ids,
    int32_t *input_winning_stat,
    // hebbian part, information from the next ANDN layer
    EXTERNAL_REAL_DT *output,
    int32_t *output_winner_ids,
    int32_t *output_prewinner_ids,
    int32_t *output_winning_stat,
    uint32_t n_output_detectors,
    // gradients that we need to calculate
    EXTERNAL_REAL_DT *target_weights_gradients
) {
    __TRACE__("ANDN_RUNTIME_CONTEXT_CLASS::backward\n");
    if(this->batch_size != batch_size) {
        throw py::value_error("batch_size on backward pass doesn't match the current context batch_size");
    }

    PROF_START(ANDN_RUNTIME_BACKWARD_HEBB_PROFILER_OP);
    _ensure_firing_buffer_size(static_cast<uint64_t>(n_output_detectors) * this->max_backward_groups_per_neuron);
    firing_buffer->clear();

    __DETAILED_TRACE__("[ANDN_RUNTIME_CONTEXT_CLASS::backward] n_output_detectors %d, max_backward_groups_per_neuron %d\n", n_output_detectors, this->max_backward_groups_per_neuron);

    dim3 numBlocks((n_output_detectors + ANDN_RUNTIME_KERNELS_TPB - 1) / ANDN_RUNTIME_KERNELS_TPB, this->batch_size);
    GRID_CALL_SHARED_MEM(
        numBlocks, backfire_detectors, ANDN_RUNTIME_KERNELS_TPB, ANDN_RUNTIME_KERNELS_TPB * sizeof(uint32_t),
        output, this->n_outputs,
        output_winner_ids, n_output_detectors,
        output_winning_stat, 0,
        reinterpret_cast<NoDelaysIndexedSynapsesInfo *>(output_neuron_synapses_infos),
        firing_buffer->firings_ptr(),
        firing_buffer->counter_ptr(),
        this->backward_group_size,
        this->andn_data,
        device
    );
    firing_buffer->update_counter();

    uint64_t n_firings = firing_buffer->number_of_firings();
    numBlocks = dim3((n_firings + ANDN_RUNTIME_KERNELS_TPB - 1) / ANDN_RUNTIME_KERNELS_TPB, 1);
    GRID_CALL_NO_SHARED_MEM(
        numBlocks, calculate_hebbian_gradients, ANDN_RUNTIME_KERNELS_TPB,
        weights, this->first_synapse_id,
        firing_buffer->firings_ptr(),
        n_firings,
        target_weights_gradients,
        this->n_inputs,
        input_winning_stat,
        input,
        this->first_synapse_meta_lr,
        this->base_synapse_metas,
        1.0,
        this->andn_data
        #ifdef INTEGERS_INSTEAD_OF_FLOATS
        , this->int_rescaler
        #else
        , 0.0
        #endif
    );

    if(anti_hebb_coeff > 0.0) {
        firing_buffer->clear();
        numBlocks = dim3((n_output_detectors + ANDN_RUNTIME_KERNELS_TPB - 1) / ANDN_RUNTIME_KERNELS_TPB, this->batch_size);
        GRID_CALL_SHARED_MEM(
            numBlocks, backfire_detectors, ANDN_RUNTIME_KERNELS_TPB, ANDN_RUNTIME_KERNELS_TPB * sizeof(uint32_t),
            output, this->n_outputs,
            output_prewinner_ids, n_output_detectors,
            output_winning_stat, -1,
            reinterpret_cast<NoDelaysIndexedSynapsesInfo *>(output_neuron_synapses_infos),
            firing_buffer->firings_ptr(),
            firing_buffer->counter_ptr(),
            this->backward_group_size,
            this->andn_data,
            device
        );
        firing_buffer->update_counter();

        n_firings = firing_buffer->number_of_firings();
        numBlocks = dim3((n_firings + ANDN_RUNTIME_KERNELS_TPB - 1) / ANDN_RUNTIME_KERNELS_TPB, 1);
        GRID_CALL_NO_SHARED_MEM(
            numBlocks, calculate_hebbian_gradients, ANDN_RUNTIME_KERNELS_TPB,
            weights, this->first_synapse_id,
            firing_buffer->firings_ptr(),
            n_firings,
            target_weights_gradients,
            this->n_inputs,
            input_winning_stat,
            input,
            this->first_synapse_meta_lr,
            this->base_synapse_metas,
            -anti_hebb_coeff,
            this->andn_data
            #ifdef INTEGERS_INSTEAD_OF_FLOATS
            , this->int_rescaler
            #else
            , 0.0
            #endif
        );
    }
    PROF_END(ANDN_RUNTIME_BACKWARD_HEBB_PROFILER_OP);

    #ifdef INTEGERS_INSTEAD_OF_FLOATS
    PROF_START(ANDN_RUNTIME_CONVERT_OUTPUTS_PROFILER_OP);
    numBlocks = dim3((this->n_weights + ANDN_RUNTIME_KERNELS_TPB - 1) / ANDN_RUNTIME_KERNELS_TPB, 1);
    GRID_CALL_NO_SHARED_MEM(
        numBlocks, convert_integers_to_floats, ANDN_RUNTIME_KERNELS_TPB,
        target_weights_gradients,
        this->n_weights,
        this->int_rescaler
    );
    PROF_END(ANDN_RUNTIME_CONVERT_OUTPUTS_PROFILER_OP);
    #endif
}

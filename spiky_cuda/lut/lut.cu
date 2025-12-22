#include "lut_runtime.cuh"
#include <random>

namespace {
#include "aux/lut_compile_time_kernels_logic.cu"
}

namespace py = pybind11;

#define LUTM_CLASS_NAME PFX(LUTDataManager)

class __attribute__((visibility("hidden"))) LUTM_CLASS_NAME {
    friend py::tuple PFX(pickle_lut_neuron_manager)(const LUTM_CLASS_NAME& ldm);
    friend std::unique_ptr<LUTM_CLASS_NAME> PFX(unpickle_lut_neuron_manager)(py::tuple t);
public:
    LUTM_CLASS_NAME(
        uint32_t n_inputs, uint32_t n_outputs,
        uint32_t n_detectors, uint32_t n_anchors_per_detector,
        uint32_t sequence_length,
        uint32_t positional_dim,
        uint64_t initial_synapse_capacity,
        uint32_t forward_group_size,
        uint32_t backward_group_size
        #ifdef INTEGERS_INSTEAD_OF_FLOATS
        , double int_rescaler
        #endif
    ) :
        #ifdef ENABLE_PROFILING
        profiler(N_LUT_PROFILER_OPS),
        #endif
        host_device_allocator(initial_synapse_capacity * 2 * sizeof(NeuronIndex_t) + MAX_N_SYNAPSE_METAS * sizeof(BaseSynapseMeta)),
        only_host_allocator(1024),
        connections_manager(nullptr), runtime_context(nullptr),
        detector_connections_allocator(nullptr),
        detector_connections_manager(nullptr),
        n_inputs(n_inputs),
        n_outputs(n_outputs),
        n_detectors(n_detectors),
        n_anchors_per_detector(n_anchors_per_detector),
        sequence_length(sequence_length),
        positional_dim(positional_dim),
        n_lookup_neurons(n_detectors * (1 << (n_anchors_per_detector + ((sequence_length > 1) ? (n_anchors_per_detector + positional_dim) : 0)))),
        forward_group_size(forward_group_size),
        backward_group_size(backward_group_size)
    {
        // Validate inputs
        if(n_inputs == 0) {
            throw py::value_error("n_inputs == 0");
        }
        if(n_outputs == 0) {
            throw py::value_error("n_outputs == 0");
        }
        if(n_detectors == 0) {
            throw py::value_error("n_detectors == 0");
        }
        if(n_anchors_per_detector == 0) {
            throw py::value_error("n_anchors_per_detector == 0");
        }
        if(sequence_length == 0) {
            throw py::value_error("sequence_length == 0");
        }
        if(forward_group_size == 0) {
            throw py::value_error("forward_group_size == 0");
        }
        if(forward_group_size > MAX_SYNAPSE_GROUP_SIZE) {
            throw py::value_error("forward_group_size > MAX_SYNAPSE_GROUP_SIZE");
        }
        if(backward_group_size == 0) {
            throw py::value_error("backward_group_size == 0");
        }
        if(backward_group_size > MAX_SYNAPSE_GROUP_SIZE) {
            throw py::value_error("backward_group_size > MAX_SYNAPSE_GROUP_SIZE");
        }

        this->base_synapse_metas_id = host_device_allocator.allocate(MAX_N_SYNAPSE_METAS * sizeof(BaseSynapseMeta), SYNAPSE_METAS_MEMORY_LABEL);
        this->global_connections_meta_id = only_host_allocator.allocate(sizeof(GlobalConnectionsMeta), 0);
        GlobalConnectionsMeta* gc_meta = reinterpret_cast<GlobalConnectionsMeta *>(only_host_allocator.data + global_connections_meta_id);
        memset(gc_meta, 0, sizeof(GlobalConnectionsMeta));
        this->n_synapse_metas = 0;
        this->n_synapses = 0;
        this->lookup_neuron_synapses_infos_id = 0;

        #ifdef ENABLE_PROFILING
        profiler.register_operation_type(LUT_RUNTIME_FORWARD_NON_SEQ_PROFILER_OP, "lut::runtime::forward_non_seq");
        profiler.register_operation_type(LUT_RUNTIME_FORWARD_NON_SEQ_CHECK_DETECTORS_PROFILER_OP, "lut::runtime::forward_non_seq::check_detectors");
        profiler.register_operation_type(LUT_RUNTIME_FORWARD_NON_SEQ_FILL_OUTPUTS_SPARSE_PROFILER_OP, "lut::runtime::forward_non_seq::fill_outputs_sparse");
        profiler.register_operation_type(LUT_RUNTIME_FORWARD_NON_SEQ_FILL_OUTPUTS_FC_PROFILER_OP, "lut::runtime::forward_non_seq::fill_outputs_fc");
        profiler.register_operation_type(LUT_RUNTIME_FORWARD_NON_SEQ_EVAL_PROFILER_OP, "lut::runtime::forward_non_seq_eval");
        profiler.register_operation_type(LUT_RUNTIME_FORWARD_NON_SEQ_EVAL_CHECK_DETECTORS_PROFILER_OP, "lut::runtime::forward_non_seq_eval::check_detectors");
        profiler.register_operation_type(LUT_RUNTIME_CONVERT_OUTPUTS_PROFILER_OP, "lut::runtime::convert_outputs");
        profiler.register_operation_type(LUT_RUNTIME_BACKWARD_NON_SEQ_BACKPROP_PROFILER_OP, "lut::runtime::backward_non_seq::backprop");
        profiler.register_operation_type(LUT_RUNTIME_BACKWARD_NON_SEQ_PROPAGATE_DETECTORS_SPARSE_PROFILER_OP, "lut::runtime::backward_non_seq::propagate_detectors_sparse");
        profiler.register_operation_type(LUT_RUNTIME_BACKWARD_NON_SEQ_GATHER_GRADIENTS_SPARSE_PROFILER_OP, "lut::runtime::backward_non_seq::gather_gradients_sparse");
        profiler.register_operation_type(LUT_RUNTIME_BACKWARD_NON_SEQ_PROPAGATE_DETECTORS_FC_PROFILER_OP, "lut::runtime::backward_non_seq::propagate_detectors_fc");
        profiler.register_operation_type(LUT_RUNTIME_BACKWARD_NON_SEQ_GATHER_GRADIENTS_FC_PROFILER_OP, "lut::runtime::backward_non_seq::gather_gradients_fc");
        profiler.register_operation_type(LUT_RUNTIME_FORWARD_SEQ_PROFILER_OP, "lut::runtime::forward_seq");
        profiler.register_operation_type(LUT_RUNTIME_FORWARD_SEQ_CHECK_DETECTORS_PROFILER_OP, "lut::runtime::forward_seq::check_detectors");
        profiler.register_operation_type(LUT_RUNTIME_FORWARD_SEQ_CHECK_POSITIONAL_EMBEDDINGS_PROFILER_OP, "lut::runtime::forward_seq::check_positional_embeddings");
        profiler.register_operation_type(LUT_RUNTIME_FORWARD_SEQ_FILL_OUTPUTS_SPARSE_PROFILER_OP, "lut::runtime::forward_seq::fill_outputs_sparse");
        profiler.register_operation_type(LUT_RUNTIME_FORWARD_SEQ_FILL_OUTPUTS_FC_PROFILER_OP, "lut::runtime::forward_seq::fill_outputs_fc");
        profiler.register_operation_type(LUT_RUNTIME_FORWARD_SEQ_EVAL_PROFILER_OP, "lut::runtime::forward_seq_eval");
        profiler.register_operation_type(LUT_RUNTIME_FORWARD_SEQ_EVAL_CHECK_DETECTORS_PROFILER_OP, "lut::runtime::forward_seq_eval::check_detectors");
        profiler.register_operation_type(LUT_RUNTIME_FORWARD_SEQ_EVAL_CHECK_POSITIONAL_EMBEDDINGS_PROFILER_OP, "lut::runtime::forward_seq_eval::check_positional_embeddings");
        profiler.register_operation_type(LUT_RUNTIME_BACKWARD_SEQ_PROFILER_OP, "lut::runtime::backward_seq");
        profiler.register_operation_type(LUT_RUNTIME_BACKWARD_SEQ_PROPAGATE_THROUGH_DETECTORS_SPARSE_PROFILER_OP, "lut::runtime::backward_seq::propagate_through_detectors_sparse");
        profiler.register_operation_type(LUT_RUNTIME_BACKWARD_SEQ_PROPAGATE_THROUGH_DETECTORS_FC_PROFILER_OP, "lut::runtime::backward_seq::propagate_through_detectors_fc");
        profiler.register_operation_type(LUT_RUNTIME_BACKWARD_SEQ_GATHER_W_GRADIENTS_SPARSE_PROFILER_OP, "lut::runtime::backward_seq::gather_w_gradients_sparse");
        profiler.register_operation_type(LUT_RUNTIME_BACKWARD_SEQ_GATHER_W_GRADIENTS_FC_PROFILER_OP, "lut::runtime::backward_seq::gather_w_gradients_fc");
        #endif

        weights_allocator = new SimpleAllocator(initial_synapse_capacity * sizeof(EXTERNAL_REAL_DT));
        // detector_connections_allocator will be created when first detector connection is added

        #ifdef INTEGERS_INSTEAD_OF_FLOATS
        this->int_rescaler = int_rescaler;
        #endif
    }

    REAL_DT get_smallest_distinguishable_fraction()
    {
        #ifdef INTEGERS_INSTEAD_OF_FLOATS
        return (1.0 / DENOMINATOR32) / int_rescaler;
        #else
        return 0.0;
        #endif
    }

    double get_epsilon()
    {
        return (double) EPS;
    }

    auto get_summations_data_type() {
        return SUMMATION_DT_STR;
    }

    uint32_t register_synapse_meta(
        REAL_DT learning_rate,
        REAL_DT min_synaptic_weight, REAL_DT max_synaptic_weight,
        REAL_DT initial_noise_level, REAL_DT initial_weight
    ) {
        __TRACE__("lutm_register_synapse_meta\n");
        checkOnHostDuringPrepare();

        if(this->lookup_neuron_synapses_infos_id != 0) {
            throw py::value_error("can't register new synapse metas after neurons were initialized");
        }
        if(detector_connections_manager != nullptr) {
            throw py::value_error("can't register new synapse metas after detector connections were added");
        }
        if(learning_rate < 0.0) {
            throw py::value_error("learning_rate < 0");
        }
        if(learning_rate > 1.0) {
            throw py::value_error("learning_rate > 1");
        }
        #ifdef INTEGERS_INSTEAD_OF_FLOATS
        REAL_DT denominator_reciproc = 1.0 / DENOMINATOR32;
        if((learning_rate > 0.0) && ((learning_rate * int_rescaler) < denominator_reciproc)) {
            std::ostringstream os;
            os << "too small value for 'learning_rate' " << learning_rate << " (less than smallest distinguishable fraction " << denominator_reciproc << ")";
            throw py::value_error(os.str());
        }
        #endif
        if(min_synaptic_weight > max_synaptic_weight) {
            throw py::value_error("min_synaptic_weight > max_synaptic_weight");
        }
        if(initial_weight < min_synaptic_weight) {
            throw py::value_error("initial_weight < min_synaptic_weight");
        }
        if(initial_weight > max_synaptic_weight) {
            throw py::value_error("initial_weight > max_synaptic_weight");
        }
        BaseSynapseMeta target_base_meta = {
            learning_rate,
            0, 0,
            min_synaptic_weight, max_synaptic_weight,
            initial_noise_level, initial_weight,
            this->forward_group_size, 0
        };
        BaseSynapseMeta *current_base_synapse_meta = BaseSynapseMetas(this->base_synapse_metas_id, host_device_allocator.data);
        uint32_t i=0;
        for(;i < this->n_synapse_metas;i++, current_base_synapse_meta++) {
            if(!memcmp(&target_base_meta, current_base_synapse_meta, sizeof(BaseSynapseMeta))) {
                return i;
            }
        }
        if(this->n_synapse_metas == MAX_N_SYNAPSE_METAS) {
            throw py::value_error("too many different synapse metas");
        }
        this->n_synapse_metas++;
        memcpy(current_base_synapse_meta, &target_base_meta, sizeof(BaseSynapseMeta));
        return i;
    }

    void initialize_neurons(bool is_fully_connected) {
        __TRACE__("lutm_initialize_neurons\n");
        checkOnHostDuringPrepare();
        if(this->n_synapse_metas == 0) {
            throw py::value_error("no synapse metas were registered");
        }

        if(is_fully_connected) {
            if(n_synapse_metas > 1) {
                throw py::value_error("fully connected mode is not compatible with multiple synapse metas");
            }
            this->lookup_neuron_synapses_infos_id = std::numeric_limits<uint64_t>::max();
            connections_manager = nullptr;
            GlobalConnectionsMeta* gc_meta = reinterpret_cast<GlobalConnectionsMeta *>(only_host_allocator.data + global_connections_meta_id);
            gc_meta->first_synapse_id = 0;
            gc_meta->last_synapse_id = (static_cast<uint64_t>(this->n_lookup_neurons) * this->n_outputs - 1) * SizeOfSynapse(true);
            gc_meta->max_forward_groups_per_neuron = (this->n_outputs + this->forward_group_size - 1) / this->forward_group_size;
            delete this->weights_allocator;
            this->weights_allocator = nullptr;
        } else {
            this->lookup_neuron_synapses_infos_id = host_device_allocator.allocate(sizeof(IndexedSynapsesInfo) * this->n_lookup_neurons, NEURON_INFOS_MEMORY_LABEL);

            IndexedSynapsesInfo *current_neuron_info = IndexedSynapsesInfos(this->lookup_neuron_synapses_infos_id, host_device_allocator.data);
            memset(current_neuron_info, 0, sizeof(IndexedSynapsesInfo) * this->n_lookup_neurons);

            connections_manager = new ConnectionsManager(
                #ifdef ENABLE_PROFILING
                profiler,
                #endif
                host_device_allocator, only_host_allocator, true,
                this->base_synapse_metas_id, this->global_connections_meta_id,
                this->lookup_neuron_synapses_infos_id, this->n_lookup_neurons, 0,
                0, 0, this->n_lookup_neurons,  // No backward synapses for output neurons
                this->n_synapse_metas
            );
        }
    }

    uint32_t finalize_detector_connections() {
        if(detector_connections_manager == nullptr) {
            throw py::value_error("detector connections manager not initialized, add detector connections first");
        }

        // finalize detector connections manager, we need it only for backward synapses
        detector_connections_manager->finalize(
            0,  // random_seed not needed for detector connections
            false, false,
            false,
            true
        );

        return detector_connections_manager->count_max_input_synapses_per_neuron();
    }

    uint32_t get_number_of_inputs() {
        __TRACE__("lutm_get_number_of_inputs\n");
        return this->n_inputs;
    }

    uint32_t get_number_of_outputs() {
        __TRACE__("lutm_get_number_of_outputs\n");
        return this->n_outputs;
    }

    uint32_t get_number_of_detectors() {
        __TRACE__("lutm_get_number_of_detectors\n");
        return this->n_detectors;
    }

    uint32_t get_number_of_lookup_neurons() {
        __TRACE__("lutm_get_number_of_lookup_neurons\n");
        return this->n_lookup_neurons;
    }

    uint32_t get_number_of_anchors_per_detector() {
        __TRACE__("lutm_get_number_of_anchors_per_detector\n");
        return this->n_anchors_per_detector;
    }

    uint64_t get_number_of_synapses() {
        __TRACE__("lutm_get_number_of_synapses\n");
        return this->n_synapses;
    }

    uint64_t get_weights_dimension() {
        __TRACE__("lutm_get_weights_dimension\n");
        GlobalConnectionsMeta* gc_meta = reinterpret_cast<GlobalConnectionsMeta *>(only_host_allocator.data + global_connections_meta_id);
        return N_WEIGHTS(gc_meta, true);
    }

    uint32_t get_max_forward_groups_per_neuron() {
        __TRACE__("lutm_get_max_forward_groups_per_neuron\n");
        GlobalConnectionsMeta* gc_meta = reinterpret_cast<GlobalConnectionsMeta *>(only_host_allocator.data + global_connections_meta_id);
        return gc_meta->max_forward_groups_per_neuron;
    }

    void to_device(int device) { // -1 - cpu
        host_device_allocator.to_device(device);
        if(connections_manager != nullptr) {
            delete connections_manager;
            connections_manager = new ConnectionsManager(
                #ifdef ENABLE_PROFILING
                profiler,
                #endif
                host_device_allocator, only_host_allocator, true,
                this->base_synapse_metas_id, this->global_connections_meta_id,
                this->lookup_neuron_synapses_infos_id, this->n_lookup_neurons, 0,
                0, 0, this->n_lookup_neurons,  // No backward synapses for output neurons
                this->n_synapse_metas
            );
        }
        // weights_allocator can be nullptr after compile, so check is needed
        if(weights_allocator != nullptr) {
            weights_allocator->to_device(device);
        }
        // detector_connections_allocator should not exist after compile, but check just in case
        if(detector_connections_allocator != nullptr) {
            detector_connections_allocator->to_device(device);
        }
        if(runtime_context != nullptr) {
            delete runtime_context;
            runtime_context = nullptr;
        }
    }

    void add_detector_connections(
        const torch::Tensor &connections_buffer,
        uint32_t single_input_group_size,
        int ids_shift,
        std::optional<uint32_t> &random_seed
    ) {
        int device = host_device_allocator.device;
        checkTensor(connections_buffer, "connections_buffer", false, device, sizeof(int32_t));

        // Initialize detector connections manager if not already done
        if(detector_connections_manager == nullptr) {
            // Estimate capacity for detector connections
            uint64_t detector_connections_capacity = static_cast<uint64_t>(this->n_inputs) * this->n_detectors;

            detector_connections_allocator = new SimpleAllocator(
                detector_connections_capacity * 2 * sizeof(NeuronIndex_t) +
                sizeof(IndexedSynapsesInfo) * (this->n_inputs + this->n_detectors) +
                sizeof(BaseSynapseMeta)  // For fake synapse meta
            );
            detector_connections_allocator->to_device(device);

            // Allocate fake synapse meta for detector connections (required by ConnectionsManager)
            NeuronDataId_t detector_synapse_metas_id =
                detector_connections_allocator->allocate(sizeof(BaseSynapseMeta), 0);
            BaseSynapseMeta fake_synapse_meta = {
                0.0,  // lr (not used for detector connections)
                0,    // min_delay
                0,    // max_delay
                0.0,  // min_synaptic_weight
                1.0,  // max_synaptic_weight
                0.0,  // initial_noise_level
                0.0,  // initial_weight
                this->forward_group_size,
                this->backward_group_size
            };
            BaseSynapseMeta *synapse_meta_ptr = BaseSynapseMetas(detector_synapse_metas_id, detector_connections_allocator->data);
            if(device == -1) {
                memcpy(synapse_meta_ptr, &fake_synapse_meta, sizeof(BaseSynapseMeta));
            } else {
                #ifndef NO_CUDA
                c10::cuda::CUDAGuard guard(device);
                cudaMemcpy(synapse_meta_ptr, &fake_synapse_meta, sizeof(BaseSynapseMeta), cudaMemcpyHostToDevice);
                #endif
            }

            // Allocate neuron infos for input and detector neurons on detector allocator
            NeuronDataId_t input_neuron_synapses_infos_id =
                detector_connections_allocator->allocate(sizeof(IndexedSynapsesInfo) * this->n_inputs, 0);
            NeuronDataId_t detector_neuron_synapses_infos_id =
                detector_connections_allocator->allocate(sizeof(IndexedSynapsesInfo) * this->n_detectors, 0);

            // Initialize neuron infos to zero
            IndexedSynapsesInfo *input_infos = IndexedSynapsesInfos(input_neuron_synapses_infos_id, detector_connections_allocator->data);
            if(device == -1) {
                memset(input_infos, 0, sizeof(IndexedSynapsesInfo) * this->n_inputs);
            } else {
                #ifndef NO_CUDA
                c10::cuda::CUDAGuard guard(device);
                cudaMemset(input_infos, 0, sizeof(IndexedSynapsesInfo) * this->n_inputs);
                #endif
            }
            IndexedSynapsesInfo *detector_infos = IndexedSynapsesInfos(detector_neuron_synapses_infos_id, detector_connections_allocator->data);
            if(device == -1) {
                memset(detector_infos, 0, sizeof(IndexedSynapsesInfo) * this->n_detectors);
            } else {
                #ifndef NO_CUDA
                c10::cuda::CUDAGuard guard(device);
                cudaMemset(detector_infos, 0, sizeof(IndexedSynapsesInfo) * this->n_detectors);
                #endif
            }

            // Allocate global connections meta for detector connections on main only_host_allocator
            NeuronDataId_t detector_global_connections_meta_id = only_host_allocator.allocate(sizeof(GlobalConnectionsMeta), 0);
            GlobalConnectionsMeta* detector_gc_meta = reinterpret_cast<GlobalConnectionsMeta *>(only_host_allocator.data + detector_global_connections_meta_id);
            memset(detector_gc_meta, 0, sizeof(GlobalConnectionsMeta));

            // Create detector connections manager
            detector_connections_manager = new ConnectionsManager(
                #ifdef ENABLE_PROFILING
                profiler,
                #endif
                *detector_connections_allocator, only_host_allocator, true,
                detector_synapse_metas_id, detector_global_connections_meta_id,
                input_neuron_synapses_infos_id, this->n_inputs, 0,
                detector_neuron_synapses_infos_id, this->n_detectors, this->n_inputs,
                1  // One fake synapse meta
            );
        }

        std::optional<const torch::Tensor> none;
        // Detector connections don't use weights, so we pass nullptr for weights allocator
        detector_connections_manager->add_connections(
            connections_buffer, none,
            single_input_group_size,
            ids_shift,
            nullptr,  // No weights for detector connections
            random_seed ? random_seed.value() : std::random_device{}()
        );
    }

    void initialize_detectors(
        const torch::Tensor &encoded_pairs_permutations,
        uint32_t max_n_inputs_per_detector,
        torch::Tensor &detector_anchors,
        bool compact_mode
    ) {
        int device = host_device_allocator.device;
        checkTensor(encoded_pairs_permutations, "encoded_pairs_permutations", false, device, sizeof(int32_t));
        checkTensor(detector_anchors, "detector_anchors", false, device, sizeof(int32_t));
        if(detector_connections_manager == nullptr) {
            throw py::value_error("detector_connections_manager not initialized");
        }

        uint32_t max_pairs_per_detector;
        if(compact_mode) {
            max_pairs_per_detector = max_n_inputs_per_detector;
            if((encoded_pairs_permutations.numel() % max_pairs_per_detector) != 0) {
                throw py::value_error("(encoded_pairs_permutations.numel() % max_n_inputs_per_detector) != 0");
            }
        } else {
            max_pairs_per_detector = max_n_inputs_per_detector * (max_n_inputs_per_detector - 1);
            if((encoded_pairs_permutations.numel() % max_pairs_per_detector) != 0) {
                throw py::value_error("(encoded_pairs_permutations.numel() % (max_n_inputs_per_detector * (max_n_inputs_per_detector - 1))) != 0");
            }
        }

        uint32_t provided_n_detectors = encoded_pairs_permutations.numel() / max_pairs_per_detector;
        if(provided_n_detectors != this->n_detectors) {
            throw py::value_error("provided_n_detectors != this->n_detectors");
        }

        // Validate detector_anchors tensor size
        int64_t expected_anchors_size = static_cast<int64_t>(this->n_detectors) * this->n_anchors_per_detector * 2;
        if(detector_anchors.numel() != expected_anchors_size) {
            throw py::value_error("detector_anchors.numel() != n_detectors * n_anchors_per_detector * 2");
        }

        // Initialize detector_anchors to zero
        uint64_t detector_infos_memsize = static_cast<uint64_t>(this->n_detectors) * this->n_anchors_per_detector * sizeof(AnchorsPair);
        AnchorsPair* detector_infos = reinterpret_cast<AnchorsPair *>(detector_anchors.data_ptr());
        if(device == -1) {
            memset(detector_infos, 0, detector_infos_memsize);
        } else {
            #ifndef NO_CUDA
            c10::cuda::CUDAGuard guard(device);
            cudaMemset(detector_infos, 0, detector_infos_memsize);
            #endif
        }

        // Prepare encoded pairs data
        uint32_t* encoded_pairs_data = reinterpret_cast<uint32_t *>(encoded_pairs_permutations.data_ptr());

        // Allocate error counter
        uint32_t* error_counter;
        if(device == -1) {
            error_counter = reinterpret_cast<uint32_t *>(PyMem_Malloc(sizeof(uint32_t)));
            *error_counter = 0;
        } else {
            #ifndef NO_CUDA
            c10::cuda::CUDAGuard guard(device);
            cudaHostAlloc(&error_counter, sizeof(uint32_t), cudaHostAllocMapped);
            *error_counter = 0;
            #endif
        }

        // Call kernel to prepare detectors
        NeuronDataId_t detector_neuron_synapses_infos_id = detector_connections_manager->get_backward_neuron_infos_id();
        NoDelaysIndexedSynapsesInfo* backward_indexed_synapses_ptr =
            reinterpret_cast<NoDelaysIndexedSynapsesInfo *>(
                IndexedSynapsesInfos(
                    detector_neuron_synapses_infos_id, detector_connections_allocator->data
                )
            );

        dim3 numBlocks((this->n_detectors + LUT_COMPILE_TIME_KERNELS_TPB - 1) / LUT_COMPILE_TIME_KERNELS_TPB, 1);
        if(compact_mode) {
            GRID_CALL_NO_SHARED_MEM(
                numBlocks, prepare_detectors_compact, LUT_COMPILE_TIME_KERNELS_TPB,
                encoded_pairs_data,
                this->n_detectors,
                max_pairs_per_detector,
                this->n_anchors_per_detector,
                backward_indexed_synapses_ptr,
                this->backward_group_size,
                detector_infos,
                detector_connections_allocator->data,
                error_counter
            );
        } else {
            GRID_CALL_NO_SHARED_MEM(
                numBlocks, prepare_detectors, LUT_COMPILE_TIME_KERNELS_TPB,
                encoded_pairs_data,
                this->n_detectors,
                max_pairs_per_detector,
                this->n_anchors_per_detector,
                backward_indexed_synapses_ptr,
                this->backward_group_size,
                detector_infos,
                detector_connections_allocator->data,
                error_counter
            );
        }

        // Check for errors
        if(device == -1) {
            if(*error_counter > 0) {
                PyMem_Free(error_counter);
                throw py::value_error("Some detectors have <= 1 input synapses");
            }
            PyMem_Free(error_counter);
        } else {
            #ifndef NO_CUDA
            c10::cuda::CUDAGuard guard(device);
            if(*error_counter > 0) {
                cudaFreeHost(error_counter);
                throw py::value_error("Some detectors have <= 1 input synapses");
            }
            cudaFreeHost(error_counter);
            #endif
        }

        // Finalize and destroy detector connections manager and allocator
        delete detector_connections_manager;
        detector_connections_manager = nullptr;
        delete detector_connections_allocator;
        detector_connections_allocator = nullptr;
    }

    void add_lookup_connections(
        const torch::Tensor &connections_buffer,
        uint32_t single_input_group_size,
        int ids_shift,
        std::optional<uint32_t> &random_seed
    ) {
        if(lookup_neuron_synapses_infos_id == std::numeric_limits<uint64_t>::max()) {
            throw py::value_error("can't add lookup connections in fully connected mode");
        }
        checkConnectionsManagerIsInitialized();
        std::optional<const torch::Tensor> none;
        this->n_synapses += connections_manager->add_connections(
            connections_buffer, none,
            single_input_group_size,
            ids_shift,
            weights_allocator,
            random_seed ? random_seed.value() : 0
        );
    }

    void compile(
        bool only_trainable_backwards,
        torch::Tensor &weights,
        std::optional<uint32_t> &random_seed
    ) {
        if(lookup_neuron_synapses_infos_id == std::numeric_limits<uint64_t>::max()) {
            throw py::value_error("nothing to compile in fully connected mode");
        }

        if(random_seed && random_seed.value() == 0) {
            throw py::value_error("random_seed should be greater than 0");
        }
        checkConnectionsManagerIsInitialized();
        checkTensor(weights, "weights", true, host_device_allocator.device);

        GlobalConnectionsMeta* gc_meta = reinterpret_cast<GlobalConnectionsMeta *>(only_host_allocator.data + global_connections_meta_id);
        if(weights.numel() != N_WEIGHTS(gc_meta, true)) {
            throw py::value_error("wrong weights.numel()");
        }

        connections_manager->finalize(
            random_seed ? random_seed.value() : 0,
            true, true,
            only_trainable_backwards,
            true
        );

        EXTERNAL_REAL_DT* weights_data = reinterpret_cast<EXTERNAL_REAL_DT *>(weights.data_ptr());

        if(host_device_allocator.device == -1) {
            memcpy(
                weights_data,
                this->weights_allocator->data,
                weights.numel() * sizeof(EXTERNAL_REAL_DT)
            );
        } else {
            #ifndef NO_CUDA
            c10::cuda::CUDAGuard guard(host_device_allocator.device);
            cuMemcpyDtoD(
                (CUdeviceptr) weights_data,
                (CUdeviceptr) this->weights_allocator->data,
                weights.numel() * sizeof(EXTERNAL_REAL_DT)
            );
            #endif
        }
        delete this->weights_allocator;
        this->weights_allocator = nullptr;
    }

    void forward_step(
        const torch::Tensor &r_weights,
        uint32_t batch_size,
        const torch::Tensor &r_input,
        const torch::Tensor &r_detector_anchors,
        torch::Tensor &w_output,
        torch::Tensor &w_lookup_indices,
        std::optional<torch::Tensor> &r_stream_handles,
        std::optional<torch::Tensor> &w_min_anchor_deltas,
        std::optional<torch::Tensor> &w_min_anchor_delta_indices
    ) {
        py::gil_scoped_release gil_guard;
        __TRACE__("lutm_forward_step\n");
        checkTensor(r_weights, "r_weights", true, host_device_allocator.device);
        checkTensor(r_input, "r_input", true, host_device_allocator.device);
        checkTensor(r_detector_anchors, "r_detector_anchors", false, host_device_allocator.device, sizeof(int32_t));
        checkTensor(w_output, "w_output", true, host_device_allocator.device);
        checkTensor(w_lookup_indices, "w_lookup_indices", false, host_device_allocator.device, sizeof(int32_t));
        if(w_min_anchor_deltas.has_value()) {
            checkTensor(w_min_anchor_deltas.value(), "w_min_anchor_deltas", true, host_device_allocator.device);
        }
        if(w_min_anchor_delta_indices.has_value()) {
            checkTensor(w_min_anchor_delta_indices.value(), "w_min_anchor_delta_indices", false, host_device_allocator.device, sizeof(int32_t));
        }
        if(r_stream_handles.has_value()) {
            checkTensor(r_stream_handles.value(), "r_stream_handles", false, -1, sizeof(int64_t));
            if(r_stream_handles.value().numel() < 3) {
                throw py::value_error("r_stream_handles must have at least 3 elements");
            }
        }
        if(batch_size == 0) {
            throw py::value_error("batch_size == 0");
        }

        if(this->runtime_context == nullptr) {
            GlobalConnectionsMeta* gc_meta = reinterpret_cast<GlobalConnectionsMeta *>(only_host_allocator.data + global_connections_meta_id);
            IndexedSynapsesInfo *synapse_infos = nullptr;
            if(this->lookup_neuron_synapses_infos_id != std::numeric_limits<uint64_t>::max()) {
                synapse_infos = IndexedSynapsesInfos(this->lookup_neuron_synapses_infos_id, host_device_allocator.data);
            }
            this->runtime_context = new LUT_RUNTIME_CONTEXT_CLASS(
                host_device_allocator.data,
                host_device_allocator.device,
                this->n_inputs,
                this->n_outputs,
                this->n_detectors,
                this->n_anchors_per_detector,
                this->n_lookup_neurons,
                this->sequence_length,
                this->positional_dim,
                this->forward_group_size,
                this->backward_group_size,
                gc_meta->max_forward_groups_per_neuron,
                #ifdef INTEGERS_INSTEAD_OF_FLOATS
                N_WEIGHTS(gc_meta, true),
                int_rescaler,
                #endif
                #ifdef ENABLE_PROFILING
                this->profiler,
                #endif
                BaseSynapseMetas(this->base_synapse_metas_id, host_device_allocator.data),
                synapse_infos,
                gc_meta->first_synapse_id
            );
        }

        #ifndef NO_CUDA
        cudaStream_t *cuda_streams_ptr = nullptr;
        if(r_stream_handles.has_value() && host_device_allocator.device != -1) {
            cuda_streams_ptr = reinterpret_cast<cudaStream_t *>(r_stream_handles.value().data_ptr());
        }
        #endif
        this->runtime_context->forward_step(
            reinterpret_cast<EXTERNAL_REAL_DT *>(r_weights.data_ptr()),
            batch_size,
            reinterpret_cast<EXTERNAL_REAL_DT *>(r_input.data_ptr()),
            reinterpret_cast<AnchorsPair *>(r_detector_anchors.data_ptr()),
            reinterpret_cast<EXTERNAL_REAL_DT *>(w_output.data_ptr()),
            reinterpret_cast<int32_t *>(w_lookup_indices.data_ptr()),
            w_min_anchor_deltas.has_value() ? reinterpret_cast<EXTERNAL_REAL_DT *>(w_min_anchor_deltas.value().data_ptr()) : nullptr,
            w_min_anchor_delta_indices.has_value() ? reinterpret_cast<int32_t *>(w_min_anchor_delta_indices.value().data_ptr()) : nullptr
            #ifndef NO_CUDA
            , cuda_streams_ptr
            #endif
        );
    }

    void forward_step_concat(
        const torch::Tensor &r_weights,
        uint32_t batch_size,
        const torch::Tensor &r_input,
        const torch::Tensor &r_detector_anchors,
        torch::Tensor &w_output,
        torch::Tensor &w_lookup_indices,
        std::optional<torch::Tensor> &r_positional_embeddings,
        std::optional<torch::Tensor> &w_positional_lookup_indices,
        std::optional<torch::Tensor> &w_min_anchor_deltas,
        std::optional<torch::Tensor> &w_min_anchor_delta_indices,
        std::optional<torch::Tensor> &w_positional_min_deltas,
        std::optional<torch::Tensor> &w_positional_min_delta_indices,
        std::optional<torch::Tensor> &r_stream_handles
    ) {
        py::gil_scoped_release gil_guard;
        __TRACE__("lutm_forward_step_concat\n");
        checkTensor(r_weights, "r_weights", true, host_device_allocator.device);
        if(this->positional_dim > 0) {
            if(!r_positional_embeddings.has_value()) {
                throw py::value_error("r_positional_embeddings must be provided when positional_dim > 0");
            }
            checkTensor(r_positional_embeddings.value(), "r_positional_embeddings", true, host_device_allocator.device);
        } else {
            if(r_positional_embeddings.has_value()) {
                throw py::value_error("r_positional_embeddings must be None when positional_dim == 0");
            }
        }
        checkTensor(r_input, "r_input", true, host_device_allocator.device);
        checkTensor(r_detector_anchors, "r_detector_anchors", false, host_device_allocator.device, sizeof(int32_t));
        checkTensor(w_output, "w_output", true, host_device_allocator.device);
        checkTensor(w_lookup_indices, "w_lookup_indices", false, host_device_allocator.device, sizeof(int32_t));
        if(this->positional_dim > 0) {
            if(!w_positional_lookup_indices.has_value()) {
                throw py::value_error("w_positional_lookup_indices must be provided when positional_dim > 0");
            }
            checkTensor(w_positional_lookup_indices.value(), "w_positional_lookup_indices", false, host_device_allocator.device, sizeof(int32_t));
        }
        if(w_min_anchor_deltas.has_value()) {
            checkTensor(w_min_anchor_deltas.value(), "w_min_anchor_deltas", true, host_device_allocator.device);
        }
        if(w_min_anchor_delta_indices.has_value()) {
            checkTensor(w_min_anchor_delta_indices.value(), "w_min_anchor_delta_indices", false, host_device_allocator.device, sizeof(int32_t));
        }
        if(this->positional_dim > 0) {
            if(w_positional_min_deltas.has_value()) {
                checkTensor(w_positional_min_deltas.value(), "w_positional_min_deltas", true, host_device_allocator.device);
            }
            if(w_positional_min_delta_indices.has_value()) {
                checkTensor(w_positional_min_delta_indices.value(), "w_positional_min_delta_indices", false, host_device_allocator.device, sizeof(int32_t));
            }
        } else {
            if(w_positional_min_deltas.has_value()) {
                throw py::value_error("w_positional_min_deltas must be None when positional_dim == 0");
            }
            if(w_positional_min_delta_indices.has_value()) {
                throw py::value_error("w_positional_min_delta_indices must be None when positional_dim == 0");
            }
        }
        if(r_stream_handles.has_value()) {
            checkTensor(r_stream_handles.value(), "r_stream_handles", false, -1, sizeof(int64_t));
            if(r_stream_handles.value().numel() < 3) {
                throw py::value_error("r_stream_handles must have at least 3 elements");
            }
        }
        if(batch_size == 0) {
            throw py::value_error("batch_size == 0");
        }

        if(this->runtime_context == nullptr) {
            GlobalConnectionsMeta* gc_meta = reinterpret_cast<GlobalConnectionsMeta *>(only_host_allocator.data + global_connections_meta_id);
            IndexedSynapsesInfo *synapse_infos = nullptr;
            if(this->lookup_neuron_synapses_infos_id != std::numeric_limits<uint64_t>::max()) {
                synapse_infos = IndexedSynapsesInfos(this->lookup_neuron_synapses_infos_id, host_device_allocator.data);
            }
            this->runtime_context = new LUT_RUNTIME_CONTEXT_CLASS(
                host_device_allocator.data,
                host_device_allocator.device,
                this->n_inputs,
                this->n_outputs,
                this->n_detectors,
                this->n_anchors_per_detector,
                this->n_lookup_neurons,
                this->sequence_length,
                this->positional_dim,
                this->forward_group_size,
                this->backward_group_size,
                gc_meta->max_forward_groups_per_neuron,
                #ifdef INTEGERS_INSTEAD_OF_FLOATS
                N_WEIGHTS(gc_meta, true),
                int_rescaler,
                #endif
                #ifdef ENABLE_PROFILING
                this->profiler,
                #endif
                BaseSynapseMetas(this->base_synapse_metas_id, host_device_allocator.data),
                synapse_infos,
                gc_meta->first_synapse_id
            );
        }

        #ifndef NO_CUDA
        cudaStream_t *cuda_streams_ptr = nullptr;
        if(r_stream_handles.has_value() && host_device_allocator.device != -1) {
            cuda_streams_ptr = reinterpret_cast<cudaStream_t *>(r_stream_handles.value().data_ptr());
        }
        #endif

        this->runtime_context->forward_step_concat(
            reinterpret_cast<EXTERNAL_REAL_DT *>(r_weights.data_ptr()),
            batch_size,
            reinterpret_cast<EXTERNAL_REAL_DT *>(r_input.data_ptr()),
            reinterpret_cast<AnchorsPair *>(r_detector_anchors.data_ptr()),
            reinterpret_cast<EXTERNAL_REAL_DT *>(w_output.data_ptr()),
            reinterpret_cast<int32_t *>(w_lookup_indices.data_ptr()),
            (this->positional_dim > 0 && r_positional_embeddings.has_value()) ? reinterpret_cast<EXTERNAL_REAL_DT *>(r_positional_embeddings.value().data_ptr()) : nullptr,
            (this->positional_dim > 0 && w_positional_lookup_indices.has_value()) ? reinterpret_cast<int32_t *>(w_positional_lookup_indices.value().data_ptr()) : nullptr,
            w_min_anchor_deltas.has_value() ? reinterpret_cast<EXTERNAL_REAL_DT *>(w_min_anchor_deltas.value().data_ptr()) : nullptr,
            w_min_anchor_delta_indices.has_value() ? reinterpret_cast<int32_t *>(w_min_anchor_delta_indices.value().data_ptr()) : nullptr,
            (this->positional_dim > 0 && w_positional_min_deltas.has_value()) ? reinterpret_cast<EXTERNAL_REAL_DT *>(w_positional_min_deltas.value().data_ptr()) : nullptr,
            (this->positional_dim > 0 && w_positional_min_delta_indices.has_value()) ? reinterpret_cast<int32_t *>(w_positional_min_delta_indices.value().data_ptr()) : nullptr
            #ifndef NO_CUDA
            , cuda_streams_ptr
            #endif
        );
    }

    void backward_backprop(
        const torch::Tensor &r_weights,
        uint32_t batch_size,
        const torch::Tensor &r_output_gradients,
        const torch::Tensor &r_detector_anchors,
        const torch::Tensor &r_lookup_indices,
        const torch::Tensor &r_min_anchor_deltas,
        const torch::Tensor &r_min_anchor_delta_indices,
        torch::Tensor &w_input_gradients,
        double external_lr,
        std::optional<torch::Tensor> w_weights_gradients,
        std::optional<torch::Tensor> r_stream_handles
    ) {
        py::gil_scoped_release gil_guard;
        if(batch_size == 0) {
            throw py::value_error("batch_size == 0");
        }
        if(this->runtime_context == nullptr) {
            throw py::value_error("no active context");
        }
        checkTensor(r_weights, "r_weights", true, host_device_allocator.device);
        checkTensor(w_input_gradients, "w_input_gradients", true, host_device_allocator.device);
        checkTensor(r_output_gradients, "r_output_gradients", true, host_device_allocator.device);
        checkTensor(r_detector_anchors, "r_detector_anchors", false, host_device_allocator.device, sizeof(int32_t));
        checkTensor(r_lookup_indices, "r_lookup_indices", false, host_device_allocator.device, sizeof(int32_t));
        checkTensor(r_min_anchor_deltas, "r_min_anchor_deltas", true, host_device_allocator.device);
        checkTensor(r_min_anchor_delta_indices, "r_min_anchor_delta_indices", false, host_device_allocator.device, sizeof(int32_t));
        if(w_weights_gradients.has_value()) {
            checkTensor(w_weights_gradients.value(), "w_weights_gradients", true, host_device_allocator.device);
        }
        if(r_stream_handles.has_value()) {
            checkTensor(r_stream_handles.value(), "r_stream_handles", false, -1, sizeof(int64_t));
            if(r_stream_handles.value().numel() < 3) {
                throw py::value_error("r_stream_handles must have at least 3 elements");
            }
        }

        #ifndef NO_CUDA
        cudaStream_t *cuda_streams_ptr = nullptr;
        if(r_stream_handles.has_value() && host_device_allocator.device != -1) {
            cuda_streams_ptr = reinterpret_cast<cudaStream_t *>(r_stream_handles.value().data_ptr());
        }
        #endif
        this->runtime_context->backward_backprop(
            reinterpret_cast<EXTERNAL_REAL_DT *>(r_weights.data_ptr()),
            batch_size,
            reinterpret_cast<EXTERNAL_REAL_DT *>(r_output_gradients.data_ptr()),
            reinterpret_cast<AnchorsPair *>(r_detector_anchors.data_ptr()),
            reinterpret_cast<int32_t *>(r_lookup_indices.data_ptr()),
            reinterpret_cast<EXTERNAL_REAL_DT *>(r_min_anchor_deltas.data_ptr()),
            reinterpret_cast<int32_t *>(r_min_anchor_delta_indices.data_ptr()),
            reinterpret_cast<EXTERNAL_REAL_DT *>(w_input_gradients.data_ptr()),
            static_cast<EXTERNAL_REAL_DT>(external_lr),
            w_weights_gradients.has_value() ? reinterpret_cast<EXTERNAL_REAL_DT *>(w_weights_gradients.value().data_ptr()) : nullptr
            #ifndef NO_CUDA
            , cuda_streams_ptr
            #endif
        );
    }

    void backward_backprop_concat(
        const torch::Tensor &r_weights,
        uint32_t batch_size,
        const torch::Tensor &r_output_gradients,
        const torch::Tensor &r_detector_anchors,
        const torch::Tensor &r_lookup_indices,
        const torch::Tensor &r_min_anchor_deltas,
        const torch::Tensor &r_min_anchor_delta_indices,
        torch::Tensor &w_input_gradients,
        double external_lr,
        std::optional<torch::Tensor> &r_positional_lookup_indices,
        std::optional<torch::Tensor> &r_positional_min_deltas,
        std::optional<torch::Tensor> &r_positional_min_delta_indices,
        std::optional<torch::Tensor> &w_positional_embeddings_gradients,
        std::optional<torch::Tensor> &w_weights_gradients,
        std::optional<torch::Tensor> &r_stream_handles
    ) {
        py::gil_scoped_release gil_guard;
        if(batch_size == 0) {
            throw py::value_error("batch_size == 0");
        }
        if(this->runtime_context == nullptr) {
            throw py::value_error("no active context");
        }
        checkTensor(r_weights, "r_weights", true, host_device_allocator.device);
        checkTensor(w_input_gradients, "w_input_gradients", true, host_device_allocator.device);
        if(this->positional_dim > 0) {
            if(!w_positional_embeddings_gradients.has_value()) {
                throw py::value_error("w_positional_embeddings_gradients must be provided when positional_dim > 0");
            }
            checkTensor(w_positional_embeddings_gradients.value(), "w_positional_embeddings_gradients", true, host_device_allocator.device);
        } else {
            if(w_positional_embeddings_gradients.has_value()) {
                throw py::value_error("w_positional_embeddings_gradients must be None when positional_dim == 0");
            }
        }
        checkTensor(r_output_gradients, "r_output_gradients", true, host_device_allocator.device);
        checkTensor(r_detector_anchors, "r_detector_anchors", false, host_device_allocator.device, sizeof(int32_t));
        checkTensor(r_lookup_indices, "r_lookup_indices", false, host_device_allocator.device, sizeof(int32_t));
        checkTensor(r_min_anchor_deltas, "r_min_anchor_deltas", true, host_device_allocator.device);
        checkTensor(r_min_anchor_delta_indices, "r_min_anchor_delta_indices", false, host_device_allocator.device, sizeof(int32_t));
        if(this->positional_dim > 0) {
            if(!r_positional_lookup_indices.has_value()) {
                throw py::value_error("r_positional_lookup_indices must be provided when positional_dim > 0");
            }
            checkTensor(r_positional_lookup_indices.value(), "r_positional_lookup_indices", false, host_device_allocator.device, sizeof(int32_t));
            if(!r_positional_min_deltas.has_value()) {
                throw py::value_error("r_positional_min_deltas must be provided when positional_dim > 0");
            }
            checkTensor(r_positional_min_deltas.value(), "r_positional_min_deltas", true, host_device_allocator.device);
            if(!r_positional_min_delta_indices.has_value()) {
                throw py::value_error("r_positional_min_delta_indices must be provided when positional_dim > 0");
            }
            checkTensor(r_positional_min_delta_indices.value(), "r_positional_min_delta_indices", false, host_device_allocator.device, sizeof(int32_t));
        }
        if(w_weights_gradients.has_value()) {
            checkTensor(w_weights_gradients.value(), "w_weights_gradients", true, host_device_allocator.device);
        }
        if(r_stream_handles.has_value()) {
            checkTensor(r_stream_handles.value(), "r_stream_handles", false, -1, sizeof(int64_t));
            if(r_stream_handles.value().numel() < 3) {
                throw py::value_error("r_stream_handles must have at least 3 elements");
            }
        }

        #ifndef NO_CUDA
        cudaStream_t *cuda_streams_ptr = nullptr;
        if(r_stream_handles.has_value() && host_device_allocator.device != -1) {
            cuda_streams_ptr = reinterpret_cast<cudaStream_t *>(r_stream_handles.value().data_ptr());
        }
        #endif
        this->runtime_context->backward_backprop_concat(
            reinterpret_cast<EXTERNAL_REAL_DT *>(r_weights.data_ptr()),
            batch_size,
            reinterpret_cast<EXTERNAL_REAL_DT *>(r_output_gradients.data_ptr()),
            reinterpret_cast<AnchorsPair *>(r_detector_anchors.data_ptr()),
            reinterpret_cast<int32_t *>(r_lookup_indices.data_ptr()),
            reinterpret_cast<EXTERNAL_REAL_DT *>(r_min_anchor_deltas.data_ptr()),
            reinterpret_cast<int32_t *>(r_min_anchor_delta_indices.data_ptr()),
            reinterpret_cast<EXTERNAL_REAL_DT *>(w_input_gradients.data_ptr()),
            static_cast<EXTERNAL_REAL_DT>(external_lr),
            (this->positional_dim > 0 && r_positional_lookup_indices.has_value()) ? reinterpret_cast<int32_t *>(r_positional_lookup_indices.value().data_ptr()) : nullptr,
            (this->positional_dim > 0 && r_positional_min_deltas.has_value()) ? reinterpret_cast<EXTERNAL_REAL_DT *>(r_positional_min_deltas.value().data_ptr()) : nullptr,
            (this->positional_dim > 0 && r_positional_min_delta_indices.has_value()) ? reinterpret_cast<int32_t *>(r_positional_min_delta_indices.value().data_ptr()) : nullptr,
            (this->positional_dim > 0 && w_positional_embeddings_gradients.has_value()) ? reinterpret_cast<EXTERNAL_REAL_DT *>(w_positional_embeddings_gradients.value().data_ptr()) : nullptr,
            w_weights_gradients.has_value() ? reinterpret_cast<EXTERNAL_REAL_DT *>(w_weights_gradients.value().data_ptr()) : nullptr
            #ifndef NO_CUDA
            , cuda_streams_ptr
            #endif
        );
    }

    uint64_t count_synapses(
        const torch::Tensor &neuron_indices_to_process
    ) {
        if(lookup_neuron_synapses_infos_id == std::numeric_limits<uint64_t>::max()) {
            throw py::value_error("count_synapses: there is nothing to count in fully connected mode");
        }
        checkConnectionsManagerIsInitialized();
        return connections_manager->count_synapses(
            neuron_indices_to_process,
            true
        );
    }

    void export_synapses(
        const torch::Tensor &weights,
        const torch::Tensor &neuron_indices_to_process,
        torch::Tensor &target_internal_source_indices,
        torch::Tensor &target_weights,
        torch::Tensor &target_internal_target_indices,
        std::optional<torch::Tensor> &target_synapse_meta_indices
    ) {
        if(lookup_neuron_synapses_infos_id == std::numeric_limits<uint64_t>::max()) {
            throw py::value_error("export_synapses: there is nothing to export in fully connected mode");
        }
        checkConnectionsManagerIsInitialized();
        checkTensor(weights, "weights", true, host_device_allocator.device);

        EXTERNAL_REAL_DT* weights_data = reinterpret_cast<EXTERNAL_REAL_DT *>(weights.data_ptr());
        std::optional<torch::Tensor> none;
        connections_manager->export_synapses(
            neuron_indices_to_process,
            target_internal_source_indices,
            target_weights,
            target_internal_target_indices,
            true,
            none,
            target_synapse_meta_indices,
            weights_data
        );
    }

    auto __repr__() {
        std::ostringstream os;
        GlobalConnectionsMeta* gc_meta = reinterpret_cast<GlobalConnectionsMeta *>(only_host_allocator.data + global_connections_meta_id);
        os << "LUTDataManager(host_device: " <<
            host_device_allocator.allocated <<
            ", " << host_device_allocator.used <<
            "; host_only: " <<
            only_host_allocator.allocated <<
            ", " << only_host_allocator.used <<
            "; summation type: " << SUMMATION_DT_STR <<
            "; smallest distinguishable fraction: " << get_smallest_distinguishable_fraction() <<
            "; n_synapses: " << n_synapses <<
            "; n_detectors: " << n_detectors <<
            "; n_lookup_neurons: " << n_lookup_neurons <<
            "; first_synapse_id: " << gc_meta->first_synapse_id <<
            "; last_synapse_id: " << gc_meta->last_synapse_id <<
            "; n_forward_groups: " << gc_meta->n_forward_groups <<
            "; n_backward_groups: " << gc_meta->n_backward_groups <<
        ")";
        return os.str();
    }

    auto get_memory_stats() {
        std::ostringstream os;
        for(int i=0; i < N_LUT_MEMORY_LABELS;i++) {
            switch(i) {
                case SYNAPSE_METAS_MEMORY_LABEL:
                    os << "synapse metas: ";
                    break;
                case NEURON_INFOS_MEMORY_LABEL:
                    os << "neuron infos: ";
                    break;
                case DETECTORS_MEMORY_LABEL:
                    os << "detectors: ";
                    break;
                case FORWARD_SYNAPSES_MEMORY_LABEL:
                    os << "forward main synapses: ";
                    break;
                case BACKWARD_SYNAPSES_MEMORY_LABEL:
                    os << "backward synapses: ";
                    break;
                case DELAYS_INFO_MEMORY_LABEL:
                    os << "delays info: ";
                    break;
            }
            os << host_device_allocator.get_label_stat(i) / 1024;
            os << " KB\n";
        }
        return os.str();
    }

    auto get_profiling_stats() {
        #ifdef ENABLE_PROFILING
            std::ostringstream os;
            os << profiler.get_stats_as_string();
            return os.str();
        #else
            return "profiler is disabled";
        #endif
    }

    void reset_profiler() {
        #ifdef ENABLE_PROFILING
        profiler.reset();
        #endif
    }

    ~LUTM_CLASS_NAME() {
        if(connections_manager != nullptr) {
            delete connections_manager;
        }
        if(runtime_context != nullptr) {
            delete runtime_context;
        }
        if(detector_connections_manager != nullptr) {
            delete detector_connections_manager;
        }
        if(detector_connections_allocator != nullptr) {
            delete detector_connections_allocator;
        }
    }

    void checkOnHostDuringPrepare()
    {
        if(host_device_allocator.device != -1) {
            throw std::runtime_error("LUTDataManager must be in the host memory at this point");
        }
    }

    void checkConnectionsManagerIsInitialized()
    {
        if(connections_manager == nullptr) {
            throw py::value_error("connections_manager is not initialized, use initialize_neurons(...) method");
        }
    }

private:
    LUTM_CLASS_NAME(
        uint8_t* host_device_data, NeuronDataId_t host_device_used,
        uint8_t* only_host_data, NeuronDataId_t only_host_used,
        uint32_t n_inputs,
        uint32_t n_outputs,
        uint32_t n_detectors,
        uint32_t n_anchors_per_detector,
        uint32_t sequence_length,
        uint32_t positional_dim,
        uint32_t forward_group_size,
        uint32_t backward_group_size,
        NeuronDataId_t base_synapse_metas_id,
        uint32_t n_synapse_metas,
        uint64_t n_synapses,
        NeuronDataId_t lookup_neuron_synapses_infos_id,
        NeuronDataId_t global_connections_meta_id
        #ifdef INTEGERS_INSTEAD_OF_FLOATS
        , double int_rescaler
        #endif
    ) :
        #ifdef ENABLE_PROFILING
        profiler(N_LUT_PROFILER_OPS),
        #endif
        host_device_allocator(host_device_data, host_device_used),
        only_host_allocator(only_host_data, only_host_used),
        runtime_context(nullptr),
        weights_allocator(nullptr),
        detector_connections_allocator(nullptr),
        detector_connections_manager(nullptr),
        n_inputs(n_inputs),
        n_outputs(n_outputs),
        n_detectors(n_detectors),
        n_anchors_per_detector(n_anchors_per_detector),
        sequence_length(sequence_length),
        positional_dim(positional_dim),
        n_lookup_neurons(n_detectors * (1U << (n_anchors_per_detector + ((sequence_length > 1) ? (n_anchors_per_detector + positional_dim) : 0)))),
        forward_group_size(forward_group_size),
        backward_group_size(backward_group_size),
        base_synapse_metas_id(base_synapse_metas_id),
        n_synapse_metas(n_synapse_metas),
        n_synapses(n_synapses),
        lookup_neuron_synapses_infos_id(lookup_neuron_synapses_infos_id),
        global_connections_meta_id(global_connections_meta_id)
    {
        connections_manager = new ConnectionsManager(
            #ifdef ENABLE_PROFILING
            profiler,
            #endif
            host_device_allocator, only_host_allocator, true,
            base_synapse_metas_id, global_connections_meta_id,
            lookup_neuron_synapses_infos_id, n_lookup_neurons, 0,
            0, 0, n_lookup_neurons,  // No backward synapses for output neurons
            n_synapse_metas
        );
        #ifdef INTEGERS_INSTEAD_OF_FLOATS
        this->int_rescaler = int_rescaler;
        #endif
    }

    #ifdef ENABLE_PROFILING
    SimpleProfiler profiler;
    #endif
    SimpleAllocator host_device_allocator;
    SimpleAllocator only_host_allocator;
    ConnectionsManager* connections_manager;
    LUT_RUNTIME_CONTEXT_CLASS *runtime_context;
    SimpleAllocator* weights_allocator;
    SimpleAllocator* detector_connections_allocator;
    ConnectionsManager* detector_connections_manager;

    uint32_t n_inputs;
    uint32_t n_outputs;
    uint32_t n_detectors;
    uint32_t n_anchors_per_detector;
    uint32_t sequence_length;
    uint32_t positional_dim;
    uint32_t n_lookup_neurons;
    uint32_t forward_group_size;
    uint32_t backward_group_size;
    NeuronDataId_t base_synapse_metas_id;
    uint32_t n_synapse_metas;
    uint64_t n_synapses;
    NeuronDataId_t lookup_neuron_synapses_infos_id;
    NeuronDataId_t global_connections_meta_id;
    #ifdef INTEGERS_INSTEAD_OF_FLOATS
    double int_rescaler;
    #endif
};

py::tuple PFX(pickle_lut_neuron_manager)(const LUTM_CLASS_NAME& ldm) {
    if(ldm.host_device_allocator.device != -1) {
        throw std::runtime_error("net must be on CPU before serialization");
    }
    return py::make_tuple(
        py::reinterpret_steal<py::object>(
            PyBytes_FromStringAndSize((char *) ldm.host_device_allocator.data, ldm.host_device_allocator.used)
        ),
        py::reinterpret_steal<py::object>(
            PyBytes_FromStringAndSize((char *) ldm.only_host_allocator.data, ldm.only_host_allocator.used)
        ),
        ldm.n_inputs,
        ldm.n_outputs,
        ldm.n_detectors,
        ldm.n_anchors_per_detector,
        ldm.sequence_length,
        ldm.positional_dim,
        ldm.forward_group_size,
        ldm.backward_group_size,
        ldm.base_synapse_metas_id,
        ldm.n_synapse_metas,
        ldm.n_synapses,
        ldm.lookup_neuron_synapses_infos_id,
        ldm.global_connections_meta_id
        #ifdef INTEGERS_INSTEAD_OF_FLOATS
        , ldm.int_rescaler
        #endif
    );
}

std::unique_ptr<LUTM_CLASS_NAME> PFX(unpickle_lut_neuron_manager)(py::tuple t) {
    char *buf;
    Py_ssize_t host_device_used;
    PyBytes_AsStringAndSize(t[0].ptr(), &buf, &host_device_used);
    uint8_t *host_device_data = (uint8_t *) PyMem_Malloc(host_device_used);
    memcpy(host_device_data, buf, host_device_used);

    Py_ssize_t only_host_used;
    PyBytes_AsStringAndSize(t[1].ptr(), &buf, &only_host_used);
    uint8_t *only_host_data = (uint8_t *) PyMem_Malloc(only_host_used);
    memcpy(only_host_data, buf, only_host_used);

    return std::unique_ptr<LUTM_CLASS_NAME>(
        new LUTM_CLASS_NAME(
            host_device_data, (NeuronDataId_t) host_device_used,
            only_host_data, (NeuronDataId_t) only_host_used,
            t[2].cast<uint32_t>(),         // n_inputs
            t[3].cast<uint32_t>(),         // n_outputs
            t[4].cast<uint32_t>(),         // n_detectors
            t[5].cast<uint32_t>(),         // n_anchors_per_detector
            t[6].cast<uint32_t>(),         // sequence_length
            t[7].cast<uint32_t>(),         // positional_dim
            t[8].cast<uint32_t>(),         // forward_group_size
            t[9].cast<uint32_t>(),         // backward_group_size
            t[10].cast<NeuronDataId_t>(),   // base_synapse_metas_id
            t[11].cast<uint32_t>(),        // n_synapse_metas
            t[12].cast<uint64_t>(),        // n_synapses
            t[13].cast<NeuronDataId_t>(),  // lookup_neuron_synapses_infos_id
            t[14].cast<NeuronDataId_t>()   // global_connections_meta_id
            #ifdef INTEGERS_INSTEAD_OF_FLOATS
            , t[15].cast<double>()         // int_rescaler
            #endif
        )
    );
}


void PFX(PB_LUTDataManager)(py::module& m) {
    #ifdef INTEGERS_INSTEAD_OF_FLOATS
    py::class_<LUTDataManagerI>(m, "LUTDataManagerI")
        .def(py::init<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint64_t, uint32_t, uint32_t, double>())
    #else
    py::class_<LUTDataManagerF>(m, "LUTDataManagerF")
        .def(py::init<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint64_t, uint32_t, uint32_t>())
    #endif
        .def("get_smallest_distinguishable_fraction", &LUTM_CLASS_NAME::get_smallest_distinguishable_fraction,
            "Returns smallest fraction that can exist inside data manager in integers mode. Returns 0.0 in floats mode.")
        .def("get_epsilon", &LUTM_CLASS_NAME::get_epsilon,
            "Returns super small real number used inside the engine")
        .def("get_summations_data_type", &LUTM_CLASS_NAME::get_summations_data_type,
            "Returns string 'float32' in floats mode and string 'int32' in integers mode")
        .def("register_synapse_meta", &LUTM_CLASS_NAME::register_synapse_meta,
            "Register synapse meta",
            py::arg("learning_rate"),
            py::arg("min_synaptic_weight"),
            py::arg("max_synaptic_weight"),
            py::arg("initial_noise_level"),
            py::arg("initial_weight"))
        .def("initialize_neurons", &LUTM_CLASS_NAME::initialize_neurons,
            py::arg("is_fully_connected"),
            "Initialize neurons")
        .def("initialize_detectors", &LUTM_CLASS_NAME::initialize_detectors,
            "Initialize detectors",
            py::arg("encoded_pairs_permutations"),
            py::arg("max_n_inputs_per_detector"),
            py::arg("detector_anchors"),
            py::arg("compact_mode"))
        .def("finalize_detector_connections", &LUTM_CLASS_NAME::finalize_detector_connections,
            "Finalize detector connections and return max number of inputs per detector")
        .def("get_number_of_inputs", &LUTM_CLASS_NAME::get_number_of_inputs,
            "Get number of input neurons")
        .def("get_number_of_outputs", &LUTM_CLASS_NAME::get_number_of_outputs,
            "Get number of output neurons")
        .def("get_number_of_detectors", &LUTM_CLASS_NAME::get_number_of_detectors,
            "Get number of detectors")
        .def("get_number_of_lookup_neurons", &LUTM_CLASS_NAME::get_number_of_lookup_neurons,
            "Get number of lookup neurons")
        .def("get_number_of_anchors_per_detector", &LUTM_CLASS_NAME::get_number_of_anchors_per_detector,
            "Get number of anchors per detector")
        .def("get_number_of_synapses", &LUTM_CLASS_NAME::get_number_of_synapses,
            "Get number of synapses")
        .def("get_weights_dimension", &LUTM_CLASS_NAME::get_weights_dimension,
            "Get the length of the weights array (it's greater than number of synapses because of small holes)")
        .def("get_max_forward_groups_per_neuron", &LUTM_CLASS_NAME::get_max_forward_groups_per_neuron,
            "Get maximum number of forward groups per neuron")
        .def("to_device", &LUTM_CLASS_NAME::to_device,
            "Move to device",
            py::arg("device"))
        .def("add_detector_connections", &LUTM_CLASS_NAME::add_detector_connections,
            "Add detector connections",
            py::arg("connections_buffer"),
            py::arg("single_input_group_size"),
            py::arg("ids_shift"),
            py::arg("random_seed") = py::none())
        .def("add_lookup_connections", &LUTM_CLASS_NAME::add_lookup_connections,
            "Add lookup synapses",
            py::arg("connections_buffer"),
            py::arg("single_input_group_size"),
            py::arg("ids_shift"),
            py::arg("random_seed") = py::none())
        .def("compile", &LUTM_CLASS_NAME::compile,
            "compile the network",
            py::arg("only_trainable_backwards"),
            py::arg("weights"),
            py::arg("random_seed") = py::none())
        .def("forward_step", &LUTM_CLASS_NAME::forward_step,
            "Forward step",
            py::arg("r_weights"),
            py::arg("batch_size"),
            py::arg("r_input"),
            py::arg("r_detector_anchors"),
            py::arg("w_output"),
            py::arg("w_lookup_indices"),
            py::arg("r_stream_handles") = py::none(),
            py::arg("w_min_anchor_deltas") = py::none(),
            py::arg("w_min_anchor_delta_indices") = py::none())
        .def("forward_step_concat", &LUTM_CLASS_NAME::forward_step_concat,
            "Forward step concat for fully connected mode",
            py::arg("r_weights"),
            py::arg("batch_size"),
            py::arg("r_input"),
            py::arg("r_detector_anchors"),
            py::arg("w_output"),
            py::arg("w_lookup_indices"),
            py::arg("r_positional_embeddings") = py::none(),
            py::arg("w_positional_lookup_indices") = py::none(),
            py::arg("w_min_anchor_deltas") = py::none(),
            py::arg("w_min_anchor_delta_indices") = py::none(),
            py::arg("w_positional_min_deltas") = py::none(),
            py::arg("w_positional_min_delta_indices") = py::none(),
            py::arg("r_stream_handles") = py::none())
        .def("backward_backprop", &LUTM_CLASS_NAME::backward_backprop,
            "Gradients back propagation",
            py::arg("r_weights"),
            py::arg("batch_size"),
            py::arg("r_output_gradients"),
            py::arg("r_detector_anchors"),
            py::arg("r_lookup_indices"),
            py::arg("r_min_anchor_deltas"),
            py::arg("r_min_anchor_delta_indices"),
            py::arg("w_input_gradients"),
            py::arg("external_lr"),
            py::arg("w_weights_gradients") = py::none(),
            py::arg("r_stream_handles") = py::none())
        .def("backward_backprop_concat", &LUTM_CLASS_NAME::backward_backprop_concat,
            "Gradients back propagation for fully connected sequence mode",
            py::arg("r_weights"),
            py::arg("batch_size"),
            py::arg("r_output_gradients"),
            py::arg("r_detector_anchors"),
            py::arg("r_lookup_indices"),
            py::arg("r_min_anchor_deltas"),
            py::arg("r_min_anchor_delta_indices"),
            py::arg("w_input_gradients"),
            py::arg("external_lr"),
            py::arg("r_positional_lookup_indices") = py::none(),
            py::arg("r_positional_min_deltas") = py::none(),
            py::arg("r_positional_min_delta_indices") = py::none(),
            py::arg("w_positional_embeddings_gradients") = py::none(),
            py::arg("w_weights_gradients") = py::none(),
            py::arg("r_stream_handles") = py::none())
        .def("count_synapses", &LUTM_CLASS_NAME::count_synapses,
            "Count forward or backward synapses for the set of neurons",
            py::arg("neuron_indices_to_process"))
        .def("export_synapses", &LUTM_CLASS_NAME::export_synapses,
            "Export all synapses for a set of neurons",
            py::arg("weights"),
            py::arg("neuron_indices_to_process"),
            py::arg("target_internal_source_indices"),
            py::arg("target_weights"),
            py::arg("target_internal_target_indices"),
            py::arg("target_synapse_meta_indices") = py::none())
        .def("__repr__", &LUTM_CLASS_NAME::__repr__)
        .def("get_memory_stats", &LUTM_CLASS_NAME::get_memory_stats)
        .def("get_profiling_stats", &LUTM_CLASS_NAME::get_profiling_stats)
        .def("reset_profiler", &LUTM_CLASS_NAME::reset_profiler)
        .def(py::pickle(
            &PFX(pickle_lut_neuron_manager),
            &PFX(unpickle_lut_neuron_manager)
        ));
}


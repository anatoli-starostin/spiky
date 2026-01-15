#include "andn_runtime.cuh"
#include <limits.h>
#include <random>

namespace py = pybind11;

#define ANDM_CLASS_NAME PFX(ANDNDataManager)

class __attribute__((visibility("hidden"))) ANDM_CLASS_NAME {
    friend py::tuple PFX(pickle_andn_neuron_manager)(const ANDM_CLASS_NAME& ndm);
    friend std::unique_ptr<ANDM_CLASS_NAME> PFX(unpickle_andn_neuron_manager)(py::tuple t);
public:
    ANDM_CLASS_NAME(
        uint32_t n_inputs, uint32_t n_outputs,
        uint64_t initial_synapse_capacity,
        uint32_t forward_group_size, uint32_t backward_group_size,
        bool spiking_inhibition
        #ifdef INTEGERS_INSTEAD_OF_FLOATS
        , double int_rescaler
        #endif
    ) :
        #ifdef ENABLE_PROFILING
        profiler(N_ANDN_PROFILER_OPS),
        #endif
        host_device_allocator(initial_synapse_capacity * 2 * sizeof(NeuronIndex_t) + MAX_N_SYNAPSE_METAS * sizeof(BaseSynapseMeta)),
        only_host_allocator(1024),
        connections_manager(nullptr), runtime_context(nullptr),
        n_inputs(n_inputs),
        n_outputs(n_outputs),
        forward_group_size(forward_group_size),
        backward_group_size(backward_group_size),
        spiking_inhibition(spiking_inhibition)
    {
        if(this->n_outputs > 0) {
            this->base_synapse_metas_id = host_device_allocator.allocate(MAX_N_SYNAPSE_METAS * sizeof(BaseSynapseMeta), SYNAPSE_METAS_MEMORY_LABEL);
        } else {
            this->base_synapse_metas_id = 0;
        }
        this->global_connections_meta_id = only_host_allocator.allocate(sizeof(GlobalConnectionsMeta), 0);
        GlobalConnectionsMeta* gc_meta = reinterpret_cast<GlobalConnectionsMeta *>(only_host_allocator.data + global_connections_meta_id);
        memset(gc_meta, 0, sizeof(GlobalConnectionsMeta));
        this->n_synapse_metas = 0;
        this->n_synapses = 0;
        this->input_neuron_synapses_infos_id = 0;
        this->output_neuron_synapses_infos_id = 0;
        this->detectors_id = 0;
        this->n_detectors = 0;
        this->max_inputs_per_detector = 0;

        #ifdef ENABLE_PROFILING
        if(n_outputs == 0) {
            // only to intialise profiler properly
            ConnectionsManager *fake_manager = new ConnectionsManager(
                #ifdef ENABLE_PROFILING
                profiler,
                #endif
                host_device_allocator, only_host_allocator, true,
                0, 0,
                0, 0, 0, 0, 0, 0, 0
            );
            delete fake_manager;
        }
        profiler.register_operation_type(ANDN_RUNTIME_FIRE_DETECTORS_PROFILER_OP, "andn::runtime::fire_detectors");
        profiler.register_operation_type(ANDN_RUNTIME_FIRE_INPUTS_PROFILER_OP, "andn::runtime::fire_inputs");
        profiler.register_operation_type(ANDN_RUNTIME_FILL_OUTPUTS_PROFILER_OP, "andn::runtime::fill_outputs");
        profiler.register_operation_type(ANDN_RUNTIME_CONVERT_OUTPUTS_PROFILER_OP, "andn::runtime::convert_outputs");
        profiler.register_operation_type(ANDN_RUNTIME_BACKWARD_HEBB_PROFILER_OP, "andn::runtime::backward_hebb");
        profiler.register_operation_type(ANDN_RUNTIME_BACKWARD_GRAD_PROFILER_OP, "andn::runtime::backward_grad");
        #endif

        if(n_inputs == 0) {
            throw py::value_error("n_inputs == 0");
        }

        // n_outputs may be 0 in a special detectors only case

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

        if(n_outputs > 0) {
            weights_allocator = new SimpleAllocator(initial_synapse_capacity * sizeof(EXTERNAL_REAL_DT));
        } else {
            weights_allocator = nullptr;
        }

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
        __TRACE__("andm_register_synapse_meta\n");
        checkOnHostDuringPrepare();
        if(this->n_outputs == 0) {
            throw py::value_error("can't register synapse metas in only detectors mode");
        }

        if(this->input_neuron_synapses_infos_id != 0) {
            throw py::value_error("can't register new synapse metas after neurons were initialized");
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
            this->forward_group_size, this->backward_group_size
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

    void initialize_neurons() {
        __TRACE__("andm_initialize_neurons\n");
        checkOnHostDuringPrepare();

        this->input_neuron_synapses_infos_id = host_device_allocator.allocate(sizeof(IndexedSynapsesInfo) * this->n_inputs, NEURON_INFOS_MEMORY_LABEL);
        if(this->n_outputs > 0) {
            this->output_neuron_synapses_infos_id = host_device_allocator.allocate(sizeof(IndexedSynapsesInfo) * this->n_outputs, NEURON_INFOS_MEMORY_LABEL);
        }

        IndexedSynapsesInfo *current_neuron_info = IndexedSynapsesInfos(this->input_neuron_synapses_infos_id, host_device_allocator.data);
        memset(current_neuron_info, 0, sizeof(IndexedSynapsesInfo) * this->n_inputs);
        if(this->n_outputs > 0) {
            current_neuron_info = IndexedSynapsesInfos(this->output_neuron_synapses_infos_id, host_device_allocator.data);
            memset(current_neuron_info, 0, sizeof(IndexedSynapsesInfo) * this->n_outputs);
            connections_manager = new ConnectionsManager(
                #ifdef ENABLE_PROFILING
                profiler,
                #endif
                host_device_allocator, only_host_allocator, true,
                this->base_synapse_metas_id, this->global_connections_meta_id,
                this->input_neuron_synapses_infos_id, this->n_inputs, 0,
                this->output_neuron_synapses_infos_id, this->n_outputs, this->n_inputs,
                this->n_synapse_metas
            );
        }
    }

    void initialize_detectors(
        const torch::Tensor &detectors_data,
        uint32_t max_inputs_per_detector
    ) {
        checkOnHostDuringPrepare();
        checkTensor(detectors_data, "detectors_data", false, host_device_allocator.device, sizeof(int32_t));
        if((detectors_data.numel() % max_inputs_per_detector) != 0) {
            throw py::value_error("(detectors_data.numel() % max_inputs_per_detector) != 0");
        }
        this->n_detectors = detectors_data.numel() / max_inputs_per_detector;
        this->max_inputs_per_detector = max_inputs_per_detector;

        uint64_t memsize = static_cast<uint64_t>(detectors_data.numel()) * sizeof(int32_t);
        this->detectors_id = host_device_allocator.allocate(memsize, DETECTORS_MEMORY_LABEL);

        int32_t* detectors_internal_ptr = reinterpret_cast<int32_t *>(host_device_allocator.data + this->detectors_id);
        memcpy(
            detectors_internal_ptr,
            detectors_data.data_ptr(),
            memsize
        );
    }

    uint32_t get_number_of_inputs() {
        __TRACE__("andm_get_number_of_inputs\n");
        return this->n_inputs;
    }

    uint32_t get_number_of_outputs() {
        __TRACE__("andm_get_number_of_outputs\n");
        return this->n_outputs;
    }

    uint64_t get_number_of_synapses() {
        __TRACE__("andm_get_number_of_synapses\n");
        return this->n_synapses;
    }

    uint64_t get_weights_dimension() {
        __TRACE__("andm_get_weights_dimension\n");
        if(this->n_outputs == 0) {
            return 0;
        }

        GlobalConnectionsMeta* gc_meta = reinterpret_cast<GlobalConnectionsMeta *>(only_host_allocator.data + global_connections_meta_id);
        return N_WEIGHTS(gc_meta, true);
    }

    uint64_t get_number_of_detectors() {
        __TRACE__("andm_get_number_of_detectors\n");
        return this->n_detectors;
    }

    uint64_t get_max_inputs_per_detector() {
        __TRACE__("andm_get_max_inputs_per_detector\n");
        return this->max_inputs_per_detector;
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
                this->input_neuron_synapses_infos_id, this->n_inputs, 0,
                this->output_neuron_synapses_infos_id, this->n_outputs, this->n_inputs,
                this->n_synapse_metas
            );
        }
        if(weights_allocator != nullptr) {
            weights_allocator->to_device(device);
        }
        if(runtime_context != nullptr) {
            delete runtime_context;
            runtime_context = nullptr;
        }
    }

    void add_connections(
        const torch::Tensor &connections_buffer,
        uint32_t single_input_group_size,
        int ids_shift,
        std::optional<uint32_t> &random_seed
    ) {
        checkConnectionsManagerIsInitialized();
        if(this->n_outputs == 0) {
            throw py::value_error("can't add connections in only detectors mode");
        }
        std::optional<const torch::Tensor> none;
        this->n_synapses += connections_manager->add_connections(
            connections_buffer, none,
            single_input_group_size,
            ids_shift,
            weights_allocator,
            random_seed ? random_seed.value() : std::random_device{}()
        );
    }

    void compile(
        bool only_trainable_backwards,
        torch::Tensor &weights,
        std::optional<uint32_t> &random_seed
    ) {
        if(this->n_outputs == 0) {
            throw py::value_error("nothing to compile in only detectors mode");
        } else {
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
    }

    void forward(
        const torch::Tensor &weights,
        uint32_t batch_size,
        const torch::Tensor &input,
        torch::Tensor &target_output,
        torch::Tensor &target_input_winner_ids,
        torch::Tensor &target_input_prewinner_ids,
        torch::Tensor &target_input_winning_stat
    ) {
        __TRACE__("andm_forward\n");
        checkTensor(weights, "weights", true, host_device_allocator.device);
        checkTensor(input, "input", true, host_device_allocator.device);
        if(this->n_outputs > 0) {
            checkTensor(target_output, "target_output", true, host_device_allocator.device);
        } else if(this->n_detectors == 0) {
            throw py::value_error("n_detectors should be > 0 in only detectors mode");
        }
        checkTensor(target_input_winner_ids, "target_input_winner_ids", false, host_device_allocator.device, sizeof(int32_t));
        checkTensor(target_input_prewinner_ids, "target_input_prewinner_ids", false, host_device_allocator.device, sizeof(int32_t));
        checkTensor(target_input_winning_stat, "target_input_winning_stat", false, host_device_allocator.device, sizeof(int32_t));
        if(batch_size == 0) {
            throw py::value_error("batch_size == 0");
        }
        if((this->n_outputs == 0) && (weights.numel() > 0)) {
            throw py::value_error("weights should be empty tensor in only detectors mode");
        }

        if(this->runtime_context == nullptr) {
            GlobalConnectionsMeta* gc_meta = reinterpret_cast<GlobalConnectionsMeta *>(only_host_allocator.data + global_connections_meta_id);
            this->runtime_context = new ANDN_RUNTIME_CONTEXT_CLASS(
                host_device_allocator.data,
                host_device_allocator.device,
                this->n_inputs,
                this->n_outputs,
                this->n_detectors,
                this->max_inputs_per_detector,
                this->forward_group_size,
                this->backward_group_size,
                this->spiking_inhibition,
                gc_meta->max_forward_groups_per_neuron,
                gc_meta->max_backward_groups_per_neuron,
                #ifdef INTEGERS_INSTEAD_OF_FLOATS
                N_WEIGHTS(gc_meta, true),
                int_rescaler,
                #endif
                #ifdef ENABLE_PROFILING
                this->profiler,
                #endif
                this->base_synapse_metas_id == 0 ? nullptr : BaseSynapseMetas(this->base_synapse_metas_id, host_device_allocator.data),
                IndexedSynapsesInfos(this->input_neuron_synapses_infos_id, host_device_allocator.data),
                reinterpret_cast<int32_t *>(host_device_allocator.data + this->detectors_id),
                this->output_neuron_synapses_infos_id == 0 ? nullptr : IndexedSynapsesInfos(this->output_neuron_synapses_infos_id, host_device_allocator.data),
                gc_meta->first_synapse_id
            );
        }

        this->runtime_context->forward(
            reinterpret_cast<EXTERNAL_REAL_DT *>(weights.data_ptr()),
            batch_size,
            reinterpret_cast<EXTERNAL_REAL_DT *>(input.data_ptr()),
            reinterpret_cast<int32_t *>(target_input_winner_ids.data_ptr()),
            reinterpret_cast<int32_t *>(target_input_prewinner_ids.data_ptr()),
            reinterpret_cast<int32_t *>(target_input_winning_stat.data_ptr()),
            reinterpret_cast<EXTERNAL_REAL_DT *>(target_output.data_ptr())
        );
    }

    void backward_backprop(
        const torch::Tensor &weights,
        uint32_t batch_size,
        const torch::Tensor &output_gradients,
        const torch::Tensor &input,
        const torch::Tensor &input_winner_ids,
        const torch::Tensor &input_prewinner_ids,
        const torch::Tensor &input_winning_stat,
        torch::Tensor &target_input_gradients,
        torch::Tensor &target_weights_gradients
    ) {
        if(batch_size == 0) {
            throw py::value_error("batch_size == 0");
        }
        if(this->n_outputs == 0) {
            if(output_gradients.numel() == 0) {
                throw py::value_error("output_gradients.numel() should be > 0 in only detectors mode");
            }
            if(weights.numel() > 0) {
                throw py::value_error("weights should be empty tensor in only detectors mode");
            }
            if(target_weights_gradients.numel() > 0) {
                throw py::value_error("target_weights_gradients should be empty tensor in only detectors mode");
            }
        }
        if(this->runtime_context == nullptr) {
            throw py::value_error("no active context");
        }
        checkTensor(weights, "weights", true, host_device_allocator.device);
        checkTensor(target_weights_gradients, "target_weights_gradients", true, host_device_allocator.device);
        checkTensor(target_input_gradients, "target_input_gradients", true, host_device_allocator.device);
        checkTensor(output_gradients, "output_gradients", true, host_device_allocator.device);
        checkTensor(input, "input", true, host_device_allocator.device);
        checkTensor(input_winner_ids, "input_winner_ids", false, host_device_allocator.device, sizeof(int32_t));
        checkTensor(input_prewinner_ids, "input_prewinner_ids", false, host_device_allocator.device, sizeof(int32_t));
        checkTensor(input_winning_stat, "input_winning_stat", false, host_device_allocator.device, sizeof(int32_t));

        this->runtime_context->backward_backprop(
            reinterpret_cast<EXTERNAL_REAL_DT *>(weights.data_ptr()),
            batch_size,
            reinterpret_cast<EXTERNAL_REAL_DT *>(output_gradients.data_ptr()),
            reinterpret_cast<EXTERNAL_REAL_DT *>(input.data_ptr()),
            reinterpret_cast<int32_t *>(input_winner_ids.data_ptr()),
            reinterpret_cast<int32_t *>(input_prewinner_ids.data_ptr()),
            reinterpret_cast<int32_t *>(input_winning_stat.data_ptr()),
            reinterpret_cast<EXTERNAL_REAL_DT *>(target_input_gradients.data_ptr()),
            reinterpret_cast<EXTERNAL_REAL_DT *>(target_weights_gradients.data_ptr())
        );
    }

    void backward_hebb(
        const torch::Tensor &weights,
        uint32_t batch_size,
        double anti_hebb_coeff,
        const torch::Tensor &input,
        const torch::Tensor &input_winner_ids,
        const torch::Tensor &input_prewinner_ids,
        const torch::Tensor &input_winning_stat,
        const torch::Tensor &output,
        const torch::Tensor &output_winner_ids,
        const torch::Tensor &output_prewinner_ids,
        const torch::Tensor &output_winning_stat,
        bool spiking_output_inhibition,
        torch::Tensor &target_weights_gradients
    ) {
        if(this->n_outputs == 0) {
            throw py::value_error("backward_hebb is not allowed in only detectors mode");
        }
        if(batch_size == 0) {
            throw py::value_error("batch_size == 0");
        }
        if(this->runtime_context == nullptr) {
            throw py::value_error("no active context");
        }

        checkTensor(weights, "weights", true, host_device_allocator.device);
        checkTensor(target_weights_gradients, "target_weights_gradients", true, host_device_allocator.device);
        checkTensor(input, "input", true, host_device_allocator.device);
        checkTensor(input_winner_ids, "input_winner_ids", false, host_device_allocator.device, sizeof(int32_t));
        checkTensor(input_prewinner_ids, "input_prewinner_ids", false, host_device_allocator.device, sizeof(int32_t));
        checkTensor(input_winning_stat, "input_winning_stat", false, host_device_allocator.device, sizeof(int32_t));
        checkTensor(output, "output", true, host_device_allocator.device);
        checkTensor(output_winner_ids, "output_winner_ids", false, host_device_allocator.device, sizeof(int32_t));
        checkTensor(output_prewinner_ids, "output_prewinner_ids", false, host_device_allocator.device, sizeof(int32_t));
        checkTensor(output_winning_stat, "output_winning_stat", false, host_device_allocator.device, sizeof(int32_t));

        this->runtime_context->backward_hebb(
            reinterpret_cast<EXTERNAL_REAL_DT *>(weights.data_ptr()),
            batch_size,
            anti_hebb_coeff,
            reinterpret_cast<EXTERNAL_REAL_DT *>(input.data_ptr()),
            reinterpret_cast<int32_t *>(input_winner_ids.data_ptr()),
            reinterpret_cast<int32_t *>(input_prewinner_ids.data_ptr()),
            reinterpret_cast<int32_t *>(input_winning_stat.data_ptr()),
            reinterpret_cast<EXTERNAL_REAL_DT *>(output.data_ptr()),
            reinterpret_cast<int32_t *>(output_winner_ids.data_ptr()),
            reinterpret_cast<int32_t *>(output_prewinner_ids.data_ptr()),
            reinterpret_cast<int32_t *>(output_winning_stat.data_ptr()),
            output_winner_ids.numel() / batch_size,
            spiking_output_inhibition,
            reinterpret_cast<EXTERNAL_REAL_DT *>(target_weights_gradients.data_ptr())
        );
    }

    uint64_t count_synapses(
        const torch::Tensor &neuron_indices_to_process,
        bool forward_or_backward
    ) {
        if(this->n_outputs == 0) {
            return 0;
        }

        checkConnectionsManagerIsInitialized();
        return connections_manager->count_synapses(
            neuron_indices_to_process,
            forward_or_backward
        );
    }

    void export_synapses(
        const torch::Tensor &weights,
        const torch::Tensor &neuron_indices_to_process,
        torch::Tensor &target_internal_source_indices,
        torch::Tensor &target_weights,
        torch::Tensor &target_internal_target_indices,
        bool forward_or_backward,
        std::optional<torch::Tensor> &target_synapse_meta_indices
    ) {
        if(this->n_outputs == 0) {
            throw py::value_error("nothing to export in only detectors mode");
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
            forward_or_backward,
            none,
            target_synapse_meta_indices,
            weights_data
        );
    }

    auto __repr__() {
        std::ostringstream os;
        GlobalConnectionsMeta* gc_meta = reinterpret_cast<GlobalConnectionsMeta *>(only_host_allocator.data + global_connections_meta_id);
        os << "ANDNDataManager(host_device: " <<
            host_device_allocator.allocated <<
            ", " << host_device_allocator.used <<
            "; host_only: " <<
            only_host_allocator.allocated <<
            ", " << only_host_allocator.used <<
            "; summation type: " << SUMMATION_DT_STR <<
            "; smallest distinguishable fraction: " << get_smallest_distinguishable_fraction() <<
            "; n_synapses: " << n_synapses <<
            "; n_detectors: " << n_detectors <<
            "; first_synapse_id: " << gc_meta->first_synapse_id <<
            "; last_synapse_id: " << gc_meta->last_synapse_id <<
            "; n_forward_groups: " << gc_meta->n_forward_groups <<
            "; n_backward_groups: " << gc_meta->n_backward_groups <<
        ")";
        return os.str();
    }

    auto get_memory_stats() {
        std::ostringstream os;
        for(int i=0; i < N_ANDN_MEMORY_LABELS;i++) {
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

    ~ANDM_CLASS_NAME() {
        if(connections_manager != nullptr) {
            delete connections_manager;
        }
        if(runtime_context != nullptr) {
            delete runtime_context;
        }
    }

    void checkOnHostDuringPrepare()
    {
        if(host_device_allocator.device != -1) {
            throw std::runtime_error("ANDNDataManager must be in the host memory at this point");
        }
    }

    void checkConnectionsManagerIsInitialized()
    {
        if(connections_manager == nullptr) {
            throw py::value_error("connections_manager is not initialized, use initialize_neurons(...) method");
        }
    }

private:
    ANDM_CLASS_NAME(
        uint8_t* host_device_data, NeuronDataId_t host_device_used,
        uint8_t* only_host_data, NeuronDataId_t only_host_used,
        uint32_t n_inputs,
        uint32_t n_outputs,
        uint32_t forward_group_size,
        uint32_t backward_group_size,
        bool spiking_inhibition,
        NeuronDataId_t base_synapse_metas_id,
        uint32_t n_synapse_metas,
        uint64_t n_synapses,
        NeuronDataId_t input_neuron_synapses_infos_id,
        NeuronDataId_t output_neuron_synapses_infos_id,
        NeuronDataId_t detectors_id,
        uint32_t n_detectors,
        uint32_t max_inputs_per_detector,
        NeuronDataId_t global_connections_meta_id
        #ifdef INTEGERS_INSTEAD_OF_FLOATS
        , double int_rescaler
        #endif
    ) :
        #ifdef ENABLE_PROFILING
        profiler(N_ANDN_PROFILER_OPS),
        #endif
        host_device_allocator(host_device_data, host_device_used),
        only_host_allocator(only_host_data, only_host_used),
        runtime_context(nullptr),
        weights_allocator(nullptr),
        n_inputs(n_inputs),
        n_outputs(n_outputs),
        forward_group_size(forward_group_size),
        backward_group_size(backward_group_size),
        spiking_inhibition(spiking_inhibition),
        base_synapse_metas_id(base_synapse_metas_id),
        n_synapse_metas(n_synapse_metas),
        n_synapses(n_synapses),
        input_neuron_synapses_infos_id(input_neuron_synapses_infos_id),
        output_neuron_synapses_infos_id(output_neuron_synapses_infos_id),
        detectors_id(detectors_id),
        n_detectors(n_detectors),
        max_inputs_per_detector(max_inputs_per_detector),
        global_connections_meta_id(global_connections_meta_id)
    {
        if(this->n_outputs > 0) {
            connections_manager = new ConnectionsManager(
                #ifdef ENABLE_PROFILING
                profiler,
                #endif
                host_device_allocator, only_host_allocator, true,
                this->base_synapse_metas_id, this->global_connections_meta_id,
                this->input_neuron_synapses_infos_id, this->n_inputs, 0,
                this->output_neuron_synapses_infos_id, this->n_outputs, this->n_inputs,
                this->n_synapse_metas
            );
        } else {
            connections_manager = nullptr;
        }
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
    ANDN_RUNTIME_CONTEXT_CLASS *runtime_context;
    SimpleAllocator* weights_allocator;

    uint32_t n_inputs;
    uint32_t n_outputs;
    uint32_t forward_group_size;
    uint32_t backward_group_size;
    bool spiking_inhibition;
    NeuronDataId_t base_synapse_metas_id;
    uint32_t n_synapse_metas;
    uint64_t n_synapses;
    NeuronDataId_t input_neuron_synapses_infos_id;
    NeuronDataId_t output_neuron_synapses_infos_id;
    NeuronDataId_t detectors_id;
    uint32_t n_detectors;
    uint32_t max_inputs_per_detector;
    NeuronDataId_t global_connections_meta_id;
    #ifdef INTEGERS_INSTEAD_OF_FLOATS
    double int_rescaler;
    #endif
};

py::tuple PFX(pickle_andn_neuron_manager)(const ANDM_CLASS_NAME& ndm) {
    if(ndm.host_device_allocator.device != -1) {
        throw std::runtime_error("net must be on CPU before serialization");
    }
    return py::make_tuple(
        py::reinterpret_steal<py::object>(
            PyBytes_FromStringAndSize((char *) ndm.host_device_allocator.data, ndm.host_device_allocator.used)
        ),
        py::reinterpret_steal<py::object>(
            PyBytes_FromStringAndSize((char *) ndm.only_host_allocator.data, ndm.only_host_allocator.used)
        ),
        ndm.n_inputs,
        ndm.n_outputs,
        ndm.forward_group_size,
        ndm.backward_group_size,
        ndm.spiking_inhibition,
        ndm.base_synapse_metas_id,
        ndm.n_synapse_metas,
        ndm.n_synapses,
        ndm.input_neuron_synapses_infos_id,
        ndm.output_neuron_synapses_infos_id,
        ndm.detectors_id,
        ndm.n_detectors,
        ndm.max_inputs_per_detector,
        ndm.global_connections_meta_id
        #ifdef INTEGERS_INSTEAD_OF_FLOATS
        , ndm.int_rescaler
        #endif
    );
}

std::unique_ptr<ANDM_CLASS_NAME> PFX(unpickle_andn_neuron_manager)(py::tuple t) {
    char *buf;
    Py_ssize_t host_device_used;
    PyBytes_AsStringAndSize(t[0].ptr(), &buf, &host_device_used);
    uint8_t *host_device_data = (uint8_t *) PyMem_Malloc(host_device_used);
    memcpy(host_device_data, buf, host_device_used);

    Py_ssize_t only_host_used;
    PyBytes_AsStringAndSize(t[1].ptr(), &buf, &only_host_used);
    uint8_t *only_host_data = (uint8_t *) PyMem_Malloc(only_host_used);
    memcpy(only_host_data, buf, only_host_used);

    return std::unique_ptr<ANDM_CLASS_NAME>(
        new ANDM_CLASS_NAME(
            host_device_data, (NeuronDataId_t) host_device_used,
            only_host_data, (NeuronDataId_t) only_host_used,
            t[2].cast<uint32_t>(),         // n_inputs
            t[3].cast<uint32_t>(),         // n_outputs
            t[4].cast<uint32_t>(),         // forward_group_size
            t[5].cast<uint32_t>(),         // backward_group_size
            t[6].cast<bool>(),             // spiking_inhibition
            t[7].cast<NeuronDataId_t>(),   // base_synapse_metas_id
            t[8].cast<uint32_t>(),         // n_synapse_metas
            t[9].cast<uint64_t>(),         // n_synapses
            t[10].cast<NeuronDataId_t>(),  // input_neuron_synapses_infos_id
            t[11].cast<NeuronDataId_t>(),  // output_neuron_synapses_infos_id
            t[12].cast<NeuronDataId_t>(),  // detectors_id
            t[13].cast<uint32_t>(),        // n_detectors
            t[14].cast<uint32_t>(),        // max_inputs_per_detector
            t[15].cast<NeuronDataId_t>()   // global_connections_meta_id
            #ifdef INTEGERS_INSTEAD_OF_FLOATS
            , t[16].cast<double>()         // int_rescaler
            #endif
        )
    );
}


void PFX(PB_ANDNDataManager)(py::module& m) {
    #ifdef INTEGERS_INSTEAD_OF_FLOATS
    py::class_<ANDNDataManagerI>(m, "ANDNDataManagerI")
        .def(py::init<uint32_t, uint32_t, uint64_t, uint32_t, uint32_t, bool, double>(),
            py::arg("n_inputs"),
            py::arg("n_outputs"),
            py::arg("initial_synapse_capacity"),
            py::arg("forward_group_size"),
            py::arg("backward_group_size"),
            py::arg("spiking_inhibition"),
            py::arg("int_rescaler"))
    #else
    py::class_<ANDNDataManagerF>(m, "ANDNDataManagerF")
        .def(py::init<uint32_t, uint32_t, uint64_t, uint32_t, uint32_t, bool>(),
            py::arg("n_inputs"),
            py::arg("n_outputs"),
            py::arg("initial_synapse_capacity"),
            py::arg("forward_group_size"),
            py::arg("backward_group_size"),
            py::arg("spiking_inhibition"))
    #endif
        .def("get_smallest_distinguishable_fraction", &ANDM_CLASS_NAME::get_smallest_distinguishable_fraction,
            "Returns smallest fraction that can exist inside data manager in integers mode. Returns 0.0 in floats mode.")
        .def("get_epsilon", &ANDM_CLASS_NAME::get_epsilon,
            "Returns super small real number used inside the engine")
        .def("get_summations_data_type", &ANDM_CLASS_NAME::get_summations_data_type,
            "Returns string 'float32' in floats mode and string 'int32' in integers mode")
        .def("register_synapse_meta", &ANDM_CLASS_NAME::register_synapse_meta,
            "Register synapse meta",
            py::arg("learning_rate"),
            py::arg("min_synaptic_weight"),
            py::arg("max_synaptic_weight"),
            py::arg("initial_noise_level"),
            py::arg("initial_weight"))
        .def("initialize_neurons", &ANDM_CLASS_NAME::initialize_neurons,
            "Initialize neurons")
        .def("initialize_detectors", &ANDM_CLASS_NAME::initialize_detectors,
            "Initialize detectors",
            py::arg("detectors_data"),
            py::arg("max_inputs_per_detector"))
        .def("get_number_of_inputs", &ANDM_CLASS_NAME::get_number_of_inputs,
            "Get number of input neurons")
        .def("get_number_of_outputs", &ANDM_CLASS_NAME::get_number_of_outputs,
            "Get number of output neurons")
        .def("get_number_of_synapses", &ANDM_CLASS_NAME::get_number_of_synapses,
            "Get number of synapses")
        .def("get_weights_dimension", &ANDM_CLASS_NAME::get_weights_dimension,
            "Get the length of the weights array (it's greater than number of synapses because of small holes)")
        .def("get_number_of_detectors", &ANDM_CLASS_NAME::get_number_of_detectors,
            "Get number of detectors")
        .def("get_max_inputs_per_detector", &ANDM_CLASS_NAME::get_max_inputs_per_detector,
            "Get max inputs per detector")
        .def("to_device", &ANDM_CLASS_NAME::to_device,
            "Move to device",
            py::arg("device"))
        .def("add_connections", &ANDM_CLASS_NAME::add_connections,
            "Add synapses",
            py::arg("connections_buffer"),
            py::arg("single_input_group_size"),
            py::arg("ids_shift"),
            py::arg("random_seed") = py::none())
        .def("compile", &ANDM_CLASS_NAME::compile,
            "compile the network",
            py::arg("only_trainable_backwards"),
            py::arg("weights"),
            py::arg("random_seed") = py::none())
        .def("forward", &ANDM_CLASS_NAME::forward,
            "Forward pass",
            py::arg("weights"),
            py::arg("batch_size"),
            py::arg("input"),
            py::arg("target_output"),
            py::arg("target_input_winner_ids"),
            py::arg("target_input_prewinner_ids"),
            py::arg("target_input_winning_stat"))
        .def("backward_backprop", &ANDM_CLASS_NAME::backward_backprop,
            "Gradients back propagation",
            py::arg("weights"),
            py::arg("batch_size"),
            py::arg("output_gradients"),
            py::arg("input"),
            py::arg("input_winner_ids"),
            py::arg("input_prewinner_ids"),
            py::arg("input_winning_stat"),
            py::arg("target_input_gradients"),
            py::arg("target_weights_gradients"))
        .def("backward_hebb", &ANDM_CLASS_NAME::backward_hebb,
            "Hebbian quasi gradients",
            py::arg("weights"),
            py::arg("batch_size"),
            py::arg("anti_hebb_coeff"),
            py::arg("input"),
            py::arg("input_winner_ids"),
            py::arg("input_prewinner_ids"),
            py::arg("input_winning_stat"),
            py::arg("output"),
            py::arg("output_winner_ids"),
            py::arg("output_prewinner_ids"),
            py::arg("output_winning_stat"),
            py::arg("spiking_output_inhibition"),
            py::arg("target_weights_gradients"))
        .def("count_synapses", &ANDM_CLASS_NAME::count_synapses,
            "Count forward or backward synapses for the set of neurons",
            py::arg("neuron_indices_to_process"),
            py::arg("forward_or_backward"))
        .def("export_synapses", &ANDM_CLASS_NAME::export_synapses,
            "Export all synapses for a set of neurons",
            py::arg("weights"),
            py::arg("neuron_indices_to_process"),
            py::arg("target_internal_source_indices"),
            py::arg("target_weights"),
            py::arg("target_internal_target_indices"),
            py::arg("forward_or_backward"),
            py::arg("target_synapse_meta_indices") = py::none())
        .def("__repr__", &ANDM_CLASS_NAME::__repr__)
        .def("get_memory_stats", &ANDM_CLASS_NAME::get_memory_stats)
        .def("get_profiling_stats", &ANDM_CLASS_NAME::get_profiling_stats)
        .def("reset_profiler", &ANDM_CLASS_NAME::reset_profiler)
        .def(py::pickle(
            &PFX(pickle_andn_neuron_manager),
            &PFX(unpickle_andn_neuron_manager)
        ));
}

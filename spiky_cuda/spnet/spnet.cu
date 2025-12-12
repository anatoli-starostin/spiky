#include "spnet_runtime.cuh"
#include <limits.h>
#include <random>

namespace py = pybind11;

#define SPNDM_CLASS_NAME PFX(SPNetDataManager)

class __attribute__((visibility("hidden"))) SPNDM_CLASS_NAME {
    friend py::tuple PFX(pickle_spnet_neuron_manager)(const SPNDM_CLASS_NAME& ndm);
    friend std::unique_ptr<SPNDM_CLASS_NAME> PFX(unpickle_spnet_neuron_manager)(py::tuple t);
public:
    SPNDM_CLASS_NAME(uint64_t initial_synapse_capacity, REAL_DT stdp_threshold) :
        #ifdef ENABLE_PROFILING
        profiler(N_SPNET_PROFILER_OPS),
        #endif
        host_device_allocator(initial_synapse_capacity * 2 * sizeof(SynapseInfo) + MAX_N_SYNAPSE_METAS * (sizeof(BaseSynapseMeta) + sizeof(SPNetSynapseMeta))),
        only_host_allocator(1024),
        connections_manager(nullptr), runtime_context(nullptr),
        stdp_threshold(stdp_threshold)
    {
        this->base_synapse_metas_id = host_device_allocator.allocate(MAX_N_SYNAPSE_METAS * sizeof(BaseSynapseMeta), SYNAPSE_METAS_MEMORY_LABEL);
        this->spnet_synapse_metas_id = host_device_allocator.allocate(MAX_N_SYNAPSE_METAS * sizeof(SPNetSynapseMeta), SYNAPSE_METAS_MEMORY_LABEL);
        this->global_connections_meta_id = only_host_allocator.allocate(sizeof(GlobalConnectionsMeta), 0);
        GlobalConnectionsMeta* gc_meta = reinterpret_cast<GlobalConnectionsMeta *>(only_host_allocator.data + global_connections_meta_id);
        memset(gc_meta, 0, sizeof(GlobalConnectionsMeta));
        this->neuron_meta_host_infos_id = 0;
        this->n_synapse_metas = 0;
        this->n_synapses = 0;
        this->max_delay = 0;
        this->neuron_metas_id = host_device_allocator.allocate(MAX_NEURON_METAS * sizeof(NeuronMeta), NEURON_METAS_MEMORY_LABEL);
        this->n_neuron_metas = 0;
        this->forward_neuron_infos_id = 0;
        this->backward_neuron_infos_id = 0;
        this->neurons_to_ltd_table_shift_id = 0;
        this->stdp_tables_id = 0;
        this->n_neurons = 0;

        if(stdp_threshold < 0.0) {
            throw py::value_error("stdp_threshold < 0.0");
        }
        if(stdp_threshold > 0.1) {
            throw py::value_error("stdp_threshold > 0.1");
        }

        #ifdef ENABLE_PROFILING
        profiler.register_operation_type(SPNET_RUNTIME_PROCESS_TICK_PROFILER_OP, "spnet::runtime::process_tick");
        profiler.register_operation_type(SPNET_RUNTIME_APPLY_INPUT_PROFILER_OP, "spnet::runtime::apply_input");
        profiler.register_operation_type(SPNET_RUNTIME_DETECT_SPIKES_PROFILER_OP, "spnet::runtime::detect_spikes");
        profiler.register_operation_type(SPNET_RUNTIME_FIRE_SPIKES_PROFILER_OP, "spnet::runtime::fire_spikes");
        profiler.register_operation_type(SPNET_RUNTIME_EULER_STEPS_PROFILER_OP, "spnet::runtime::euler_steps");
        profiler.register_operation_type(SPNET_RUNTIME_CALCULATE_LTP_PROFILER_OP, "spnet::runtime::calculate_ltp");
        profiler.register_operation_type(SPNET_RUNTIME_APPLY_WEIGHT_DELTAS_PROFILER_OP, "spnet::runtime::apply_weight_deltas");
        #endif
    }

    REAL_DT get_smallest_distinguishable_fraction()
    {
        #ifdef INTEGERS_INSTEAD_OF_FLOATS
        return 1.0 / DENOMINATOR32;
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
        uint32_t min_delay, uint32_t max_delay,
        REAL_DT min_synaptic_weight, REAL_DT max_synaptic_weight,
        REAL_DT initial_noise_level, REAL_DT initial_weight,
        REAL_DT weight_decay, REAL_DT weight_scaling_cf,
        uint32_t _forward_group_size, uint32_t _backward_group_size
    ) {
        __TRACE__("spndm_register_synapse_meta\n");
        checkOnHostDuringPrepare();
        if(this->n_neurons > 0) {
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
        if((learning_rate > 0.0) && (learning_rate < denominator_reciproc)) {
            std::ostringstream os;
            os << "too small value for 'learning_rate' " << learning_rate << " (less than smallest distinguishable fraction " << denominator_reciproc << ")";
            throw py::value_error(os.str());
        }
        #endif
        if(min_delay > MAX_DELAY) {
            std::ostringstream os;
            os << "min_delay > " << MAX_DELAY;
            throw py::value_error(os.str());
        }
        if(max_delay > MAX_DELAY) {
            std::ostringstream os;
            os << "max_delay > " << MAX_DELAY;
            throw py::value_error(os.str());
        }
        if(min_delay > max_delay) {
            throw py::value_error("min_delay > max_delay");
        }
        if(min_synaptic_weight > max_synaptic_weight) {
            throw py::value_error("min_synaptic_weight > max_synaptic_weight");
        }
        if(initial_weight < min_synaptic_weight) {
            throw py::value_error("initial_weight < min_synaptic_weight");
        }
        if(initial_weight > max_synaptic_weight) {
            throw py::value_error("initial_weight > max_synaptic_weight");
        }
        if(weight_decay <= 0.0) {
            throw py::value_error("weight_decay <= 0.0");
        }
        if(weight_decay > 1.0) {
            throw py::value_error("weight_decay > 1.0");
        }
        if(weight_scaling_cf < 0.0) {
            throw py::value_error("weight_scaling_cf < 0.0");
        }
        if(weight_scaling_cf > 1.0) {
            throw py::value_error("weight_scaling_cf > 1.0");
        }
        if(_forward_group_size == 0) {
            throw py::value_error("_forward_group_size == 0");
        }
        if(_forward_group_size > MAX_SYNAPSE_GROUP_SIZE) {
            throw py::value_error("_forward_group_size > MAX_SYNAPSE_GROUP_SIZE");
        }
        if(_backward_group_size > MAX_SYNAPSE_GROUP_SIZE) {
            throw py::value_error("_backward_group_size > MAX_SYNAPSE_GROUP_SIZE");
        }
        BaseSynapseMeta target_base_meta = {
            learning_rate,
            min_delay, max_delay,
            min_synaptic_weight, max_synaptic_weight,
            initial_noise_level, initial_weight,
            _forward_group_size, _backward_group_size
        };
        SPNetSynapseMeta target_spnet_meta = {
            weight_decay, weight_scaling_cf
        };
        BaseSynapseMeta *current_base_synapse_meta = BaseSynapseMetas(this->base_synapse_metas_id, host_device_allocator.data);
        SPNetSynapseMeta *current_spnet_synapse_meta = SPNetSynapseMetas(this->spnet_synapse_metas_id, host_device_allocator.data);
        uint32_t i=0;
        for(;i < this->n_synapse_metas;i++, current_base_synapse_meta++, current_spnet_synapse_meta++) {
            if(!memcmp(&target_base_meta, current_base_synapse_meta, sizeof(BaseSynapseMeta)) && !memcmp(&target_spnet_meta, current_spnet_synapse_meta, sizeof(SPNetSynapseMeta))) {
                return i;
            }
        }
        if(this->n_synapse_metas == MAX_N_SYNAPSE_METAS) {
            throw py::value_error("too many different synapse metas");
        }
        this->n_synapse_metas++;
        memcpy(current_base_synapse_meta, &target_base_meta, sizeof(BaseSynapseMeta));
        memcpy(current_spnet_synapse_meta, &target_spnet_meta, sizeof(SPNetSynapseMeta));
        if(max_delay > this->max_delay) {
            this->max_delay = max_delay;
        }
        return i;
    }

    uint32_t _count_stdp_horizon(REAL_DT start_val, REAL_DT decay) {
        uint32_t res = static_cast<uint32_t>(std::floor(std::log(this->stdp_threshold / start_val) / std::log(decay))) + 1;
        if(res > MAX_STDP_HORIZON) {
            res = MAX_STDP_HORIZON;
        }
        return res;
    }

    uint32_t register_neuron_meta(
        uint32_t neuron_type,
        REAL_DT cf_2,
        REAL_DT cf_1,
        REAL_DT cf_0,
        REAL_DT a,
        REAL_DT b,
        REAL_DT c,
        REAL_DT d,
        REAL_DT spike_threshold,
        REAL_DT stdp_decay,
        REAL_DT ltp_max,
        REAL_DT ltd_max
    ) {
        __TRACE__("spndm_register_neuron_meta\n");
        checkOnHostDuringPrepare();
        if(this->n_neurons > 0) {
            throw py::value_error("can't register new neuron metas after neurons were initialized");
        }
        if(stdp_decay < 0.0) {
            throw py::value_error("stdp_decay < 0.0");
        }
        if(stdp_decay > 1.0) {
            throw py::value_error("stdp_decay > 1.0");
        }
        if(ltp_max < 0.0) {
            throw py::value_error("ltp_max < 0.0");
        }
        if(ltd_max < 0.0) {
            throw py::value_error("ltd_max < 0.0");
        }

        NeuronMeta target_meta = {
            neuron_type, cf_2, cf_1, cf_0, a, b, c, d,
            spike_threshold, stdp_decay, ltp_max, ltd_max
        };
        NeuronMeta *current_neuron_meta = NeuronMetas(this->neuron_metas_id, host_device_allocator.data);
        uint32_t i=0;
        for(;i < this->n_neuron_metas;i++, current_neuron_meta++) {
            if(!memcmp(&target_meta, current_neuron_meta, sizeof(NeuronMeta))) {
                return i;
            }
        }
        if(this->n_neuron_metas == MAX_NEURON_METAS) {
            throw py::value_error("too many different neuron metas");
        }
        this->n_neuron_metas++;
        memcpy(current_neuron_meta, &target_meta, sizeof(NeuronMeta));
        return i;
    }

    uint32_t initialize_neurons(
        const torch::Tensor &neuron_counts_by_meta_id
    ) {
        __TRACE__("spndm_initialize_neurons\n");
        checkOnHostDuringPrepare();
        checkTensor(neuron_counts_by_meta_id, "neuron_counts_by_meta_id", false, host_device_allocator.device, sizeof(uint32_t));

        // prepare stdp tables

        uint32_t* nm_to_ltp = (uint32_t *) PyMem_Malloc(this->n_synapse_metas * sizeof(uint32_t));
        uint32_t* nm_to_ltd = (uint32_t *) PyMem_Malloc(this->n_synapse_metas * sizeof(uint32_t));

        uint32_t stdp_capacity = 0;

        NeuronMeta* neuron_metas = NeuronMetas(this->neuron_metas_id, host_device_allocator.data);
        for(uint32_t i=0;i < this->n_neuron_metas;i++) {
            NeuronMeta current_nm = neuron_metas[i];

            uint32_t j=0;
            for(;j < i;j++) {
                NeuronMeta prev_nm = neuron_metas[j];

                if(prev_nm.stdp_decay == current_nm.stdp_decay) {
                    if(current_nm.ltp_max == prev_nm.ltp_max) {
                        nm_to_ltp[i] = nm_to_ltp[j];
                        break;
                    }

                    if(current_nm.ltp_max == prev_nm.ltd_max) {
                        nm_to_ltp[i] = nm_to_ltd[j];
                        break;
                    }
                }
            }

            if(j == i) {
                uint32_t ltp_horizon = _count_stdp_horizon(current_nm.ltp_max, current_nm.stdp_decay);
                nm_to_ltp[i] = stdp_capacity;
                stdp_capacity += sizeof(STDPTable) + ltp_horizon * sizeof(SUMMATION32_DT);
            }

            if(current_nm.ltd_max == current_nm.ltp_max) {
                nm_to_ltd[i] = nm_to_ltp[i];
                continue;
            }

            j=0;
            for(;j < i;j++) {
                NeuronMeta prev_nm = neuron_metas[j];

                if(prev_nm.stdp_decay == current_nm.stdp_decay) {
                    if(current_nm.ltd_max == prev_nm.ltd_max) {
                        nm_to_ltd[i] = nm_to_ltd[j];
                        break;
                    }

                    if(current_nm.ltd_max == prev_nm.ltp_max) {
                        nm_to_ltd[i] = nm_to_ltp[j];
                        break;
                    }
                }
            }

            if(j == i) {
                uint32_t ltd_horizon = _count_stdp_horizon(current_nm.ltd_max, current_nm.stdp_decay);
                nm_to_ltd[i] = stdp_capacity;
                stdp_capacity += sizeof(STDPTable) + ltd_horizon * sizeof(SUMMATION32_DT);
            }
        }

        this->stdp_tables_id = host_device_allocator.allocate(stdp_capacity, NEURON_METAS_MEMORY_LABEL);
        memset(host_device_allocator.data + this->stdp_tables_id, 0, stdp_capacity);

        for(uint32_t i=0;i < this->n_neuron_metas;i++) {
            STDPTable *stdp_table_ptr = GetSTDPTable(this->stdp_tables_id, nm_to_ltp[i], host_device_allocator.data);
            if(stdp_table_ptr->n_ticks == 0) {
                NeuronMeta current_nm = neuron_metas[i];
                stdp_table_ptr->n_ticks = _count_stdp_horizon(current_nm.ltp_max, current_nm.stdp_decay);
                SUMMATION32_DT* values = STDPTableValues(stdp_table_ptr);
                double v = current_nm.ltp_max;
                for(uint32_t j=0;j < stdp_table_ptr->n_ticks;j++, v*=current_nm.stdp_decay) {
                    #ifdef INTEGERS_INSTEAD_OF_FLOATS
                    values[j] = static_cast<SUMMATION32_DT>(v * DENOMINATOR32);
                    #else
                    values[j] = static_cast<SUMMATION32_DT>(v);
                    #endif
                }
            }
            stdp_table_ptr = GetSTDPTable(this->stdp_tables_id, nm_to_ltd[i], host_device_allocator.data);
            if(stdp_table_ptr->n_ticks == 0) {
                NeuronMeta current_nm = neuron_metas[i];
                stdp_table_ptr->n_ticks = _count_stdp_horizon(current_nm.ltd_max, current_nm.stdp_decay);
                SUMMATION32_DT* values = STDPTableValues(stdp_table_ptr);
                double v = current_nm.ltd_max;
                for(uint32_t j=0;j < stdp_table_ptr->n_ticks;j++, v*=current_nm.stdp_decay) {
                    #ifdef INTEGERS_INSTEAD_OF_FLOATS
                    values[j] = static_cast<SUMMATION32_DT>(v * DENOMINATOR32);
                    #else
                    values[j] = static_cast<SUMMATION32_DT>(v);
                    #endif
                }
            }
        }

        uint32_t *neuron_counts_by_meta_id_data = reinterpret_cast<uint32_t *>(neuron_counts_by_meta_id.data_ptr());
        this->n_neurons = NEURON_ALIGNMENT_CONSTANT;
        this->neuron_meta_host_infos_id = only_host_allocator.allocate(neuron_counts_by_meta_id.numel() * sizeof(NeuronMetaHostInfo), 0);
        NeuronMetaHostInfo* neuron_meta_host_infos = NeuronMetaHostInfos(this->neuron_meta_host_infos_id, only_host_allocator.data);
        neuron_metas = NeuronMetas(this->neuron_metas_id, host_device_allocator.data);

        for (uint32_t i = 0; i < neuron_counts_by_meta_id.numel(); i++) {
            uint32_t neuron_count = neuron_counts_by_meta_id_data[i];
            if((neuron_count % 4) > 0) {
                throw py::value_error("(neuron_count % 4) > 0");
            }
            STDPTable *stdp_table_ptr = GetSTDPTable(this->stdp_tables_id, nm_to_ltp[i], host_device_allocator.data);
            neuron_meta_host_infos[i] = NeuronMetaHostInfo{
                this->n_neurons, neuron_count, 0, 0, stdp_table_ptr->n_ticks, nm_to_ltp[i], neuron_metas[i]
            };
            this->n_neurons += neuron_count;
        }
        __TRACE__("spndm_initialize_neurons: n_neurons %d\n", this->n_neurons);

        this->forward_neuron_infos_id = host_device_allocator.allocate(sizeof(IndexedSynapsesInfo) * this->n_neurons, NEURON_INFOS_MEMORY_LABEL);
        this->backward_neuron_infos_id = host_device_allocator.allocate(sizeof(IndexedSynapsesInfo) * this->n_neurons, NEURON_INFOS_MEMORY_LABEL);
        this->neurons_to_ltd_table_shift_id = host_device_allocator.allocate(sizeof(uint32_t) * (this->n_neurons / 4), NEURON_INFOS_MEMORY_LABEL);

        IndexedSynapsesInfo *current_neuron_info = IndexedSynapsesInfos(this->forward_neuron_infos_id, host_device_allocator.data);
        memset(current_neuron_info, 0, sizeof(IndexedSynapsesInfo) * this->n_neurons);
        current_neuron_info = IndexedSynapsesInfos(this->backward_neuron_infos_id, host_device_allocator.data);
        memset(current_neuron_info, 0, sizeof(IndexedSynapsesInfo) * this->n_neurons);
        uint32_t* neurons_to_ltd_table_shifts = (uint32_t *)(host_device_allocator.data + this->neurons_to_ltd_table_shift_id);

        uint32_t current_meta_counter = 0;
        for (
            uint32_t i=0, j=NEURON_ALIGNMENT_CONSTANT;
            (i < neuron_counts_by_meta_id.numel()) && (j < this->n_neurons);
            j++
        ) {
            if((j % 4) == 0) {
                neurons_to_ltd_table_shifts[j / 4] = nm_to_ltd[i];
            }
            current_meta_counter++;
            if(current_meta_counter == neuron_counts_by_meta_id_data[i]) {
                i++;
                current_meta_counter = 0;
            }
        }

        PyMem_Free(nm_to_ltd);
        PyMem_Free(nm_to_ltp);

        connections_manager = new ConnectionsManager(
            #ifdef ENABLE_PROFILING
            profiler,
            #endif
            host_device_allocator, only_host_allocator, false,
            this->base_synapse_metas_id,
            this->global_connections_meta_id,
            this->forward_neuron_infos_id + NEURON_ALIGNMENT_CONSTANT * sizeof(IndexedSynapsesInfo), this->n_neurons - NEURON_ALIGNMENT_CONSTANT, NEURON_ALIGNMENT_CONSTANT,
            this->backward_neuron_infos_id + NEURON_ALIGNMENT_CONSTANT * sizeof(IndexedSynapsesInfo), this->n_neurons - NEURON_ALIGNMENT_CONSTANT, NEURON_ALIGNMENT_CONSTANT,
            this->n_synapse_metas
        );

        return NEURON_ALIGNMENT_CONSTANT;
    }

    uint32_t get_total_number_of_neurons() {
        __TRACE__("get_total_number_of_neurons\n");
        return this->n_neurons;
    }

    uint64_t get_total_number_of_synapses() {
        __TRACE__("get_total_number_of_synapses\n");
        return this->n_synapses;
    }

    uint64_t get_max_delay() {
        __TRACE__("get_max_delay\n");
        return this->max_delay;
    }

    void to_device(int device) { // -1 - cpu
        host_device_allocator.to_device(device);
        if(connections_manager != nullptr) {
            delete connections_manager;
            connections_manager = new ConnectionsManager(
                #ifdef ENABLE_PROFILING
                profiler,
                #endif
                host_device_allocator, only_host_allocator, false,
                this->base_synapse_metas_id,
                this->global_connections_meta_id,
                this->forward_neuron_infos_id + NEURON_ALIGNMENT_CONSTANT * sizeof(IndexedSynapsesInfo), this->n_neurons - NEURON_ALIGNMENT_CONSTANT, NEURON_ALIGNMENT_CONSTANT,
                this->backward_neuron_infos_id + NEURON_ALIGNMENT_CONSTANT * sizeof(IndexedSynapsesInfo), this->n_neurons - NEURON_ALIGNMENT_CONSTANT, NEURON_ALIGNMENT_CONSTANT,
                this->n_synapse_metas
            );
        }
        if(runtime_context != nullptr) {
            delete runtime_context;
            runtime_context = nullptr;
        }
    }

    void add_connections(
        const torch::Tensor &connections_buffer,
        uint32_t single_input_group_size,
        std::optional<uint32_t> &random_seed
    ) {
        checkConnectionsManagerIsInitialized();
        std::optional<const torch::Tensor> none;
        this->n_synapses += connections_manager->add_connections(
            connections_buffer, none,
            single_input_group_size,
            0, nullptr,
            random_seed ? random_seed.value() : std::random_device{}()
        );
    }

    void compile(
        bool only_trainable_backwards,
        std::optional<uint32_t> &random_seed
    ) {
        if(random_seed && random_seed.value() == 0) {
            throw py::value_error("random_seed should be greater than 0");
        }
        checkConnectionsManagerIsInitialized();
        connections_manager->finalize(
            random_seed ? random_seed.value() : 0,
            true, true,
            only_trainable_backwards,
            false
        );
        NeuronMetaHostInfo* cur_nm_host_info = NeuronMetaHostInfos(this->neuron_meta_host_infos_id, only_host_allocator.data);
        for(uint32_t i=0;i < this->n_neuron_metas;i++, cur_nm_host_info++) {
            cur_nm_host_info->max_forward_delay_range = connections_manager->calculate_max_delay_range(
                cur_nm_host_info->n_neurons,
                cur_nm_host_info->first_neuron_id - NEURON_ALIGNMENT_CONSTANT,
                true
            );
            cur_nm_host_info->max_backward_delay_range = connections_manager->calculate_max_delay_range(
                cur_nm_host_info->n_neurons,
                cur_nm_host_info->first_neuron_id - NEURON_ALIGNMENT_CONSTANT,
                false
            );
        }
    }

    uint64_t process_ticks(
        uint32_t n_ticks_to_process,
        uint32_t batch_size,
        uint32_t n_input_ticks,
        const torch::Tensor &input_values,
        bool do_train,
        bool do_reset_context,
        bool do_apply_deltas,
        bool do_record_voltage,
        uint32_t stdp_period,
        std::optional<const torch::Tensor> &input_ids,
        std::optional<const torch::Tensor> &sparse_input
    ) {
        __TRACE__("spndm_process_ticks: start\n");
        checkTensor(input_values, "input_values", true, host_device_allocator.device);
        if(input_ids) {
            checkTensor(input_ids.value(), "input_ids", false, host_device_allocator.device, sizeof(NeuronIndex_t));
        }
        if(sparse_input) {
            checkTensor(sparse_input.value(), "sparse_input", false, host_device_allocator.device, sizeof(int));
        }
        if(batch_size == 0) {
            throw py::value_error("batch_size == 0");
        }
        if(stdp_period == 0) {
            throw py::value_error("stdp_period == 0");
        }
        if(stdp_period > 64) {
            throw py::value_error("stdp_period > 64");
        }

        // input_values dimensions:
        // 1. dense case (sparse_input is None)
        //    [batch_size, input_ids.numel(), n_input_ticks]
        //    assert n_input_ticks == input_values.numel() / (batch_size * input_ids.numel())
        // 2. sparse case (sparse_input is not None, input_ids is not None)
        //    [batch_size, input_ids.numel(), max_ticks_per_neuron]
        //    max_ticks_per_neuron = sparse_input.numel() / (batch_size * input_ids.numel())
        //    sparse_input.max() < n_input_ticks
        //    sparse_input may contain -1s that should be skipped
        // 3. sparse case transposed (sparse_input is not None, input_ids is None)
        //    [batch_size, n_input_ticks, max_neurons_per_tick]
        //    max_neurons_per_tick = sparse_input.numel() / (batch_size * n_input_ticks)
        //    sparse_input.max() < n_neurons
        //    sparse_input may contain zeros or -1s that should be skipped

        uint32_t max_ticks_per_neuron = 0;
        uint32_t max_neurons_per_tick = 0;
        if(sparse_input) {
            if(input_ids) {
                if((sparse_input.value().numel() % (batch_size * input_ids.value().numel())) > 0) {
                    throw py::value_error("(sparse_input.value().numel() % (batch_size * input_ids.value().numel())) > 0");
                }
                max_ticks_per_neuron = sparse_input.value().numel() / (batch_size * input_ids.value().numel());
            } else {
                if((sparse_input.value().numel() % (batch_size * n_input_ticks)) > 0) {
                    throw py::value_error("(sparse_input.value().numel() % (batch_size * n_input_ticks)) > 0");
                }
                max_neurons_per_tick = sparse_input.value().numel() / (batch_size * n_input_ticks);
            }
            if(input_values.numel() != sparse_input.value().numel()) {
                throw py::value_error("input_values.numel() != sparse_input.value().numel()");
            }
        } else {
            if((input_values.numel() / (batch_size * input_ids.value().numel())) != n_input_ticks) {
                throw py::value_error("(input_values.numel() / (batch_size * input_ids.value().numel())) != n_input_ticks");
            }
        }

        EXTERNAL_REAL_DT *input_values_data = reinterpret_cast<EXTERNAL_REAL_DT *>(input_values.data_ptr());
        NeuronIndex_t *input_ids_data = nullptr;
        if(input_ids) {
            input_ids_data = reinterpret_cast<NeuronIndex_t *>(input_ids.value().data_ptr());
        }
        int *sparse_input_data = nullptr;
        if(sparse_input) {
            sparse_input_data = reinterpret_cast<int *>(sparse_input.value().data_ptr());
        }

        __TRACE__("spndm_process_ticks: after checks\n");

        if(do_train) {
            if(this->runtime_context != nullptr && (!this->runtime_context->is_train() || do_reset_context)) {
                __DETAILED_TRACE__("spndm_process_ticks: deleting runtime_context\n");
                delete this->runtime_context;
                this->runtime_context = nullptr;
            }

            if(this->runtime_context == nullptr) {
                __DETAILED_TRACE__("spndm_process_ticks: creating new runtime_context\n");
                GlobalConnectionsMeta* gc_meta = reinterpret_cast<GlobalConnectionsMeta *>(only_host_allocator.data + global_connections_meta_id);
                this->runtime_context = new SPNET_RUNTIME_CONTEXT_CLASS(
                    host_device_allocator.data,
                    host_device_allocator.device,
                    this->n_neurons,
                    this->n_neuron_metas,
                    #ifdef ENABLE_PROFILING
                    this->profiler,
                    #endif
                    BaseSynapseMetas(this->base_synapse_metas_id, host_device_allocator.data),
                    SPNetSynapseMetas(this->spnet_synapse_metas_id, host_device_allocator.data),
                    NeuronMetaHostInfos(this->neuron_meta_host_infos_id, only_host_allocator.data),
                    IndexedSynapsesInfos(this->forward_neuron_infos_id, host_device_allocator.data),
                    IndexedSynapsesInfos(this->backward_neuron_infos_id, host_device_allocator.data),
                    this->max_delay + 1,
                    this->stdp_tables_id,
                    (uint32_t *)(host_device_allocator.data + this->neurons_to_ltd_table_shift_id),
                    gc_meta->first_synapse_id,
                    gc_meta->last_synapse_id
                );
            }

            bool reset_happened = runtime_context->adjust_to_batch(
                batch_size, n_ticks_to_process, n_input_ticks, do_record_voltage, stdp_period
            );

            if(reset_happened || do_reset_context) {
                runtime_context->initialize_neuron_states();
            } else {
                runtime_context->scroll_ticks();
            }

            if(sparse_input_data != nullptr) {
                if(max_ticks_per_neuron > 0) {
                    runtime_context->import_sparse_input(
                        sparse_input_data, input_values_data, max_ticks_per_neuron,
                        input_ids_data, input_ids.value().numel()
                    );
                } else {
                    runtime_context->import_sparse_input_transposed(
                        sparse_input_data, input_values_data, max_neurons_per_tick
                    );
                }
            } else {
                runtime_context->import_dense_input(input_values_data, input_ids_data, input_ids.value().numel());
            }

            if(do_reset_context) {
                runtime_context->reset_weight_deltas();
            }
            for(uint32_t i=0; i < n_ticks_to_process; i++)  {
                runtime_context->process_tick();
            }
            if(do_apply_deltas) {
                runtime_context->apply_weight_deltas();
            }
        } else {
            if(this->runtime_context != nullptr && (this->runtime_context->is_train() || do_reset_context)) {
                delete this->runtime_context;
                this->runtime_context = nullptr;
            }
            if(this->runtime_context == nullptr) {
                this->runtime_context = new SPNET_RUNTIME_CONTEXT_CLASS(
                    host_device_allocator.data,
                    host_device_allocator.device,
                    this->n_neurons,
                    this->n_neuron_metas,
                    #ifdef ENABLE_PROFILING
                    this->profiler,
                    #endif
                    BaseSynapseMetas(this->base_synapse_metas_id, host_device_allocator.data),
                    SPNetSynapseMetas(this->spnet_synapse_metas_id, host_device_allocator.data),
                    NeuronMetaHostInfos(this->neuron_meta_host_infos_id, only_host_allocator.data),
                    IndexedSynapsesInfos(this->forward_neuron_infos_id, host_device_allocator.data),
                    this->max_delay + 1
                );
            }

            bool reset_happened = runtime_context->adjust_to_batch(
                batch_size, n_ticks_to_process, n_input_ticks, do_record_voltage, stdp_period
            );
            if(reset_happened || do_reset_context) {
                runtime_context->initialize_neuron_states();
            } else {
                runtime_context->scroll_ticks();
            }

            if(sparse_input_data != nullptr) {
                if(max_ticks_per_neuron > 0) {
                    runtime_context->import_sparse_input(
                        sparse_input_data, input_values_data, max_ticks_per_neuron,
                        input_ids_data, input_ids.value().numel()
                    );
                } else {
                    runtime_context->import_sparse_input_transposed(
                        sparse_input_data, input_values_data, max_neurons_per_tick
                    );
                }
            } else {
                runtime_context->import_dense_input(input_values_data, input_ids_data, input_ids.value().numel());
            }

            for(uint32_t i=0; i < n_ticks_to_process; i++)  {
                runtime_context->process_tick();
            }
        }

        return runtime_context->n_generated_spikes();
    }

    void export_neuron_state_info(
        torch::Tensor &target_tensor, // batch_size * (last_tick - first_tick + 1) * neuron_ids.numel()
        const torch::Tensor &neuron_ids,
        uint32_t batch_size,
        uint32_t export_mode,
        uint32_t first_tick,
        uint32_t last_tick
    ) {
        __TRACE__("spndm_export_neuron_state_info\n");

        if(export_mode > 1) {
            throw py::value_error("export_mode > 1");
        }

        if(this->runtime_context == nullptr) {
            throw py::value_error("no active context on export");
        }
        checkTensor(neuron_ids, "neuron_ids", false, host_device_allocator.device, sizeof(NeuronIndex_t));
        checkTensor(target_tensor, "target_tensor", true, host_device_allocator.device);

        if(batch_size > runtime_context->get_batch_size()) {
            throw py::value_error("batch_size > active_ctx->batch_size");
        }

        if(target_tensor.numel() != batch_size * (last_tick - first_tick + 1) * neuron_ids.numel()) {
            throw py::value_error("target_tensor.numel() != batch_size * (last_tick - first_tick + 1) * neuron_ids.numel()");
        }

        EXTERNAL_REAL_DT *target_tensor_data = (EXTERNAL_REAL_DT *) target_tensor.data_ptr<EXTERNAL_REAL_DT>();
        NeuronIndex_t *neuron_ids_data = reinterpret_cast<NeuronIndex_t *>(neuron_ids.data_ptr());
        __TRACE__("before active_ctx->export_neuron_state_info\n");
        runtime_context->export_neuron_state_info(
            target_tensor_data,
            batch_size,
            neuron_ids.numel(),
            neuron_ids_data,
            static_cast<SPNET_RUNTIME_CONTEXT_CLASS::ExportMode>(export_mode),
            first_tick, last_tick
        );
    }

    uint64_t count_synapses(
        const torch::Tensor &neuron_indices_to_process,
        bool forward_or_backward
    ) {
        checkConnectionsManagerIsInitialized();
        return connections_manager->count_synapses(
            neuron_indices_to_process,
            forward_or_backward
        );
    }

    void export_synapses(
        const torch::Tensor &neuron_indices_to_process,
        torch::Tensor &target_internal_source_indices,
        torch::Tensor &target_synapse_meta_indices,
        torch::Tensor &target_weights,
        torch::Tensor &target_delays,
        torch::Tensor &target_internal_target_indices,
        bool forward_or_backward
    ) {
        checkConnectionsManagerIsInitialized();
        std::optional<torch::Tensor> wrp_delays(target_delays);
        std::optional<torch::Tensor> wrp_synapse_meta_indices(target_synapse_meta_indices);
        connections_manager->export_synapses(
            neuron_indices_to_process,
            target_internal_source_indices,
            target_weights,
            target_internal_target_indices,
            forward_or_backward,
            wrp_delays,
            wrp_synapse_meta_indices,
            nullptr
        );
    }

    uint32_t count_max_input_synapses_per_neuron(const torch::Tensor &neuron_indices)
    {
        checkConnectionsManagerIsInitialized();
        return connections_manager->count_max_input_synapses_per_neuron(neuron_indices);
    }

    void export_input_synaptic_weights(
        torch::Tensor &target_weights,
        const torch::Tensor &neuron_indices
    )
    {
        checkConnectionsManagerIsInitialized();
        std::optional<const torch::Tensor> none;
        connections_manager->export_input_synaptic_weights(
            target_weights,
            neuron_indices,
            none,
            nullptr
        );
    }

    auto __repr__() {
        std::ostringstream os;
        GlobalConnectionsMeta* gc_meta = reinterpret_cast<GlobalConnectionsMeta *>(only_host_allocator.data + global_connections_meta_id);
        os << "SPNetDataManager(host_device: " <<
            host_device_allocator.allocated <<
            ", " << host_device_allocator.used <<
            "; host_only: " <<
            only_host_allocator.allocated <<
            ", " << only_host_allocator.used <<
            "; summation type: " << SUMMATION_DT_STR <<
            "; smallest distinguishable fraction: " << get_smallest_distinguishable_fraction() <<
            "; n_synapses: " << n_synapses <<
            "; max_delay: " << max_delay <<
            "; first_synapse_id: " << gc_meta->first_synapse_id <<
            "; last_synapse_id: " << gc_meta->last_synapse_id <<
            "; n_forward_groups: " << gc_meta->n_forward_groups <<
            "; n_backward_groups: " << gc_meta->n_backward_groups <<
        ")";
        return os.str();
    }

    auto get_memory_stats() {
        std::ostringstream os;
        for(int i=0; i < N_SPNET_MEMORY_LABELS;i++) {
            switch(i) {
                case SYNAPSE_METAS_MEMORY_LABEL:
                    os << "synapse metas: ";
                    break;
                case NEURON_METAS_MEMORY_LABEL:
                    os << "neuron metas: ";
                    break;
                case NEURON_INFOS_MEMORY_LABEL:
                    os << "neuron infos: ";
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
            if(runtime_context != nullptr) {
                os << "process_tick by neuron_metas:\n";
                os << runtime_context->get_process_tick_profiler()->get_stats_as_string();
            }
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

    #ifdef ENABLE_PROFILING
    auto _error_counter() {
        return connections_manager->_error_counter();
    }

    void _test_math(
        torch::Tensor &I,
        torch::Tensor &U,
        torch::Tensor &V,
        uint32_t nm_index
    ) {
        checkTensor(I, "I", true, host_device_allocator.device);
        checkTensor(U, "U", true, host_device_allocator.device);
        checkTensor(V, "V", true, host_device_allocator.device);

        if(I.numel() != U.numel()) {
            throw py::value_error("I.numel() != U.numel()");
        }

        if(U.numel() != V.numel()) {
            throw py::value_error("U.numel() != V.numel()");
        }

        if(this->runtime_context == nullptr) {
            this->runtime_context = new SPNET_RUNTIME_CONTEXT_CLASS(
                host_device_allocator.data,
                host_device_allocator.device,
                this->n_neurons,
                this->n_neuron_metas,
                this->profiler,
                BaseSynapseMetas(this->base_synapse_metas_id, host_device_allocator.data),
                SPNetSynapseMetas(this->spnet_synapse_metas_id, host_device_allocator.data),
                NeuronMetaHostInfos(this->neuron_meta_host_infos_id, only_host_allocator.data),
                IndexedSynapsesInfos(this->forward_neuron_infos_id, host_device_allocator.data),
                this->max_delay + 1
            );
        }

        this->runtime_context->test_math(
            reinterpret_cast<REAL_DT *>(I.data_ptr()),
            reinterpret_cast<REAL_DT *>(U.data_ptr()),
            reinterpret_cast<REAL_DT *>(V.data_ptr()),
            I.numel(),
            nm_index
        );
    }
    #endif

    ~SPNDM_CLASS_NAME() {
        if(connections_manager != nullptr) {
            delete connections_manager;
        }
        if(runtime_context != nullptr) {
            delete runtime_context;
        }
    }

private:
    #ifdef ENABLE_PROFILING
    SimpleProfiler profiler;
    #endif
    SimpleAllocator host_device_allocator;
    SimpleAllocator only_host_allocator;

    ConnectionsManager* connections_manager;
    SPNET_RUNTIME_CONTEXT_CLASS *runtime_context;

    NeuronDataId_t base_synapse_metas_id;
    NeuronDataId_t spnet_synapse_metas_id;
    NeuronDataId_t neuron_meta_host_infos_id;
    uint32_t n_synapse_metas;
    uint64_t n_synapses;
    uint64_t max_delay;
    NeuronDataId_t neuron_metas_id;
    uint32_t n_neuron_metas;
    NeuronDataId_t forward_neuron_infos_id;
    NeuronDataId_t backward_neuron_infos_id;
    NeuronDataId_t neurons_to_ltd_table_shift_id;
    NeuronDataId_t stdp_tables_id;
    uint32_t n_neurons;
    NeuronDataId_t global_connections_meta_id;

    REAL_DT stdp_threshold;

    SPNDM_CLASS_NAME(
        uint8_t* host_device_data, NeuronDataId_t host_device_used,
        uint8_t* only_host_data, NeuronDataId_t only_host_used,
        NeuronDataId_t _base_synapse_metas_id,
        NeuronDataId_t _spnet_synapse_metas_id,
        NeuronDataId_t _neuron_meta_host_infos_id,
        uint32_t _n_synapse_metas,
        uint64_t _n_synapses,
        uint64_t _max_delay,
        NeuronDataId_t _neuron_metas_id,
        uint32_t _n_neuron_metas,
        NeuronDataId_t _forward_neuron_infos_id,
        NeuronDataId_t _backward_neuron_infos_id,
        NeuronDataId_t _neurons_to_ltd_table_shift_id,
        NeuronDataId_t _stdp_tables_id,
        uint32_t _n_neurons,
        NeuronDataId_t _global_connections_meta_id,
        REAL_DT _stdp_threshold
    ) :
        #ifdef ENABLE_PROFILING
        profiler(N_SPNET_PROFILER_OPS),
        #endif
        host_device_allocator(host_device_data, host_device_used),
        only_host_allocator(only_host_data, only_host_used),
        runtime_context(nullptr),
        base_synapse_metas_id(_base_synapse_metas_id),
        spnet_synapse_metas_id(_spnet_synapse_metas_id),
        neuron_meta_host_infos_id(_neuron_meta_host_infos_id),
        n_synapse_metas(_n_synapse_metas),
        n_synapses(_n_synapses),
        max_delay(_max_delay),
        neuron_metas_id(_neuron_metas_id),
        n_neuron_metas(_n_neuron_metas),
        forward_neuron_infos_id(_forward_neuron_infos_id),
        backward_neuron_infos_id(_backward_neuron_infos_id),
        neurons_to_ltd_table_shift_id(_neurons_to_ltd_table_shift_id),
        stdp_tables_id(_stdp_tables_id),
        n_neurons(_n_neurons),
        global_connections_meta_id(_global_connections_meta_id),
        stdp_threshold(_stdp_threshold)
    {
        connections_manager = new ConnectionsManager(
            #ifdef ENABLE_PROFILING
            this->profiler,
            #endif
            this->host_device_allocator, this->only_host_allocator, false,
            this->base_synapse_metas_id,
            this->global_connections_meta_id,
            this->forward_neuron_infos_id + NEURON_ALIGNMENT_CONSTANT * sizeof(IndexedSynapsesInfo), this->n_neurons - NEURON_ALIGNMENT_CONSTANT, NEURON_ALIGNMENT_CONSTANT,
            this->backward_neuron_infos_id + NEURON_ALIGNMENT_CONSTANT * sizeof(IndexedSynapsesInfo), this->n_neurons - NEURON_ALIGNMENT_CONSTANT, NEURON_ALIGNMENT_CONSTANT,
            this->n_synapse_metas
        );
    }

    void checkOnHostDuringPrepare()
    {
        if(host_device_allocator.device != -1) {
            throw std::runtime_error("NeuronDataManager must be in the host memory during neural data preparation");
        }
    }

    void checkConnectionsManagerIsInitialized()
    {
        if(connections_manager == nullptr) {
            throw py::value_error("connections_manager is not initialized, use initialize_neurons(...) method");
        }
    }
};

py::tuple PFX(pickle_spnet_neuron_manager)(const SPNDM_CLASS_NAME& ndm) {
    if(ndm.host_device_allocator.device != -1) {
        throw std::runtime_error("neural net must be on CPU before serialization");
    }
    return py::make_tuple(
        py::reinterpret_steal<py::object>(
            PyBytes_FromStringAndSize((char *) ndm.host_device_allocator.data, ndm.host_device_allocator.used)
        ),
        py::reinterpret_steal<py::object>(
            PyBytes_FromStringAndSize((char *) ndm.only_host_allocator.data, ndm.only_host_allocator.used)
        ),
        ndm.base_synapse_metas_id,
        ndm.spnet_synapse_metas_id,
        ndm.neuron_meta_host_infos_id,
        ndm.n_synapse_metas,
        ndm.n_synapses,
        ndm.max_delay,
        ndm.neuron_metas_id,
        ndm.n_neuron_metas,
        ndm.forward_neuron_infos_id,
        ndm.backward_neuron_infos_id,
        ndm.neurons_to_ltd_table_shift_id,
        ndm.stdp_tables_id,
        ndm.n_neurons,
        ndm.global_connections_meta_id,
        ndm.stdp_threshold
    );
}

std::unique_ptr<SPNDM_CLASS_NAME> PFX(unpickle_spnet_neuron_manager)(py::tuple t) {
    char *buf;
    Py_ssize_t host_device_used;
    PyBytes_AsStringAndSize(t[0].ptr(), &buf, &host_device_used);
    uint8_t *host_device_data = (uint8_t *) PyMem_Malloc(host_device_used);
    memcpy(host_device_data, buf, host_device_used);

    Py_ssize_t only_host_used;
    PyBytes_AsStringAndSize(t[1].ptr(), &buf, &only_host_used);
    uint8_t *only_host_data = (uint8_t *) PyMem_Malloc(only_host_used);
    memcpy(only_host_data, buf, only_host_used);

    return std::unique_ptr<SPNDM_CLASS_NAME>(
        new SPNDM_CLASS_NAME(
            host_device_data, (NeuronDataId_t) host_device_used,
            only_host_data, (NeuronDataId_t) only_host_used,
            t[2].cast<NeuronDataId_t>(),
            t[3].cast<NeuronDataId_t>(),
            t[4].cast<NeuronDataId_t>(),
            t[5].cast<uint32_t>(),
            t[6].cast<uint32_t>(),
            t[7].cast<uint32_t>(),
            t[8].cast<NeuronDataId_t>(),
            t[9].cast<uint32_t>(),
            t[10].cast<NeuronDataId_t>(),
            t[11].cast<NeuronDataId_t>(),
            t[12].cast<NeuronDataId_t>(),
            t[13].cast<NeuronDataId_t>(),
            t[14].cast<uint32_t>(),
            t[15].cast<NeuronDataId_t>(),
            t[16].cast<REAL_DT>()
        )
    );
}

void PFX(PB_SPNetDataManager)(py::module& m) {
    #ifdef INTEGERS_INSTEAD_OF_FLOATS
    py::class_<SPNetDataManagerI>(m, "SPNetDataManagerI")
    #else
    py::class_<SPNetDataManagerF>(m, "SPNetDataManagerF")
    #endif
        .def(py::init<uint64_t, REAL_DT>())
        .def("get_smallest_distinguishable_fraction", &SPNDM_CLASS_NAME::get_smallest_distinguishable_fraction,
            "Returns smallest fraction that can exist inside data manager in integers mode. Returns 0.0 in floats mode.")
        .def("get_epsilon", &SPNDM_CLASS_NAME::get_epsilon,
            "Returns super small real number used inside the engine")
        .def("get_summations_data_type", &SPNDM_CLASS_NAME::get_summations_data_type,
            "Returns string 'float32' in floats mode and string 'int32' in integers mode")
        .def("register_synapse_meta", &SPNDM_CLASS_NAME::register_synapse_meta,
            "Register synapse meta",
            py::arg("learning_rate"),
            py::arg("min_delay"),
            py::arg("max_delay"),
            py::arg("min_synaptic_weight"),
            py::arg("max_synaptic_weight"),
            py::arg("initial_noise_level"),
            py::arg("initial_weight"),
            py::arg("weight_decay"),
            py::arg("weight_scaling_cf"),
            py::arg("_forward_group_size"),
            py::arg("_backward_group_size"))
        .def("register_neuron_meta", &SPNDM_CLASS_NAME::register_neuron_meta,
            "Register neuron meta",
            py::arg("neuron_type"),
            py::arg("cf_2"),
            py::arg("cf_1"),
            py::arg("cf_0"),
            py::arg("a"),
            py::arg("b"),
            py::arg("c"),
            py::arg("d"),
            py::arg("spike_threshold"),
            py::arg("stdp_decay"),
            py::arg("ltp_max"),
            py::arg("ltd_max"))
        .def("initialize_neurons", &SPNDM_CLASS_NAME::initialize_neurons,
            "Initialize neurons",
            py::arg("neuron_counts_by_meta_id"))
        .def("get_total_number_of_neurons", &SPNDM_CLASS_NAME::get_total_number_of_neurons,
            "Get total number of neurons")
        .def("get_total_number_of_synapses", &SPNDM_CLASS_NAME::get_total_number_of_synapses,
            "Get total number of synapses")
        .def("get_max_delay", &SPNDM_CLASS_NAME::get_max_delay,
            "Get max synaptic delay (according to registered synapse metas)")
        .def("to_device", &SPNDM_CLASS_NAME::to_device)
        .def("add_connections", &SPNDM_CLASS_NAME::add_connections,
            "Add connections from ConnectionsBuffer",
            py::arg("connections_buffer"),
            py::arg("single_input_group_size"),
            py::arg("random_seed") = py::none())
        .def("compile", &SPNDM_CLASS_NAME::compile,
            "Finalize synapse groups (calculate backward synapses)",
            py::arg("only_trainable_backwards"),
            py::arg("random_seed") = py::none())
        .def("process_ticks", &SPNDM_CLASS_NAME::process_ticks,
            "Calculate n input ticks taking into account given external input",
            py::arg("n_ticks_to_process"),
            py::arg("batch_size"),
            py::arg("n_input_ticks"),
            py::arg("input_values"),
            py::arg("do_train"),
            py::arg("do_reset_context"),
            py::arg("do_apply_deltas"),
            py::arg("do_record_voltage"),
            py::arg("stdp_period"),
            py::arg("input_ids") = py::none(),
            py::arg("sparse_input") = py::none())
        .def("export_neuron_state_info", &SPNDM_CLASS_NAME::export_neuron_state_info,
            "Export neuron state info",
            py::arg("target_tensor"),
            py::arg("internal_neuron_indices"),
            py::arg("batch_size"),
            py::arg("export_mode"),
            py::arg("first_tick"),
            py::arg("last_tick"))
        .def("count_max_input_synapses_per_neuron", &SPNDM_CLASS_NAME::count_max_input_synapses_per_neuron,
            "Counts maximum number of input synapses per neuron for a set of neurons",
            py::arg("neuron_indices"))
        .def("export_input_synaptic_weights", &SPNDM_CLASS_NAME::export_input_synaptic_weights,
            "Export input synaptic weights of a set of neurons",
            py::arg("target_weights"),
            py::arg("neuron_indices"))
        .def("count_synapses", &SPNDM_CLASS_NAME::count_synapses,
            "Count forward or backward synapses for the set of neurons",
            py::arg("neuron_indices_to_process"),
            py::arg("forward_or_backward"))
        .def("export_synapses", &SPNDM_CLASS_NAME::export_synapses,
            "Export all synapses",
            py::arg("neuron_indices_to_process"),
            py::arg("target_internal_source_indices"),
            py::arg("target_synapse_meta_indices"),
            py::arg("target_weights"),
            py::arg("target_delays"),
            py::arg("target_internal_target_indices"),
            py::arg("forward_or_backward"))
        .def("__repr__", &SPNDM_CLASS_NAME::__repr__)
        .def("get_memory_stats", &SPNDM_CLASS_NAME::get_memory_stats)
        .def("get_profiling_stats", &SPNDM_CLASS_NAME::get_profiling_stats)
        .def("reset_profiler", &SPNDM_CLASS_NAME::reset_profiler)
        #ifdef ENABLE_PROFILING
        .def("_error_counter", &SPNDM_CLASS_NAME::_error_counter)
        .def("_test_math", &SPNDM_CLASS_NAME::_test_math,
            "test math (aux method)",
            py::arg("I"),
            py::arg("U"),
            py::arg("V"),
            py::arg("nm_index"))
        #endif
        .def(py::pickle(
            &PFX(pickle_spnet_neuron_manager),
            &PFX(unpickle_spnet_neuron_manager)
        ));
}

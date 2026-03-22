
#include "loading_functions.h"




struct __attribute__((packed)) BinaryTrainingSample 
{
    float o[3];
    float d[3];
    float beta_before_rgb[3];
    float L_after_rgb[3];
    float T_after[3];
    float tMax;
};


struct __attribute__((packed)) BinaryTrainingSampleSpectral
{
    float o[3];
    float d[3];
    // float beta_before_rgb[3];
    // float L_after_rgb[3];
    float T_after[4];
    float tMax;
};

void load_dataset_to_testbed_fixedsize(Testbed& testbed, const fs::path& path, DatasetType dataset_type, uint64_t sampleCnt)
{
    std::ifstream f{native_string(path), std::ios::in | std::ios::out | std::ios::binary };
    if(!f.is_open())
        throw std::runtime_error("Failed to open file");

    f.clear();
    f.seekg(0, std::ios::beg);

    uint64_t totalCount = 0;
    f.read(reinterpret_cast<char *>(&totalCount), sizeof(uint64_t));

    uint64_t sampleCount = std::min(totalCount, (uint64_t)UINT32_MAX);
    sampleCount = std::min(sampleCount, sampleCnt);

    tlog::info() << "Loading " << sampleCount << " samples as " 
                    << (dataset_type == DatasetType::Training ? "training" : 
                        dataset_type == DatasetType::Validation ? "validation" : "test") 
                    << " data from " << path.str();

    float **input_cpu_ptr;
    float **target_cpu_ptr;
    uint64_t* n_samples_ptr;
    GPUMemory<float>* input_gpu_ptr;
    GPUMemory<float> *target_gpu_ptr;

    switch(dataset_type) {
        case DatasetType::Training:
            input_cpu_ptr = &testbed.m_volume_training_inputs_cpu;
            target_cpu_ptr = &testbed.m_volume_training_targets_cpu;
            n_samples_ptr = &testbed.m_n_volume_training_samples;
            input_gpu_ptr = &testbed.m_volume_training_inputs;
            target_gpu_ptr = &testbed.m_volume_training_targets;
            break;
        case DatasetType::Validation:
            input_cpu_ptr = &testbed.m_volume_validation_inputs_cpu;
            target_cpu_ptr = &testbed.m_volume_validation_targets_cpu;
            n_samples_ptr = &testbed.m_n_volume_validation_samples;
            input_gpu_ptr = &testbed.m_volume_validation_inputs;
            target_gpu_ptr = &testbed.m_volume_validation_targets;
            break;
        case DatasetType::Test:
            input_cpu_ptr = &testbed.m_volume_test_inputs_cpu;
            target_cpu_ptr = &testbed.m_volume_test_targets_cpu;
            n_samples_ptr = &testbed.m_n_volume_test_samples;
            input_gpu_ptr = &testbed.m_volume_test_inputs;
            target_gpu_ptr = &testbed.m_volume_test_targets;
            break;
        default:
            break;
        }

    if(*input_cpu_ptr) {
        delete[] *input_cpu_ptr;
        *input_cpu_ptr = nullptr;
    }
    if(*target_cpu_ptr) {
        delete[] *target_cpu_ptr;
        *target_cpu_ptr = nullptr;
    }

    *n_samples_ptr = sampleCount;
    *input_cpu_ptr = new float[sampleCount * N_VOLUME_INPUT_DIMS];
    *target_cpu_ptr = new float[sampleCount * N_VOLUME_TARGET_DIMS];


    vec3 min_bound = vec3(1e30f);
    vec3 max_bound = vec3(-1e30f);
    float minTMax = 1e30f;
    float maxTMax = -1e30f;

    auto& input_cpu_buffer = *input_cpu_ptr;
    auto &target_cpu_buffer = *target_cpu_ptr;
    std::vector<BinaryTrainingSample> binarySampleBuffer;
    binarySampleBuffer.resize(sampleCount);

    f.read(reinterpret_cast<char *>(binarySampleBuffer.data()),
        sizeof(BinaryTrainingSample) * sampleCount);


    if(dataset_type == DatasetType::Training)
    {
        std::random_device rd;
        std::mt19937 shuffle_rng(rd());
        // default_rng_t shuffle_rng{testbed.m_seed};
        std::shuffle(binarySampleBuffer.begin(), binarySampleBuffer.end(), shuffle_rng);
    }

    bool normalize_data = (dataset_type == DatasetType::Training);

    auto processSample = [&](const BinaryTrainingSample& bs, int idx
        )
        {
            // Calculate scene BB to normalize rays
            vec3 rayO = vec3(bs.o[0], bs.o[1], bs.o[2]);
            
            // Check for nans or infs
            if(!std::isfinite(bs.o[0]) || !std::isfinite(bs.o[1]) || !std::isfinite(bs.o[2]))
            {
                tlog::warning() << "found Nan/Inf input ray position data";
                rayO = vec3(0.f, 0.f, 0.f);
            }
            
            constexpr float MAX_SCENE_DIST = 160.f;
            float tMaxVal = std::min(bs.tMax, MAX_SCENE_DIST);
            if(!std::isfinite(bs.tMax))
            {
                tlog::warning() << "found Nan/Inf input tmax data";
                tMaxVal = 0.f;
            }

            if(normalize_data)
            {
                min_bound = min(min_bound, rayO);
                max_bound = max(max_bound, rayO);
                minTMax = min(minTMax, tMaxVal);
                maxTMax = max(maxTMax, tMaxVal);
            }

            input_cpu_buffer[idx * N_VOLUME_INPUT_DIMS + POS_OFFSET + 0] = rayO.x;
            input_cpu_buffer[idx * N_VOLUME_INPUT_DIMS + POS_OFFSET + 1] = rayO.y;
            input_cpu_buffer[idx * N_VOLUME_INPUT_DIMS + POS_OFFSET + 2] = rayO.z;
            
            // Normalize direction vector and transform from [-1,1] to [0,1] range
            // SphericalHarmonics expects input in [0,1] and internally maps to [-1,1]
            vec3 rayD = vec3(bs.d[0], bs.d[1], bs.d[2]);
            float dirLen = std::sqrt(rayD.x * rayD.x + rayD.y * rayD.y + rayD.z * rayD.z);
            if (dirLen > 1e-6f) {
                rayD = rayD / dirLen;  // Normalize to unit length
            }
            if(!std::isfinite(rayD.x) || !std::isfinite(rayD.y) || !std::isfinite(rayD.z))
            {
                tlog::warning() << "found Nan/Inf input ray direction data";
                rayD = vec3(0.f, 0.f, 0.f);
            }
            // Map from [-1, 1] to [0, 1] for SphericalHarmonics encoding
            input_cpu_buffer[idx * N_VOLUME_INPUT_DIMS + DIR_OFFSET + 0] = rayD.x * 0.5f + 0.5f;
            input_cpu_buffer[idx * N_VOLUME_INPUT_DIMS + DIR_OFFSET + 1] = rayD.y * 0.5f + 0.5f;
            input_cpu_buffer[idx * N_VOLUME_INPUT_DIMS + DIR_OFFSET + 2] = rayD.z * 0.5f + 0.5f;
            
            input_cpu_buffer[idx * N_VOLUME_INPUT_DIMS + TMAX_OFFSET] = tMaxVal;


            //TODO: Check if i need to do some clamping here as well
            constexpr float epsilon = 1e-6f;
            constexpr float MAX_RADIANCE = 1000.f; //for now, just hard clamping this
            for (int c = 0; c < 3; ++c) {
                if(bs.beta_before_rgb[c] > epsilon)
                {
                    float val = std::max(0.f, bs.L_after_rgb[c]);
                    float L_clamped = std::min(MAX_RADIANCE, val / bs.beta_before_rgb[c]);
                    float L_log = std::log(L_clamped + 1.f);
                    if(!std::isfinite(L_log))
                    {
                        tlog::warning() << "found Nan/Inf target radiance data";
                        L_log = 0.f;
                    }
                    target_cpu_buffer[idx * N_VOLUME_TARGET_DIMS + L_OFFSET + c] = L_log;
                } else {
                    target_cpu_buffer[idx * N_VOLUME_TARGET_DIMS + L_OFFSET + c] = 0.f;
                }
            }

            vec3 T_after = vec3(bs.T_after[0], bs.T_after[1], bs.T_after[2]);
            if (!std::isfinite(bs.T_after[0]) || !std::isfinite(bs.T_after[1]) ||
                !std::isfinite(bs.T_after[2])) {
                tlog::warning() << "found Nan/Inf target T_after data";
                T_after = vec3(0.f, 0.f, 0.f);
            }
            target_cpu_buffer[idx * N_VOLUME_TARGET_DIMS + T_OFFSET + 0] = std::min(1.f, std::max(0.f, T_after[0]));
            target_cpu_buffer[idx * N_VOLUME_TARGET_DIMS + T_OFFSET + 1] = std::min(1.f, std::max(0.f, T_after[1]));
            target_cpu_buffer[idx * N_VOLUME_TARGET_DIMS + T_OFFSET + 2] = std::min(1.f, std::max(0.f, T_after[2]));
    };

    for(uint64_t i = 0; i < sampleCount; ++i)
    {
        processSample(binarySampleBuffer[i], i);
    }

    tlog::success() << "Read " << sampleCount << " samples";
    CUDA_CHECK_THROW(cudaDeviceSynchronize());
    CUDA_CHECK_THROW(cudaGetLastError()); // Clear any prior errors

    tlog::info() << "Scene AABB: [" << min_bound.x << ", " << min_bound.y << ", "
                 << min_bound.z << "] to [" << max_bound.x << ", " << max_bound.y << ", "
                 << max_bound.z << "]";

    float scale, tMax_scale, tMax_offset;
    vec3 offset;

    // Compute and store normalization values

    // Use stored normalization parameters from training data
    scale = testbed.m_volume_training_inputs_scale;
    offset = testbed.m_volume_training_inputs_offset;
    tMax_scale = testbed.m_volume_training_inputs_tMax_scale;
    tMax_offset = testbed.m_volume_training_inputs_tMax_offset;

    if(scale == 0.0f) {
        tlog::warning() << "Normalization parameters not set! Load training data first.";
    }

    // Apply normalization
    for(uint64_t i = 0; i < sampleCount; ++i)
    {
        auto index = i * N_VOLUME_INPUT_DIMS;
        
        for(int dim = 0; dim < 3; ++dim)
        {
            const float val = input_cpu_buffer[index + POS_OFFSET + dim];
            input_cpu_buffer[index + POS_OFFSET + dim] = val * scale + offset[dim];
        }

        const float tMax_val = input_cpu_buffer[index + TMAX_OFFSET];
        input_cpu_buffer[index + TMAX_OFFSET] = (tMax_val - tMax_offset) * tMax_scale;
    }

    const auto input_bytes_to_copy = (*n_samples_ptr) * N_VOLUME_INPUT_DIMS;
    const auto target_bytes_to_copy = (*n_samples_ptr) * N_VOLUME_TARGET_DIMS;

    auto &input_gpu_buffer = *input_gpu_ptr;
    auto &target_gpu_buffer = *target_gpu_ptr;


    input_gpu_buffer.resize(input_bytes_to_copy);
    target_gpu_buffer.resize(target_bytes_to_copy);

    input_gpu_buffer.copy_from_host(input_cpu_buffer, input_bytes_to_copy);
    target_gpu_buffer.copy_from_host(target_cpu_buffer, target_bytes_to_copy);


    CUDA_CHECK_THROW(cudaDeviceSynchronize());
    tlog::success() << "Uploaded to GPU";
}

void load_datasets_to_testbed(Testbed& testbed, const std::vector<fs::path>& paths, DatasetType dataset_type)
{
    if(paths.empty()) return;

    // Calculate total sample cnt first
    uint64_t total_sample_cnt = 0;
    std::vector<uint64_t> sample_cnts;
    sample_cnts.reserve(paths.size());

    for(const auto& path : paths)
    {
        std::ifstream f{native_string(path), std::ios::in | std::ios::out | std::ios::binary };
        if(!f.is_open())
            throw std::runtime_error("Failed to open file path" + path.str());
        f.clear();
        f.seekg(0, std::ios::beg);

        uint64_t cnt = 0;
        f.read(reinterpret_cast<char *>(&cnt), sizeof(uint64_t));

        sample_cnts.push_back(cnt);
        total_sample_cnt += cnt;
    }

    tlog::info() << "Loading " << paths.size()
                 << " files. Total training samples: " << total_sample_cnt;

    //TODO: Complete this if this is required



    float **input_cpu_ptr;
    float **target_cpu_ptr;
    uint64_t* n_samples_ptr;
    
    input_cpu_ptr = &testbed.m_volume_training_inputs_cpu;
    target_cpu_ptr = &testbed.m_volume_training_targets_cpu;
    n_samples_ptr = &testbed.m_n_volume_training_samples;
    
    if(*input_cpu_ptr) {
        if(testbed.m_stream_training_data_from_CPU)
        {
            cudaFreeHost(*input_cpu_ptr);
            cudaFreeHost(*target_cpu_ptr);
        }
        else
        {
            delete[] *input_cpu_ptr;
            delete[] *target_cpu_ptr;
        }
        *input_cpu_ptr = nullptr;
        *target_cpu_ptr = nullptr;
    }
    
    
    
    testbed.m_stream_training_data_from_CPU = true; // Just in case we do not have enough VRAM, enforcing this to be true here
    *n_samples_ptr = total_sample_cnt;
    const auto training_input_bytes_size = total_sample_cnt * N_VOLUME_INPUT_DIMS * sizeof(float);
    const auto training_target_bytes_size = total_sample_cnt * N_VOLUME_TARGET_DIMS * sizeof(float);
    CUDA_CHECK_THROW(cudaMallocHost((void **)input_cpu_ptr, training_input_bytes_size));
    CUDA_CHECK_THROW(cudaMallocHost((void**)target_cpu_ptr, training_target_bytes_size));


    vec3 min_bound = vec3(1e30f);
    vec3 max_bound = vec3(-1e30f);
    float minTMax = 1e30f;
    float maxTMax = -1e30f;

    auto& input_cpu_buffer = *input_cpu_ptr;
    auto &target_cpu_buffer = *target_cpu_ptr;
    uint64_t ptr = 0;
    for (size_t i = 0; i < paths.size(); ++i) {
        uint64_t fileCount = sample_cnts[i];
        if(fileCount == 0) continue;

        tlog::info() << "Processing file " << (i + 1) << "/" << paths.size() << ":" << paths[i].filename();

        std::ifstream f{native_string(paths[i]), std::ios::in | std::ios::binary};
        f.seekg(sizeof(uint64_t));

        std::vector<BinaryTrainingSample> batch(fileCount);
        f.read(reinterpret_cast<char*>(batch.data()), sizeof(BinaryTrainingSample) * fileCount);

        std::random_device rd;
        std::mt19937 shuffle_rng(rd());
        // default_rng_t shuffle_rng{testbed.m_seed};
        std::shuffle(batch.begin(), batch.end(), shuffle_rng);

        for (uint64_t j = 0; j < fileCount; ++j)
        {
            const auto &bs = batch[j];
            uint64_t idx = ptr + j;
 // Calculate scene BB to normalize rays
            vec3 rayO = vec3(bs.o[0], bs.o[1], bs.o[2]);

            constexpr float MAX_SCENE_DIST = 160.f;
            float tMaxVal = std::min(bs.tMax, MAX_SCENE_DIST);

            min_bound = min(min_bound, rayO);
            max_bound = max(max_bound, rayO);
            minTMax = min(minTMax, tMaxVal);
            maxTMax = max(maxTMax, tMaxVal);

            input_cpu_buffer[idx * N_VOLUME_INPUT_DIMS + POS_OFFSET + 0] = rayO.x;
            input_cpu_buffer[idx * N_VOLUME_INPUT_DIMS + POS_OFFSET + 1] = rayO.y;
            input_cpu_buffer[idx * N_VOLUME_INPUT_DIMS + POS_OFFSET + 2] = rayO.z;
            
            // Normalize direction vector and transform from [-1,1] to [0,1] range
            // SphericalHarmonics expects input in [0,1] and internally maps to [-1,1]
            vec3 rayD = vec3(bs.d[0], bs.d[1], bs.d[2]);
            float dirLen = std::sqrt(rayD.x * rayD.x + rayD.y * rayD.y + rayD.z * rayD.z);
            if (dirLen > 1e-6f) {
                rayD = rayD / dirLen;  // Normalize to unit length
            }
            // Map from [-1, 1] to [0, 1] for SphericalHarmonics encoding
            input_cpu_buffer[idx * N_VOLUME_INPUT_DIMS + DIR_OFFSET + 0] = rayD.x * 0.5f + 0.5f;
            input_cpu_buffer[idx * N_VOLUME_INPUT_DIMS + DIR_OFFSET + 1] = rayD.y * 0.5f + 0.5f;
            input_cpu_buffer[idx * N_VOLUME_INPUT_DIMS + DIR_OFFSET + 2] = rayD.z * 0.5f + 0.5f;
            
            input_cpu_buffer[idx * N_VOLUME_INPUT_DIMS + TMAX_OFFSET] = tMaxVal;

            constexpr float epsilon = 1e-6f;
            constexpr float MAX_RADIANCE = 1000.f; //for now, just hard clamping this
            for (int c = 0; c < 3; ++c) {
                if(bs.beta_before_rgb[c] > epsilon)
                {
                    float val = std::max(0.f, bs.L_after_rgb[c]);
                    float L_clamped = std::min(MAX_RADIANCE, val / bs.beta_before_rgb[c]);
                    float L_log = std::log(L_clamped + 1.f);
                    target_cpu_buffer[idx * N_VOLUME_TARGET_DIMS + L_OFFSET + c] = L_log;
                } else {
                    target_cpu_buffer[idx * N_VOLUME_TARGET_DIMS + L_OFFSET + c] = 0.f;
                }
            }

            target_cpu_buffer[idx * N_VOLUME_TARGET_DIMS + T_OFFSET + 0] = std::min(1.f, std::max(0.f, bs.T_after[0]));
            target_cpu_buffer[idx * N_VOLUME_TARGET_DIMS + T_OFFSET + 1] = std::min(1.f, std::max(0.f, bs.T_after[1]));
            target_cpu_buffer[idx * N_VOLUME_TARGET_DIMS + T_OFFSET + 2] = std::min(1.f, std::max(0.f, bs.T_after[2]));
        }

        ptr += fileCount;
    }


    //INFO: Calculating Normalization data
    // Compute scale and offset to fit in [0, 1]
    float scale, tMax_scale, tMax_offset;
    vec3 offset;
    vec3 size = max_bound - min_bound;
    float max_dim = std::max({size.x, size.y, size.z});

    if(max_dim == 0) max_dim = 1.f;

    scale = 1.f / max_dim;

    vec3 centre = (max_bound + min_bound) * 0.5f;
    offset = vec3(0.5f) - centre * scale;

    float tMax_range = maxTMax - minTMax;

    if(tMax_range == 0.f)
    {
        tMax_range = 1.f;
        tlog::warning() << "tMax range is 0, no normalization applied";
    }

    tMax_scale = 1.f / tMax_range;
    tMax_offset = minTMax;


    // Store normalization params in testbed
    testbed.m_volume_training_inputs_scale = scale;
    testbed.m_volume_training_inputs_offset = offset;
    testbed.m_volume_training_inputs_tMax_scale = tMax_scale;
    testbed.m_volume_training_inputs_tMax_offset = tMax_offset;

    tlog::info() << "Scene AABB: [" << min_bound.x << ", " << min_bound.y << ", "
            << min_bound.z << "] to [" << max_bound.x << ", " << max_bound.y << ", "
            << max_bound.z << "]";
    tlog::info() << "Auto-normalizing rays: scale=" << scale << " offset=[" << offset.x
            << ", " << offset.y << ", " << offset.z << "]";
    tlog::info() << "Auto-normalizing tmax: scale=" << tMax_scale << " with range=[ "
            << minTMax << ", " << maxTMax << " ]\n";

    //INFO: Applying Normalization 
    for(uint64_t i = 0; i < total_sample_cnt; ++i)
    {
        auto index = i * N_VOLUME_INPUT_DIMS;
        
        for(int dim = 0; dim < 3; ++dim)
        {
            const float val = input_cpu_buffer[index + POS_OFFSET + dim];
            input_cpu_buffer[index + POS_OFFSET + dim] = val * scale + offset[dim];
        }

        const float tMax_val = input_cpu_buffer[index + TMAX_OFFSET];
        input_cpu_buffer[index + TMAX_OFFSET] = (tMax_val - tMax_offset) * tMax_scale;
    }

    const auto input_bytes_to_copy = (*n_samples_ptr) * N_VOLUME_INPUT_DIMS * sizeof(float);
    const auto target_bytes_to_copy = (*n_samples_ptr) * N_VOLUME_TARGET_DIMS * sizeof(float);


    //GPU Upload or Pinned memory streaming
    tlog::info() << "Training data too large for VRAM ( " << (input_bytes_to_copy + target_bytes_to_copy)/1e9 
                << " GB). Using CPU memory streaming instead.";
            

    GPUMemory<float>* input_gpu_ptr = &testbed.m_volume_training_inputs;
    GPUMemory<float>* target_gpu_ptr = &testbed.m_volume_training_targets;
    input_gpu_ptr->resize(0);
    target_gpu_ptr->resize(0);
}

void load_dataset_to_testbed(Testbed& testbed, const fs::path& path, DatasetType dataset_type, bool should_shuffle)
{
    std::ifstream f{native_string(path), std::ios::in | std::ios::out | std::ios::binary };
    if(!f.is_open())
        throw std::runtime_error("Failed to open file");

    f.clear();
    f.seekg(0, std::ios::beg);

    uint64_t totalCount = 0;
    f.read(reinterpret_cast<char *>(&totalCount), sizeof(uint64_t));

    uint64_t sampleCount = std::min(totalCount, (uint64_t)UINT32_MAX);


    tlog::info() << "Loading " << sampleCount << " samples as " 
                    << (dataset_type == DatasetType::Training ? "training" : 
                        dataset_type == DatasetType::Validation ? "validation" : "test") 
                    << " data from " << path.str();

    float **input_cpu_ptr;
    float **target_cpu_ptr;
    uint64_t* n_samples_ptr;
    GPUMemory<float>* input_gpu_ptr;
    GPUMemory<float> *target_gpu_ptr;

    switch(dataset_type) {
        case DatasetType::Training:
            input_cpu_ptr = &testbed.m_volume_training_inputs_cpu;
            target_cpu_ptr = &testbed.m_volume_training_targets_cpu;
            n_samples_ptr = &testbed.m_n_volume_training_samples;
            input_gpu_ptr = &testbed.m_volume_training_inputs;
            target_gpu_ptr = &testbed.m_volume_training_targets;
            break;
        case DatasetType::Validation:
            input_cpu_ptr = &testbed.m_volume_validation_inputs_cpu;
            target_cpu_ptr = &testbed.m_volume_validation_targets_cpu;
            n_samples_ptr = &testbed.m_n_volume_validation_samples;
            input_gpu_ptr = &testbed.m_volume_validation_inputs;
            target_gpu_ptr = &testbed.m_volume_validation_targets;
            break;
        case DatasetType::Test:
            input_cpu_ptr = &testbed.m_volume_test_inputs_cpu;
            target_cpu_ptr = &testbed.m_volume_test_targets_cpu;
            n_samples_ptr = &testbed.m_n_volume_test_samples;
            input_gpu_ptr = &testbed.m_volume_test_inputs;
            target_gpu_ptr = &testbed.m_volume_test_targets;
            break;
        default:
            break;
        }

    if(*input_cpu_ptr) {
        delete[] *input_cpu_ptr;
        *input_cpu_ptr = nullptr;
    }
    if(*target_cpu_ptr) {
        delete[] *target_cpu_ptr;
        *target_cpu_ptr = nullptr;
    }
    *n_samples_ptr = sampleCount;

    if((testbed.m_stream_training_data_from_CPU || testbed.m_stream_test_data_from_CPU) && (dataset_type == DatasetType::Training || dataset_type == DatasetType::Test))
    {
        const auto training_input_bytes_size = sampleCount * N_VOLUME_INPUT_DIMS * sizeof(float);
        const auto training_target_bytes_size = sampleCount * N_VOLUME_TARGET_DIMS * sizeof(float);
        CUDA_CHECK_THROW(cudaMallocHost((void **)input_cpu_ptr, training_input_bytes_size));
        CUDA_CHECK_THROW(cudaMallocHost((void**)target_cpu_ptr, training_target_bytes_size));
    } else {
        *input_cpu_ptr = new float[sampleCount * N_VOLUME_INPUT_DIMS];
        *target_cpu_ptr = new float[sampleCount * N_VOLUME_TARGET_DIMS];
    }

    vec3 min_bound = vec3(1e30f);
    vec3 max_bound = vec3(-1e30f);
    float minTMax = 1e30f;
    float maxTMax = -1e30f;
    float maxTAfter = -1e30f;

    auto& input_cpu_buffer = *input_cpu_ptr;
    auto &target_cpu_buffer = *target_cpu_ptr;
    std::vector<BinaryTrainingSample> binarySampleBuffer;
    binarySampleBuffer.resize(sampleCount);

    f.read(reinterpret_cast<char *>(binarySampleBuffer.data()),
        sizeof(BinaryTrainingSample) * sampleCount);


    if(dataset_type == DatasetType::Training || should_shuffle)
    {
        std::random_device rd;
        std::mt19937 shuffle_rng(rd());
        // default_rng_t shuffle_rng{testbed.m_seed};
        std::shuffle(binarySampleBuffer.begin(), binarySampleBuffer.end(), shuffle_rng);
    }

    bool normalize_data = (dataset_type == DatasetType::Training);

    auto processSample = [&](const BinaryTrainingSample& bs, int idx
        )
        {
            // Calculate scene BB to normalize rays
            vec3 rayO = vec3(bs.o[0], bs.o[1], bs.o[2]);

            constexpr float MAX_SCENE_DIST = 160.f;
            float tMaxVal = std::min(bs.tMax, MAX_SCENE_DIST);

            if(normalize_data)
            {
                min_bound = min(min_bound, rayO);
                max_bound = max(max_bound, rayO);
                minTMax = min(minTMax, tMaxVal);
                maxTMax = max(maxTMax, tMaxVal);
            }

            input_cpu_buffer[idx * N_VOLUME_INPUT_DIMS + POS_OFFSET + 0] = rayO.x;
            input_cpu_buffer[idx * N_VOLUME_INPUT_DIMS + POS_OFFSET + 1] = rayO.y;
            input_cpu_buffer[idx * N_VOLUME_INPUT_DIMS + POS_OFFSET + 2] = rayO.z;
            
            // Normalize direction vector and transform from [-1,1] to [0,1] range
            // SphericalHarmonics expects input in [0,1] and internally maps to [-1,1]
            vec3 rayD = vec3(bs.d[0], bs.d[1], bs.d[2]);
            float dirLen = std::sqrt(rayD.x * rayD.x + rayD.y * rayD.y + rayD.z * rayD.z);
            if (dirLen > 1e-6f) {
                rayD = rayD / dirLen;  // Normalize to unit length
            }
            // Map from [-1, 1] to [0, 1] for SphericalHarmonics encoding
            input_cpu_buffer[idx * N_VOLUME_INPUT_DIMS + DIR_OFFSET + 0] = rayD.x * 0.5f + 0.5f;
            input_cpu_buffer[idx * N_VOLUME_INPUT_DIMS + DIR_OFFSET + 1] = rayD.y * 0.5f + 0.5f;
            input_cpu_buffer[idx * N_VOLUME_INPUT_DIMS + DIR_OFFSET + 2] = rayD.z * 0.5f + 0.5f;
            
            input_cpu_buffer[idx * N_VOLUME_INPUT_DIMS + TMAX_OFFSET] = tMaxVal;


            //TODO: Check if i need to do some clamping here as well
            constexpr float epsilon = 1e-6f;
            constexpr float MAX_RADIANCE = 1000.f; //for now, just hard clamping this
            for (int c = 0; c < 3; ++c) {
                if(bs.beta_before_rgb[c] > epsilon)
                {
                    float val = std::max(0.f, bs.L_after_rgb[c]);
                    float L_clamped = std::min(MAX_RADIANCE, val / bs.beta_before_rgb[c]);
                    float L_log = std::log(L_clamped + 1.f);
                    target_cpu_buffer[idx * N_VOLUME_TARGET_DIMS + L_OFFSET + c] = L_log;
                } else {
                    target_cpu_buffer[idx * N_VOLUME_TARGET_DIMS + L_OFFSET + c] = 0.f;
                }
            }

            maxTAfter = std::max({maxTAfter, bs.T_after[0], bs.T_after[1], bs.T_after[2]});

            target_cpu_buffer[idx * N_VOLUME_TARGET_DIMS + T_OFFSET + 0] = bs.T_after[0];
            target_cpu_buffer[idx * N_VOLUME_TARGET_DIMS + T_OFFSET + 1] = bs.T_after[1];
            target_cpu_buffer[idx * N_VOLUME_TARGET_DIMS + T_OFFSET + 2] = bs.T_after[2];
    };

    for(uint64_t i = 0; i < sampleCount; ++i)
    {
        processSample(binarySampleBuffer[i], i);
    }

    tlog::success() << "Read " << sampleCount << " samples";
    CUDA_CHECK_THROW(cudaDeviceSynchronize());
    CUDA_CHECK_THROW(cudaGetLastError()); // Clear any prior errors

    tlog::info() << "Scene AABB: [" << min_bound.x << ", " << min_bound.y << ", "
                 << min_bound.z << "] to [" << max_bound.x << ", " << max_bound.y << ", "
                 << max_bound.z << "]";

    float scale, tMax_scale, tMax_offset;
    vec3 offset;

    // Compute and store normalization values
    if(dataset_type == DatasetType::Training)
    {
        // Compute scale and offset to fit in [0, 1]
        vec3 size = max_bound - min_bound;
        float max_dim = std::max({size.x, size.y, size.z});

        if(max_dim == 0) max_dim = 1.f;
        float padded_range = max_dim;

        scale = 1.f / padded_range;

        vec3 centre = (max_bound + min_bound) * 0.5f;
        offset = vec3(0.5f) - centre * scale;

        float tMax_range = maxTMax - minTMax;

        if(tMax_range == 0.f)
        {
            tMax_range = 1.f;
            tlog::warning() << "tMax range is 0, no normalization applied";
        }

        tMax_scale = 1.f / tMax_range;
        tMax_offset = minTMax;


        // Store normalization params in testbed
        testbed.m_volume_training_inputs_scale = scale;
        testbed.m_volume_training_inputs_offset = offset;
        testbed.m_volume_training_inputs_tMax_scale = tMax_scale;
        testbed.m_volume_training_inputs_tMax_offset = tMax_offset;
        testbed.m_volume_training_inputs_tAfter_scale = maxTAfter;

        tlog::info() << "Scene AABB: [" << min_bound.x << ", " << min_bound.y << ", "
                << min_bound.z << "] to [" << max_bound.x << ", " << max_bound.y << ", "
                << max_bound.z << "]";
        tlog::info() << "Auto-normalizing rays: scale=" << scale << " offset=[" << offset.x
                << ", " << offset.y << ", " << offset.z << "]";
        tlog::info() << "Auto-normalizing tmax: scale=" << tMax_scale << " with range=[ "
                << minTMax << ", " << maxTMax << " ]\n";
        tlog::info() << "Max T_after is: " << maxTAfter;

    }
    else
    {
        // Use stored normalization parameters from training data
        scale = testbed.m_volume_training_inputs_scale;
        offset = testbed.m_volume_training_inputs_offset;
        tMax_scale = testbed.m_volume_training_inputs_tMax_scale;
        tMax_offset = testbed.m_volume_training_inputs_tMax_offset;

        if(scale == 0.0f) {
            tlog::warning() << "Normalization parameters not set! Load training data first.";
        }
    }

    // Apply normalization
    for(uint64_t i = 0; i < sampleCount; ++i)
    {
        auto index = i * N_VOLUME_INPUT_DIMS;
        auto target_index = i * N_VOLUME_TARGET_DIMS;

        for(int dim = 0; dim < 3; ++dim)
        {
            const float val = input_cpu_buffer[index + POS_OFFSET + dim];
            input_cpu_buffer[index + POS_OFFSET + dim] = val * scale + offset[dim];
        }

        const float tMax_val = input_cpu_buffer[index + TMAX_OFFSET];
        input_cpu_buffer[index + TMAX_OFFSET] = (tMax_val - tMax_offset) * tMax_scale;

        for(int dim = 0; dim < 3; ++dim)
        {
            const float val = target_cpu_buffer[target_index + T_OFFSET + dim];
            target_cpu_buffer[target_index + T_OFFSET + dim] = val / maxTAfter;
        }
    }


    if((testbed.m_stream_training_data_from_CPU || testbed.m_stream_test_data_from_CPU) && (dataset_type == DatasetType::Training || dataset_type == DatasetType::Test))
    {
        const auto input_bytes_to_copy = (*n_samples_ptr) * N_VOLUME_INPUT_DIMS * sizeof(float);
        const auto target_bytes_to_copy = (*n_samples_ptr) * N_VOLUME_TARGET_DIMS * sizeof(float);

        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        bool space_in_vram = (input_bytes_to_copy + target_bytes_to_copy) < (free_mem - 4ULL * 1024 * 1024 * 1024);

        if(space_in_vram && false)
        {
            auto &input_gpu_buffer = *input_gpu_ptr;
            auto &target_gpu_buffer = *target_gpu_ptr;
        
        
            input_gpu_buffer.resize(input_bytes_to_copy);
            target_gpu_buffer.resize(target_bytes_to_copy);
            
            input_gpu_buffer.copy_from_host(input_cpu_buffer, input_bytes_to_copy);
            target_gpu_buffer.copy_from_host(target_cpu_buffer, target_bytes_to_copy);
            
            
            CUDA_CHECK_THROW(cudaDeviceSynchronize());
            tlog::success() << "Uploaded to GPU";
        }
        else
        {
            tlog::info() << "Training data too large for VRAM ( " << (input_bytes_to_copy + target_bytes_to_copy)/1e9 
                << " GB). Using CPU memory streaming instead.";
            
            input_gpu_ptr->resize(0);
            target_gpu_ptr->resize(0);
        }
    } else {
        const auto input_bytes_to_copy = (*n_samples_ptr) * N_VOLUME_INPUT_DIMS;
        const auto target_bytes_to_copy = (*n_samples_ptr) * N_VOLUME_TARGET_DIMS;
        
        auto &input_gpu_buffer = *input_gpu_ptr;
        auto &target_gpu_buffer = *target_gpu_ptr;
        
        
        input_gpu_buffer.resize(input_bytes_to_copy);
        target_gpu_buffer.resize(target_bytes_to_copy);
        
        input_gpu_buffer.copy_from_host(input_cpu_buffer, input_bytes_to_copy);
        target_gpu_buffer.copy_from_host(target_cpu_buffer, target_bytes_to_copy);
        
        
        CUDA_CHECK_THROW(cudaDeviceSynchronize());
        tlog::success() << "Uploaded to GPU";
    }
}

void load_dataset_to_testbed_spectral(Testbed& testbed, const fs::path& path, DatasetType dataset_type, bool should_shuffle)
{
    std::ifstream f{native_string(path), std::ios::in | std::ios::out | std::ios::binary };
    if(!f.is_open())
        throw std::runtime_error("Failed to open file");

    f.clear();
    f.seekg(0, std::ios::beg);

    uint64_t totalCount = 0;
    f.read(reinterpret_cast<char *>(&totalCount), sizeof(uint64_t));

    uint64_t sampleCount = std::min(totalCount, (uint64_t)UINT32_MAX);


    tlog::info() << "Loading " << sampleCount << " samples as " 
                    << (dataset_type == DatasetType::Training ? "training" : 
                        dataset_type == DatasetType::Validation ? "validation" : "test") 
                    << " data from " << path.str();

    float **input_cpu_ptr;
    float **target_cpu_ptr;
    uint64_t* n_samples_ptr;
    GPUMemory<float>* input_gpu_ptr;
    GPUMemory<float> *target_gpu_ptr;

    switch(dataset_type) {
        case DatasetType::Training:
            input_cpu_ptr = &testbed.m_volume_training_inputs_cpu;
            target_cpu_ptr = &testbed.m_volume_training_targets_cpu;
            n_samples_ptr = &testbed.m_n_volume_training_samples;
            input_gpu_ptr = &testbed.m_volume_training_inputs;
            target_gpu_ptr = &testbed.m_volume_training_targets;
            break;
        case DatasetType::Validation:
            input_cpu_ptr = &testbed.m_volume_validation_inputs_cpu;
            target_cpu_ptr = &testbed.m_volume_validation_targets_cpu;
            n_samples_ptr = &testbed.m_n_volume_validation_samples;
            input_gpu_ptr = &testbed.m_volume_validation_inputs;
            target_gpu_ptr = &testbed.m_volume_validation_targets;
            break;
        case DatasetType::Test:
            input_cpu_ptr = &testbed.m_volume_test_inputs_cpu;
            target_cpu_ptr = &testbed.m_volume_test_targets_cpu;
            n_samples_ptr = &testbed.m_n_volume_test_samples;
            input_gpu_ptr = &testbed.m_volume_test_inputs;
            target_gpu_ptr = &testbed.m_volume_test_targets;
            break;
        default:
            break;
        }

    if(*input_cpu_ptr) {
        delete[] *input_cpu_ptr;
        *input_cpu_ptr = nullptr;
    }
    if(*target_cpu_ptr) {
        delete[] *target_cpu_ptr;
        *target_cpu_ptr = nullptr;
    }
    *n_samples_ptr = sampleCount;

    if((testbed.m_stream_training_data_from_CPU || testbed.m_stream_test_data_from_CPU) && (dataset_type == DatasetType::Training || dataset_type == DatasetType::Test))
    {
        const auto training_input_bytes_size = sampleCount * N_VOLUME_INPUT_DIMS * sizeof(float);
        const auto training_target_bytes_size = sampleCount * N_VOLUME_TARGET_DIMS_SPECTRAL_TRANSMISSION_ONLY * sizeof(float);
        CUDA_CHECK_THROW(cudaMallocHost((void **)input_cpu_ptr, training_input_bytes_size));
        CUDA_CHECK_THROW(cudaMallocHost((void**)target_cpu_ptr, training_target_bytes_size));
    } else {
        *input_cpu_ptr = new float[sampleCount * N_VOLUME_INPUT_DIMS];
        *target_cpu_ptr = new float[sampleCount * N_VOLUME_TARGET_DIMS_SPECTRAL_TRANSMISSION_ONLY];
    }

    vec3 min_bound = vec3(1e30f);
    vec3 max_bound = vec3(-1e30f);
    float minTMax = 1e30f;
    float maxTMax = -1e30f;
    float maxTAfter = -1e30f;

    auto& input_cpu_buffer = *input_cpu_ptr;
    auto &target_cpu_buffer = *target_cpu_ptr;
    std::vector<BinaryTrainingSampleSpectral> binarySampleBuffer;
    binarySampleBuffer.resize(sampleCount);

    f.read(reinterpret_cast<char *>(binarySampleBuffer.data()),
        sizeof(BinaryTrainingSampleSpectral) * sampleCount);


    if(dataset_type == DatasetType::Training || should_shuffle)
    {
        std::random_device rd;
        std::mt19937 shuffle_rng(rd());
        // default_rng_t shuffle_rng{testbed.m_seed};
        std::shuffle(binarySampleBuffer.begin(), binarySampleBuffer.end(), shuffle_rng);
    }

    bool normalize_data = (dataset_type == DatasetType::Training);

    auto processSample = [&](const BinaryTrainingSampleSpectral& bs, int idx
        )
        {
            // Calculate scene BB to normalize rays
            vec3 rayO = vec3(bs.o[0], bs.o[1], bs.o[2]);

            constexpr float MAX_SCENE_DIST = 160.f;
            float tMaxVal = std::min(bs.tMax, MAX_SCENE_DIST);
            if(tMaxVal < 0)
            {
                tlog::error() << "Tmax value is less than 0! " << tMaxVal
                              << ". Setting to zero...";
                tMaxVal = 0.f;
            }
            if(normalize_data)
            {
                min_bound = min(min_bound, rayO);
                max_bound = max(max_bound, rayO);
                minTMax = min(minTMax, tMaxVal);
                maxTMax = max(maxTMax, tMaxVal);
            }

            input_cpu_buffer[idx * N_VOLUME_INPUT_DIMS + POS_OFFSET + 0] = rayO.x;
            input_cpu_buffer[idx * N_VOLUME_INPUT_DIMS + POS_OFFSET + 1] = rayO.y;
            input_cpu_buffer[idx * N_VOLUME_INPUT_DIMS + POS_OFFSET + 2] = rayO.z;
            
            // Normalize direction vector and transform from [-1,1] to [0,1] range
            // SphericalHarmonics expects input in [0,1] and internally maps to [-1,1]
            vec3 rayD = vec3(bs.d[0], bs.d[1], bs.d[2]);
            float dirLen = std::sqrt(rayD.x * rayD.x + rayD.y * rayD.y + rayD.z * rayD.z);
            if (dirLen > 1e-6f) {
                rayD = rayD / dirLen;  // Normalize to unit length
            }
            // Map from [-1, 1] to [0, 1] for SphericalHarmonics encoding
            input_cpu_buffer[idx * N_VOLUME_INPUT_DIMS + DIR_OFFSET + 0] = rayD.x * 0.5f + 0.5f;
            input_cpu_buffer[idx * N_VOLUME_INPUT_DIMS + DIR_OFFSET + 1] = rayD.y * 0.5f + 0.5f;
            input_cpu_buffer[idx * N_VOLUME_INPUT_DIMS + DIR_OFFSET + 2] = rayD.z * 0.5f + 0.5f;
            
            input_cpu_buffer[idx * N_VOLUME_INPUT_DIMS + TMAX_OFFSET] = tMaxVal;


            //TODO: Check if i need to do some clamping here as well
            constexpr float epsilon = 1e-6f;
            constexpr float MAX_RADIANCE = 1000.f; //for now, just hard clamping this
            // for (int c = 0; c < 3; ++c) {
            //     if(bs.beta_before_rgb[c] > epsilon)
            //     {
            //         float val = std::max(0.f, bs.L_after_rgb[c]);
            //         float L_clamped = std::min(MAX_RADIANCE, val / bs.beta_before_rgb[c]);
            //         float L_log = std::log(L_clamped + 1.f);
            //         target_cpu_buffer[idx * N_VOLUME_TARGET_DIMS + L_OFFSET + c] = L_log;
            //     } else {
            //         target_cpu_buffer[idx * N_VOLUME_TARGET_DIMS + L_OFFSET + c] = 0.f;
            //     }
            // }

            maxTAfter = std::max({maxTAfter, bs.T_after[0], bs.T_after[1], bs.T_after[2], bs.T_after[3]});

            target_cpu_buffer[idx * N_VOLUME_TARGET_DIMS_SPECTRAL_TRANSMISSION_ONLY + 0] = bs.T_after[0];
            target_cpu_buffer[idx * N_VOLUME_TARGET_DIMS_SPECTRAL_TRANSMISSION_ONLY + 1] = bs.T_after[1];
            target_cpu_buffer[idx * N_VOLUME_TARGET_DIMS_SPECTRAL_TRANSMISSION_ONLY + 2] = bs.T_after[2];
            target_cpu_buffer[idx * N_VOLUME_TARGET_DIMS_SPECTRAL_TRANSMISSION_ONLY + 3] = bs.T_after[3];
    };

    for(uint64_t i = 0; i < sampleCount; ++i)
    {
        processSample(binarySampleBuffer[i], i);
    }

    tlog::success() << "Read " << sampleCount << " samples";
    CUDA_CHECK_THROW(cudaDeviceSynchronize());
    CUDA_CHECK_THROW(cudaGetLastError()); // Clear any prior errors

    tlog::info() << "Scene AABB: [" << min_bound.x << ", " << min_bound.y << ", "
                 << min_bound.z << "] to [" << max_bound.x << ", " << max_bound.y << ", "
                 << max_bound.z << "]";

    float scale, tMax_scale, tMax_offset;
    vec3 offset;

    // Compute and store normalization values
    if(dataset_type == DatasetType::Training)
    {
        // Compute scale and offset to fit in [0, 1]
        vec3 size = max_bound - min_bound;
        float max_dim = std::max({size.x, size.y, size.z});

        if(max_dim == 0) max_dim = 1.f;
        float padded_range = max_dim;

        scale = 1.f / padded_range;

        vec3 centre = (max_bound + min_bound) * 0.5f;
        offset = vec3(0.5f) - centre * scale;

        float tMax_range = maxTMax - minTMax;

        if(tMax_range == 0.f)
        {
            tMax_range = 1.f;
            tlog::warning() << "tMax range is 0, no normalization applied";
        }

        tMax_scale = 1.f / tMax_range;
        tMax_offset = minTMax;


        // Store normalization params in testbed
        testbed.m_volume_training_inputs_scale = scale;
        testbed.m_volume_training_inputs_offset = offset;
        testbed.m_volume_training_inputs_tMax_scale = tMax_scale;
        testbed.m_volume_training_inputs_tMax_offset = tMax_offset;
        testbed.m_volume_training_inputs_tAfter_scale = maxTAfter;

        tlog::info() << "Scene AABB: [" << min_bound.x << ", " << min_bound.y << ", "
                << min_bound.z << "] to [" << max_bound.x << ", " << max_bound.y << ", "
                << max_bound.z << "]";
        tlog::info() << "Auto-normalizing rays: scale=" << scale << " offset=[" << offset.x
                << ", " << offset.y << ", " << offset.z << "]";
        tlog::info() << "Auto-normalizing tmax: scale=" << tMax_scale << " with range=[ "
                << minTMax << ", " << maxTMax << " ]\n";
        tlog::info() << "Max T_after is: " << maxTAfter;

    }
    else
    {
        // Use stored normalization parameters from training data
        scale = testbed.m_volume_training_inputs_scale;
        offset = testbed.m_volume_training_inputs_offset;
        tMax_scale = testbed.m_volume_training_inputs_tMax_scale;
        tMax_offset = testbed.m_volume_training_inputs_tMax_offset;
        maxTAfter = testbed.m_volume_training_inputs_tAfter_scale;

        if(scale == 0.0f) {
            tlog::warning() << "Normalization parameters not set! Load training data first.";
        }
    }

    // Apply normalization
    for(uint64_t i = 0; i < sampleCount; ++i)
    {
        auto index = i * N_VOLUME_INPUT_DIMS;
        auto target_index = i * N_VOLUME_TARGET_DIMS_SPECTRAL_TRANSMISSION_ONLY;

        for(int dim = 0; dim < 3; ++dim)
        {
            const float val = input_cpu_buffer[index + POS_OFFSET + dim];
            input_cpu_buffer[index + POS_OFFSET + dim] = val * scale + offset[dim];
        }



        const float tMax_val = input_cpu_buffer[index + TMAX_OFFSET];
        input_cpu_buffer[index + TMAX_OFFSET] = (tMax_val - tMax_offset) * tMax_scale;

        for(int dim = 0; dim < 3; ++dim)
        {
            const float val = target_cpu_buffer[target_index + T_OFFSET + dim];
            target_cpu_buffer[target_index + T_OFFSET + dim] = val / maxTAfter;
        }
    }

    if((testbed.m_stream_training_data_from_CPU || testbed.m_stream_test_data_from_CPU) && (dataset_type == DatasetType::Training || dataset_type == DatasetType::Test))
    {
        const auto input_bytes_to_copy =
            (*n_samples_ptr) * N_VOLUME_INPUT_DIMS * sizeof(float);
        const auto target_bytes_to_copy = (*n_samples_ptr) * N_VOLUME_TARGET_DIMS_SPECTRAL_TRANSMISSION_ONLY * sizeof(float);

        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        bool space_in_vram = (input_bytes_to_copy + target_bytes_to_copy) < (free_mem - 4ULL * 1024 * 1024 * 1024);

        if(space_in_vram && false)
        {
            auto &input_gpu_buffer = *input_gpu_ptr;
            auto &target_gpu_buffer = *target_gpu_ptr;
        
        
            input_gpu_buffer.resize(input_bytes_to_copy);
            target_gpu_buffer.resize(target_bytes_to_copy);
            
            input_gpu_buffer.copy_from_host(input_cpu_buffer, input_bytes_to_copy);
            target_gpu_buffer.copy_from_host(target_cpu_buffer, target_bytes_to_copy);
            
            
            CUDA_CHECK_THROW(cudaDeviceSynchronize());
            tlog::success() << "Uploaded to GPU";
        }
        else
        {
            tlog::info() << "Training data too large for VRAM ( " << (input_bytes_to_copy + target_bytes_to_copy)/1e9 
                << " GB). Using CPU memory streaming instead.";
            
            input_gpu_ptr->resize(0);
            target_gpu_ptr->resize(0);
        }
    } else {
        const auto input_bytes_to_copy = (*n_samples_ptr) * N_VOLUME_INPUT_DIMS;
        const auto target_bytes_to_copy = (*n_samples_ptr) * N_VOLUME_TARGET_DIMS_SPECTRAL_TRANSMISSION_ONLY;
        
        auto &input_gpu_buffer = *input_gpu_ptr;
        auto &target_gpu_buffer = *target_gpu_ptr;
        
        
        input_gpu_buffer.resize(input_bytes_to_copy);
        target_gpu_buffer.resize(target_bytes_to_copy);
        
        input_gpu_buffer.copy_from_host(input_cpu_buffer, input_bytes_to_copy);
        target_gpu_buffer.copy_from_host(target_cpu_buffer, target_bytes_to_copy);
        
        
        CUDA_CHECK_THROW(cudaDeviceSynchronize());
        tlog::success() << "Uploaded to GPU";
    }
}



void load_nerfdataset(Testbed& testbed, const fs::path& data_path)
{
    std::vector<fs::path> paths;
    if(data_path.is_directory())
    {
        for(const auto& path : fs::directory{data_path})
        {
            if(path.is_file() && equals_case_insensitive(path.extension(), "txt"))
            {
                paths.emplace_back(path);
            }
        }   
    }
    else if(equals_case_insensitive(data_path.extension(), "txt") || equals_case_insensitive(data_path.extension(), "bin"))
    {
        paths.emplace_back(data_path);
    }
    else
    {
        throw std::runtime_error{"Nerf data path must be text or directory"};
    }

    //TODO: Consider switching to outputting as jsons instead
    testbed.m_nerf.training.dataset.has_rays = true;
    // nerf_data.has_rays = true;
    // TODO: Extend this so it can take multiple files at once
    for(size_t i = 0; i < paths.size(); ++i)
    {
        // load_dataset_to_testbed(testbed, paths[i], DatasetType::Training);
        if(testbed.m_volume_training_spectral_only)
        {
            load_dataset_to_testbed_spectral(testbed, paths[i], DatasetType::Training);
        }
        else
        {
            load_dataset_to_testbed(testbed, paths[i], DatasetType::Training);
        }
    }
    return;
}

#ifdef PBRT_GPU_BUILD_RENDERER

/**
*
*Focusing on your Instant-NGP workflow, here’s the plan: outline how to feed your per-ray radiance data into the standard hash-encoding training loop, then sketch how to reuse the encoder with a separate fully connected head.
*
*Training Instant-NGP’s multiresolution hash encoding with your ray/radiance pairs
*Prepare data: For each camera ray, store its origin, direction, (optional) timestamp, and ground-truth radiance RGB from your file. Instant-NGP expects either images plus camera intrinsics/extrinsics or explicit samples. You can adapt the data_loader to ingest your ray-wise dataset (e.g., convert to the .msgpack format with per-ray metadata or extend the NeRF training code to read your CSV/binary).
*Configure training: Launch the Instant-NGP NeRF mode, but disable volume density supervision if you already have final radiance. Instead, treat your data as direct RGB targets by:
*Feeding each sample’s 3D position and view direction through the multires hash encoding (Instant-NGP’s default).
*Using the built-in tiny MLP decoder to predict RGB.
*Applying an L2 or sRGB-space loss between predicted and ground-truth radiance. You can modify testbed.cu to skip volumetric accumulation and just train on “single-hit” samples.
*Workflow steps:
*Export your dataset to Instant-NGP’s expected format (positions, directions, RGB, optional exposure).
*Run instant-ngp with --mode nerf --scene <your_data>.
*Monitor training; the hash encoding plus tiny MLP will learn a mapping from ray params to radiance directly.
*Using Instant-NGP as an encoder plus an extra fully connected network
*Concept: Treat the multiresolution hash grid + first few MLP layers as a feature extractor. Freeze or fine-tune it, then attach a deeper fully connected head tailored to your radiance prediction (e.g., more layers, custom activation, uncertainty outputs).
*Implementation sketch:
*Modify Instant-NGP’s network config to expose the hash encoding output (encoding->n_output_dims()).
*Build a new FC module (e.g., in network.cu) that takes the encoded features and optional auxiliary inputs (material ID, viewing angle) and predicts RGB.
*During training, forward pass = encode inputs → feed to FC head → compute loss vs ground truth.
*Optionally pretrain the hash encoder using standard Instant-NGP, then load its weights and train only the FC head (or fine-tune both jointly).
*Training loop: identical data pipeline, but your optimizer updates both the hash-table entries and the FC head unless you freeze one. Instant-NGP already supports custom network configs via descriptors *.jsonyou can define a “composite” network where the “encoding” is Instant-NGP’s hash grid and the “network” block is your extended MLP.
*This gives you a high-level recipe both for straight Instant-NGP training on your ray/radiance dumps and for a hybrid approach with an additional prediction head.
*
*
*
*
*
*Updating task completion status
*Starting: Describe hybrid hash+MLP setup (3/3)
*
*Completing todo3 description
*Completed: Describe hybrid hash+MLP setup (3/3)
*
*Acknowledging truncation and considering summary
*Picking up where the message clipped:
*
*Multires hash training recap – Convert each camera ray (origin, direction, optional time/exposure) plus its ground-truth RGB into Instant-NGP’s dataset format (e.g., msgpack). Run NeRF mode, but swap the volumetric loss for a direct RGB regression: ray params → hash encoding → built-in tiny MLP → predicted radiance → L2/Huber loss to the recorded RGB. You can either extend the existing data loader or emit pseudo-images with matching intrinsics/extrinsics so Instant-NGP’s default loader consumes them unchanged.
*
*Hash encoder + extra fully connected head – Treat Instant-NGP’s multires hash grid as a learnable feature extractor. Capture its encoded feature vector (and optionally the first tiny-MLP layer output), then feed that into your own deeper fully connected network:
*
*Modify the network config JSON so the “encoding” block is the hash grid and the “network” block references a new FC module you add in network.cu.
*Forward pass: encode 3D position and view direction → concatenate any auxiliary inputs (BRDF tags, sample depth, etc.) → run through your FC layers to emit RGB (and whatever else you need, like uncertainty).
*Training: optimize both the hash table entries and the FC weights jointly, or pretrain the hash encoder with standard Instant-NGP and freeze it while you train only the FC head.
*This gives you the same ultra-fast spatial encoding while letting a richer head model the final radiance mapping.
*
*
*
*
*
*
*
*
*
*
*
*
*/

#include <neural-graphics-primitives/testbed.h>
#include <neural-graphics-primitives/nerf_loader.h>
#include <tiny-cuda-nn/loss.h>
#include <tiny-cuda-nn/reduce_sum.h>
#include <args/args.hxx>
#include <tiny-cuda-nn/common.h>
#include <memory>
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <fstream>
#include <filesystem/directory.h>
#include <random>
#include <numeric>
using namespace tcnn;
using namespace args;
using namespace ngp;

// TODO(main-nn-mirror-plan):
// 1. Recreate the dataset ingestion pipeline from instant-ngp's load_nerf:
//    - Parse your per-ray exports (origin, direction, metadata, radiance) and fill Ray arrays + float RGB buffers.
//    - Normalize rays via result.nerf_ray_to_ngp equivalent so they align with the hash grid coordinate frame.
// 2. Mirror NerfDataset plumbing:
//    - Allocate metadata/pixelmemory/raymemory vectors, set n_images based on your chunking, and call set_training_image.
//    - Populate n_extra_learnable_dims / per-frame metadata for any auxiliary scalars you plan to feed the network.
// 3. Hook into Testbed/TestbedNerf training loop:
//    - Ensure has_rays=true so batching pulls explicit rays instead of regenerating from cameras.
//    - Pipe your radiance buffer into target_rgbs so the loss compares network output to your supervised values directly.
// 4. Extend network config if needed:
//    - Adjust encoding dims or attach the extra FC head before backprop to accommodate additional metadata channels.




//Adapting the loader to treat each (position, ray, metadata) tuple with a known radiance as the training sample involves three main moves: redefine the dataset you emit from load_nerf, flow the new attributes through the NerfDataset/Testbed plumbing, and teach the training step to build network inputs directly from those per-ray payloads instead of camera poses.
//
//Data Model Shift (nerf_loader.cu + format tooling)
//
//Keep the existing JSON envelope but point each frame at your precomputed ray dump instead of RGB images; you can still emit one pseudo-frame per chunk so you reuse the threading and progress logic.
//Populate LoadedImageInfo::rays for every pixel with a struct that already contains the start position, direction, and any per-ray extras (Ray currently holds o, d, l, t, cone_angle, pdf). If you need more fields (e.g., BRDF roughness, wavelength), extend the Ray definition in include/neural-graphics-primitives/nerf_loader.h and the downstream CUDA structs (nerf_training.cuh) so the GPU kernels can read them.
//Store the ground-truth radiance directly in the pixel buffer: instead of loading PNGs, serialize your float RGB (or spectral) into an EXR-equivalent block and set image_type = EImageDataType::Float so set_training_image uploads it untouched. If you do not have a texture per ray, you can bypass image loading entirely by synthesizing a buffer and calling result.set_training_image manually with your radiance array.
//Metadata & Extra Inputs (NerfDataset, TrainingImageMetadata)
//
//Use result.n_extra_learnable_dims and metadata[i].light_dir/extra_dims_gpu to feed side-channel values. The NeRF pipeline already supports “appearanceper image; you can repurpose them to carry arbitrary per-ray scalars by writing into TrainingImageMetadata::extra_metadata and bumping n_extra_learnable_dims. embeddings
//When you call result.nerf_ray_to_ngp(dst.rays[px]), append your extra metadata conversion there (or add a sibling helper) so all rays are normalized to the hash grid’s coordinate system before they reach CUDA.
//Training Pipeline Touchpoints (testbed_nerf.cu, nerf_training.cuh)
//
//Testbed::load_training_data ultimately invokes train_nerf_accumulate_gradients with batches drawn from m_training_data.rays_*. Ensure m_training_data.has_rays stays true so the sampling path uses your explicit origins/directions rather than regenerating rays from camera intrinsics.
//In generate_training_batch_nerf (inside testbed_nerf.cu), tweak the packing of NerfPosition so it copies your per-ray metadata into NerfNetworkInput::extra_dims (there is already logic guarded by n_extra_learnable_dims). This is where you can, for example, stuff ray differentials, time, or surface parameters that the hash encoding should see.
//Because you already know the target radiance, disable exposure/tonemapping steps by setting m_nerf.training.dataset.is_hdr = true and bypassing tonemap when copying pixels; otherwise the loader will convert your floats to 8-bit linear.
//If you want to skip the notion of “frames” entirely, consider creating a parallel loader (e.g., load_raybundle_dataset) that returns a NerfDataset with n_images = num_chunks and metadata[i].resolution = {N,1} so every chunk is just a flat list of rays. The rest of the pipeline treats it identically.
//Ground Truth Integration & Loss
//
//Since the network now trains on precomputed radiance rather than rendered pixels, make sure Testbed::loss_fn_nerf still compares predicted RGB against target_rgbs read from your float buffer; no change needed if you kept the pixel pathway.
//If radiance is spectral or multi-channel, increase NerfPayload::DIM_COL and adjust network_config.encoding.n_dims_to_encode so the MLP output matches your channel count.
//Once these hooks are in place, you can iteratively phase out camera-derived math: set dummy transforms in the JSON, keep enable_ray_loading = true, and rely entirely on your ray bundles + radiance arrays. Next steps could be (1) design a compact binary that packs {origin, direction, metadata, radiance}, (2) extend the Ray struct and GPU kernels to read it, and (3) add a bespoke loader entry point (CLI flag or new TestbedMode) so you don’t have to spoof NeRF JSON forever.


//!TODO: List & Recommendations
//Here is the list of things you must do to ensure correct training:
//
//1. Normalize Your Latent Parameters (CRITICAL)
//You need to scale sampleIdx and finalDepth to the [0, 1] range.
//
//Why: To prevent numerical instability.
//How:
//Find the maximum sample count (e.g., max_samples) and maximum depth (e.g., max_depth) in your dataset.
//In main_nn.cu, when populating the buffers, convert the values to float and divide by the maximums.
//Action: Change TrainingImageMetadata pointers to const float* instead of const int*, and perform the normalization in main_nn.cu.
//2. Verify Network Input Dimensions
//Ensure the network actually sees these dimensions.
//
//How: In testbed.cu, look for the log output starting with Color model:. It should show something like ... + 16 + 2 --> ... (where 2 is your extra dims).
//Action: Run the program and check the console output.
//3. Check "One-Blob" Encoding
//If you are using the default "OneBlob" encoding for the extra dimensions (which is common for latent codes), it expects inputs in [0, 1].
//
//Action: Ensure your normalized values are strictly within [0, 1].
//4. Validate Data Alignment
//Ensure that sampleIdx and finalDepth actually align with the rays and rgbas.
//
//How: In main_nn.cu, you are grouping rays by sampleIdx. Ensure that frameRayData construction preserves the 1:1 mapping between rays[i] and sampleIndices[i]. (Your current code looks correct for this, but double-check your parsing logic).
//**Recommended Fix for Normalization (
//in main_nn.cu)**
//
//I recommend modifying main_nn.cu to normalize these values before uploading them.

struct __attribute__((packed)) BinaryTrainingSample 
{
    float o[3];
    float d[3];
    float beta_before_rgb[3];
    float L_after_rgb[3];
    float T_after[3];
    float tMax;
};


enum class DatasetType
{
    Training,
    Validation,
    Test
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
            
            constexpr float MAX_SCENE_DIST = 50.f;
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

            constexpr float MAX_SCENE_DIST = 50.f;
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

void load_dataset_to_testbed(Testbed& testbed, const fs::path& path, DatasetType dataset_type)
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

            constexpr float MAX_SCENE_DIST = 50.f;
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

            target_cpu_buffer[idx * N_VOLUME_TARGET_DIMS + T_OFFSET + 0] = std::min(1.f, std::max(0.f, bs.T_after[0]));
            target_cpu_buffer[idx * N_VOLUME_TARGET_DIMS + T_OFFSET + 1] = std::min(1.f, std::max(0.f, bs.T_after[1]));
            target_cpu_buffer[idx * N_VOLUME_TARGET_DIMS + T_OFFSET + 2] = std::min(1.f, std::max(0.f, bs.T_after[2]));
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
        
        for(int dim = 0; dim < 3; ++dim)
        {
            const float val = input_cpu_buffer[index + POS_OFFSET + dim];
            input_cpu_buffer[index + POS_OFFSET + dim] = val * scale + offset[dim];
        }

        const float tMax_val = input_cpu_buffer[index + TMAX_OFFSET];
        input_cpu_buffer[index + TMAX_OFFSET] = (tMax_val - tMax_offset) * tMax_scale;
    }


    if((testbed.m_stream_training_data_from_CPU || testbed.m_stream_test_data_from_CPU) && (dataset_type == DatasetType::Training || dataset_type == DatasetType::Test))
    {
        const auto input_bytes_to_copy = (*n_samples_ptr) * N_VOLUME_INPUT_DIMS * sizeof(float);
        const auto target_bytes_to_copy = (*n_samples_ptr) * N_VOLUME_TARGET_DIMS * sizeof(float);

        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        bool space_in_vram = (input_bytes_to_copy + target_bytes_to_copy) < (free_mem - 4ULL * 1024 * 1024 * 1024);

        if(space_in_vram)
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

void load_training_to_testbed(Testbed& testbed, const fs::path& path)
{
    std::ifstream f{native_string(path), std::ios::in | std::ios::out | std::ios::binary };
    if(!f.is_open())
        throw std::runtime_error("Failed to open file");

    testbed.m_n_volume_training_samples = 0;
    f.clear();
    f.seekg(0, std::ios::beg);

    uint64_t totalCount = 0;
    f.read(reinterpret_cast<char *>(&totalCount), sizeof(uint64_t));



    uint64_t trainCount = (uint64_t)(0.8f * totalCount);
    uint64_t testAndValidationCount = totalCount - trainCount;
    uint64_t testCount = (uint64_t)(0.5f * testAndValidationCount);
    uint64_t validationCount = testAndValidationCount - testCount;

    testbed.m_n_volume_training_samples = std::min(trainCount, (uint64_t)UINT32_MAX);
    testbed.m_n_volume_validation_samples = std::min(validationCount, (uint64_t)UINT32_MAX);
    testbed.m_n_volume_test_samples = std::min(testCount, (uint64_t)UINT32_MAX);
    
    if(testbed.m_stream_training_data_from_CPU)
    {
        testbed.m_volume_training_inputs_cpu = new float[testbed.m_n_volume_training_samples * N_VOLUME_INPUT_DIMS];
        testbed.m_volume_training_targets_cpu = new float[testbed.m_n_volume_training_samples * N_VOLUME_TARGET_DIMS];
        
        testbed.m_volume_validation_inputs_cpu = new float[testbed.m_n_volume_validation_samples * N_VOLUME_INPUT_DIMS];
        testbed.m_volume_validation_targets_cpu = new float[testbed.m_n_volume_validation_samples * N_VOLUME_TARGET_DIMS];
        
        testbed.m_volume_test_inputs_cpu = new float[testbed.m_n_volume_test_samples * N_VOLUME_INPUT_DIMS];
        testbed.m_volume_test_targets_cpu = new float[testbed.m_n_volume_test_samples * N_VOLUME_TARGET_DIMS];
    }
    else
    {
        //TODO: Add functionality to extend this to also copy over any samples already loaded if i want to load multiple files
        // Currently just allocates new float buffer
        testbed.m_volume_training_inputs_cpu = new float[testbed.m_n_volume_training_samples * N_VOLUME_INPUT_DIMS];
        testbed.m_volume_training_targets_cpu = new float[testbed.m_n_volume_training_samples * N_VOLUME_TARGET_DIMS];
        
        testbed.m_volume_validation_inputs_cpu = new float[testbed.m_n_volume_validation_samples * N_VOLUME_INPUT_DIMS];
        testbed.m_volume_validation_targets_cpu = new float[testbed.m_n_volume_validation_samples * N_VOLUME_TARGET_DIMS];
        
        testbed.m_volume_test_inputs_cpu = new float[testbed.m_n_volume_test_samples * N_VOLUME_INPUT_DIMS];
        testbed.m_volume_test_targets_cpu = new float[testbed.m_n_volume_test_samples * N_VOLUME_TARGET_DIMS];
    }

    const uint64_t trainEnd = trainCount;
    const uint64_t validationEnd = trainCount + validationCount;

    if(trainCount == 0)
        return;

    
    auto &input_cpu_buffer = testbed.m_volume_training_inputs_cpu;
    auto &target_cpu_buffer = testbed.m_volume_training_targets_cpu;

    vec3 min_bound = vec3(1e30f);
    vec3 max_bound = vec3(-1e30f);
    float minTMax = 1e30f;
    float maxTMax = -1e30f;

    std::vector<BinaryTrainingSample> binarySampleBuffer;
    binarySampleBuffer.resize(totalCount);

    f.read(reinterpret_cast<char *>(binarySampleBuffer.data()),
           sizeof(BinaryTrainingSample) * totalCount);

    // Shuffling sample buffer so its random distribution (the way its logged is correlated to how it samples pixels row by row)
    std::random_device rd;
    std::mt19937 shuffle_rng(rd());
    // default_rng_t shuffle_rng{testbed.m_seed};
    std::shuffle(binarySampleBuffer.begin(), binarySampleBuffer.end(), shuffle_rng);


    auto processSample = [](const BinaryTrainingSample& bs, 
        float* input_buf, float* target_buf,
        int idx, vec3& min_bound, vec3& max_bound, float& minTMax, float& maxTMax
        )
        {
            // Calculate scene BB to normalize rays
            vec3 rayO = vec3(bs.o[0], bs.o[1], bs.o[2]);
            min_bound = min(min_bound, rayO);
            max_bound = max(max_bound, rayO);

            constexpr float MAX_SCENE_DIST = 50.f;
            float tMaxVal = std::min(bs.tMax, MAX_SCENE_DIST);

            minTMax = min(minTMax, tMaxVal);
            maxTMax = max(maxTMax, tMaxVal);

            input_buf[idx * N_VOLUME_INPUT_DIMS + POS_OFFSET + 0] = rayO.x;
            input_buf[idx * N_VOLUME_INPUT_DIMS + POS_OFFSET + 1] = rayO.y;
            input_buf[idx * N_VOLUME_INPUT_DIMS + POS_OFFSET + 2] = rayO.z;
            
            // Normalize direction vector and transform from [-1,1] to [0,1] range
            // SphericalHarmonics expects input in [0,1] and internally maps to [-1,1]
            vec3 rayD = vec3(bs.d[0], bs.d[1], bs.d[2]);
            float dirLen = std::sqrt(rayD.x * rayD.x + rayD.y * rayD.y + rayD.z * rayD.z);
            if (dirLen > 1e-6f) {
                rayD = rayD / dirLen;  // Normalize to unit length
            }
            // Map from [-1, 1] to [0, 1] for SphericalHarmonics encoding
            input_buf[idx * N_VOLUME_INPUT_DIMS + DIR_OFFSET + 0] = rayD.x * 0.5f + 0.5f;
            input_buf[idx * N_VOLUME_INPUT_DIMS + DIR_OFFSET + 1] = rayD.y * 0.5f + 0.5f;
            input_buf[idx * N_VOLUME_INPUT_DIMS + DIR_OFFSET + 2] = rayD.z * 0.5f + 0.5f;
            
            input_buf[idx * N_VOLUME_INPUT_DIMS + TMAX_OFFSET] = tMaxVal;


            //TODO: Check if i need to do some clamping here as well
            constexpr float epsilon = 1e-6f;
            constexpr float MAX_RADIANCE = 1000.f; //for now, just hard clamping this
            for (int c = 0; c < 3; ++c) {
                if(bs.beta_before_rgb[c] > epsilon)
                {
                    float val = std::max(0.f, bs.L_after_rgb[c]);
                    float L_clamped = std::min(MAX_RADIANCE, val / bs.beta_before_rgb[c]);
                    float L_log = std::log(L_clamped + 1.f);
                    target_buf[idx * N_VOLUME_TARGET_DIMS + L_OFFSET + c] = L_log;
                } else {
                    target_buf[idx * N_VOLUME_TARGET_DIMS + L_OFFSET + c] = 0.f;
                }
            }

            target_buf[idx * N_VOLUME_TARGET_DIMS + T_OFFSET + 0] = std::min(1.f, std::max(0.f, bs.T_after[0]));
            target_buf[idx * N_VOLUME_TARGET_DIMS + T_OFFSET + 1] = std::min(1.f, std::max(0.f, bs.T_after[1]));
            target_buf[idx * N_VOLUME_TARGET_DIMS + T_OFFSET + 2] = std::min(1.f, std::max(0.f, bs.T_after[2]));
            
        };

    for(uint64_t i = 0; i < trainEnd; ++i)
    {
        processSample(binarySampleBuffer[i], testbed.m_volume_training_inputs_cpu,
            testbed.m_volume_training_targets_cpu, i, min_bound, max_bound, minTMax, maxTMax
        );
    }

    vec3 dummy_min_bound = vec3(0.0);
    vec3 dummy_max_bound = vec3(0.0);
    float dummy_tmax = 0.f, dummy_tMin = 0.f;

    for(uint64_t i = trainEnd; i < validationEnd; ++i)
    {
        uint64_t localIdx = i - trainEnd;
        processSample(binarySampleBuffer[i], testbed.m_volume_validation_inputs_cpu,
            testbed.m_volume_validation_targets_cpu, localIdx, dummy_min_bound, dummy_max_bound, dummy_tMin, dummy_tmax
        );
    }

    for (uint64_t i = validationEnd; i < totalCount; ++i)
    {
        uint64_t localIdx = i - validationEnd;
        processSample(binarySampleBuffer[i], testbed.m_volume_test_inputs_cpu, 
            testbed.m_volume_test_targets_cpu, localIdx, dummy_min_bound, dummy_max_bound, dummy_tMin, dummy_tmax
        );
    }

        tlog::success() << "Read " << totalCount << " samples \n";
    tlog::success() << "Total Training, Validation and Test Count: [ " << trainCount
                    << ", " << validationCount << ", " << testCount << " ]";
    CUDA_CHECK_THROW(cudaDeviceSynchronize());
    CUDA_CHECK_THROW(cudaGetLastError()); // Clear any prior errors



    // Compute scale and offset to fit in [0, 1]
    tlog::info() << "Scene AABB: [" << min_bound.x << ", " << min_bound.y << ", "
                 << min_bound.z << "] to [" << max_bound.x << ", " << max_bound.y << ", "
                 << max_bound.z << "]";

    vec3 size = max_bound - min_bound;
    float max_dim = std::max({size.x, size.y, size.z});

    if(max_dim == 0)
        max_dim = 1.0f;

    float scale = 1.0f / max_dim;

    vec3 centre = (max_bound + min_bound) * 0.5f;
    vec3 offset = vec3(0.5f) - centre * scale;


    tlog::info() << "Auto-normalizing rays: scale=" << scale << " offset=[" << offset.x
                << ", " << offset.y << ", " << offset.z << "]";
    
    auto &validation_input_cpu_buffer = testbed.m_volume_validation_inputs_cpu;
    auto &validation_target_cpu_buffer = testbed.m_volume_validation_targets_cpu;
    auto& test_input_cpu_buffer = testbed.m_volume_test_inputs_cpu;
    auto& test_target_cpu_buffer = testbed.m_volume_test_targets_cpu;

    for(int i = 0; i < trainCount; ++i)
    {
        auto index = i * N_VOLUME_INPUT_DIMS + POS_OFFSET;
        for(int dim = 0; dim < 3; ++dim)
        {
            const float val = input_cpu_buffer[index + dim];
            input_cpu_buffer[index + dim] = val * scale + offset[dim];
        }
    }


    for (int i = 0; i < validationCount; ++i) {
        auto index = i * N_VOLUME_INPUT_DIMS + POS_OFFSET;
        for(int dim = 0; dim < 3; ++dim)
        {
            const float val = validation_input_cpu_buffer[index + dim];
            validation_input_cpu_buffer[index + dim] = val * scale + offset[dim];
        }
    }

    for (int i = 0; i < testCount; ++i)
    {
        auto index = i * N_VOLUME_INPUT_DIMS + POS_OFFSET;
        for(int dim = 0; dim < 3; ++dim)
        {
            const float val = test_input_cpu_buffer[index + dim];
            test_input_cpu_buffer[index + dim] = val * scale + offset[dim];
        }
    }

    testbed.m_volume_training_inputs_scale = scale;
    testbed.m_volume_training_inputs_offset = offset;

    // Compute tMax normalization
    float tMax_range = maxTMax - minTMax;
    if(tMax_range == 0.0f)
    {
        tMax_range = 1.0f;
        tlog::error() << "tmax range is 0. This will result in NO normalization...";
    }

    float tMax_scale = 1.0f / tMax_range;
    for (int i = 0; i < trainCount; ++i) {
        auto index = i * N_VOLUME_INPUT_DIMS + TMAX_OFFSET;
        const float val = input_cpu_buffer[index];
        input_cpu_buffer[index] = (val - minTMax) * tMax_scale;
    }

    for (int i = 0; i < validationCount; ++i) {
        auto index = i * N_VOLUME_INPUT_DIMS + TMAX_OFFSET;
        const float val = validation_input_cpu_buffer[index];
        validation_input_cpu_buffer[index] = (val - minTMax) * tMax_scale;
    }

    for (int i = 0; i < testCount; ++i) {
        auto index = i * N_VOLUME_INPUT_DIMS + TMAX_OFFSET;
        const float val = test_input_cpu_buffer[index];
        test_input_cpu_buffer[index] = (val - minTMax) * tMax_scale;
    }


    tlog::info() << "Auto-normalizing tmax: scale=" << tMax_scale << " with range=[ "
         << minTMax << ", " << maxTMax << " ]\n";

    testbed.m_volume_training_inputs_tMax_scale = tMax_scale;
    testbed.m_volume_training_inputs_tMax_offset = minTMax;

    // Load into GPU buffers
    const auto input_bytes_to_copy =
        testbed.m_n_volume_training_samples * N_VOLUME_INPUT_DIMS;
    const auto target_bytes_to_copy =
        testbed.m_n_volume_training_samples * N_VOLUME_TARGET_DIMS;
    auto &input_gpu_buffer = testbed.m_volume_training_inputs;
    auto &target_gpu_buffer = testbed.m_volume_training_targets;

    input_gpu_buffer.resize(input_bytes_to_copy);
    target_gpu_buffer.resize(target_bytes_to_copy);

    input_gpu_buffer.copy_from_host(input_cpu_buffer, input_bytes_to_copy);
    target_gpu_buffer.copy_from_host(target_cpu_buffer, target_bytes_to_copy);


    const auto validation_input_bytes_to_copy =
        testbed.m_n_volume_validation_samples * N_VOLUME_INPUT_DIMS;
    const auto validation_target_bytes_to_copy =
        testbed.m_n_volume_validation_samples * N_VOLUME_TARGET_DIMS;
    auto &validation_input_gpu_buffer = testbed.m_volume_validation_inputs;
    auto &validation_target_gpu_buffer = testbed.m_volume_validation_targets;

    validation_input_gpu_buffer.resize(validation_input_bytes_to_copy);
    validation_target_gpu_buffer.resize(validation_target_bytes_to_copy);

    validation_input_gpu_buffer.copy_from_host(validation_input_cpu_buffer, validation_input_bytes_to_copy);
    validation_target_gpu_buffer.copy_from_host(validation_target_cpu_buffer, validation_target_bytes_to_copy);


    const auto test_input_bytes_to_copy =
        testbed.m_n_volume_test_samples * N_VOLUME_INPUT_DIMS;
    const auto test_target_bytes_to_copy =
        testbed.m_n_volume_test_samples * N_VOLUME_TARGET_DIMS;
    auto &test_input_gpu_buffer = testbed.m_volume_test_inputs;
    auto &test_target_gpu_buffer = testbed.m_volume_test_targets;

    test_input_gpu_buffer.resize(test_input_bytes_to_copy);
    test_target_gpu_buffer.resize(test_target_bytes_to_copy);

    test_input_gpu_buffer.copy_from_host(test_input_cpu_buffer, test_input_bytes_to_copy);
    test_target_gpu_buffer.copy_from_host(test_target_cpu_buffer, test_target_bytes_to_copy);
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
        load_dataset_to_testbed(testbed, paths[i], DatasetType::Training);
    }
    return;
}

Testbed::ValidationTestResults evaluate_pure_loss(Testbed& testbed, const GPUMemory<float>& inputs, const GPUMemory<float>& targets, size_t n_samples) {
    if(!testbed.m_network) return {-1.f, -1.f};
    
    cudaStream_t stream = 0; 
    const uint32_t padded_output = testbed.m_network->padded_output_width();
    const uint32_t max_batch_size = 1 << 18; // 256k batch

    GPUMemory<float> loss_values(max_batch_size * padded_output);
    GPUMemory<network_precision_t> predictions(max_batch_size * padded_output);
    GPUMemory<network_precision_t> dummy_gradients(max_batch_size * padded_output);

    const uint32_t n_batches = (uint32_t)tcnn::div_round_up(n_samples, (uint64_t)max_batch_size);

    float total_loss = 0.f;
    uint64_t total_samples = 0;

    for (uint32_t batch_idx = 0; batch_idx < n_batches; ++batch_idx) {
        uint64_t offset = (uint64_t)batch_idx * max_batch_size;
        uint32_t batch_size = (uint32_t)std::min((uint64_t)max_batch_size, n_samples - offset);
        
        // Ensure batch size is granularity aligned
        batch_size = (batch_size / tcnn::BATCH_SIZE_GRANULARITY) * tcnn::BATCH_SIZE_GRANULARITY;

        if (batch_size == 0) continue;

        GPUMatrix<float> input_matrix((float*)inputs.data() + offset * N_VOLUME_INPUT_DIMS, N_VOLUME_INPUT_DIMS, batch_size);
        GPUMatrix<float> target_matrix((float*)targets.data() + offset * N_VOLUME_TARGET_DIMS, N_VOLUME_TARGET_DIMS, batch_size);
        
        GPUMatrix<network_precision_t> predictions_matrix(predictions.data(), padded_output, batch_size);
        GPUMatrix<float> loss_values_matrix(loss_values.data(), padded_output, batch_size);
        GPUMatrix<network_precision_t> dummy_gradients_matrix(dummy_gradients.data(), padded_output, batch_size);

        testbed.m_network->inference_mixed_precision(stream, input_matrix, predictions_matrix, true);
        testbed.m_loss->evaluate(stream, 1.f, predictions_matrix, target_matrix, loss_values_matrix, dummy_gradients_matrix, nullptr);

        float batch_loss = tcnn::reduce_sum(loss_values.data(), batch_size * padded_output, stream);
        
        total_loss += batch_loss * (batch_size * N_VOLUME_TARGET_DIMS);
        total_samples += batch_size;
    }

    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
    if (total_samples == 0) return {-1.f, -1.f};
    Testbed::ValidationTestResults results;

    float mse = total_loss / (total_samples * N_VOLUME_TARGET_DIMS);
	float psnr = -10.f * std::log10(std::max(mse, 1e-10f));
    // MSE calculation
    return {mse, psnr};
}


int main(int argc, char** argv)
{
    std::vector<std::string> arguments;
    try {
        for (int i = 0; i < argc; ++i) {
#ifdef _WIN32
            arguments.emplace_back(ngp::utf16_to_utf8(argv[i]));
#else
            arguments.emplace_back(argv[i]);
#endif
        }
    }   catch (const std::exception& e) {
		tlog::error() << "Uncaught exception: " << e.what();
		return 1;
	}

    

    ArgumentParser parser{
        "Instant Neural Graphics Primitives\n"
        "Version " NGP_VERSION,
        "",
	};

    ValueFlag<std::string> model_flag{
        parser, 
        "MODEL_PATH",
        "Model path to load",
        {
            "model-path"
        },
    };

    ValueFlag<std::string> data_flag{
        parser, "TRAINING_DATA_PATH", "Path to training data binary file", {"data-path"}};

        
    ValueFlag<int> train_epoch_flag
    {
        parser,
        "TRAINING_ITERATIONS",
        "How many epochs to train for (default 40k batch iterations)",
        {
            "n-epochs"
        }
    };
    
    Flag validation_flag
    {
        parser,
        "SHOULD_VALIDATE",
        "Should do validation testing afterwards",
        {
            "validate"
        }
    };
        
    ValueFlag<std::string> validation_path_flag
    {
        parser,
        "VALIDATION_PATH",
        "Path to Validation Data file",
        {
            "validation-file-path"
        }
    };

    ValueFlag<std::string> test_path_flag
    {
        parser,
        "TEST_PATH",
        "Path to Test data file",
        {
            "test-file-path"
        }
    };


    Flag test_flag
    {
        parser,
        "SHOULD DO TESTING",
        "Should do testing",
        {
            "test"
        }
    };

    ValueFlag<std::string> save_path_flag
    {
        parser,
        "MODEL_SAVE_PATH",
        "Where to save the model at the end of training",
        {
            "save-model-path"
        }
    };

    ValueFlag<std::string> model_msgpack_flag
    {
        parser,
        "MODEL_MSGPACK_PATH",
        "Path to load json config file",
        {
            "model-msgpack-path"
        }
    };

    ValueFlag<std::string> validation_mse_results_flag
    {
        parser,
        "VALIDATION_RESULTS_FILE_PATH",
        "Path to save validation mse results",
        {
            "validation-mse-results-path"
        }
    };

    Flag perform_epoch_based_training_flag
    {
        parser,
        "EPOCH_BASED_TRAINING",
        "Whether to use epoch based training (goes over entire dataset). Default is false",
        {
            "perform-epoch-based-training"
        }
    };

    Flag perform_l1_flag
    {
        parser,
        "PERFORM_L2_LOSS",
        "Whether to use L1 loss (default uses splitl2)",
        {
            "l1-loss"
        }
    };

    Flag perform_l2_relative_flag
    {
        parser, 
        "PERFORM_L2_RELATIVE_LOSS",
        "Whether to use L2 relative loss (default uses splitl2)",
        {
            "l2-relative-loss"
        }
    };

    ValueFlag<std::string> training_metrics_export_path
    {
        parser,
        "PATH_TO_TRAINING_METRICS_EXPORT_FILE",
        "Path to export training metrics to",
        {
            "training-metrics-export-path"
        }
    };

    ValueFlag<int> training_export_freq
    {
        parser,
        "TRAINING_METRICS_EXPORT_FREQ",
        "How frequently to export training metrics (per # of iterations)",
        {
            "training-metrics-export-freq"
        }
    };

    // Flag no_gui_flag{
	// 	parser,
	// 	"NO_GUI",
	// 	"Disables the GUI and instead reports training progress on the command line.",
	// 	{"no-gui"},
	// };

    // Parse command line arguments and react to parsing
	// errors using exceptions.
	try {
		if (arguments.empty()) {
			tlog::error() << "Number of arguments must be bigger than 0.";
			return -3;
		}

		parser.Prog(arguments.front());
		parser.ParseArgs(begin(arguments) + 1, end(arguments));
	} catch (const Help&) {
		std::cout << parser;
		return 0;
	} catch (const ParseError& e) {
		std::cerr << e.what() << std::endl;
		std::cerr << parser;
		return -1;
	} catch (const ValidationError& e) {
		std::cerr << e.what() << std::endl;
		std::cerr << parser;
		return -2;
	}

    // Parse args first before any CUDA/GL init
    fs::path data_path = get(data_flag);
    // if (argc > 1) {
    //     data_path = argv[1];
    // } else {
    //     // Default or error
    //     tlog::error() << "Please provide a data path.";
    //     return 1;
    // }

    // Check DISPLAY is set for X11 forwarding
    const char* display = std::getenv("DISPLAY");
    if (!display || display[0] == '\0') {
        tlog::warning() << "DISPLAY not set. GUI may not work over remote SSH.";
        tlog::warning() << "Try: export DISPLAY=<your-local-ip>:0.0";
    } else {
        tlog::info() << "DISPLAY=" << display;
    }



    Testbed testbed;
    testbed.m_perform_epoch_based_training = perform_epoch_based_training_flag;

    // Initialize window early to establish GL context before CUDA operations
    tlog::info() << "Initializing window...";
    // testbed.init_window(1920, 1080);
    tlog::info() << "Window initialized.";
    
    // Clear any CUDA errors that may have occurred during GL initialization
    // (GL-CUDA interop can leave spurious errors)
    cudaGetLastError();
    CUDA_CHECK_THROW(cudaDeviceSynchronize());

    
    if(model_flag)
    {
        const filesystem::path &model_path = get(model_flag);
        tlog::info() << "Loading model file from " << model_path;

        testbed.load_model(model_path);

        
        testbed.m_training_data_available = true;

        CUDA_CHECK_THROW(cudaDeviceSynchronize());
        CUDA_CHECK_THROW(cudaGetLastError()); // Clear any prior errors

        testbed.set_mode(ETestbedMode::PBRT);

        CUDA_CHECK_THROW(cudaDeviceSynchronize());
        CUDA_CHECK_THROW(cudaGetLastError()); // Clear any prior errors

        testbed.set_jit_fusion(false);

        CUDA_CHECK_THROW(cudaDeviceSynchronize());
        CUDA_CHECK_THROW(cudaGetLastError()); // Clear any prior errors

        tlog::info() << "Loading dataset...";
        load_nerfdataset(testbed, data_path);

        CUDA_CHECK_THROW(cudaDeviceSynchronize());
        CUDA_CHECK_THROW(cudaGetLastError()); // Clear any prior errors

        testbed.update_imgui_paths();

        //TODO: Put this as a flag
        testbed.m_train = true;
        testbed.m_training_data_available = true;

    } else {
        //INFO: Load Nerf data, because Volume data actually refers to NanoVDB
        // std::ifstream f{native_string(data_path), std::ios::in | std::ios::binary}; // Removed as it's handled in load_nerfdataset
        
        //INFO: setting training data available to true
        testbed.m_training_data_available = true;
        
        CUDA_CHECK_THROW(cudaDeviceSynchronize());
        CUDA_CHECK_THROW(cudaGetLastError()); // Clear any prior errors
        tlog::info() << "Setting testbed mode...\n";
        // TODO: Change this to what is most appropriate
        testbed.set_mode(ETestbedMode::PBRT);
        
        nlohmann::json config = 
        {
            {"encoding", {
                {"otype", "Composite"},
                {"nested", {
                    {
                        {"n_dims_to_encode", 3},
                        {"otype", "HashGrid"},
                        {"n_levels", 16},
                        {"n_features_per_level", 2},
                        {"log2_hashmap_size", 19},
                        {"base_resolution", 16},
                        {"per_level_scale", 1.5}
                    },
                    {
                        {"n_dims_to_encode", 3},
                        {"otype", "SphericalHarmonics"},
                        {"degree", 4 }
                    },
                    {
                        {"n_dims_to_encode", 1},
                        {"otype", "Frequency"}, 
                        {"n_frequencies", 4} //TODO try having n_frequencies = 6
                    }
                }}
            }},
            {"network", {
                {"otype", "FullyFusedMLP"},
                {"activation", "ReLU"},
                {"output_activation", "None"}, 
                {"n_neurons", 128}, //TODO try having 128 neurons here
                {"n_hidden_layers", 3} //TODO try having 3 hidden layers here
            }},
            {"loss", {
                {"otype", "SplitL2Loss"},
                {"split_idx", 3}
            }},
            {"optimizer", {
                {"otype", "Adam"},
                {"learning_rate", 1e-3},
                {"beta1", 0.9},
                {"beta2", 0.99},
                {"epsilon", 1e-8},
                {"gradient_clipping_magnitude", 1.0}
            }}
        };
        
        CUDA_CHECK_THROW(cudaDeviceSynchronize());
        CUDA_CHECK_THROW(cudaGetLastError()); // Clear any prior errors
        
        tlog::info() << "setting network json...\n";
        if(model_msgpack_flag)
        {
            const fs::path config_path = get(model_msgpack_flag);
            testbed.reload_network_from_file(config_path);
            tlog::info() << "Loaded from file " << config_path;

        }
        else
        {
            testbed.reload_network_from_json(config);
            tlog::info() << "Loaded from builtin config\n";
        }

        // Disable JIT fusion to avoid CUDA_ERROR_ILLEGAL_ADDRESS during cuModuleLoadDataEx
        // This is a workaround for a potential driver/PTX compatibility issue
        testbed.set_jit_fusion(false);
        tlog::info() << "JIT fusion disabled.";
        
        CUDA_CHECK_THROW(cudaDeviceSynchronize());
        CUDA_CHECK_THROW(cudaGetLastError()); // Clear any prior errors
        tlog::info() << "Loading dataset...\n";
        load_nerfdataset(testbed, data_path);
        
        testbed.update_imgui_paths();
        
        // Reset network to ensure it picks up the new dimensions
        tlog::info() << "Resetting network...";
        CUDA_CHECK_THROW(cudaGetLastError()); // Clear any prior errors
        testbed.reset_network();
        CUDA_CHECK_THROW(cudaDeviceSynchronize());
        tlog::info() << "Network reset complete.";
        
        // Initialize training state (gradients, optimizers, etc.)
        tlog::info() << "Calling load_nerf_post...";
        CUDA_CHECK_THROW(cudaGetLastError()); // Clear any prior errors
        testbed.load_nerf_post();
        CUDA_CHECK_THROW(cudaDeviceSynchronize());
        tlog::info() << "load_nerf_post complete.";
        
        testbed.m_train = true;
        testbed.m_training_batch_size = 1 << 18;
        testbed.m_training_data_available = true;
        // nerf_data.n_extra_learnable_dims = 2;
        // nerf_data.n_extra_learnable_dims = 0;
    }

    constexpr uint64_t validation_ray_count = 1 << 20;
    if(validation_flag)
    {
        if(!validation_path_flag)
        {
            throw std::runtime_error{"No validation file given..."};
        }
        const fs::path validation_data_path = get(validation_path_flag);
        load_dataset_to_testbed_fixedsize(testbed, validation_data_path, DatasetType::Validation,
            validation_ray_count);
        
    }

    // Window already initialized at startup
    // Training loop
    uint64_t curr_frame = 0;
    constexpr uint64_t BATCH_INTERVAL = 500;
    constexpr uint64_t DEFAULT_MAX_ITERATIONS = 40000;
    uint64_t max_iterations = train_epoch_flag ? get(train_epoch_flag) : DEFAULT_MAX_ITERATIONS;
    uint32_t &curr_epoch = testbed.m_volume_training_epoch;
    uint32_t last_epoch = curr_epoch;
    std::vector<Testbed::ValidationTestResults> validation_loss_results;

    double total_training_time = 0.f;
    if(perform_l1_flag)
    {
        testbed.m_loss->m_loss_mode = ESplitLossMode::SplitL1;
    }
    else if(perform_l2_relative_flag)
    {
        testbed.m_loss->m_loss_mode = ESplitLossMode::SplitL2Relative;
    }
    ESplitLossMode originalLossMode = testbed.m_loss->m_loss_mode;
    int export_freq = 0;
    if (training_export_freq) {
        export_freq = get(training_export_freq);
    }

    // We want to also load the test set as well
    if(training_metrics_export_path && test_path_flag)
    {
        testbed.m_stream_test_data_from_CPU = true;
        const fs::path &test_path = get(test_path_flag);
        delete[] testbed.m_volume_training_inputs_cpu;
        delete[] testbed.m_volume_training_targets_cpu;
        testbed.m_volume_training_targets_cpu = nullptr;
        testbed.m_volume_training_inputs_cpu = nullptr;
        load_dataset_to_testbed(testbed, test_path, DatasetType::Test);
    }

    struct TrainingExportMetrics
    {
        float trainMseLoss;
        float trainEmaMseLoss;
        float testMSELoss;
        float testPSNR;
        double timeElapsed;
        uint64_t iteration;

    };
    std::vector<TrainingExportMetrics> exportMetrics;
    auto last_time = std::chrono::steady_clock::now();

    while (testbed.frame()) {
        auto now = std::chrono::steady_clock::now();
        total_training_time += std::chrono::duration<double>(now - last_time).count();


        if (!(curr_frame % BATCH_INTERVAL) && validation_flag) {
            tlog::info() << "Done " << curr_frame << " frames\n";
            tlog::info() << "iteration=" << testbed.m_training_step
            << " loss=" << testbed.m_loss_scalar.val()
            << " ema loss=" << testbed.m_loss_scalar.ema_val();
            testbed.m_train = false;
            testbed.m_loss->m_loss_mode = ESplitLossMode::SplitL2;
            Testbed::ValidationTestResults res = testbed.validation_test();
            validation_loss_results.push_back(res);
            testbed.m_train = true;
            testbed.m_loss->m_loss_mode = originalLossMode;
        }
        else if(training_metrics_export_path && !(curr_frame % export_freq))
        {
            tlog::info() << "Done " << curr_frame << " frames\n";
            tlog::info() << "iteration=" << testbed.m_training_step
            << " loss=" << testbed.m_loss_scalar.val()
            << " ema loss=" << testbed.m_loss_scalar.ema_val();

            testbed.m_train = false;
            testbed.m_loss->m_loss_mode = ESplitLossMode::SplitL2;
            Testbed::ValidationTestResults res = testbed.validation_test(true);
            validation_loss_results.push_back(res);
            testbed.m_train = true;
            testbed.m_loss->m_loss_mode = originalLossMode;

            TrainingExportMetrics tem;
            
            tem.trainMseLoss = testbed.m_loss_scalar.val();
            tem.trainEmaMseLoss = testbed.m_loss_scalar.ema_val();
            tem.testMSELoss = res.mse;
            tem.testPSNR = res.psnr;
            tem.timeElapsed = total_training_time;
            tem.iteration = curr_frame;
            exportMetrics.push_back(tem);
        }
        curr_frame++;
        


        if(testbed.m_perform_epoch_based_training)
        {
            if(curr_epoch >= max_iterations)
            {
                break;
            }
        }
        else if(curr_frame > max_iterations)
        {
            break;
        }
        last_time = std::chrono::steady_clock::now();
        // The frame() function handles training steps if m_train is true.
    }

    if(save_path_flag)
    {
        //For now, don't compress but include optimizer state
        const fs::path save_path = get(save_path_flag);
        testbed.save_model(save_path, true, false);
    }


    //INFO: Do validation testing here

    if(validation_flag)
    {
        testbed.m_loss->m_loss_mode = ESplitLossMode::SplitL2;
        //TODO: Consider freeing training data to free up space on the GPU
        testbed.m_volume_training_inputs.free_memory();
        testbed.m_volume_training_targets.free_memory();

        delete[] testbed.m_volume_training_inputs_cpu;
        delete[] testbed.m_volume_training_targets_cpu;

        testbed.m_volume_training_inputs_cpu = nullptr;
        testbed.m_volume_training_targets_cpu = nullptr;

        testbed.m_volume_validation_inputs.free_memory();
        testbed.m_volume_validation_targets.free_memory();

        delete[] testbed.m_volume_validation_inputs_cpu;
        delete[] testbed.m_volume_validation_targets_cpu;

        testbed.m_volume_validation_inputs_cpu = nullptr;
        testbed.m_volume_validation_targets_cpu = nullptr;

        if(!validation_path_flag)
        {
            throw std::runtime_error{"No validation file given..."};
        }
        const fs::path validation_data_path = get(validation_path_flag);

        load_dataset_to_testbed(testbed, validation_data_path, DatasetType::Validation);

        testbed.m_train = false;
        Testbed::ValidationTestResults res = testbed.validation_test();

        if(validation_mse_results_flag && model_msgpack_flag)
        {
            const fs::path validation_mse_path = get(validation_mse_results_flag);
            const fs::path model_msgpack_path = get(model_msgpack_flag);
            std::ofstream mseFile{native_string(validation_mse_path),
                                  std::ios::out | std::ios::app};
            std::string formattedStr =
                fmt::format("{},mse:{},psnr:{} dB,loss:{},ema:{},total training time: {}", 
                    native_string(model_msgpack_path), res.mse, res.psnr, 
                    testbed.m_loss_scalar.val(), testbed.m_loss_scalar.ema_val(), total_training_time);
            
            formattedStr += ",mse_series:[";
            for (size_t i = 0; i < validation_loss_results.size(); ++i) {
                if (i > 0) formattedStr += ",";
                formattedStr += fmt::format("{}", validation_loss_results[i].mse);
            }
            formattedStr += "],psnr_series:[";
            for (size_t i = 0; i < validation_loss_results.size(); ++i) {
                if (i > 0) formattedStr += ",";
                formattedStr += fmt::format("{}", validation_loss_results[i].psnr);
            }
            formattedStr += "]\n";
            
            mseFile.write(formattedStr.c_str(), formattedStr.length());
        }
    }

    if(training_metrics_export_path)
    {
        const fs::path& metrics_path = get(training_metrics_export_path);
        std::ofstream metrics_file{native_string(metrics_path), std::ios::out | std::ios::app};
        for(const auto& tem : exportMetrics)
        {
            std::string formatted_string =
                fmt::format("testMse:{},testPSNR:{},mse:{},ema:{},time:{},frame:{}", 
                            tem.testMSELoss,  tem.testPSNR, tem.trainMseLoss, tem.trainEmaMseLoss,
                            tem.timeElapsed, tem.iteration);
            formatted_string += "\n";
            metrics_file.write(formatted_string.c_str(), formatted_string.length());
        }
    }

    return 0;
}
    
#endif
        
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
#include <tiny-cuda-nn/common.h>
#include <memory>
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <fstream>
#include <filesystem/directory.h>

using namespace tcnn;
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

struct MetadataExtras
{
    std::vector<GPUMemory<float>> sample_indices;
    std::vector<GPUMemory<float>> final_depths;
};

struct LoadedRayInfo {
    ivec2 res = ivec2(0);
    // EImageDataType image_type = EImageDataType::None;
    // bool white_transparent = false;
    // bool black_transparent = false;
    // uint32_t mask_color = 0;
    // void *pixels = nullptr;
    // uint16_t *depth_pixels = nullptr;
    Ray *rays = nullptr;
    // int *pixelIdx = nullptr;
    float *sampleIdx = nullptr;
    float *finalDepths = nullptr;
    // float *luminances = nullptr;
    vec4 *rgbas = nullptr;
    // float depth_scale = -1.f;
};

// batchOutput += StringPrintf(
// "%s|%s|%s|%s|%s\n",
// trainingSample.rayo.ToString(), trainingSample.rayd.ToString(),
// trainingSample.beta_before_rgb.ToString(),
// trainingSample.L_after_rgb.ToString(), trainingSample.T_after.ToString()
// );
// struct TrainingDataSample
// {
//     ivec2 res = ivec2(0);
//     vec3 *p;
//     vec3 *wo;
//     vec4 *beta_before;
//     vec4 *L_after;
//     vec4 *T_after;
// };

struct TrainingDataSample
{
    ivec2 res = ivec2(0);
    Ray *rays;
    vec4 *L_physical;
    vec4* T;
    float *tMax;
};

inline vec4 extract_rgb_from_string(const char* buf)
{
    float r, g, b;
    int read = std::sscanf(buf, "[%f %f %f", &r, &g, &b);
    return vec4(r, g, b, 1.0);
}


inline vec3 extract_vec_from_string(const char* buf)
{
    float x, y, z;
    int read = std::sscanf(buf, "[%f, %f, %f", &x, &y, &z);
    return vec3(x, y, z); 
}

int load_training_data_from_file(std::vector<TrainingDataSample>& trainingSampleData, const fs::path& path)
{
    // TODO(parser-validation): handle unreadable files gracefully and replace this std::count usage
    // with an actual '\n' count (std::count expects a value, not a predicate lambda).
    std::ifstream f{native_string(path), std::ios::in | std::ios::out};
    if (!f.is_open()) throw std::runtime_error("Failed to open file");
    
    //TODO: Find faster way to get line count, for now it is harcoded (i know how much it is)
    //auto count = std::count(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>(), '\n');
    int count = 21772800;
    f.clear();
    f.seekg(0, std::ios::beg);


    // Find header information about number of samples
    std::string header;
    std::getline(f, header);
    auto samplesPos = header.find("samples=");
    if(samplesPos != header.npos)
    {
        count = std::stoi(header.substr(samplesPos + 8));
    }

    // Create buffers to hold results of parsing
    Ray *rays = new Ray[count];
    vec4 *L_physical = new vec4[count];
    vec4 *T = new vec4[count];
    float *tMax = new float[count];

    // TODO(radiance-buffer): capture full RGB (or spectral) outputs per ray and pack them exactly
    // like Instant-NGP's pixel tensors instead of storing only a single luminance.
    // TODO(rgba-layout): allocate a contiguous float buffer sized n_rays*4, write RGB into
    // channels [0..2], set A=1 (or unused), and feed it to result.set_training_image so it mirrors
    // LoadedImageInfo::pixels.

    //we will set 1D array for the luminances/radiances. frames will be equal to sampleIdx i guess.
    int currSampleIdx = 0;
    int rayCount = 0;
    int ptr = 0;
    int frameIdx = 0;
    int linesRead = 0;
    for (std::string line; std::getline(f, line);)
    {
        if(line.find("version") != std::string::npos)
            continue;
        linesRead++;

        char rayOBuf[64];
        char rayDBuf[64];
        char betaBeforeBuf[64];
        char LAfterBuf[64];
        char TAfterBuf[64];
        float tMaxVal;
        int sampleIdx;
        // "rayo|rayd|beta_before|L_after|T_after|tMax|sampleIdx"

        int read = std::sscanf(line.c_str(), 
            "%[^]]]|%[^]]]|%[^]]]|%[^]]]|%[^]]]|%f|%d",
            rayOBuf, rayDBuf, betaBeforeBuf, LAfterBuf, TAfterBuf, &tMaxVal, &sampleIdx
        );
        if(read < 6)
            throw std::runtime_error(std::string("Failed to parse line, only ") + std::to_string(read) + " items read");

        vec3 rayO = extract_vec_from_string(rayOBuf);
        vec3 rayD = extract_vec_from_string(rayDBuf);
        vec4 betaBefore = extract_rgb_from_string(betaBeforeBuf);
        vec4 LAfter = extract_rgb_from_string(LAfterBuf);
        vec4 TAfter = extract_rgb_from_string(TAfterBuf);

        rays[ptr] = {rayO, rayD};
        
        const float epsilon = 1e-6f;
        for(int c = 0; c < 3; ++c)
        {
            if(betaBefore[c] > epsilon)
            {
                L_physical[ptr][c] = LAfter[c] / betaBefore[c];
            }
            else
            {
                L_physical[ptr][c] = 0.f;
            }
        }

        T[ptr] = TAfter;
        tMax[ptr] = tMaxVal;

        rayCount++;
        ++ptr;
        if(currSampleIdx != sampleIdx || rayCount >= 4096)
        {
            Ray *ray_sample = new Ray[rayCount];
            vec4 *L_physical_sample = new vec4[rayCount];
            vec4 *T_sample = new vec4[rayCount];
            float *tMax_sample = new float[rayCount];

            memcpy(ray_sample, rays, rayCount * sizeof(Ray));
            memcpy(L_physical_sample, L_physical, rayCount * sizeof(vec4));
            memcpy(T_sample, T, rayCount * sizeof(vec4));
            memcpy(tMax_sample, tMax, rayCount * sizeof(float));
            TrainingDataSample sampleData;
            sampleData.res = {rayCount, 1};
            sampleData.rays = ray_sample;
            sampleData.L_physical = L_physical_sample;
            sampleData.T = T_sample;
            sampleData.tMax = tMax_sample;
            trainingSampleData.push_back(sampleData);
            frameIdx++;
            currSampleIdx = sampleIdx;
            ptr = 0;
            rayCount = 0;
        }
        else
        {
            ++ptr;
        }
    }

    // last samples were not considered:
    {
        Ray *ray_sample = new Ray[rayCount];
        vec4 *L_physical_sample = new vec4[rayCount];
        vec4 *T_sample = new vec4[rayCount];
        float *tMax_sample = new float[rayCount];

        // memcpy(frameRay, rays, rayCount*sizeof(Ray));
        // memcpy(frameSampleIndices, sampleIndices, rayCount*sizeof(float));
        // memcpy(frameFinalDepths, finalDepths, rayCount*sizeof(float));
        // memcpy(frameRGBAs, rgbas, rayCount*sizeof(vec4));

        memcpy(ray_sample, rays, rayCount * sizeof(Ray));
        memcpy(L_physical_sample, L_physical, rayCount * sizeof(vec4));
        memcpy(T_sample, T, rayCount * sizeof(vec4));
        memcpy(tMax_sample, tMax, rayCount * sizeof(float));
        TrainingDataSample sampleData;
        sampleData.res = {rayCount, 1};
        sampleData.rays = ray_sample;
        sampleData.L_physical = L_physical_sample;
        sampleData.T = T_sample;
        sampleData.tMax;
        trainingSampleData.push_back(sampleData);
        frameIdx++;
        ptr = 0;
        rayCount = 0;
    }
    
    tlog::success() << "Loaded training sample of size: " << trainingSampleData.size() << "\n";        
    tlog::success() << "Read " << linesRead << " training samples\n";
    CUDA_CHECK_THROW(cudaDeviceSynchronize());

    delete[] rays;
    delete[] L_physical;
    delete[] T;
    delete[] tMax;
    return frameIdx;
}

int load_ray_data_from_file(std::vector<LoadedRayInfo>& frameRayData, const fs::path& path)
{
    // TODO(parser-validation): handle unreadable files gracefully and replace this std::count usage
    // with an actual '\n' count (std::count expects a value, not a predicate lambda).
    std::ifstream f{native_string(path), std::ios::in | std::ios::out};
    if (!f.is_open()) throw std::runtime_error("Failed to open file");
    //TODO: Find faster way to get line count, for now it is harcoded (i know how much it is)

    //auto count = std::count(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>(), '\n');
    const int count = 21772800;
    f.clear();
    f.seekg(0, std::ios::beg);

    // Create buffers to hold results of parsing
    Ray* rays = new Ray[count];
    float* sampleIndices = new float[count];
    float* finalDepths = new float[count];
    vec4* rgbas = new vec4[count];



    // TODO(radiance-buffer): capture full RGB (or spectral) outputs per ray and pack them exactly
    // like Instant-NGP's pixel tensors instead of storing only a single luminance.
    // TODO(rgba-layout): allocate a contiguous float buffer sized n_rays*4, write RGB into
    // channels [0..2], set A=1 (or unused), and feed it to result.set_training_image so it mirrors
    // LoadedImageInfo::pixels.

    //we will set 1D array for the luminances/radiances. frames will be equal to sampleIdx i guess.
    int currSampleIdx = 0;
    int rayCount = 0;
    int ptr = 0;
    int frameIdx = 0;
    int linesRead = 0;
    for (std::string line; std::getline(f, line);)
    {
        linesRead++;
        
        char oBuf[64];
        char dBuf[64];
        int pixelIdx;
        int sampleIdx;
        int finalDepth;
        char lumBuf[128];
        char rgbBuf[64];

        int read = std::sscanf(line.c_str(), 
            "Pixel (o: %[^]]], d: %[^]]]) PixelIdx (%d) Sample (%d) FinalDepth (%d) Luminance (%[^]]]) RGB (%[^]]])",
            oBuf, dBuf, &pixelIdx, &sampleIdx, &finalDepth, lumBuf, rgbBuf
        );
        if(read != 7)
            throw std::runtime_error(std::string("Failed to parse line, only ") + std::to_string(read) + " items read");

        vec3 o;
        vec3 d;
        vec4 rgba;
        float ox, oy, oz;
        float dx, dy, dz;
        float r, g, b;
        read = std::sscanf(oBuf, "[%f, %f, %f", &ox, &oy, &oz);
        read = std::sscanf(dBuf, "[%f, %f, %f", &dx, &dy, &dz);
        read = std::sscanf(rgbBuf, "[%f %f %f", &r, &g, &b);


        o = vec3(ox, oy, oz);
        d = vec3(dx, dy, dz);
        rgba = vec4(r, g, b, 1.0);

        rays[ptr] = {o, d};
        sampleIndices[ptr] = (float)sampleIdx / 16.f;
        finalDepths[ptr] = (float)finalDepth / 5.f;
        rgbas[ptr] = rgba;
        rayCount++;



        if(currSampleIdx != sampleIdx)
        {
            Ray* frameRay = new Ray[rayCount];
            float* frameSampleIndices = new float[rayCount];
            float* frameFinalDepths = new float[rayCount];
            vec4* frameRGBAs = new vec4[rayCount];

            memcpy(frameRay, rays, rayCount*sizeof(Ray));
            memcpy(frameSampleIndices, sampleIndices, rayCount*sizeof(float));
            memcpy(frameFinalDepths, finalDepths, rayCount*sizeof(float));
            memcpy(frameRGBAs, rgbas, rayCount*sizeof(vec4));
            ivec2 res{rayCount, 1};
            LoadedRayInfo info;
            info.res = res;
            info.rays = frameRay;
            info.sampleIdx = frameSampleIndices;
            info.finalDepths = frameFinalDepths;
            info.rgbas = frameRGBAs;
            frameRayData.push_back(info);
            frameIdx++;
            currSampleIdx = sampleIdx;
            ptr = 0;
            rayCount = 0;
        }
        else
        {
            ++ptr;
        }


    }

    // last samples were not considered:
    {

        Ray* frameRay = new Ray[rayCount];
        float* frameSampleIndices = new float[rayCount];
        float* frameFinalDepths = new float[rayCount];
        vec4* frameRGBAs = new vec4[rayCount];
        
        memcpy(frameRay, rays, rayCount*sizeof(Ray));
        memcpy(frameSampleIndices, sampleIndices, rayCount*sizeof(float));
        memcpy(frameFinalDepths, finalDepths, rayCount*sizeof(float));
        memcpy(frameRGBAs, rgbas, rayCount*sizeof(vec4));
        ivec2 res{rayCount, 1};
        LoadedRayInfo info;
        info.res = res;
        info.rays = frameRay;
        info.sampleIdx = frameSampleIndices;
        info.finalDepths = frameFinalDepths;
        info.rgbas = frameRGBAs;
        frameRayData.push_back(info);
        frameIdx++;
        // currSampleIdx = sampleIdx; // Removed as sampleIdx is out of scope and not needed here
        ptr = 0;
        rayCount = 0;
    }
    
    tlog::success() << "Loaded " << frameRayData.size() << "\n";        
    tlog::success() << "Read " << linesRead << "\n";
    CUDA_CHECK_THROW(cudaDeviceSynchronize());

    delete[] rays;
    delete[] sampleIndices;
    delete[] finalDepths;
    delete[] rgbas;

    return frameIdx;
}


void load_metadata(NerfDataset& nerf_data, MetadataExtras& meta_extras, int frameIdx, int rayCount, LoadedRayInfo frame)
{
    auto& sample_buf = meta_extras.sample_indices[frameIdx];
    auto& final_depth_buf = meta_extras.final_depths[frameIdx];

    sample_buf.resize(rayCount);
    final_depth_buf.resize(rayCount);

    sample_buf.copy_from_host(frame.sampleIdx, rayCount);
    final_depth_buf.copy_from_host(frame.finalDepths, rayCount);
    // Setup metadata buffers
    nerf_data.metadata[frameIdx].sample_indices = sample_buf.data();
    nerf_data.metadata[frameIdx].final_depths = final_depth_buf.data();
    
    tlog::info() << "Frame " << frameIdx << ": rays=" << rayCount 
                    << " sample_indices=" << (void*)sample_buf.data() 
                    << " final_depths=" << (void*)final_depth_buf.data();
}

struct __attribute__((packed)) BinaryTrainingSample 
{
    float o[3];
    float d[3];
    float beta_before_rgb[3];
    float L_after_rgb[3];
    float T_after[3];
    float tMax;
};

void load_training_to_testbed(Testbed& testbed, const fs::path& path)
{
    std::ifstream f{native_string(path), std::ios::in | std::ios::out | std::ios::binary };
    if(!f.is_open())
        throw std::runtime_error("Failed to open file");

    testbed.m_n_volume_training_samples = 0;
    f.clear();
    f.seekg(0, std::ios::beg);


    // std::string header;
    // std::getline(f, header);
    // auto samplesPos = header.find("samples=");
    // if(samplesPos != header.npos)
    // {
    //     testbed.m_n_volume_training_samples = std::min(std::stoul(header.substr(samplesPos + 8)), (uint64_t)UINT32_MAX);
    // }
    uint64_t count = 0;
    f.read(reinterpret_cast<char *>(&count), sizeof(uint64_t));

    testbed.m_n_volume_training_samples = std::min(count, (uint64_t)UINT32_MAX);

    testbed.m_volume_training_inputs_cpu = new float[testbed.m_n_volume_training_samples * N_VOLUME_INPUT_DIMS];
    testbed.m_volume_training_targets_cpu = new float[testbed.m_n_volume_training_samples * N_VOLUME_TARGET_DIMS];

    if(count == 0)
        return;

    
    auto &input_cpu_buffer = testbed.m_volume_training_inputs_cpu;
    auto &target_cpu_buffer = testbed.m_volume_training_targets_cpu;

    vec3 min_bound = vec3(1e30f);
    vec3 max_bound = vec3(-1e30f);
    float minTMax = 1e30f;
    float maxTMax = -1e30f;

    std::vector<BinaryTrainingSample> binarySampleBuffer;
    binarySampleBuffer.resize(count);

    f.read(reinterpret_cast<char *>(binarySampleBuffer.data()),
           sizeof(BinaryTrainingSample) * count);

        

    int ptr = 0;
    for (const auto& bs : binarySampleBuffer) {
        // Calculate scene BB to normalize rays

        vec3 rayO = vec3(bs.o[0], bs.o[1], bs.o[2]);
        min_bound = min(min_bound, rayO);
        max_bound = max(max_bound, rayO);

        float tMaxVal = bs.tMax;
        minTMax = min(minTMax, tMaxVal);
        maxTMax = max(maxTMax, tMaxVal);

        input_cpu_buffer[ptr * N_VOLUME_INPUT_DIMS + POS_OFFSET + 0] = rayO.x;
        input_cpu_buffer[ptr * N_VOLUME_INPUT_DIMS + POS_OFFSET + 1] = rayO.y;
        input_cpu_buffer[ptr * N_VOLUME_INPUT_DIMS + POS_OFFSET + 2] = rayO.z;
        input_cpu_buffer[ptr * N_VOLUME_INPUT_DIMS + DIR_OFFSET + 0] = bs.d[0];
        input_cpu_buffer[ptr * N_VOLUME_INPUT_DIMS + DIR_OFFSET + 1] = bs.d[1];
        input_cpu_buffer[ptr * N_VOLUME_INPUT_DIMS + DIR_OFFSET + 2] = bs.d[2];
        input_cpu_buffer[ptr * N_VOLUME_INPUT_DIMS + TMAX_OFFSET] = tMaxVal;


        //TODO: Check if i need to do some clamping here as well
        constexpr float epsilon = 1e-6f;

        for(int c = 0; c < 3; ++c)
        {
            if(bs.beta_before_rgb[c] > epsilon)
            {
                target_cpu_buffer[ptr * N_VOLUME_TARGET_DIMS + L_OFFSET + c] = bs.L_after_rgb[c] / bs.beta_before_rgb[c];
            }
            else
            {
                target_cpu_buffer[ptr * N_VOLUME_TARGET_DIMS + L_OFFSET + c] = 0.f;
            }
        }

        target_cpu_buffer[ptr * N_VOLUME_TARGET_DIMS + T_OFFSET + 0] = std::min(1.f, std::max(0.f, bs.T_after[0]));
        target_cpu_buffer[ptr * N_VOLUME_TARGET_DIMS + T_OFFSET + 1] = std::min(1.f, std::max(0.f, bs.T_after[1]));
        target_cpu_buffer[ptr * N_VOLUME_TARGET_DIMS + T_OFFSET + 2] = std::min(1.f, std::max(0.f, bs.T_after[2]));
        
        ++ptr;
    }

    tlog::success() << "Read " << count << " samples \n";
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

    vec3 centre = (max_bound - min_bound) * 0.5f;
    vec3 offset = vec3(0.5f) - centre * scale;


    tlog::info() << "Auto-normalizing rays: scale=" << scale << " offset=[" << offset.x
                << ", " << offset.y << ", " << offset.z << "]";
    
    for(int i = 0; i < ptr; ++i)
    {
        auto index = i * N_VOLUME_INPUT_DIMS + POS_OFFSET;
        for(int dim = 0; dim < 3; ++dim)
        {
            const float val = input_cpu_buffer[index + dim];
            input_cpu_buffer[index + dim] = val * scale + offset[dim];
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
    for (int i = 0; i < ptr; ++i) {
        auto index = i * N_VOLUME_INPUT_DIMS + TMAX_OFFSET;
        const float val = input_cpu_buffer[index];
        input_cpu_buffer[index] = val * tMax_scale;
    }

    tlog::info() << "Auto-normalizing tmax: scale=" << tMax_scale << " with range=[ "
         << minTMax << ", " << maxTMax << " ]\n";

    testbed.m_volume_training_inputs_tMax_scale = tMax_scale;
    
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
}

void load_nerfdataset(Testbed& testbed, NerfDataset& nerf_data, MetadataExtras& meta_extras, const fs::path& data_path)
{
    // TODO(custom-dataset-export): before calling this, generate per-ray dumps containing
    // origin, direction, optional metadata, and supervised radiance. The instant-ngp loader
    // comments describe how to serialize those into rays_*.dat plus float radiance blocks.
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

    // TODO(custom-nngp-hookup): reuse the instant-ngp load_nerf to ingest your prepared files,
    // ensuring n_extra_learnable_dims and has_rays are set so the training loop sees the
    // per-ray metadata/radiance pairs.
    // TODO(nerfdataset-plumbing): actually instantiate a NerfDataset, resize xforms/metadata/pixelmemory,
    // and call set_training_image with the radiance buffers + ray pointers produced above.
    // TODO(frame-shape): for each ray chunk, set metadata[i].resolution = {n_rays, 1} (or W,H) so
    // batching knows the number of samples, and mark image_type = EImageDataType::Float, is_hdr=true.


    // std::vector<LoadedRayInfo> frameRayData;
    // std::vector<TrainingDataSample> trainingDataSamples;

    //TODO: Consider switching to outputting as jsons instead

    nerf_data.has_rays = true;
    //TODO: Extend this so it can take multiple files at once
    for(size_t i = 0; i < paths.size(); ++i)
    {
        // int added_frames = load_ray_data_from_file(frameRayData, paths[i]);

        load_training_to_testbed(testbed, paths[i]);

        // TODO(extra-metadata-pack): if pixelIdx/sampleIdx/finalDepth (or other ray tags) are required at
        // train time, either extend the Ray struct to store them or upload parallel buffers and record
        // pointers in TrainingImageMetadata so batch generation can recover them.



        //TODO: Test out both having pixelidx/sampleIdx/finalDepth in either extending the Ray, or
        // record pointers in TrainingImageMetadata. For now I will just add it to TrainingImageMetadata struct


    }
    return;
    // // Now loaded all the frame data, now time to send it to NerfDataset
    // int frameCount = frameRayData.size();
    // // Create dataset with aabb_scale=1 and is_hdr=true
    // nerf_data = create_empty_nerf_dataset(frameCount, 1, true);

    // // Override default scale/offset to ensure raw PBRT rays are not transformed
    // // (User must ensure rays are within [0,1] or [0, aabb_scale] if aabb_scale > 1)
    
    // // Calculate scene bounding box to normalize rays
    // vec3 min_bound = vec3(1e30f);
    // vec3 max_bound = vec3(-1e30f);

    // for (const auto &frame : frameRayData) {
    //     for (int i = 0; i < frame.res.x; ++i) {
    //         const auto &ray = frame.rays[i];
    //         min_bound = min(min_bound, ray.o);
    //         max_bound = max(max_bound, ray.o);
    //     }
    // }

        // Now loaded all the frame data, now time to send it to NerfDataset
//     int trainingSampleCnt = trainingDataSamples.size();
//     // Create dataset with aabb_scale=1 and is_hdr=true
//     nerf_data = create_empty_nerf_dataset(trainingSampleCnt, 1, true);
    
//     // Override default scale/offset to ensure raw PBRT rays are not transformed
//     // (User must ensure rays are within [0,1] or [0, aabb_scale] if aabb_scale > 1)
    
//     // Calculate scene bounding box to normalize rays
//     vec3 min_bound = vec3(1e30f);
//     vec3 max_bound = vec3(-1e30f);

//     for (const auto &trainingDataSample : trainingDataSamples) {
//         for (int i = 0; i < trainingDataSample.res.x; ++i) {
//             const auto &ray = trainingDataSample.rays[i];
//             min_bound = min(min_bound, ray.o);
//             max_bound = max(max_bound, ray.o);
//         }
//     }


//     tlog::info() << "Scene AABB: [" << min_bound.x << ", " << min_bound.y << ", "
//                  << min_bound.z << "] to [" << max_bound.x << ", " << max_bound.y << ", "
//                  << max_bound.z << "]";

//     // Compute scale and offset to fit in [0.05, 0.95]
//     vec3 size = max_bound - min_bound;
//     float max_dim = std::max({size.x, size.y, size.z});
//     if (max_dim == 0)
//         max_dim = 1.0f;
//     float scale = 0.9f / max_dim;
//     vec3 center = (min_bound + max_bound) * 0.5f;
//     vec3 offset = vec3(0.5f) - center * scale;

//     tlog::info() << "Auto-normalizing rays: scale=" << scale << " offset=[" << offset.x
//                  << ", " << offset.y << ", " << offset.z << "]";

//     // Apply normalization
//     for (auto &trainingDataSample : trainingDataSamples) {
//         for (int i = 0; i < trainingDataSample.res.x; ++i) {
//             auto &ray = trainingDataSample.rays[i];
//             ray.o = ray.o * scale + offset;
//         }
//     }

//     nerf_data.scale = scale;
//     nerf_data.offset = offset;
    
//     // nerf_data.n_extra_learnable_dims = 2;
//     //TODO: Consider if I want to also add in the sample and depth index data, for now just set to 0
//     nerf_data.n_extra_learnable_dims = 0;
//     nerf_data.has_rays = true;
//     // meta_extras.sample_indices.resize(trainingSampleCnt);
//     // meta_extras.final_depths.resize(trainingSampleCnt);
//     uint32_t frameIdx = 0;
//     for(auto& trainingDataSample : trainingDataSamples)
//     {
//         int rayCount = trainingDataSample.res.x;

//         for(int rIdx = 0; rIdx < rayCount; ++rIdx)
//         {
//             Ray& ray = trainingDataSample.rays[rIdx];
//             ray.d = normalize(ray.d);
//             // nerf_data.nerf_ray_to_ngp(ray);
//         }

//         //TODO(trainingDataUpload): Find a way to also upload the Transmission to a separate or same I-NGP network
//         //from trainingDataSample.T
//         nerf_data.set_training_image(frameIdx, 
//             trainingDataSample.res, 
//             trainingDataSample.L_physical, 
//             nullptr, 
//             1.0f, 
//             false, 
//             EImageDataType::Float, 
//             EDepthDataType::Float,
//             0.f,
//             false,
//             false,
//             0,            
//             trainingDataSample.rays
//         );


        
//         // INFO: Add on extra metadata information
//         // load_metadata(nerf_data, meta_extras);

//         //TODO: Check if there are any other fields that I need to set for the metadata
//         nerf_data.metadata[frameIdx].image_data_type = EImageDataType::Float;
//         nerf_data.xforms[frameIdx].start = mat4x3::identity();
//         nerf_data.xforms[frameIdx].end = mat4x3::identity();
//         nerf_data.metadata[frameIdx].lens = {};
//         nerf_data.metadata[frameIdx].resolution = trainingDataSample.res;
//         nerf_data.metadata[frameIdx].principal_point = vec2(0.5f);
//         nerf_data.metadata[frameIdx].focal_length = vec2(1000.f);
//         nerf_data.metadata[frameIdx].rolling_shutter = vec4(0.f);
//         nerf_data.metadata[frameIdx].light_dir = vec3(0.f);
        
//         nerf_data.update_metadata(frameIdx, frameIdx + 1);

//         frameIdx++;

// #ifndef _DEBUG
//         // delete[] frame.rays;
//         // delete[] frame.sampleIdx;
//         // delete[] frame.finalDepths;
//         // delete[] frame.rgbas;
//         delete[] trainingDataSample.rays;
//         delete[] trainingDataSample.L_physical;
//         delete[] trainingDataSample.T;
// #endif
//     }

    


}


int main(int argc, char** argv)
{

    // Parse args first before any CUDA/GL init
    fs::path data_path;
    if (argc > 1) {
        data_path = argv[1];
    } else {
        // Default or error
        tlog::error() << "Please provide a data path.";
        return 1;
    }

    // Check DISPLAY is set for X11 forwarding
    const char* display = std::getenv("DISPLAY");
    if (!display || display[0] == '\0') {
        tlog::warning() << "DISPLAY not set. GUI may not work over remote SSH.";
        tlog::warning() << "Try: export DISPLAY=<your-local-ip>:0.0";
    } else {
        tlog::info() << "DISPLAY=" << display;
    }

    Testbed testbed;
    
    // Initialize window early to establish GL context before CUDA operations
    tlog::info() << "Initializing window...";
    testbed.init_window(1920, 1080);
    tlog::info() << "Window initialized.";
    
    // Clear any CUDA errors that may have occurred during GL initialization
    // (GL-CUDA interop can leave spurious errors)
    cudaGetLastError();
    CUDA_CHECK_THROW(cudaDeviceSynchronize());

    

    //INFO: Load Nerf data, because Volume data actually refers to NanoVDB
    // std::ifstream f{native_string(data_path), std::ios::in | std::ios::binary}; // Removed as it's handled in load_nerfdataset

    //INFO: setting training data available to true
    testbed.m_training_data_available = true;

    auto& training = testbed.m_nerf.training;
    NerfDataset& nerf_data = training.dataset;
    nerf_data.n_extra_learnable_dims = 2;
    nerf_data.scale = 1.0;
    nerf_data.is_hdr = true;

    MetadataExtras meta_extras{};

    CUDA_CHECK_THROW(cudaDeviceSynchronize());
    CUDA_CHECK_THROW(cudaGetLastError()); // Clear any prior errors
    tlog::info() << "Setting testbed mode...\n";
    // TODO: Change this to what is most appropriate
    testbed.set_mode(ETestbedMode::PBRT);

    // Manually load config to ensure dir_encoding and rgb_network are present
    // This prevents crash at reset_network() because default config lacks these fields
    // required for NeRF mode
    // nlohmann::json c = {
    //     {"loss", {{"otype", "Huber"}}},
    //     {"optimizer",
    //      {{"otype", "Ema"},
    //       {"decay", 0.95},
    //       {"nested",
    //        {{"otype", "ExponentialDecay"},
    //         {"decay_start", 20000},
    //         {"decay_interval", 10000},
    //         {"decay_base", 0.33},
    //         {"nested",
    //          {{"otype", "Adam"},
    //           {"learning_rate", 1e-2},
    //           {"beta1", 0.9},
    //           {"beta2", 0.99},
    //           {"epsilon", 1e-15},
    //           {"l2_reg", 1e-6}}}}}}},
    //     {"encoding",
    //      {{"otype", "HashGrid"},
    //       {"n_levels", 8},
    //       {"n_features_per_level", 4},
    //       {"log2_hashmap_size", 19},
    //       {"base_resolution", 16}}},
    //     {"network",
    //      {{"otype", "FullyFusedMLP"},
    //       {"activation", "ReLU"},
    //       {"output_activation", "None"},
    //       {"n_neurons", 64},
    //       {"n_hidden_layers", 1}}},
    //     {"dir_encoding",
    //      {{"otype", "Composite"},
    //       {"nested",
    //        {{{"n_dims_to_encode", 3}, {"otype", "SphericalHarmonics"}, {"degree", 4}},
    //         {{"otype", "Identity"}}}}}},
    //     {"rgb_network",
    //      {{"otype", "FullyFusedMLP"},
    //       {"activation", "ReLU"},
    //       {"output_activation", "None"},
    //       {"n_neurons", 64},
    //       {"n_hidden_layers", 2}}}};
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
            {"n_neurons", 64}, //TODO try having 128 neurons here
            {"n_hidden_layers", 2} //TODO try having 3 hidden layers here
        }},
        {"loss", {
            {"otype", "SplitL2Loss"},
            {"split_idx", 3}
        }},
        {"optimizer", {
            {"otype", "Adam"},
            {"learning_rate", 1e-2},
            {"beta1", 0.9},
            {"beta2", 0.99},
            {"epsilon", 1e-8}
        }}
    };
    
    CUDA_CHECK_THROW(cudaDeviceSynchronize());
    CUDA_CHECK_THROW(cudaGetLastError()); // Clear any prior errors

    tlog::info() << "setting network json...\n";
    testbed.reload_network_from_json(config);

    CUDA_CHECK_THROW(cudaDeviceSynchronize());
    CUDA_CHECK_THROW(cudaGetLastError()); // Clear any prior errors
    tlog::info() << "Loading dataset...\n";
    load_nerfdataset(testbed, nerf_data, meta_extras, data_path);

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
    nerf_data.n_extra_learnable_dims = 0;

    // Window already initialized at startup
    // Training loop
    uint64_t curr_frame = 0;
    while (testbed.frame()) {
        if (!(curr_frame % 100000)) {
            tlog::info() << "Done " << curr_frame << " frames\n";
            tlog::info() << "iteration=" << testbed.m_training_step
                         << " loss=" << testbed.m_loss_scalar.val();
        }
        curr_frame++;
        // The frame() function handles training steps if m_train is true.
    }

    return 0;
}

#endif
        
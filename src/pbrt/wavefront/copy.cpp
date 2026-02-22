#include <pbrt/wavefront/integrator.h>
#include "config.h"

#include <pbrt/media.h>
#include <thread>
#include <chrono>

#include <type_traits>
#ifdef PBRT_BUILD_GPU_RENDERER
#include <pbrt/gpu/util.h>
#endif
namespace pbrt
{
    constexpr uint64_t kMaxBytes = 20 * 1024ull * 1024ull * 1024ull;
    constexpr uint64_t kBinaryTrainingSampleSize = 76;
    constexpr uint64_t kMaxSamples = kMaxBytes / kBinaryTrainingSampleSize;
    constexpr bool optimized_output = false;
    constexpr bool use_volume_training_data = true;


    void WavefrontPathIntegrator::RayLogWorker()
    {
        while(true)
        {
            RayLogBatch batch;

            // Wait for work
            {
                std::unique_lock<std::mutex> lock(rayLogMutex);

                rayLogCondition.wait(lock, [this]
                {
                    return rayLogShutdown || !rayLogWorkQueue.empty();
                });

                if(rayLogShutdown && rayLogWorkQueue.empty()) break;


                batch = std::move(rayLogWorkQueue.front());
                rayLogWorkQueue.pop();
            }

            bool skipWrite = false;
                
            // Check if we've already written enough samples
            if (rayLogSampleCnt >= kMaxSamples) {
                LOG_VERBOSE("Sample count %llu >= max %llu, skipping write\n", 
                            rayLogSampleCnt, kMaxSamples);
                continue;
            }


            // std::string batchOutput;
            // batchOutput.reserve(batch.trainingData.size() * 150);

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
                // float beta_before_spectral[NSpectrumSamples];
                // float L_after_spectral[NSpectrumSamples];
                float T_after_spectral[NSpectrumSamples];
                float tMax;
            };

            std::vector<BinaryTrainingSample> binaryBatchOutput;
            std::vector<BinaryTrainingSampleSpectral> binaryBatchOutputSpectral;


            if(false)
            {
                binaryBatchOutput.resize(batch.trainingData.size());
                LOG_VERBOSE("Allocating %d number of binary batch outputs\n",

                            batch.trainingData.size());

            }
            else
            {
                binaryBatchOutputSpectral.resize(batch.trainingData.size());

            }

            

            for(size_t i = 0; i < batch.trainingData.size(); ++i)
            {
                // const auto& sample = batch.samples[i];
                const auto& trainingSample = batch.trainingData[i];
                // const auto& L_final = batch.finalLValues[i];
                // const auto& lambda = batch.lambdas[i];

                // if(sample.beta == SampledSpectrum(0.f)) continue;

                // SampledSpectrum L_target = (L_final - sample.L) / sample.beta;
                // RGB rgb = film.ToOutputRGB(L_target, lambda);

                // if(L_target.HasNaNs() || rgb.r == Infinity || rgb.g == Infinity || rgb.b == Infinity) continue;


                if (optimized_output) {
                } else {
                    // "rayo|rayd|beta_before|L_after|T_after"
                    if(use_volume_training_data)
                    {
                        // TODO: Figure out if I want to use RGB or luminance values.
                        // batchOutput += StringPrintf(
                        //     "%s|%s|%s|%s|%s|%f|%d\n",
                        //     trainingSample.rayo.ToString(), trainingSample.rayd.ToString(),
                        //     trainingSample.beta_before_rgb.ToString(),
                        //     trainingSample.L_after_rgb.ToString(), trainingSample.T_after.ToString(),
                        //     trainingSample.tMax,
                        //     batch.sampleIndex
                        // );
                        if(false)
                        {
                            auto &binaryOutput = binaryBatchOutput[i];
                            binaryOutput.o[0] = trainingSample.rayo.x;
                            binaryOutput.o[1] = trainingSample.rayo.y;
                            binaryOutput.o[2] = trainingSample.rayo.z;

                            binaryOutput.d[0] = trainingSample.rayd.x;
                            binaryOutput.d[1] = trainingSample.rayd.y;
                            binaryOutput.d[2] = trainingSample.rayd.z;



                            binaryOutput.beta_before_rgb[0] = trainingSample.beta_before_rgb.r;
                            binaryOutput.beta_before_rgb[1] = trainingSample.beta_before_rgb.g;
                            binaryOutput.beta_before_rgb[2] = trainingSample.beta_before_rgb.b;
                            
                            binaryOutput.L_after_rgb[0] = trainingSample.L_after_rgb.r;
                            binaryOutput.L_after_rgb[1] = trainingSample.L_after_rgb.g;
                            binaryOutput.L_after_rgb[2] = trainingSample.L_after_rgb.b;
                            
                            binaryOutput.T_after[0] = trainingSample.T_after.r;
                            binaryOutput.T_after[1] = trainingSample.T_after.g;
                            binaryOutput.T_after[2] = trainingSample.T_after.b;
                        }
                        else
                        {
                            auto &binaryOutput = binaryBatchOutputSpectral[i];
                            binaryOutput.o[0] = trainingSample.rayo.x;
                            binaryOutput.o[1] = trainingSample.rayo.y;
                            binaryOutput.o[2] = trainingSample.rayo.z;

                            binaryOutput.d[0] = trainingSample.rayd.x;
                            binaryOutput.d[1] = trainingSample.rayd.y;
                            binaryOutput.d[2] = trainingSample.rayd.z;

                            // binaryOutput.beta_before_spectral[0] = trainingSample.beta_before_spectral[0];
                            // binaryOutput.beta_before_spectral[1] = trainingSample.beta_before_spectral[1];
                            // binaryOutput.beta_before_spectral[2] = trainingSample.beta_before_spectral[2];
                            // binaryOutput.beta_before_spectral[3] = trainingSample.beta_before_spectral[3];
    
                            // binaryOutput.L_after_spectral[0] = trainingSample.L_after_spectral[0];
                            // binaryOutput.L_after_spectral[1] = trainingSample.L_after_spectral[1];
                            // binaryOutput.L_after_spectral[2] = trainingSample.L_after_spectral[2];
                            // binaryOutput.L_after_spectral[3] = trainingSample.L_after_spectral[3];
    
                            binaryOutput.T_after_spectral[0] = trainingSample.T_after_spectral[0];
                            binaryOutput.T_after_spectral[1] = trainingSample.T_after_spectral[1];
                            binaryOutput.T_after_spectral[2] = trainingSample.T_after_spectral[2];
                            binaryOutput.T_after_spectral[3] = trainingSample.T_after_spectral[3];
    
                            binaryOutput.tMax = trainingSample.tMax;

                        }
                        
                    } else {
                        // batchOutput += StringPrintf(
                        //     "%s|%s|%d|%d|%d|%s|%s\n",
                        //     sample.ray.o.ToString(), sample.ray.d.ToString(), sample.pixelIdx, batch.sampleIndex,
                        //     L_target.ToString(), rgb.ToString()
                        // );
                    }
                }
            }
            if(!binaryBatchOutput.empty() || !binaryBatchOutputSpectral.empty())
            {
                LOG_VERBOSE("Binary batch output not empty, writing...\n");
                std::lock_guard<std::mutex> fileLock(rayLogFileMutex);
                if(outputRayDataFile)
                {
                    LOG_VERBOSE("OutputRayDataFile not empty, writing...\n");
                    if(false)
                    {

                        // outputRayDataFile->write(batchOutput.c_str(), batchOutput.length());
                        outputRayDataFile->seekp(0, std::ios::end);
                        outputRayDataFile->write(
                        reinterpret_cast<const char *>(binaryBatchOutput.data()),
                        binaryBatchOutput.size() * sizeof(BinaryTrainingSample));

                        
                        rayLogSampleCnt += batch.trainingData.size();
                    }
                    else
                    {
                        outputRayDataFile->seekp(0, std::ios::end);
                        outputRayDataFile->write(
                        reinterpret_cast<const char *>(binaryBatchOutputSpectral.data()),
                        binaryBatchOutputSpectral.size() * sizeof(BinaryTrainingSampleSpectral));

                        
                        rayLogSampleCnt += batch.trainingData.size();
                    }
                        
                        
                        
                    outputRayDataFile->seekp(0, std::ios::beg);
                    outputRayDataFile->write(reinterpret_cast<const char*>(&rayLogSampleCnt), sizeof(uint64_t));
                        
                    outputRayDataFile->flush();
                }
            }
        }
    }

    void WavefrontPathIntegrator::HandleRayLogging(int depthIndex, int sampleIndex)
    {
        if(!Options->useGPU) return;

        constexpr size_t maxQueuedBatches = 16;
        {
            std::unique_lock<std::mutex> lock(rayLogMutex);
            while(rayLogWorkQueue.size() >= maxQueuedBatches && !rayLogShutdown)
            {
                lock.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                lock.lock();
            }
        }

        int nItems = 0;

        CUDA_CHECK(cudaMemcpy(&nItems, pendingSamplesCnt, sizeof(int), cudaMemcpyDeviceToHost));
        if(nItems == 0) return;


        int nToCopy = std::min(nItems, pendingSamplesMaxSize);


        GPUWait();
        
        // NOTE: Due to SOA, can't easily just do a cudaMemcpy. Therefore, just copy over and do the work here.
        // TODO: Figure out a way to just be able to perform a memcpy and just copy this over from the GPU side in 1 go.
        // And then i can probably set the writing off to be in a separate thread or something like that.
        

    
        std::vector<TrainingDataSample> samples(nToCopy);
        
        CUDA_CHECK(cudaMemcpy(samples.data(), pendingRayData, nToCopy * sizeof(TrainingDataSample), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemset(pendingSamplesCnt, 0, sizeof(int)));

        if(samples.empty()) return;

        //Prepare batch
        RayLogBatch batch;

        batch.sampleIndex = sampleIndex;
        batch.trainingData = std::move(samples);


        nItems = batch.trainingData.size();
        // batch.finalLValues.resize(nItems);
        // batch.lambdas.resize(nItems);


        // Gather step
        for(size_t i = 0; i < nItems; ++i)
        {
            // int pixelIdx = batch.trainingData[i].pixelIdx;
            // Note: If outputRayData is on GPU, this loop might be slow. 
            // Ideally, you would use a CUDA kernel to gather these into a buffer first.
            // batch.finalLValues[i] = outputRayData.L[pixelIdx];
            // batch.lambdas[i] = outputRayData.lambda[pixelIdx];
        }

        //Push to queue
        {
            std::lock_guard<std::mutex> lock(rayLogMutex);
            rayLogWorkQueue.push(std::move(batch));
        }

        rayLogCondition.notify_one();

        
        // for (int i = 0; i < nItems; ++i)
        //     samples[i] = (*rayLogQueue)[i];

        //TODO: Consider using a cudaMemcpy(Async) to perform this so its slightly quicker.
        // for (const auto& sample : samples)
        // {
        //     const int pixelIndex = sample.pixelIdx;
        //     // Read data from SOA structure
        //     Point3f p = sample.ray.o;
        //     Vector3f wo = sample.ray.d;
        //     SampledSpectrum beta = sample.beta;
        //     SampledSpectrum L_prefix = sample.L;
        //     int depthIdx = sample.depthIdx;
        //     SampledSpectrum L_final = outputRayData.L[pixelIndex];
        //     SampledWavelengths lambda = outputRayData.lambda[pixelIndex];


        //     //TODO: Figure out proper way to deal with this
        //     if(sample.beta == SampledSpectrum(0.f))
        //     {
        //         continue;
        //     }

        //     SampledSpectrum L_target = (L_final - L_prefix) / sample.beta;


        //     RGB rgb = film.ToOutputRGB(L_target, lambda);

        //     if(L_target.HasNaNs() | rgb.r == Infinity | rgb.g == Infinity | rgb.b == Infinity)
        //     {
        //         continue;
        //     }
            
        //     std::string outputLine = StringPrintf(
        //         "Ray (p: %s, wo: %s) PixelIdx (%d) Sample (%d) FinalDepth (%d) Luminance (%s) RGB (%s)\n",
        //         p.ToString(), wo.ToString(), pixelIndex, sampleIndex, depthIdx, 
        //         L_target.ToString(), rgb.ToString()
        //     );

        //     batchOutput += StringPrintf(
        //     "%s|%s|%d|%d|%d|%s|%s\n",
        //     sample.ray.o.ToString(), sample.ray.d.ToString(), sample.pixelIdx, batch.sampleIndex,
        //     L_target.ToString(), rgb.ToString()
        //             );
        //     outputRayDataFile->write(outputLine.c_str(), outputLine.length());
        // }
        // outputRayDataFile->flush();


        CUDA_CHECK(cudaMemset(pendingSamplesCnt, 0, sizeof(int)));

        // rayLogQueue->Reset();
    }
    


    void WavefrontPathIntegrator::OutputRayDataToFiles()
    {
        if (!outputToFile || !outputRayDataFile || !inputRayDataFile) return;
        // INFO: Wait for GPU synchronization

#ifdef PBRT_BUILD_GPU_RENDERER
        GPUWait();
#endif
        // Copy data from GPU to CPU if using GPU rendering
        // The SOA data will be automatically synchronized when accessed on CPU

        //For now, we just set a limit on the size of the files, set to 1GB for now
                // Ensure file pointers are positioned at the end so writes append.

        auto outPos = outputRayDataFile->tellp();
        auto inPos  = inputRayDataFile->tellp();

        // If tellp() is not valid, try seeking to end and re-query.
        if (outPos == std::streampos(-1) || inPos == std::streampos(-1)) {
            outputRayDataFile->seekp(0, std::ios::end);
            inputRayDataFile->seekp(0, std::ios::end);
            outPos = outputRayDataFile->tellp();
            inPos  = inputRayDataFile->tellp();
        }

        // If either file is already >= 1 GiB, don't write anything.
        if ((outPos != std::streampos(-1) && static_cast<uint64_t>(outPos) >= kMaxBytes) ||
            (inPos  != std::streampos(-1) && static_cast<uint64_t>(inPos)  >= kMaxBytes)) {
            LOG_VERBOSE("Files are larger than 1GB, skipping outputting...\n");
            return;
        }

        // Make sure we append to the end.
        outputRayDataFile->seekp(0, std::ios::end);
        inputRayDataFile->seekp(0, std::ios::end);

        Bounds2i pixelBounds = film.PixelBounds();
        Vector2i resolution = pixelBounds.Diagonal();

        // Loop over all pixel samples and write ray data to file
        for (int pixelIndex = 0; pixelIndex < maxQueueSize; ++pixelIndex)
        {
            //TODO: Consider using a cudaMemcpy(Async) to perform this so its slightly quicker.

            Point3f iRayo = inputRayData.rayo[pixelIndex];
            Vector3f iRayd = inputRayData.rayd[pixelIndex];
            int iDepth = inputRayData.depth[pixelIndex];
            int iSampleIdx = inputRayData.sampleIdx[pixelIndex];

            // Read data from SOA structure
            Point3f oRayo = outputRayData.rayo[pixelIndex];
            Vector3f oRayd = outputRayData.rayd[pixelIndex];
            int oDepth = outputRayData.depth[pixelIndex];
            int oSampleIdx = outputRayData.sampleIdx[pixelIndex];
            SampledSpectrum L = outputRayData.L[pixelIndex];
            SampledWavelengths lambda = outputRayData.lambda[pixelIndex];
            //TODO: Set lambda in cpu side

            
            //TODO: Check if this is the correct way to get the pixelrgb
            // RGB rgb = film.GetPixelRGB(p + pixelBounds.pMin);
            RGB rgb = film.ToOutputRGB(L, lambda);

            // std::string inputLine = StringPrintf("Pixel (o: %s, d: %s) PixelIdx (%d) Sample (%d) InitialDepth (%d)\n",
            //     iRayo.ToString(), iRayd.ToString(), pixelIndex, iSampleIdx, iDepth
            // );
            // inputRayDataFile->write(inputLine.c_str(), inputLine.length());

            std::string outputLine = StringPrintf("Pixel (o: %s, d: %s) PixelIdx (%d) Sample (%d) FinalDepth (%d) Luminance (%s) RGB (%s)\n",
                oRayo.ToString(), oRayd.ToString(), pixelIndex, oSampleIdx, oDepth, L.ToString(), rgb.ToString());
            outputRayDataFile->write(outputLine.c_str(), outputLine.length());
        }

        //NOTE: Proposed Async solution
        //             Bounds2i pixelBounds = film.PixelBounds();

        //     // Allocate pinned CPU memory for faster async transfers
        //     Point2i* pPixel_cpu;
        //     Point3f* rayo_cpu;
        //     Vector3f* rayd_cpu;
        //     int* depth_cpu;
        //     int* sampleIdx_cpu;

        // #ifdef PBRT_BUILD_GPU_RENDERER
        //     if (Options->useGPU) {
        //         // Allocate pinned (page-locked) host memory for async transfer
        //         cudaMallocHost(&pPixel_cpu, maxQueueSize * sizeof(Point2i));
        //         cudaMallocHost(&rayo_cpu, maxQueueSize * sizeof(Point3f));
        //         cudaMallocHost(&rayd_cpu, maxQueueSize * sizeof(Vector3f));
        //         cudaMallocHost(&depth_cpu, maxQueueSize * sizeof(int));
        //         cudaMallocHost(&sampleIdx_cpu, maxQueueSize * sizeof(int));

        //         // Async copy from GPU to CPU (non-blocking)
        //         cudaMemcpyAsync(pPixel_cpu, outputRayData.pPixel, 
        //                     maxQueueSize * sizeof(Point2i), cudaMemcpyDeviceToHost);
        //         cudaMemcpyAsync(rayo_cpu, outputRayData.rayo, 
        //                     maxQueueSize * sizeof(Point3f), cudaMemcpyDeviceToHost);
        //         cudaMemcpyAsync(rayd_cpu, outputRayData.rayd, 
        //                     maxQueueSize * sizeof(Vector3f), cudaMemcpyDeviceToHost);
        //         cudaMemcpyAsync(depth_cpu, outputRayData.depth, 
        //                     maxQueueSize * sizeof(int), cudaMemcpyDeviceToHost);
        //         cudaMemcpyAsync(sampleIdx_cpu, outputRayData.sampleIdx, 
        //                     maxQueueSize * sizeof(int), cudaMemcpyDeviceToHost);

        //         // Synchronize only the copy stream (not the render stream)
        //         cudaStreamSynchronize(0);
        //     } else
        // #endif
        //     {
        //         // CPU rendering - just use the pointers directly
        //         pPixel_cpu = outputRayData.pPixel;
        //         rayo_cpu = outputRayData.rayo;
        //         rayd_cpu = outputRayData.rayd;
        //         depth_cpu = outputRayData.depth;
        //         sampleIdx_cpu = outputRayData.sampleIdx;
        //     }


        //         *outputRayDataFile << "Pixel: (" << pPixel.x << ", " << pPixel.y << ") "
        //                         << "Sample: " << sampleIdx_cpu[pixelIndex] << " "
        //                         << "Depth: " << depth_cpu[pixelIndex] << " "
        //                         << "Origin: (" << rayo_cpu[pixelIndex].x << ", " 
        //                         << rayo_cpu[pixelIndex].y << ", " << rayo_cpu[pixelIndex].z << ") "
        //                         << "Direction: (" << rayd_cpu[pixelIndex].x << ", " 
        //                         << rayd_cpu[pixelIndex].y << ", " << rayd_cpu[pixelIndex].z << ")\n";
        //     }

        //     outputRayDataFile->flush();

        // #ifdef PBRT_BUILD_GPU_RENDERER
        //     if (Options->useGPU) {
        //         // Free pinned memory
        //         cudaFreeHost(pPixel_cpu);
        //         cudaFreeHost(rayo_cpu);
        //         cudaFreeHost(rayd_cpu);
        //         cudaFreeHost(depth_cpu);
        //         cudaFreeHost(sampleIdx_cpu);
        //     }
        // #endif

            // Flush the file to ensure data is written
        outputRayDataFile->flush();
    }


}
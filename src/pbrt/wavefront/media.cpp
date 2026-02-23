// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/wavefront/integrator.h>
#ifdef PBRT_BUILD_GPU_RENDERER
#include <neural-graphics-primitives/testbed.h>
#endif
#include <pbrt/media.h>

#include <type_traits>

namespace pbrt
{

    // SampleMediumScatteringCallback Definition
    struct SampleMediumScatteringCallback
    {
        int wavefrontDepth;
        WavefrontPathIntegrator* integrator;
        template <typename PhaseFunction>
        void operator()()
        {
            integrator->SampleMediumScattering<PhaseFunction>(wavefrontDepth);
        }
    };


    void WavefrontPathIntegrator::TransmissionOnly(int wavefrontDepth)
    {
        RayQueue *nextRayQueue = NextRayQueue(wavefrontDepth);

        float* d_inputs = inferInputs;
        float* d_outputs = inferOutputs;
        int* d_pixelIndices = inferPixelIndices;
        SampledSpectrum *d_betaBefore = inferBetaBefore;
        SampledWavelengths* d_lambda = inferLambda;
        int* d_itemCount = inferItemCount;
        int maxBatchSize = maxInferenceBatchSize;

        int *d_slotMap = inferSlotMap;
        SampledSpectrum *d_betaAfter = inferBetaAfter;
        SampledSpectrum* d_ruBefore = inferRuBefore;
        SampledSpectrum *d_ruAfter = inferRuAfter;
        SampledSpectrum *d_rlBefore = inferRlBefore;
        SampledSpectrum *d_rlAfter = inferRlAfter;

        const float posScale = m_Testbed->m_volume_training_inputs_scale;
        const vec3 posOffset = m_Testbed->m_volume_training_inputs_offset;
        const float tMaxScale = m_Testbed->m_volume_training_inputs_tMax_scale;
        LOG_VERBOSE("TMax scale is %f", tMaxScale);
        const float tMaxOffset = m_Testbed->m_volume_training_inputs_tMax_offset;
        const float tAfterScale = m_Testbed->m_volume_training_inputs_tAfter_scale;

        // Reset item count
        Do("Reset NGP item count", PBRT_CPU_GPU_LAMBDA() { *d_itemCount = 0; });

        // Reset slotMap to -1 for all pixel indices
        int maxPixels = maxQueueSize;
        ParallelFor(
            "Reset NGP slotmap", maxPixels,
            PBRT_CPU_GPU_LAMBDA(int i){
                d_slotMap[i] = -1;
            }
        );

        auto pixelSampleState = this->pixelSampleState;

        // ******************************************
        // Step 1+2: Prepare and Normalize inputs and record slot mapping        
        // ******************************************

        ForAllQueued(
            "Pack medium samples for NGP", mediumSampleQueue, maxQueueSize,
            PBRT_CPU_GPU_LAMBDA(MediumSampleWorkItem w) {
#ifdef PBRT_IS_GPU_CODE
                int slot = atomicAdd(d_itemCount, 1);
#else
                int slot = __sync_fetch_and_add(d_itemCount, 1);
#endif

                if(slot >= maxBatchSize)
                    return;
                
                d_pixelIndices[slot] = w.pixelIndex;
                d_slotMap[w.pixelIndex] = slot;
                d_betaBefore[slot] = w.beta;
                d_lambda[slot] = w.lambda;
                d_ruBefore[slot] = w.r_u;
                d_rlBefore[slot] = w.r_l;

                constexpr int N_IN = 7;
                Point3f pos = w.ray.o;
                Vector3f dir = Normalize(w.ray.d);
                Float tMax = w.tMax;

                constexpr float MAX_SCENE_DIST = 160.f;
                float tMaxClamped = std::min(float(tMax), MAX_SCENE_DIST);

                d_inputs[slot * N_IN + 0] = float(pos.x) * posScale + posOffset.x;
                d_inputs[slot * N_IN + 1] = float(pos.y) * posScale + posOffset.y;
                d_inputs[slot * N_IN + 2] = float(pos.z) * posScale + posOffset.z;

                d_inputs[slot * N_IN + 3] = float(dir.x) * 0.5f + 0.5f;
                d_inputs[slot * N_IN + 4] = float(dir.y) * 0.5f + 0.5f;
                d_inputs[slot * N_IN + 5] = float(dir.z) * 0.5f + 0.5f;

                d_inputs[slot * N_IN + 6] = (tMaxClamped - tMaxOffset) * tMaxScale;

                if(slot < 5)
                {
                    // printf("Input slot %d pixel %d: pos=[%f %f %f] dir=[%f %f %f] posOffset=[%f %f %f] posScale=[%f]\n",
                    //     slot, d_pixelIndices[slot],
                    //     d_inputs[slot * 7 + 0], d_inputs[slot * 7 + 1], d_inputs[slot * 7 + 2],
                    //     d_inputs[slot * 7 + 3], d_inputs[slot * 7 + 4], d_inputs[slot * 7 + 5],
                    //     posOffset.x, posOffset.y, posOffset.z, posScale);
                }
            });

        #ifdef PBRT_BUILD_GPU_RENDERER
            if(Options->useGPU)
            {
                GPUWait();
            }
        #endif

            int nItems = 0;
        #ifdef PBRT_BUILD_GPU_RENDERER
            if(Options->useGPU)
            {
                CUDA_CHECK(cudaMemcpy(&nItems, d_itemCount, sizeof(int), cudaMemcpyDeviceToHost));
            } else
        #endif
            {
                nItems = *d_itemCount;
            }

            nItems = std::min(nItems, maxBatchSize);

            if(nItems == 0)
                return;

            // ******************************************
            // Step 3: Perform Inference via Instant-NGP        
            // ******************************************

            InferNGP(uint64_t(nItems), d_inputs, d_outputs);

            #ifdef PBRT_BUILD_GPU_RENDERER
                if(Options->useGPU)
                {
                    GPUWait();
                }
            #endif

            // auto outputRayData = this->outputRayData;

            Film film = this->film;

            #ifdef PBRT_BUILD_GPU_RENDERER
                if(Options->useGPU)
                {
                    GPUWait();
                }
            #endif

            // ******************************************
            // Step 4: Read network outputs and use to calculate new Le and beta values   
            // ******************************************

            ParallelFor(
                "Unpack NGP medium outputs", nItems, PBRT_CPU_GPU_LAMBDA(int slot) {
                    constexpr int N_OUT = 4;

                    int pixelIndex = d_pixelIndices[slot];
                    SampledSpectrum betaBefore = d_betaBefore[slot];
                    SampledWavelengths lambda = d_lambda[slot];
                    SampledSpectrum ruBefore = d_ruBefore[slot];
                    SampledSpectrum rlBefore = d_rlBefore[slot];

                    // float L_r = d_outputs[slot * N_OUT + 0];
                    // float L_g = d_outputs[slot * N_OUT + 1];
                    // float L_b = d_outputs[slot * N_OUT + 2];
                    // float T_r = d_outputs[slot * N_OUT + 3];
                    // float T_g = d_outputs[slot * N_OUT + 4];
                    // float T_b = d_outputs[slot * N_OUT + 5];

                    // float Lr_actual = fmaxf(0.f, expf(L_r) - 1.f);
                    // float Lg_actual = fmaxf(0.f, expf(L_g) - 1.f);
                    // float Lb_actual = fmaxf(0.f, expf(L_b) - 1.f);


                    // const RGBFilm *rgbFilm = film.CastOrNullptr<RGBFilm>();
                    // const RGBColorSpace *cs = rgbFilm ? rgbFilm->colorSpace : nullptr;

                    // RGB Le_rgb(Lr_actual, Lr_actual, Lr_actual);
                    // RGB T_rgb(T_r, T_g, T_b);
                    SampledSpectrum betaAfter = betaBefore;  // no attenutation by default
                    SampledSpectrum transmittance_output = betaBefore;
                    transmittance_output[0] = d_outputs[slot * N_OUT + 0];
                    transmittance_output[1] = d_outputs[slot * N_OUT + 1];
                    transmittance_output[2] = d_outputs[slot * N_OUT + 2];
                    transmittance_output[3] = d_outputs[slot * N_OUT + 3];

                    SampledSpectrum L_added(0.f);
                    SampledSpectrum ruAfter = ruBefore;
                    SampledSpectrum rlAfter = rlBefore;

                    if(transmittance_output[0] < 0.5f || transmittance_output[1] < 0.5f)
                    {
                        // printf("LOW T slot %d pixel %d: T=[%f %f %f %f] betaBefore=[%f %f %f %f]\n",
                        //     slot, pixelIndex,
                        //     transmittance_output[0], transmittance_output[1],
                        //     transmittance_oz`utput[2], transmittance_output[3],
                        //     betaBefore[0], betaBefore[1], betaBefore[2], betaBefore[3]);
                    }

                    for(int i = 0; i < NSpectrumSamples; ++i)
                    {
                        Float T_i = transmittance_output[i];
                        Float to_use = T_i;
                        // if (T_i < 0.75f) {
                        //     T_i = 0.f;
                        //     to_use = T_i * betaBefore[i];
                        // } else {
                        //     to_use = T_i;
                        // }
                        to_use = std::max(0.f, (T_i - 0.55f) / 0.45f);
                        betaAfter[i] = to_use * betaBefore[i];
                        ruAfter[i] = T_i * ruBefore[i]; // To get smoky effect need to use T_i or just pass thru ruAfter
                        rlAfter[i] = T_i * rlBefore[i]; // To get smoky effect need to use T_i or just pass thru rlAfter
                    }
                    
                    // if(slot < 5)
                    // {
                    //     printf("Slot %d pixel %d: T=[%f %f %f %f] betaBefore=[%f %f %f %f] betaAfter=[%f %f %f %f]\n",
                    //         slot, pixelIndex,
                    //         transmittance_output[0], transmittance_output[1], 
                    //         transmittance_output[2], transmittance_output[3],
                    //         betaBefore[0], betaBefore[1], betaBefore[2], betaBefore[3],
                    //         betaAfter[0], betaAfter[1], betaAfter[2], betaAfter[3]);
                    // }

                    // if(cs)
                    // {
                    //     RGBIlluminantSpectrum Le_spec(*cs, ClampZero(Le_rgb));
                    //     RGBUnboundedSpectrum T_spec(*cs, Clamp(T_rgb, 0.f, 1.f)); //NOTE: Should we Clamp 0,1 here?

                    // }
                    // Store attenuated beta for dispatch loop
                    d_betaAfter[slot] = betaAfter;
                    d_ruAfter[slot] = ruAfter;
                    d_rlAfter[slot] = rlAfter;

                    // SampledSpectrum Lp = pixelSampleState.L[pixelIndex];
                    // pixelSampleState.L[pixelIndex] = betaAfter;            
                });

            // ******************************************
            // Step 5: Enqueue work items based on ray values/MediumSampleWorkItem values   
            // ******************************************
            ForAllQueued(
                "Post-NGP medium dispatch", mediumSampleQueue, maxQueueSize,
                PBRT_CPU_GPU_LAMBDA(MediumSampleWorkItem w) {
                    int slot = d_slotMap[w.pixelIndex];

                    if(slot < 0 || slot >= maxBatchSize)
                        return;

                    SampledSpectrum beta = d_betaAfter[slot];
                    SampledSpectrum r_u = d_ruAfter[slot];
                    SampledSpectrum r_l = d_rlAfter[slot];

                    if(!beta)
                        return;

                    if (w.depth >= maxDepth)
                        return;

                    if(w.tMax == Infinity)
                    {
                        if(escapedRayQueue)
                        {
                            escapedRayQueue->Push(EscapedRayWorkItem{
                                w.ray.o, w.ray.d, w.depth, w.lambda, w.pixelIndex,
                                beta, (int)w.specularBounce, r_u, r_l,
                                w.prevIntrCtx
                            });
                        }
                        return;
                    }
                    // Surface intersection handling
                    Material material = w.material;
                    const MixMaterial *mix = material.CastOrNullptr<MixMaterial>();

                    while(mix)
                    {
                        SurfaceInteraction intr(w.pi, w.uv, w.wo, w.dpdus, w.dpdvs,
                                                w.dndus, w.dndvs, w.ray.time, false);
                        intr.faceIndex = w.faceIndex;
                        MaterialEvalContext ctx(intr);
                        material = mix->ChooseMaterial(BasicTextureEvaluator(), ctx);
                        mix = material.CastOrNullptr<MixMaterial>();
                    }

                    if(!material)
                    {
                        Interaction intr(w.pi, w.n);
                        intr.mediumInterface = &w.mediumInterface;
                        Ray newRay = intr.SpawnRay(w.ray.d);
                        nextRayQueue->PushIndirectRay(
                            newRay, w.depth, w.prevIntrCtx, beta, r_u, r_l,
                            w.lambda, w.etaScale, w.specularBounce,
                            w.anyNonSpecularBounces, w.pixelIndex);
                        return;
                    }

                    if(w.areaLight)
                    {
                        hitAreaLightQueue->Push(HitAreaLightWorkItem{
                            w.areaLight, Point3f(w.pi), w.n, w.uv, -w.ray.d, w.lambda,
                            w.depth, beta, r_u, r_l, w.prevIntrCtx,
                            w.specularBounce, w.pixelIndex});
                    }

                    FloatTexture displacement = material.GetDisplacement();
                    MaterialEvalQueue *q =
                        (material.CanEvaluateTextures(BasicTextureEvaluator()) &&
                         (!displacement ||
                          BasicTextureEvaluator().CanEvaluate({displacement}, {})))
                            ? basicEvalMaterialQueue
                            : universalEvalMaterialQueue;
                    auto enqueue = [=](auto ptr) {
                        using Material = typename std::remove_reference_t<decltype(*ptr)>;
                        q->Push<MaterialEvalWorkItem<Material>>(
                            MaterialEvalWorkItem<Material>{ptr,
                                                           w.pi,
                                                           w.n,
                                                           w.dpdu,
                                                           w.dpdv,
                                                           w.ray.time,
                                                           w.depth,
                                                           w.ns,
                                                           w.dpdus,
                                                           w.dpdvs,
                                                           w.dndus,
                                                           w.dndvs,
                                                           w.uv,
                                                           w.faceIndex,
                                                           w.lambda,
                                                           w.pixelIndex,
                                                           w.anyNonSpecularBounces,
                                                           -w.ray.d,
                                                           beta,
                                                           r_u,
                                                           w.etaScale,
                                                           w.mediumInterface});
                        
                    };
                    material.Dispatch(enqueue);
                });
                
        if (wavefrontDepth == maxDepth)
            return;

        ForEachType(SampleMediumScatteringCallback{ wavefrontDepth, this },
            PhaseFunction::Types());
    }

    void WavefrontPathIntegrator::SampleMediumInteractionNGP(int wavefrontDepth)
    {
        if(!haveMedia)
            return;
        if(!m_Testbed)
            return;
        if(Options->useSpectralModel)
        {
            TransmissionOnly(wavefrontDepth);
            return;
        }
        else
        {
            LOG_VERBOSE("Im here");
        }
        RayQueue *nextRayQueue = NextRayQueue(wavefrontDepth);

        float* d_inputs = inferInputs;
        float* d_outputs = inferOutputs;
        int* d_pixelIndices = inferPixelIndices;
        SampledSpectrum *d_betaBefore = inferBetaBefore;
        SampledWavelengths* d_lambda = inferLambda;
        int* d_itemCount = inferItemCount;
        int maxBatchSize = maxInferenceBatchSize;

        int *d_slotMap = inferSlotMap;
        SampledSpectrum *d_betaAfter = inferBetaAfter;
        SampledSpectrum* d_ruBefore = inferRuBefore;
        SampledSpectrum *d_ruAfter = inferRuAfter;
        SampledSpectrum *d_rlBefore = inferRlBefore;
        SampledSpectrum *d_rlAfter = inferRlAfter;
        
        const float posScale = m_Testbed->m_volume_training_inputs_scale;
        const vec3 posOffset = m_Testbed->m_volume_training_inputs_offset;
        const float tMaxScale = m_Testbed->m_volume_training_inputs_tMax_scale;
        const float tMaxOffset = m_Testbed->m_volume_training_inputs_tMax_offset;
        const float tAfterScale = m_Testbed->m_volume_training_inputs_tAfter_scale;
        LOG_VERBOSE("Tafter scale is %f", tAfterScale);
        // Reset item count
        Do("Reset NGP item count", PBRT_CPU_GPU_LAMBDA() { *d_itemCount = 0; });

        // Reset slotMap to -1 for all pixel indices
        int maxPixels = maxQueueSize;
        ParallelFor(
            "Reset NGP slotmap", maxPixels,
            PBRT_CPU_GPU_LAMBDA(int i){
                d_slotMap[i] = -1;
            }
        );

        auto pixelSampleState = this->pixelSampleState;

        // ******************************************
        // Step 1+2: Prepare and Normalize inputs and record slot mapping        
        // ******************************************

        ForAllQueued(
            "Pack medium samples for NGP", mediumSampleQueue, maxQueueSize,
            PBRT_CPU_GPU_LAMBDA(MediumSampleWorkItem w) {
#ifdef PBRT_IS_GPU_CODE
                int slot = atomicAdd(d_itemCount, 1);
#else
                int slot = __sync_fetch_and_add(d_itemCount, 1);
#endif

                if(slot >= maxBatchSize)
                    return;
                
                d_pixelIndices[slot] = w.pixelIndex;
                d_slotMap[w.pixelIndex] = slot;
                d_betaBefore[slot] = w.beta;
                d_lambda[slot] = w.lambda;
                d_ruBefore[slot] = w.r_u;
                d_rlBefore[slot] = w.r_l;

                constexpr int N_IN = 7;
                Point3f pos = w.ray.o;
                Vector3f dir = Normalize(w.ray.d);
                Float tMax = w.tMax;

                constexpr float MAX_SCENE_DIST = 160.f;
                float tMaxClamped = std::min(float(tMax), MAX_SCENE_DIST);

                d_inputs[slot * N_IN + 0] = float(pos.x) * posScale + posOffset.x;
                d_inputs[slot * N_IN + 1] = float(pos.y) * posScale + posOffset.y;
                d_inputs[slot * N_IN + 2] = float(pos.z) * posScale + posOffset.z;

                d_inputs[slot * N_IN + 3] = float(dir.x) * 0.5f + 0.5f;
                d_inputs[slot * N_IN + 4] = float(dir.y) * 0.5f + 0.5f;
                d_inputs[slot * N_IN + 5] = float(dir.z) * 0.5f + 0.5f;

                d_inputs[slot * N_IN + 6] = (tMaxClamped - tMaxOffset) * tMaxScale;
            });

        #ifdef PBRT_BUILD_GPU_RENDERER
            if(Options->useGPU)
            {
                GPUWait();
            }
        #endif

            int nItems = 0;
        #ifdef PBRT_BUILD_GPU_RENDERER
            if(Options->useGPU)
            {
                CUDA_CHECK(cudaMemcpy(&nItems, d_itemCount, sizeof(int), cudaMemcpyDeviceToHost));
            } else
        #endif
            {
                nItems = *d_itemCount;
            }

            nItems = std::min(nItems, maxBatchSize);

            if(nItems == 0)
                return;

            // ******************************************
            // Step 3: Perform Inference via Instant-NGP        
            // ******************************************

            InferNGP(uint64_t(nItems), d_inputs, d_outputs);

            #ifdef PBRT_BUILD_GPU_RENDERER
                if(Options->useGPU)
                {
                    GPUWait();
                }
            #endif

            // auto outputRayData = this->outputRayData;

            Film film = this->film;

            // ******************************************
            // Step 4: Read network outputs and use to calculate new Le and beta values   
            // ******************************************
            ParallelFor(
                "Unpack NGP medium outputs", nItems, PBRT_CPU_GPU_LAMBDA(int slot) {
                    constexpr int N_OUT = 6;

                    int pixelIndex = d_pixelIndices[slot];
                    SampledSpectrum betaBefore = d_betaBefore[slot];
                    SampledWavelengths lambda = d_lambda[slot];
                    SampledSpectrum ruBefore = d_ruBefore[slot];
                    SampledSpectrum rlBefore = d_rlBefore[slot];

                    float L_r = d_outputs[slot * N_OUT + 0];
                    float L_g = d_outputs[slot * N_OUT + 1];
                    float L_b = d_outputs[slot * N_OUT + 2];
                    float T_r = d_outputs[slot * N_OUT + 3];
                    float T_g = d_outputs[slot * N_OUT + 4];
                    float T_b = d_outputs[slot * N_OUT + 5];

                    float Lr_actual = fmaxf(0.f, expf(L_r) - 1.f);
                    float Lg_actual = fmaxf(0.f, expf(L_g) - 1.f);
                    float Lb_actual = fmaxf(0.f, expf(L_b) - 1.f);


                    const RGBFilm *rgbFilm = film.CastOrNullptr<RGBFilm>();
                    const RGBColorSpace *cs = rgbFilm ? rgbFilm->colorSpace : nullptr;

                    RGB Le_rgb(Lr_actual, Lg_actual, Lb_actual);
                    RGB T_rgb(T_r, T_g, T_b);

                    T_rgb *= tAfterScale;

                    SampledSpectrum L_added(0.f);
                    SampledSpectrum betaAfter = betaBefore;  // no attenutation by default
                    SampledSpectrum ruAfter = ruBefore;
                    SampledSpectrum rlAfter = rlBefore;
                    
                    if(cs)
                    {
                        RGBIlluminantSpectrum Le_spec(*cs, ClampZero(Le_rgb));
                        RGBUnboundedSpectrum T_spec(*cs, T_rgb); //NOTE: Should we Clamp 0,1 here?

                        for(int i = 0; i < NSpectrumSamples; ++i)
                        {
                            Float T_i = T_spec(lambda[i]);
                            L_added[i] = Le_spec(lambda[i]) * betaBefore[i];
                            betaAfter[i] = T_i * betaBefore[i];
                            ruAfter[i] = T_i * ruBefore[i];
                            rlAfter[i] = T_i * rlBefore[i];
                            if(L_added[i] > 1e-2f)
                            {
                                L_added[i] *= 10.f;
                                betaAfter[i] = 1e-3f;
                                ruAfter[i] = 1e-3f;
                                rlAfter[i] = 1e-3f;
                            }
                            // else {
                                // betaAfter[i] = 0.f;
                                // ruAfter[i] = 0.f;
                                // rlAfter[i] = 0.f;
                            // }
                        }
                    }
                    // Store attenuated beta for dispatch loop
                    d_betaAfter[slot] = betaAfter;
                    d_ruAfter[slot] = ruAfter;
                    d_rlAfter[slot] = rlAfter;

                    SampledSpectrum Lp = pixelSampleState.L[pixelIndex];
                    pixelSampleState.L[pixelIndex] = L_added;
                });

            // ******************************************
            // Step 5: Enqueue work items based on ray values/MediumSampleWorkItem values   
            // ******************************************
            ForAllQueued(
                "Post-NGP medium dispatch", mediumSampleQueue, maxQueueSize,
                PBRT_CPU_GPU_LAMBDA(MediumSampleWorkItem w) {
                    int slot = d_slotMap[w.pixelIndex];

                    if(slot < 0 || slot >= maxBatchSize)
                        return;

                    SampledSpectrum beta = d_betaAfter[slot];
                    SampledSpectrum r_u = d_ruAfter[slot];
                    SampledSpectrum r_l = d_rlAfter[slot];

                    if(!beta)
                        return;

                    if (w.depth >= maxDepth)
                        return;

                    if(w.tMax == Infinity)
                    {
                        if(escapedRayQueue)
                        {
                            escapedRayQueue->Push(EscapedRayWorkItem{
                                w.ray.o, w.ray.d, w.depth, w.lambda, w.pixelIndex,
                                beta, (int)w.specularBounce, r_u, r_l,
                                w.prevIntrCtx
                            });
                        }
                        return;
                    }
                    // Surface intersection handling
                    Material material = w.material;
                    const MixMaterial *mix = material.CastOrNullptr<MixMaterial>();

                    while(mix)
                    {
                        SurfaceInteraction intr(w.pi, w.uv, w.wo, w.dpdus, w.dpdvs,
                                                w.dndus, w.dndvs, w.ray.time, false);
                        intr.faceIndex = w.faceIndex;
                        MaterialEvalContext ctx(intr);
                        material = mix->ChooseMaterial(BasicTextureEvaluator(), ctx);
                        mix = material.CastOrNullptr<MixMaterial>();
                    }

                    if(!material)
                    {
                        Interaction intr(w.pi, w.n);
                        intr.mediumInterface = &w.mediumInterface;
                        Ray newRay = intr.SpawnRay(w.ray.d);
                        nextRayQueue->PushIndirectRay(
                            newRay, w.depth, w.prevIntrCtx, beta, r_u, r_l,
                            w.lambda, w.etaScale, w.specularBounce,
                            w.anyNonSpecularBounces, w.pixelIndex);
                        return;
                    }

                    if(w.areaLight)
                    {
                        hitAreaLightQueue->Push(HitAreaLightWorkItem{
                            w.areaLight, Point3f(w.pi), w.n, w.uv, -w.ray.d, w.lambda,
                            w.depth, beta, r_u, r_l, w.prevIntrCtx,
                            w.specularBounce, w.pixelIndex});
                    }

                    FloatTexture displacement = material.GetDisplacement();
                    MaterialEvalQueue *q =
                        (material.CanEvaluateTextures(BasicTextureEvaluator()) &&
                         (!displacement ||
                          BasicTextureEvaluator().CanEvaluate({displacement}, {})))
                            ? basicEvalMaterialQueue
                            : universalEvalMaterialQueue;
                    auto enqueue = [=](auto ptr) {
                        using Material = typename std::remove_reference_t<decltype(*ptr)>;
                        q->Push<MaterialEvalWorkItem<Material>>(
                            MaterialEvalWorkItem<Material>{ptr,
                                                           w.pi,
                                                           w.n,
                                                           w.dpdu,
                                                           w.dpdv,
                                                           w.ray.time,
                                                           w.depth,
                                                           w.ns,
                                                           w.dpdus,
                                                           w.dpdvs,
                                                           w.dndus,
                                                           w.dndvs,
                                                           w.uv,
                                                           w.faceIndex,
                                                           w.lambda,
                                                           w.pixelIndex,
                                                           w.anyNonSpecularBounces,
                                                           -w.ray.d,
                                                           beta,
                                                           r_u,
                                                           w.etaScale,
                                                           w.mediumInterface});
                        
                    };
                    material.Dispatch(enqueue);
                });
                
        if (wavefrontDepth == maxDepth)
            return;

        ForEachType(SampleMediumScatteringCallback{ wavefrontDepth, this },
            PhaseFunction::Types());

        #ifdef PBRT_BUILD_GPU_RENDERER
            if(Options->useGPU)
            {
                GPUWait();
            }
        #endif
    }

    // WavefrontPathIntegrator Participating Media Methods
    void WavefrontPathIntegrator::SampleMediumInteraction(int wavefrontDepth)
    {
        if (!haveMedia)
            return;

        RayQueue* nextRayQueue = NextRayQueue(wavefrontDepth);


        auto pendingRayData = this->pendingRayData;
        auto pendingSamplesCnt = this->pendingSamplesCnt;
        int pendingSamplesMaxSize = this->pendingSamplesMaxSize;

        ForAllQueued(
            "Sample medium interaction", mediumSampleQueue, maxQueueSize,
            PBRT_CPU_GPU_LAMBDA(MediumSampleWorkItem w) {
            Ray ray = w.ray;
            Float tMax = w.tMax;

            PBRT_DBG("Sampling medium interaction pixel index %d depth %d ray %f %f %f d "
                "%f %f "
                "%f tMax %f\n",
                w.pixelIndex, w.depth, ray.o.x, ray.o.y, ray.o.z, ray.d.x, ray.d.y,
                ray.d.z, tMax);

            SampledWavelengths lambda = w.lambda;
            SampledSpectrum beta = w.beta;
            SampledSpectrum r_u = w.r_u;
            SampledSpectrum r_l = w.r_l;
            SampledSpectrum L(0.f);
            RNG rng(Hash(ray.o, tMax), Hash(ray.d));

            PBRT_DBG("Lambdas %f %f %f %f\n", lambda[0], lambda[1], lambda[2], lambda[3]);
            PBRT_DBG("Medium sample beta %f %f %f %f r_u %f %f %f %f r_l %f %f "
                "%f %f\n",
                beta[0], beta[1], beta[2], beta[3], r_u[0], r_u[1],
                r_u[2], r_u[3], r_l[0], r_l[1], r_l[2],
                r_l[3]);

            // Sample the medium according to T_maj, the homogeneous
            // transmission function based on the majorant.
            bool scattered = false;

            RaySamples raySamples = pixelSampleState.samples[w.pixelIndex];
            Float uDist = rng.Uniform<Float>();
            Float uMode = rng.Uniform<Float>();

            // Ray and Spectral state before sampling
            const Point3f p = ray.o;
            const Vector3f wo = ray.d;
            const SampledSpectrum beta_before = w.beta;
            const SampledSpectrum L_before = pixelSampleState.L[w.pixelIndex];

            RGBFilm *rgbFilm = film.CastOrNullptr<RGBFilm>();
            const RGBColorSpace *rgbColorSpace = rgbFilm ? rgbFilm->colorSpace : nullptr;
            SampledSpectrum T_maj = SampleT_maj(
                ray, tMax, uDist, rng, lambda,
                [&](Point3f p, MediumProperties mp, SampledSpectrum sigma_maj,
                    SampledSpectrum T_maj) {
                    PBRT_DBG("Medium event T_maj %f %f %f %f sigma_a %f %f %f %f sigma_s "
                        "%f %f "
                        "%f %f\n",
                        T_maj[0], T_maj[1], T_maj[2], T_maj[3], mp.sigma_a[0],
                        mp.sigma_a[1], mp.sigma_a[2], mp.sigma_a[3], mp.sigma_s[0],
                        mp.sigma_s[1], mp.sigma_s[2], mp.sigma_s[3]);

                    // Add emission, if present.  Always do this and scale
                    // by sigma_a/sigma_maj rather than only doing it
                    // (without scaling) at absorption events.
                    if (w.depth < maxDepth && mp.Le)
                    {
                        Float pr = sigma_maj[0] * T_maj[0];
                        SampledSpectrum r_e = r_u * sigma_maj * T_maj / pr;

                        // Update _L_ for medium emission
                        if (r_e)
                            L += beta * mp.sigma_a * T_maj * mp.Le /
                            (pr * r_e.Average());
                    }

                    // Compute probabilities for each type of scattering.
                    Float pAbsorb = mp.sigma_a[0] / sigma_maj[0];
                    Float pScatter = mp.sigma_s[0] / sigma_maj[0];
                    Float pNull = std::max<Float>(0, 1 - pAbsorb - pScatter);
                    PBRT_DBG("Medium scattering probabilities: %f %f %f\n", pAbsorb,
                        pScatter, pNull);

                    // And randomly choose one.
                    int mode = SampleDiscrete({ pAbsorb, pScatter, pNull }, uMode);

                    if (mode == 0)
                    {
                        // Absorption--done.
                        PBRT_DBG("absorbed\n");
                        beta = SampledSpectrum(0.f);
                        // Tell the medium to stop traversal.
                        return false;
                    }
                    else if (mode == 1)
                    {

                        // Scattering.
                        PBRT_DBG("scattered\n");
                        Float pr = T_maj[0] * mp.sigma_s[0];
                        beta *= T_maj * mp.sigma_s / pr;
                        r_u *= T_maj * mp.sigma_s / pr;

                        // Enqueue medium scattering work.
                        // auto enqueue = [=](auto ptr)
                        //     {
                        //         using PhaseFunction = typename std::remove_const_t<
                        //             std::remove_reference_t<decltype(*ptr)>>;
                        //         mediumScatterQueue->Push(MediumScatterWorkItem<PhaseFunction>{
                        //             p, w.depth, lambda, beta, r_u, ptr, -ray.d, ray.time,
                        //                 w.etaScale, ray.medium, w.pixelIndex});
                        //     };
                        // DCHECK_RARE(1e-6f, !beta);
                        // if (beta && r_u)
                        //     mp.phase.Dispatch(enqueue);

                        scattered = true;



                        return false;
                    }
                    else
                    {
                        // Null scattering.
                        PBRT_DBG("null-scattered\n");
                        SampledSpectrum sigma_n =
                            ClampZero(sigma_maj - mp.sigma_a - mp.sigma_s);

                        Float pr = T_maj[0] * sigma_n[0];
                        beta *= T_maj * sigma_n / pr;
                        if (pr == 0)
                            beta = SampledSpectrum(0.f);
                        r_u *= T_maj * sigma_n / pr;
                        r_l *= T_maj * sigma_maj / pr;

                        uMode = rng.Uniform<Float>();

                        return beta && r_u;
                    }
                });
            // if (!scattered && beta)
            // {
            //     beta *= T_maj / T_maj[0];
            //     r_u *= T_maj / T_maj[0];
            //     r_l *= T_maj / T_maj[0];
            // }


            PBRT_DBG("Post ray medium sample L %f %f %f %f beta %f %f %f %f\n", L[0],
                L[1], L[2], L[3], beta[0], beta[1], beta[2], beta[3]);
            PBRT_DBG("Post ray medium sample r_u %f %f %f %f r_l %f %f %f %f\n",
                r_u[0], r_u[1], r_u[2], r_u[3], r_l[0],
                r_l[1], r_l[2], r_l[3]);

            // Add any emission found to its pixel sample's L value.
            if (L)
            {
                SampledSpectrum Lp = pixelSampleState.L[w.pixelIndex];
                // pixelSampleState.L[w.pixelIndex] = Lp + L;
                outputRayData.L[w.pixelIndex] = Lp + L;
                outputRayData.lambda[w.pixelIndex] = w.lambda;
                PBRT_DBG("Added emitted radiance %f %f %f %f at pixel index %d\n", L[0],
                    L[1], L[2], L[3], w.pixelIndex);

            }


            // Calculate targets after
            SampledSpectrum L_target_added = L;
            SampledSpectrum T_target = (beta_before == SampledSpectrum(0.f)) ? SampledSpectrum(0.f) : (beta / beta_before);

            Float uLog = rng.Uniform<Float>();
            constexpr Float logProbability = 0.9f;
            int shouldLog = SampleDiscrete({logProbability, 1.f - logProbability}, uLog);


            // TODO: add a field to convert to RGB as well
            if (rgbColorSpace && shouldLog == 0) {
                // RGB beta_rgb = beta.ToRGB(w.lambda, *rgbColorSpace);
                // RGB beta_before_rgb = beta_before.ToRGB(w.lambda, *rgbColorSpace);

                TrainingDataSample dataSample;
                dataSample.pixelIdx = w.pixelIndex;
                dataSample.rayo = p;
                dataSample.rayd = wo;
                dataSample.tMax = tMax;
                dataSample.L_Before = L_before;
                dataSample.L_After = L_target_added;
                dataSample.L_added = L_target_added - L_before;
                dataSample.beta_before = beta_before;
                dataSample.beta_after = T_target;
                dataSample.beta_before_rgb = beta_before.ToRGB(w.lambda, *rgbColorSpace);
                dataSample.L_after_rgb = L_target_added.ToRGB(w.lambda, *rgbColorSpace);
                dataSample.T_after = T_target.ToRGB(w.lambda, *rgbColorSpace);

                dataSample.beta_before_spectral = beta_before;
                dataSample.L_after_spectral = L_target_added;
                dataSample.T_after_spectral = T_target;

#ifdef PBRT_IS_GPU_CODE
                int index = atomicAdd(pendingSamplesCnt, 1);
#else
                int index = __sync_fetch_and_add(pendingSamplesCnt, 1);
#endif
                if(index < pendingSamplesMaxSize)
                {
                    pendingRayData[index] = dataSample;
                }

                //TODO: Add to a buffer to copy over
            }

            // There's no more work to do if there was a scattering event in
            // the medium.
            if (scattered || !beta || !r_u || w.depth == maxDepth)
                return;

            // Otherwise, enqueue bump and medium stuff...
            // FIXME: this is all basically duplicate code w/optix.cu
            if (w.tMax == Infinity)
            {
                // no intersection
                if (escapedRayQueue)
                {
                    PBRT_DBG("Adding ray to escapedRayQueue pixel index %d depth %d\n",
                        w.pixelIndex, w.depth);
                    escapedRayQueue->Push(EscapedRayWorkItem{
                        ray.o, ray.d, w.depth, lambda, w.pixelIndex, beta,
                        (int)w.specularBounce, r_u, r_l, w.prevIntrCtx });
                }
                return;
            }

            Material material = w.material;
            
            const MixMaterial* mix = material.CastOrNullptr<MixMaterial>();
            while (mix)
            {
                SurfaceInteraction intr(w.pi, w.uv, w.wo, w.dpdus, w.dpdvs, w.dndus,
                    w.dndvs, ray.time, false /* flip normal */);
                intr.faceIndex = w.faceIndex;
                MaterialEvalContext ctx(intr);
                material = mix->ChooseMaterial(BasicTextureEvaluator(), ctx);
                mix = material.CastOrNullptr<MixMaterial>();
            }

            if (!material)
            {
                Interaction intr(w.pi, w.n);
                intr.mediumInterface = &w.mediumInterface;
                Ray newRay = intr.SpawnRay(ray.d);
                nextRayQueue->PushIndirectRay(
                    newRay, w.depth, w.prevIntrCtx, beta, r_u, r_l, lambda,
                    w.etaScale, w.specularBounce, w.anyNonSpecularBounces, w.pixelIndex);
                return;
            }

            // Mimicing simplevolpath erroring if surface has bsdf as it doesn't support surface scattering


                //TODO: Check if I can actually use this, might be relevant to infinit lights/might be done 
                // In SimpleVolPathIntegrator too but for now not doing this
                if (w.areaLight)
                {
                    PBRT_DBG(
                        "Ray hit an area light: adding to hitAreaLightQueue pixel index %d "
                        "depth %d\n",
                        w.pixelIndex, w.depth);
                    hitAreaLightQueue->Push(HitAreaLightWorkItem{
                        w.areaLight, Point3f(w.pi), w.n, w.uv, -ray.d, lambda, w.depth, beta,
                        r_u, r_l, w.prevIntrCtx, w.specularBounce, w.pixelIndex });
                }

                FloatTexture displacement = material.GetDisplacement();

                MaterialEvalQueue* q =
                    (material.CanEvaluateTextures(BasicTextureEvaluator()) &&
                        (!displacement ||
                            BasicTextureEvaluator().CanEvaluate({ displacement }, {})))
                    ? basicEvalMaterialQueue
                    : universalEvalMaterialQueue;

                PBRT_DBG("Enqueuing for material eval, mtl tag %d", material.Tag());

                auto enqueue = [=](auto ptr)
                    {
                        using Material = typename std::remove_reference_t<decltype(*ptr)>;
                        q->Push<MaterialEvalWorkItem<Material>>(
                            MaterialEvalWorkItem<Material>{ptr,
                            w.pi,
                            w.n,
                            w.dpdu,
                            w.dpdv,
                            ray.time,
                            w.depth,
                            w.ns,
                            w.dpdus,
                            w.dpdvs,
                            w.dndus,
                            w.dndvs,
                            w.uv,
                            w.faceIndex,
                            lambda,
                            w.pixelIndex,
                            w.anyNonSpecularBounces,
                            -ray.d,
                            beta,
                            r_u,
                            w.etaScale,
                            w.mediumInterface});
                    };
                material.Dispatch(enqueue);
        });

        if (wavefrontDepth == maxDepth)
            return;

        ForEachType(SampleMediumScatteringCallback{ wavefrontDepth, this },
            PhaseFunction::Types());
    }

    template <typename ConcretePhaseFunction>
    void WavefrontPathIntegrator::SampleMediumScattering(int wavefrontDepth) 
    {
        RayQueue* currentRayQueue = CurrentRayQueue(wavefrontDepth);
        RayQueue* nextRayQueue = NextRayQueue(wavefrontDepth);

        std::string desc =
            std::string("Sample direct/indirect - ") + ConcretePhaseFunction::Name();
        ForAllQueued(
            desc.c_str(),
            mediumScatterQueue->Get<MediumScatterWorkItem<ConcretePhaseFunction>>(),
            maxQueueSize,
            PBRT_CPU_GPU_LAMBDA(const MediumScatterWorkItem<ConcretePhaseFunction> w) {
            RaySamples raySamples = pixelSampleState.samples[w.pixelIndex];
            Vector3f wo = w.wo;

            // Sample direct lighting at medium scattering event.  First,
            // choose a light source.
            LightSampleContext ctx(Point3fi(w.p), Normal3f(0, 0, 0), Normal3f(0, 0, 0));
            pstd::optional<SampledLight> sampledLight =
                lightSampler.Sample(ctx, raySamples.direct.uc);

            if (sampledLight)
            {
                Light light = sampledLight->light;
                // And now sample a point on the light.
                pstd::optional<LightLiSample> ls =
                    light.SampleLi(ctx, raySamples.direct.u, w.lambda, true);
                if (ls && ls->L && ls->pdf > 0)
                {
                    Vector3f wi = ls->wi;
                    SampledSpectrum beta = w.beta * w.phase->p(wo, wi);

                    PBRT_DBG("Phase phase beta %f %f %f %f\n", beta[0], beta[1], beta[2],
                        beta[3]);

                    // Compute PDFs for direct lighting MIS calculation.
                    Float lightPDF = ls->pdf * sampledLight->p;
                    Float phasePDF =
                        IsDeltaLight(light.Type()) ? 0.f : w.phase->PDF(wo, wi);
                    SampledSpectrum r_u = w.r_u * phasePDF;
                    SampledSpectrum r_l = w.r_u * lightPDF;

                    SampledSpectrum Ld = beta * ls->L;
                    Ray ray(w.p, ls->pLight.p() - w.p, w.time, w.medium);

                    // Enqueue shadow ray
                    if (!mimicSimple)
                    {
                        shadowRayQueue->Push(ShadowRayWorkItem{ ray, 1 - ShadowEpsilon,
                                                               w.lambda, Ld, r_u, r_l,
                                                               w.pixelIndex });

                        PBRT_DBG("Enqueued medium shadow ray depth %d "
                            "Ld %f %f %f %f r_u %f %f %f %f "
                            "r_l %f %f %f %f pixel index %d\n",
                            w.depth, Ld[0], Ld[1], Ld[2], Ld[3], r_u[0], r_u[1],
                            r_u[2], r_u[3], r_l[0], r_l[1], r_l[2],
                            r_l[3], w.pixelIndex);

                    }
                }
            }

            // Sample indirect lighting.
            pstd::optional<PhaseFunctionSample> phaseSample =
                w.phase->Sample_p(wo, raySamples.indirect.u);
            if (!phaseSample || phaseSample->pdf == 0)
                return;

            SampledSpectrum beta = w.beta * phaseSample->p / phaseSample->pdf;
            SampledSpectrum r_u = w.r_u;
            SampledSpectrum r_l = w.r_u / phaseSample->pdf;

            // Russian roulette
            // TODO: should we even bother? Generally beta is one here,
            // due to the way scattering events are scattered and because we're
            // sampling exactly from the phase function's distribution...
            SampledSpectrum rrBeta = beta * w.etaScale / r_u.Average();
            if (rrBeta.MaxComponentValue() < 1 && w.depth >= 1)
            {
                Float q = std::max<Float>(0, 1 - rrBeta.MaxComponentValue());
                if (raySamples.indirect.rr < q)
                {
                    PBRT_DBG("RR terminated medium indirect with q %f pixel index %d\n",
                        q, w.pixelIndex);
                    return;
                }
                beta /= 1 - q;
            }

            Ray ray(w.p, phaseSample->wi, w.time, w.medium);
            bool specularBounce = false;
            bool anyNonSpecularBounces = true;

            // Spawn indirect ray.
            nextRayQueue->PushIndirectRay(ray, w.depth + 1, ctx, beta, r_u, r_l,
                w.lambda, w.etaScale, specularBounce,
                anyNonSpecularBounces, w.pixelIndex);
            PBRT_DBG("Enqueuing indirect medium ray at depth %d pixel index %d\n",
                w.depth + 1, w.pixelIndex);
        });
    }

}  // namespace pbrt

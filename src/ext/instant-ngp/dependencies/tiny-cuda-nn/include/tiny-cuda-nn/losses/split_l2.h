#pragma once



#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/loss.h>

__device__ inline float sigmoid(float val)
{
    return 1.0f / (1.0f + expf(-val));
}

__device__ inline float sigmoid_derivative(float sigmoid_val)
{
    return sigmoid_val * (1.0f - sigmoid_val);
}

namespace tcnn
{


    template<typename T>
    __global__ void split_l2_loss(
        const uint32_t n_elements,
        const uint32_t stride,
        const uint32_t dims,
        const uint32_t split_idx, // For now this won't really be used, it will always be 3
        const float loss_scale,
        const T* __restrict__ predictions,
        const float* __restrict__ targets,
        float* __restrict__ values,
        T* __restrict__ gradients,
        const float* __restrict__ data_pdf = nullptr
    )
    {
        const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if(i >= n_elements) return;

        const uint32_t intra_elem_idx = i % stride;
        const uint32_t inter_elem_idx = i / stride;

        if(intra_elem_idx >= dims)
        {
            values[i] = 0;
            gradients[i] = 0;
            return;
        }

        const uint32_t target_idx = inter_elem_idx * dims + intra_elem_idx;
        const uint32_t n_total = n_elements / stride * dims;
        const float target_val = targets[target_idx];
        float prediction_raw = (float)predictions[i];
        prediction_raw = fminf(fmaxf(prediction_raw, -50.0f), 50.0f);
        const float pdf = data_pdf ? data_pdf[target_idx] : 1;

        
        float prediction_activated;
        float dActivation_dRaw;
        
        if(intra_elem_idx <  split_idx)
        {
            // Apply ReLU since first 3 are the scattered radiance
            prediction_activated = relu(prediction_raw);
            dActivation_dRaw = prediction_raw > 0.f ? 1.f : 0.f;
        } else {
            // Apply sigmoid since last 3 are for the transmission
            prediction_activated = sigmoid(prediction_raw);
            dActivation_dRaw = sigmoid_derivative(prediction_activated);
        }

        const float difference = prediction_activated - target_val;
        values[i] = difference * difference / pdf / n_total;

        float dL_dActivated = 2.f * difference / pdf;
        float gradient = dL_dActivated * dActivation_dRaw;

        gradients[i] = (T)(loss_scale * gradient / n_total);
    }

    template<typename T>
    class SplitL2Loss : public Loss<T>
    {
        public:
            SplitL2Loss(uint32_t split_idx = 3)
                : m_split_idx(split_idx)
            {}

            void evaluate(
                cudaStream_t stream,
                const float loss_scale,
                const GPUMatrix<T>& prediction,
                const GPUMatrix<float>& target,
                GPUMatrix<float>& values,
                GPUMatrix<T>& gradients,
                const GPUMatrix<float>* data_pdf = nullptr
            ) const override
            {
                const uint32_t dims = target.m();
                const uint32_t stride = prediction.m();

                CHECK_THROW(prediction.n() == target.n());
                CHECK_THROW(values.m() == stride);
                CHECK_THROW(gradients.m() == stride);
                CHECK_THROW(!data_pdf || data_pdf->m() == dims);
                CHECK_THROW(m_split_idx <= dims);

                linear_kernel(
                    split_l2_loss<T>, 0, stream,
                    prediction.n_elements(),
                    stride,
                    dims,
                    m_split_idx,
                    loss_scale,
                    prediction.data(),
                    target.data(),
                    values.data(),
                    gradients.data(),
                    data_pdf ? data_pdf->data() : nullptr
                );
            }

            void update_hyperparams(const json& params) override 
            {
                m_split_idx = params.value("split_idx", m_split_idx);
            }

            json hyperparams() const override
            {
                return {
                    {"otype", "SplitL2Loss"},
                    {"split_idx", m_split_idx},
                };
            }

            std::string generate_device_function(const std::string& name, uint32_t n_dims) const override
            {
                return this->generate_device_function_from_body(
                    name, n_dims,
                    dfmt(1,
                         R"(
                            auto scale = (1.0f / (float)n_elements) / pdf;
                            vec<{N_DIMS}> activated;
                            vec<{N_DIMS}> dAct_dRaw;

                            // Apply split
                            #pragma unroll
                            for(uint32_t d = 0; d < {N_DIMS}; ++d) {{
                                float raw = (float)prediction[d];

                                if(d < {SPLIT_IDX}u) {{
                                    activated[d] = fmaxf(raw, 0.f);
                                    dAct_dRaw[d] = raw > 0.f ? 1.0f : 0.f;
                                }} else {{
                                    float s = 1.0f / (1.0f + expf(-raw));
                                    activated[d] = s;
                                    dAct_dRaw[d] = s * (1.0f - s);
                                }}
                            }}
                            auto diff = activated - target;
                            if(value) {{
                                *value = diff * diff * scale;
                            }}

                            return (2.f * loss_scale) * diff * dAct_dRaw * scale;
                        )",
                         "N_DIMS"_a = n_dims,
                         "SPLIT_IDX"_a = m_split_idx
                        ));
            }
        private:
          uint32_t m_split_idx = 3;
    };
}  // namespace tcnn
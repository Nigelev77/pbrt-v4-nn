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
#include "loading_functions.h"
using namespace tcnn;
using namespace args;
using namespace ngp;





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

    ValueFlag<std::string> generate_slice_path_flag
    {
        parser,
        "GENERATE-SLICE",
        "Whether to skip training and generate slice",
        {
            "generate-slice-path"
        }
    };

    ValueFlag<std::string> generate_scatter_path_flag
    {
        parser,
        "GENERATE_SCATTER_PATH_FLAG",
        "Path to generate scatter path csv, note requires a test-data path",
        {
            "generate-scatter-path"
        }
    };

    Flag T_slice_flag
    {
        parser,
        "T_SLICE_FLAG",
        "Whether to output transmittance slices",
        {
            "t-slice"
        }
    };

    Flag use_old_transmittance_flag
    {
        parser, 
        "USE_OLD_TRANSMITTANCE_FLAG",
        "Whether to use old transmittance",
        {
            "use-old-T"
        }
    };

    Flag Use_Spectral_Mode{
        parser, "USE_SPECTRAL_FLAG", "Whether to use spectral mode", {"spectral"}};
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
    testbed.m_volume_training_spectral_only = Use_Spectral_Mode;
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

        if(!data_path.empty())
        {
            tlog::info() << "Loading dataset...";
            load_nerfdataset(testbed, data_path);
            CUDA_CHECK_THROW(cudaDeviceSynchronize());
            CUDA_CHECK_THROW(cudaGetLastError()); // Clear any prior errors

            testbed.m_train = true;
            testbed.m_training_data_available = true;
        }
        else
        {
            
        }
        testbed.update_imgui_paths();


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
        // load_dataset_to_testbed(testbed, test_path, DatasetType::Test);
        load_dataset_to_testbed_spectral(testbed, test_path, DatasetType::Test);
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

    //Only generate slice if we have provided a msgpack file path
    if(generate_slice_path_flag && model_flag)
    {
        std::string path = get(generate_slice_path_flag);
        testbed.m_train = false;
        for(int i = 6; i < 12; ++i)
        {
            std::string filename;
            if(T_slice_flag)
            {
                filename = fmt::format("{}_{}_T.png", path.c_str(), i);
            }
            else
            {
                filename = fmt::format("{}_{}.png", path.c_str(), i);
            }
            bool output_pos = i >= 9 && i <= 11;
            testbed.dump_slice_img(filename, (float)(i+4) * (0.5f/20.f), T_slice_flag, use_old_transmittance_flag, output_pos);
        }
        return 0;
    }


    if(generate_scatter_path_flag && model_flag && test_path_flag)
    {
        std::string scatter_csv_path = get(generate_scatter_path_flag);
        testbed.m_train = false;
        const fs::path& test_path = get(test_path_flag);
        load_dataset_to_testbed(testbed, test_path, DatasetType::Test, true);
        testbed.dump_validation_scatter_data(scatter_csv_path);
        return 0;
    }

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
            // testbed.m_loss->m_loss_mode = ESplitLossMode::SplitL2;
            Testbed::ValidationTestResults res = testbed.validation_test(true);
            validation_loss_results.push_back(res);
            testbed.m_train = true;
            // testbed.m_loss->m_loss_mode = originalLossMode;

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
        
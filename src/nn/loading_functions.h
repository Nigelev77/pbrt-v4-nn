#pragma once
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

enum class DatasetType
{
    Training,
    Validation,
    Test
};

void load_dataset_to_testbed_fixedsize(Testbed& testbed, const fs::path& path, DatasetType dataset_type, uint64_t sampleCnt);
void load_dataset_to_testbed_spectral(Testbed& testbed, const fs::path& path, DatasetType dataset_type, bool should_shuffle = false);
void load_dataset_to_testbed(Testbed& testbed, const fs::path& path, DatasetType dataset_type, bool should_shuffle = false);
void load_datasets_to_testbed(Testbed &testbed, const std::vector<fs::path> &paths,
                              DatasetType dataset_type);

void load_nerfdataset(Testbed& testbed, const fs::path& data_path);

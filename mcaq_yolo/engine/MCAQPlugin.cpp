#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include <string>
#include <vector>
#include <iostream>
#include <cstring>
#include <cassert>

using namespace nvinfer1;

// [Fix 1] Removed extern "C" -> use C++ linkage
// mask is for the Eq.(19) soft-mask fusion (Listing 2) — this plugin keeps the
// 4-input (input, bit_map, min, max) contract, so it is called with nullptr and
// the m(p) multiply is applied as an elementwise layer in the TensorRT graph.
void launch_spatial_quantization(
    const float* input, const float* bit_map,
    const float* min_vals, const float* max_vals,
    const float* mask,
    float* output,
    int N, int C, int H, int W,
    int tile_h, int tile_w,
    int n_tiles_h, int n_tiles_w,
    cudaStream_t stream
);

// Plugin Namespace (global configuration)
namespace {
    const char* PLUGIN_NAME = "MCAQPlugin";
    const char* PLUGIN_VERSION = "1";
}

class MCAQPlugin : public IPluginV2DynamicExt {
public:
    MCAQPlugin(int tile_h, int tile_w) : mTileH(tile_h), mTileW(tile_w) {}
    
    MCAQPlugin(const void* data, size_t length) {
        const char* d = static_cast<const char*>(data);
        mTileH = *reinterpret_cast<const int*>(d); d += sizeof(int);
        mTileW = *reinterpret_cast<const int*>(d); d += sizeof(int);
    }

    // --- Core Execution ---
    int enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
                const void* const* inputs, void* const* outputs,
                void* workspace, cudaStream_t stream) noexcept override {

        const float* inputData = static_cast<const float*>(inputs[0]);
        const float* bitMapData = static_cast<const float*>(inputs[1]);
        const float* minData = static_cast<const float*>(inputs[2]);
        const float* maxData = static_cast<const float*>(inputs[3]);
        
        float* outputData = static_cast<float*>(outputs[0]);

        Dims dims = inputDesc[0].dims;
        int N = dims.d[0];
        int C = dims.d[1];
        int H = dims.d[2];
        int W = dims.d[3];

        // actual bit_map grid size (input[1]: (N, H_tiles, W_tiles))
        Dims mapDims = inputDesc[1].dims;
        int nTilesH = mapDims.d[1];
        int nTilesW = mapDims.d[2];

        launch_spatial_quantization(
            inputData, bitMapData, minData, maxData, /*mask=*/nullptr,
            outputData, N, C, H, W, mTileH, mTileW, nTilesH, nTilesW, stream
        );

        return 0;
    }

    // --- Configuration ---
    IPluginV2DynamicExt* clone() const noexcept override { return new MCAQPlugin(mTileH, mTileW); }
    
    DimsExprs getOutputDimensions(int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept override {
        return inputs[0];
    }

    bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override {
        return inOut[pos].type == DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
    }

    void configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs) noexcept override {}

    // Pure virtual functions of IPluginV2DynamicExt / IPluginV2 — without an
    // implementation the class stays abstract and instantiation (new MCAQPlugin)
    // will not even compile.
    int getNbOutputs() const noexcept override { return 1; }

    size_t getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs,
                            const PluginTensorDesc* outputs, int nbOutputs) const noexcept override {
        return 0;  // the kernel produces everything in-place — no workspace needed
    }

    size_t getSerializationSize() const noexcept override { return sizeof(mTileH) + sizeof(mTileW); }
    
    void serialize(void* buffer) const noexcept override {
        char* d = static_cast<char*>(buffer);
        *reinterpret_cast<int*>(d) = mTileH; d += sizeof(int);
        *reinterpret_cast<int*>(d) = mTileW; d += sizeof(int);
    }

    const char* getPluginType() const noexcept override { return PLUGIN_NAME; }
    const char* getPluginVersion() const noexcept override { return PLUGIN_VERSION; }
    int initialize() noexcept override { return 0; }
    void terminate() noexcept override {}
    void destroy() noexcept override { delete this; }
    void setPluginNamespace(const char* libNamespace) noexcept override { mNamespace = libNamespace; }
    const char* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }
    DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const noexcept override { return inputTypes[0]; }

private:
    int mTileH, mTileW;
    std::string mNamespace;
};

// [Fix 2] Add the Plugin Creator and the registration macro (required)
class MCAQPluginCreator : public IPluginCreator {
public:
    MCAQPluginCreator() {
        // Initialize plugin fields (parameter definitions)
        mPluginAttributes.emplace_back(PluginField("tile_h", nullptr, PluginFieldType::kINT32, 1));
        mPluginAttributes.emplace_back(PluginField("tile_w", nullptr, PluginFieldType::kINT32, 1));
        
        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    const char* getPluginName() const noexcept override { return PLUGIN_NAME; }
    const char* getPluginVersion() const noexcept override { return PLUGIN_VERSION; }
    const PluginFieldCollection* getFieldNames() noexcept override { return &mFC; }

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override {
        int tile_h = 32; // Default values
        int tile_w = 32;
        
        for (int i = 0; i < fc->nbFields; ++i) {
            const char* attrName = fc->fields[i].name;
            if (!strcmp(attrName, "tile_h")) {
                tile_h = *static_cast<const int*>(fc->fields[i].data);
            } else if (!strcmp(attrName, "tile_w")) {
                tile_w = *static_cast<const int*>(fc->fields[i].data);
            }
        }
        return new MCAQPlugin(tile_h, tile_w);
    }

    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override {
        return new MCAQPlugin(serialData, serialLength);
    }

    void setPluginNamespace(const char* libNamespace) noexcept override { mNamespace = libNamespace; }
    const char* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }

private:
    std::string mNamespace;
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
};

// Static member definitions
PluginFieldCollection MCAQPluginCreator::mFC{};
std::vector<PluginField> MCAQPluginCreator::mPluginAttributes;

// Register the plugin with TensorRT
REGISTER_TENSORRT_PLUGIN(MCAQPluginCreator);
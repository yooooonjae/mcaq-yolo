#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include <string>
#include <vector>
#include <iostream>
#include <cstring>
#include <cassert>

using namespace nvinfer1;

// [수정 1] extern "C" 제거 -> C++ 링킹 사용
// mask는 Eq.(19)의 soft mask 융합용(Listing 2) — 이 플러그인은 4-input
// (input, bit_map, min, max) 계약을 유지하므로 nullptr로 호출하고,
// m(p) 곱은 TensorRT 그래프의 elementwise 레이어로 적용한다.
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

// Plugin Namespace (전역 설정)
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

        // 실제 bit_map 그리드 크기 (input[1]: (N, H_tiles, W_tiles))
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

    // IPluginV2DynamicExt / IPluginV2의 순수 가상 함수 — 구현이 없으면
    // 추상 클래스가 되어 인스턴스화(new MCAQPlugin) 자체가 컴파일되지 않는다.
    int getNbOutputs() const noexcept override { return 1; }

    size_t getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs,
                            const PluginTensorDesc* outputs, int nbOutputs) const noexcept override {
        return 0;  // 커널이 전부 in-place 산출 — 별도 workspace 불필요
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

// [수정 2] Plugin Creator 및 등록 매크로 추가 (필수)
class MCAQPluginCreator : public IPluginCreator {
public:
    MCAQPluginCreator() {
        // Plugin Field 초기화 (파라미터 정의)
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

// TensorRT에 플러그인 등록
REGISTER_TENSORRT_PLUGIN(MCAQPluginCreator);
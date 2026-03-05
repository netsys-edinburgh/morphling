//
// Created by Yuhao Chen on 2023/7/12.
//

#ifndef CONFIDANT_SUBMODEL_H
#define CONFIDANT_SUBMODEL_H

#include <string>
#include "MNN/expr/Module.hpp"

using namespace MNN::Express;
namespace Confidant {
    class SingleModel : public Module {
    public:
        SingleModel() = default;
        virtual ~SingleModel() = default;
        virtual void loadParam(std::string& weightsBasePath, bool isTrainable = false) = 0;
        virtual std::vector<VARP> getLoss(std::vector<VARP>& logits, std::vector<VARP>& labels) = 0;
        virtual float getOutputDataSizeByIdx(int idx, int batchSize = 8, int seqLen = 128) { return 0.0; }
        virtual float getModelTimeByIdx(int idx, int batchSize = 8, int seqLen = 128) { return 0.0; }
    };

    class SubModel : public Module {
    public:
        SubModel() = default;
        virtual ~SubModel() = default;
        virtual void loadParamByLayer(int layer, std::string& weightsBasePath, int startLayer = 0, bool isTrainable = false) = 0;
        virtual std::vector<VARP> getParamsByLayer(int layer, bool isTrainable) = 0; // layer is the index of the local model, not the origin model
        virtual std::vector<VARP> getLoss(std::vector<VARP>& logits, std::vector<VARP>& labels) = 0;
        int getLayerNum() { return endLayer - startLayer + 1; };
        int getStartLayer() { return startLayer; };
        int getEndLayer() { return endLayer; };
        std::vector<std::shared_ptr<Module>> layers;
    protected:
        int startLayer; // the layer index of the first layer in the sub-model
        int endLayer; // the layer index of the last layer in the sub-model
    };
}

#endif //CONFIDANT_SUBMODEL_H

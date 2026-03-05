//
// Created by Yuhao Chen on 2023/2/23.
//

#ifndef CONFIDANT_OPTIMIZER_H
#define CONFIDANT_OPTIMIZER_H
#include <string>
#include <map>
#include "ParameterOptimizer.hpp"

using namespace MNN;
using namespace MNN::Train;

namespace Confidant {
    /**
     * The class that wraps the optimizer with the weight version
     */
    class OptimizerWeightVersion {
    public:
        OptimizerWeightVersion(std::string& optName, std::map<std::string, double>& optArgs);
        void addWeight();

        double getLearningRate();
        void setLearningRate(double lr);

        int getLatestVersion();
        void resetOptimizer();

        bool loadVersionWeights(int version);
        std::vector<Express::VARP> step(Express::VARP loss, Express::VARP input, Express::VARP lossGrad, std::map<Express::VARP, Express::VARP>& grads);

        static std::shared_ptr<OptimizerWeightVersion> opt;
    private:
        void initOptimizer(std::string &optName, std::map<std::string, double> &optArgs);
        void initWeightPool();

        std::map<int, std::vector<Express::VARP>> weightPool;
        std::shared_ptr<ParameterOptimizer> baseOpt;
        int latestVersion;
        int batchCounter;
        int updateInterval;
    };

}



#endif //CONFIDANT_OPTIMIZER_H

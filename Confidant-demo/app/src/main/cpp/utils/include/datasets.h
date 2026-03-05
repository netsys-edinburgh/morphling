//
// Created by Yuhao Chen on 2023/1/10.
//

#ifndef CONFIDANT_DATASETS_H
#define CONFIDANT_DATASETS_H

#include <string>
#include <DataLoader.hpp>

namespace Confidant {
    class Datasets {
    public:
        Datasets(std::string basePath, std::string name, std::string path, int trainBatchSize=64, int testBatchSize=8);
        static std::shared_ptr<MNN::Train::DataLoader> trainSetLoader;
        static std::shared_ptr<MNN::Train::DataLoader> testSetLoader;
        static Datasets *datasets;
    };
}

#endif //CONFIDANT_DATASETS_H

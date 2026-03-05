//
// Created by Yuhao Chen on 2023/1/10.
//

#include "datasets.h"
#include "Conll2003Dataset.h"
#include "AlpacaDataset.h"
# include "SQuADDataset.h"
#include "LambdaTransform.hpp"
#include "log.h"
#include "commonStates.h"

namespace Confidant {
    using namespace MNN::Train;
    std::shared_ptr<DataLoader> Datasets::trainSetLoader = nullptr;
    std::shared_ptr<DataLoader> Datasets::testSetLoader = nullptr;
    Datasets *Datasets::datasets = nullptr;

    Datasets::Datasets(std::string basePath, std::string name, std::string path, int trainBatchSize, int testBatchSize) {
        CommonStates::setModelDatasetPath(basePath);
        DatasetPtr trainDataset, testDataset;
        const int trainNumWorkers = 0;
        const int testNumWorkers = 0;
        std::string fullPath = basePath + name + path;
        LOGI("Dataset path: %s", fullPath.c_str());

        if (name == "conll2003") {
            trainDataset = Conll2003Dataset::create(basePath, fullPath, Conll2003Dataset::Mode::TRAIN);
            testDataset = Conll2003Dataset::create(basePath, fullPath, Conll2003Dataset::Mode::TEST);
        } else if (name == "Alpaca"){
            trainDataset = AlpacaDataset::create(basePath, AlpacaDataset::Mode::TRAIN_TOKENIZED);
            testDataset = AlpacaDataset::create(basePath, AlpacaDataset::Mode::TEST_TOKENIZED);
        } else if (name == "SQuAD"){
            trainDataset = SQuADDataset::create(basePath, SQuADDataset::Mode::TRAIN_TOKENIZED);
            testDataset = SQuADDataset::create(basePath, SQuADDataset::Mode::TEST_TOKENIZED);
        }
        else {
            MNN_PRINT("Unknown dataset name: %s\n", name.c_str());
            MNN_ASSERT(false);
        }

        trainSetLoader = std::shared_ptr<DataLoader>(trainDataset.createLoader(trainBatchSize, true, false, trainNumWorkers));
        testSetLoader = std::shared_ptr<DataLoader>(testDataset.createLoader(testBatchSize, true, false, testNumWorkers));
    }
}

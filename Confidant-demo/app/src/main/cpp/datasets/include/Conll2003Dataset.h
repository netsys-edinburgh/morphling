//
// Created by Yuhao Chen on 2023/7/3.
//

#ifndef MNN_CONLL2003DATASET_H
#define MNN_CONLL2003DATASET_H

#include <string>
#include "Dataset.hpp"
#include "Example.hpp"
#include <fstream>
#include "json.h"

namespace MNN {
    namespace Train {
        struct Conll2003Data {
            int guid;
            std::vector<int> inputIds;
            std::vector<int> attentionMask;
            std::vector<int> labels;
            std::vector<int> tokenTypeIds;
        };


        class MNN_PUBLIC Conll2003Dataset : public Dataset {
        public:
            enum Mode { TRAIN, TEST };

            Example get(size_t index) override;

            Conll2003Data getOneData();
            size_t size() override;

            void readDataset(const std::string basePath, const std::string& root, bool train = true);
            static DatasetPtr create(const std::string basePath, const std::string path, Mode mode = Mode::TRAIN);
            void fileCheck(const std::string& path);
            int maxLen;
        private:
            explicit Conll2003Dataset(const std::string basePath, const std::string path, Mode mode = Mode::TRAIN);
            std::ifstream dataFile;
            Json::Value jsonRoot;

            int guid;
            bool isTrain;
            std::vector<VARP> vocabIds;
            std::vector<VARP> nerIds;
            Mode mode;
        };
    }
}

#endif //MNN_CONLL2003DATASET_H
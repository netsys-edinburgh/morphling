//
// Created by Yuhao Chen on 2023/11/5.
//

#ifndef MNN_SQUAD_H
#define MNN_SQUAD_H

#include <string>
#include "Dataset.hpp"
#include "Example.hpp"
#include <fstream>
#include "json.h"

namespace MNN {
    namespace Train {

        struct SQuADData {
            int guid;
            std::vector<int> inputIds;
            std::vector<int> attentionMask;
            std::vector<int> answerStart;
            std::vector<std::string> answerText;
            std::string title;
            std::string context;
            std::string question;
            std::vector<int> startPos;
            std::vector<int> endPos;
        };

        class MNN_PUBLIC SQuADDataset : public Dataset {
        public:
            // Currently only two modes are supported
            enum Mode { TRAIN_TOKENIZED, TEST_TOKENIZED };

            Example get(size_t index) override;

            SQuADData getOneData();
            size_t size() override;

            void readDataset(const std::string& root);
            static DatasetPtr create(const std::string path, Mode mode = Mode::TRAIN_TOKENIZED, int maxLen = 256);
            int maxLen;
        private:
            explicit SQuADDataset(const std::string path, Mode mode = Mode::TRAIN_TOKENIZED, int maxLen = 256);
            std::ifstream dataFile;
            Json::Value jsonRoot;
            int guid;
            bool isTrain;
            Mode mode;
        };
    }
}

#endif //MNN_SQUAD_H

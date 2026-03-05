//
// Created by Yuhao Chen on 2023/11/1.
//

#ifndef MNN_ALPACA_H
#define MNN_ALPACA_H

#include <string>
#include "Dataset.hpp"
#include "Example.hpp"
#include <fstream>
#include "Tokenizer.h"
#include "json.h"

namespace MNN {
    namespace Train {
        std::string generatePrompt(std::string& instruction, std::string& input);

        struct AlpacaData {
            int guid;
            std::vector<int> inputIds;
            std::vector<int> inputIdsNoResp;
            std::vector<int> labels;
            std::string input;
            std::string instruction;
            std::string output;
        };

        class MNN_PUBLIC AlpacaDataset : public Dataset {
        public:
            enum Mode { TRAIN, TRAIN_TOKENIZED, TEST, TEST_TOKENIZED };

            Example get(size_t index) override;

            AlpacaData getOneData();
            size_t size() override;

            void readDataset(const std::string& root);
            static DatasetPtr create(const std::string path, Mode mode = Mode::TRAIN_TOKENIZED, int maxLen = 256);
            int maxLen;
        private:
            explicit AlpacaDataset(const std::string path, Mode mode = Mode::TRAIN_TOKENIZED, int maxLen = 256);
            std::string alpacaTestTokenizedFileName;
            std::string alpacaTrainTokenizedFileName;
            std::ifstream dataFile;
            Json::Value jsonRoot;
            int guid;
            bool isTrain;
            Mode mode;
            std::shared_ptr<SentencePiece> tokenizer;
        };
    }
}
#endif //MNN_ALPACA_H

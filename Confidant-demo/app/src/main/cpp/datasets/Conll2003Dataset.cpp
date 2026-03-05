//
// Created by Yuhao Chen on 2023/7/3.
//

#include "Conll2003Dataset.h"
#include <string>
#include <sstream>
#include <MNN/expr/Module.hpp>
#include "log.h"


namespace MNN {
    namespace Train {
        // referenced from huggingface/datasets
        // https://huggingface.co/datasets/conll2003/blob/main/conll2003.py
        const int32_t kTrainSize = 14041;
        const int32_t kTestSize = 3453;
        const int32_t kValidSize = 3250;

        const char* conllTrainTokenizedFileName = "conll2003_train_tokenized.json";
        const char* conllTestTokenizedFileName = "conll2003_test_tokenized.json";

        std::string conll2003JoinPaths(std::string head, const std::string& tail) {
            if (head.back() != '/') {
                head.push_back('/');
            }
            head += tail;
            return head;
        }

        void Conll2003Dataset::readDataset(const std::string basePath, const std::string& root, bool train) {
            const auto path = conll2003JoinPaths(root, train ? conllTrainTokenizedFileName : conllTestTokenizedFileName);
            std::string vocabPath = basePath + "vocab_files/bert-base-cased-vocab.txt";
            fileCheck(path);
            fileCheck(vocabPath);

            this->maxLen = 180;

            this->dataFile.open(path, std::ios::binary);
            // std::ifstream textData(path, std::ios::binary);
            if (!this->dataFile.is_open()) {
                MNN_PRINT("Error opening dataset file at %s", path.c_str());
                MNN_ASSERT(false);
            }

            Json::CharReaderBuilder reader;

            std::string errs;
            Json::parseFromStream(reader, this->dataFile, &this->jsonRoot, &errs);
            if (!errs.empty()) {
                MNN_PRINT("Error parsing JSON dataset file at %s", path.c_str());
                MNN_ASSERT(false);
            }
        }

        DatasetPtr Conll2003Dataset::create(const std::string basePath, const std::string path, Mode mode) {
            DatasetPtr res;
            res.mDataset.reset(new Conll2003Dataset(basePath, path, mode));
            return res;
        }

        Conll2003Dataset::Conll2003Dataset(const std::string basePath, const std::string path, MNN::Train::Conll2003Dataset::Mode mode) {
            this->isTrain = mode == Mode::TRAIN;
            this->mode = mode;
            this->guid = 0;
            readDataset(basePath, path, mode == Mode::TRAIN);
        }

        Example Conll2003Dataset::get(size_t index) {
            auto curData = getOneData();

            int ignoreId = -100;
            while (curData.inputIds.size() < maxLen) {
                curData.inputIds.emplace_back(0);
            }

            while (curData.labels.size() < maxLen) {
                curData.labels.emplace_back(ignoreId);
            }

            auto inputIdsVARP = _Input({maxLen}, NCHW, halide_type_of<int>());
            auto inputPtr = inputIdsVARP->writeMap<int>();
            ::memcpy(inputPtr, curData.inputIds.data(), curData.inputIds.size() * sizeof(int));

            auto labelsVARP = _Input({maxLen}, NCHW, halide_type_of<int>());
            auto labelPtr = labelsVARP->writeMap<int>();
            ::memcpy(labelPtr, curData.labels.data(), curData.labels.size() * sizeof(int));

            return {{inputIdsVARP}, {labelsVARP}};
        }

        size_t Conll2003Dataset::size() {
            return isTrain ? kTrainSize : kTestSize;
        }

        Conll2003Data Conll2003Dataset::getOneData() {
            if (guid >= this->size()) {
                guid = 0;
            }

            std::string line;
            Conll2003Data cur = {};
            cur.guid = guid;

            auto curItem = jsonRoot[guid];

            // read token ids from json file
            std::string inputIdsStr = curItem["input_ids"].asString();
            std::vector<int> inputIds;
            for (int i = 1; i < (int) inputIdsStr.size() - 1; i++) {
                if (inputIdsStr[i] == ',' || inputIdsStr[i] == ' ') {
                    continue;
                }
                int curNum = 0;
                while (i < inputIdsStr.size() - 1 && inputIdsStr[i] != ',') {
                    curNum = curNum * 10 + inputIdsStr[i] - '0';
                    i++;
                }
                inputIds.emplace_back(curNum);
            }
            cur.inputIds = std::move(inputIds);

            // parse attention mask
            std::string attnMaskStr = curItem["attention_mask"].asString();
            std::vector<int> attnMask;
            for (int i = 1; i < attnMaskStr.size() - 1; i++) {
                if (attnMaskStr[i] == ',' || attnMaskStr[i] == ' ') {
                    continue;
                }
                int curNum = 0;
                while (i < attnMaskStr.size() - 1 && attnMaskStr[i] != ',') {
                    curNum = curNum * 10 + attnMaskStr[i] - '0';
                    i++;
                }
                attnMask.emplace_back(curNum);
            }
            cur.attentionMask = std::move(attnMask);

            // parse labels, labels may contain negative numbers
            std::string labelsStr = curItem["labels"].asString();
            std::vector<int> labels;
            for (int i = 1; i < labelsStr.size() - 1; i++) {
                if (labelsStr[i] == ',' || labelsStr[i] == ' ') {
                    continue;
                }
                int curNum = 0;
                int sign = 1;
                if (labelsStr[i] == '-') {
                    sign = -1;
                    i++;
                }
                while (i < labelsStr.size() - 1 && labelsStr[i] != ',') {
                    curNum = curNum * 10 + labelsStr[i] - '0';
                    i++;
                }
                labels.emplace_back(sign * curNum);
            }
            cur.labels = std::move(labels);

            // parse token type ids
            std::string tokenTypeIdsStr = curItem["token_type_ids"].asString();
            std::vector<int> tokenTypeIds;
            for (int i = 1; i < tokenTypeIdsStr.size() - 1; i++) {
                if (tokenTypeIdsStr[i] == ',' || tokenTypeIdsStr[i] == ' ') {
                    continue;
                }
                int curNum = 0;
                while (i < tokenTypeIdsStr.size() - 1 && tokenTypeIdsStr[i] != ',') {
                    curNum = curNum * 10 + tokenTypeIdsStr[i] - '0';
                    i++;
                }
                tokenTypeIds.emplace_back(curNum);
            }
            cur.tokenTypeIds = std::move(tokenTypeIds);

            guid++;

            return cur;
        }

        void Conll2003Dataset::fileCheck(const std::string &path) {
            std::ifstream file(path);
            if(file.good()){
                LOGI("File exists: %s", path.c_str());
            } else{
                LOGI("File does not exist: %s", path.c_str());
            }
        }
    }
}
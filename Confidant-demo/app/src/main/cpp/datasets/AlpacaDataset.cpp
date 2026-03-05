//
// Created by Yuhao Chen on 2023/11/1.
//

#include "AlpacaDataset.h"
#include "Tokenizer.h"
#include <fstream>
#include "commonStates.h"
namespace MNN {
    namespace Train {
        std::string generatePrompt(std::string& instruction, std::string& input) {
            if (input.size() > 0) {
                return "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n" +
                       instruction + "\n\n### Input:\n" + input + "\n\n### Response:";
            }
            return "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n" +
                   instruction + "\n\n### Response:";
        }

        const int32_t alpacaTrainSize = 49759;
        const int32_t alpacaTestSize = 2000;
        const char* alpacaTrainFileName = "Alpaca/train.json";
        const char* alpacaTestFileName = "Alpaca/test.json";

//        const char* alpacaTestTokenizedFileName = "";
//        const char* alpacaTrainTokenizedFileName = "";
//        const char* alpacaTestTokenizedFileName = "Alpaca/alpaca_tokenized_llama3_test_zy.json";
//        const char* alpacaTrainTokenizedFileName = "Alpaca/alpaca_tokenized_llama3_train_zy.json";
//
//        const char* alpacaTrainTokenizedFileName = "Alpaca/alpaca_tokenized_train_zy.json";
//        const char* alpacaTestTokenizedFileName = "Alpaca/alpaca_tokenized_test_zy.json";

        std::string alpacaJoinPaths(std::string head, const std::string& tail) {
            if (head.back() != '/') {
                head.push_back('/');
            }
            head += tail;
            return head;
        }

        void AlpacaDataset::readDataset(const std::string& root) {
            // We read the tokenized json file now
            std::string tail = "";
            if (this->mode == TRAIN) {
                tail = alpacaTrainFileName;
            } else if (this->mode == TRAIN_TOKENIZED) {
                tail = alpacaTrainTokenizedFileName;
            } else if (this->mode == TEST) {
                tail = alpacaTestFileName;
            } else if (this->mode == TEST_TOKENIZED) {
                tail = alpacaTestTokenizedFileName;
            } else {
                MNN_PRINT("Alpaca mode error!\n");
                MNN_ASSERT(false);
            }

            // Here we directly load the tokenized json file
            const auto path = alpacaJoinPaths(root, tail);

            this->dataFile.open(path, std::ios::binary);
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

        DatasetPtr AlpacaDataset::create(const std::string path, Mode mode, int maxLen) {
            DatasetPtr res;
            res.mDataset.reset(new AlpacaDataset(path, mode, maxLen));
            return res;
        }

        AlpacaDataset::AlpacaDataset(const std::string path, MNN::Train::AlpacaDataset::Mode mode, int maxLen) {
            this->isTrain = (mode == Mode::TRAIN || mode == Mode::TRAIN_TOKENIZED);
            this->mode = mode;
            this->guid = 0;
            this->maxLen = maxLen;

            Confidant::CommonStates::ModelName globalModelName = Confidant::CommonStates::getGlobalModelName();

            if (globalModelName == Confidant::CommonStates::ModelName::LLaMA){
                alpacaTrainTokenizedFileName = "Alpaca/alpaca_tokenized_train_zy.json";
                alpacaTestTokenizedFileName = "Alpaca/alpaca_tokenized_test_zy.json";
            } else if(globalModelName == Confidant::CommonStates::ModelName::Phi2){
                alpacaTrainTokenizedFileName = "Alpaca/alpaca_tokenized_llama3_train_zy.json";
                alpacaTestTokenizedFileName = "Alpaca/alpaca_tokenized_llama3_test_zy.json";
            }

            readDataset(path);
        }

        Example AlpacaDataset::get(size_t index) {
            auto curData = getOneData();

            // strip input_ids to maxLen
            if (curData.inputIds.size() > maxLen) {
                curData.inputIds.resize(maxLen);
            }

            if (curData.labels.size() > maxLen) {
                curData.labels.resize(maxLen);
            }

            // pad input_id to maxLen by 0
            while (curData.inputIds.size() < maxLen) {
                curData.inputIds.emplace_back(0);
            }

            // pad labels to maxLen by -1
            while (curData.labels.size() < maxLen) {
                curData.labels.emplace_back(-100);
            }

            auto inputIdsVARP = _Input({maxLen}, NCHW, halide_type_of<int>());
            auto inputPtr = inputIdsVARP->writeMap<int>();
            ::memcpy(inputPtr, curData.inputIds.data(), curData.inputIds.size() * sizeof(int));

            auto labelsVARP = _Input({maxLen}, NCHW, halide_type_of<int>());
            auto labelPtr = labelsVARP->writeMap<int>();
            ::memcpy(labelPtr, curData.labels.data(), curData.labels.size() * sizeof(int));

            return {{inputIdsVARP}, {labelsVARP}};
        }
//
        size_t AlpacaDataset::size() {
            return isTrain ? alpacaTrainSize : alpacaTestSize;
        }

        AlpacaData AlpacaDataset::getOneData() {
            if (guid >= this->size()) {
                guid = 0;
            }

            std::string line;
            AlpacaData cur = {};
            cur.guid = guid;

            auto curItem = jsonRoot[guid];
            cur.instruction = curItem["instruction"].asString();
            cur.input = curItem["input"].asString();
            cur.output = curItem["output"].asString();

            if (this->mode == TRAIN_TOKENIZED || this->mode == TEST_TOKENIZED) {
                // read token ids from json file
                std::string inputIdsStr = curItem["input_ids"].asString();
                // parse string into vector, the string is like [1, ..., ]
                std::vector<int> inputIds;
                for (int i = 1; i < inputIdsStr.size() - 1; i++) {
                    if (inputIdsStr[i] == ',' or inputIdsStr[i] == ' ' ) {
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

                std::string inputIdsNoRespStr = curItem["input_ids_no_response"].asString();
                std::vector<int> inputIdsNoResp;
                for (int i = 1; i < inputIdsNoRespStr.size() - 1; i++) {
                    if (inputIdsNoRespStr[i] == ',' or inputIdsNoRespStr[i] == ' ') {
                        continue;
                    }
                    int curNum = 0;
                    while (i < inputIdsNoRespStr.size() - 1 && inputIdsNoRespStr[i] != ',') {
                        curNum = curNum * 10 + inputIdsNoRespStr[i] - '0';
                        i++;
                    }
                    inputIdsNoResp.emplace_back(curNum);
                }
                cur.inputIdsNoResp = std::move(inputIdsNoResp);

                std::string labelsStr = curItem["labels"].asString();
                std::vector<int> labels;
                for (int i = 1; i < labelsStr.size() - 1; i++) {
                    if (labelsStr[i] == ',' or labelsStr[i] == ' ' ) {
                        continue;
                    }
                    int curNum = 0;
                    while (i < labelsStr.size() - 1 && labelsStr[i] != ',') {
                        curNum = curNum * 10 + labelsStr[i] - '0';
                        i++;
                    }
                    labels.emplace_back(curNum);
                }
                cur.labels = std::move(labels);
            } else {
                // parse through tokenizer
                std::string fullPrompt = generatePrompt(cur.instruction, cur.input);
                std::string fullPromptAndResponse = fullPrompt + cur.output;

                auto encodedFullPrompt = tokenizer->encode(fullPrompt);
                tokenizer->add_bos( encodedFullPrompt);

                auto encodedFullPromptAndResponse = tokenizer->encode(fullPromptAndResponse);
                tokenizer->add_bos(encodedFullPromptAndResponse);
                tokenizer->add_eos(encodedFullPromptAndResponse);

                cur.inputIds = encodedFullPromptAndResponse;
                cur.inputIdsNoResp = encodedFullPrompt;
                cur.labels = std::move(encodedFullPromptAndResponse);
            }
            guid++;

            return cur;
        }
    }
}
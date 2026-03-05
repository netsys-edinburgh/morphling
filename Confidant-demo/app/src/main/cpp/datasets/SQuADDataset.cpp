//
// Created by Yuhao Chen on 2023/11/5.
//

#include "SQuADDataset.h"
#include <fstream>

namespace MNN {
    namespace Train {
        const int32_t squadTrainSize = 87599;
        const int32_t squadTestSize = 10570;

        const char* squadTrainTokenizedFileName = "SQuAD/squad_train_tokenized.json";
        const char* squadTestTokenizedFileName = "SQuAD/squad_test_tokenized.json";

        std::string squadJoinPaths(std::string head, const std::string& tail) {
            if (head.back() != '/') {
                head.push_back('/');
            }
            head += tail;
            return head;
        }

        void SQuADDataset::readDataset(const std::string& root) {
            // We read the tokenized json file now
            std::string tail = "";
            if (this->mode == TRAIN_TOKENIZED) {
                tail = squadTrainTokenizedFileName;
            } else if (this->mode == TEST_TOKENIZED) {
                tail = squadTestTokenizedFileName;
            } else {
                MNN_PRINT("SQuAD mode error!\n");
                MNN_ASSERT(false);
            }

            const auto path = squadJoinPaths(root, tail);

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

        DatasetPtr SQuADDataset::create(const std::string path, Mode mode, int maxLen) {
            DatasetPtr res;
            res.mDataset.reset(new SQuADDataset(path, mode, maxLen));
            return res;
        }

        SQuADDataset::SQuADDataset(const std::string path, MNN::Train::SQuADDataset::Mode mode, int maxLen) {
            this->isTrain = mode == Mode::TRAIN_TOKENIZED;
            this->mode = mode;
            this->guid = 0;
            this->maxLen = maxLen;
            readDataset(path);
        }

        Example SQuADDataset::get(size_t index) {
            auto curData = getOneData();

            // TODO: SQuAD padding id
            while (curData.inputIds.size() < maxLen) {
                curData.inputIds.emplace_back(-1);
            }

            auto inputIdsVARP = _Input({maxLen}, NCHW, halide_type_of<int>());
            auto inputPtr = inputIdsVARP->writeMap<int>();
            ::memcpy(inputPtr, curData.inputIds.data(), curData.inputIds.size() * sizeof(int));

            int labelLen = mode == TRAIN_TOKENIZED ? 1 : 3;
            auto startPosVARP = _Input({labelLen}, NCHW, halide_type_of<int>());
            auto endPosVARP = _Input({labelLen}, NCHW, halide_type_of<int>());

            auto startPosPtr = startPosVARP->writeMap<int>();
            auto endPosPtr = endPosVARP->writeMap<int>();

            ::memcpy(startPosPtr, curData.startPos.data(), curData.startPos.size() * sizeof(int));
            ::memcpy(endPosPtr, curData.endPos.data(), curData.endPos.size() * sizeof(int));
//            auto labelPtr = labelsVARP->writeMap<int>();
//            ::memcpy(labelPtr, curData.answerStart.data(), curData.answerStart.size() * sizeof(int));

            return {{inputIdsVARP}, {startPosVARP, endPosVARP}};
        }
//
        size_t SQuADDataset::size() {
            return isTrain ? squadTrainSize : squadTestSize;
        }

        SQuADData SQuADDataset::getOneData() {
            if (guid >= this->size()) {
                guid = 0;
            }

            std::string line;
            SQuADData cur = {};
            cur.guid = guid;

            auto curItem = jsonRoot[guid];
            cur.title = curItem["title"].asString();
            cur.context = curItem["context"].asString();
            cur.question = curItem["question"].asString();

            // read token ids from json file
            std::string inputIdsStr = curItem["input_ids"].asString();
            std::vector<int> inputIds;
            for (int i = 1; i < inputIdsStr.size() - 1; i++) {
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

            // parse answer_text
            std::string ansTextStr = curItem["answer_text"].asString();
            std::vector<std::string> ansText;
            for (int i = 1; i < ansTextStr.size() - 1; i++) {
                if (ansTextStr[i] == '\'') {
                    continue;
                }
                std::string curStr = "";
                while (i < ansTextStr.size() - 1 && ansTextStr[i] != ',') {
                    curStr += ansTextStr[i];
                    i++;
                }
                ansText.emplace_back(curStr);
            }
            cur.answerText = std::move(ansText);

            // parse answer start
            std::string ansStartStr = curItem["answer_start"].asString();
            std::vector<int> ansStart;
            for (int i = 1; i < ansStartStr.size() - 1; i++) {
                if (ansStartStr[i] == ',' || ansStartStr[i] == ' ') {
                    continue;
                }
                int curNum = 0;
                while (i < ansStartStr.size() - 1 && ansStartStr[i] != ',') {
                    curNum = curNum * 10 + ansStartStr[i] - '0';
                    i++;
                }
                ansStart.emplace_back(curNum);
            }
            cur.answerStart = std::move(ansStart);

            // parse start pos and end pos
            std::string startPosStr = curItem["start_pos"].asString();
            std::vector<int> startPos;
            for (int i = 1; i < startPosStr.size() - 1; i++) {
                if (startPosStr[i] == ',' || startPosStr[i] == ' ') {
                    continue;
                }
                int curNum = 0;
                while (i < startPosStr.size() - 1 && startPosStr[i] != ',') {
                    curNum = curNum * 10 + startPosStr[i] - '0';
                    i++;
                }
                startPos.emplace_back(curNum);
            }
            cur.startPos = std::move(startPos);

            std::string endPosStr = curItem["end_pos"].asString();
            std::vector<int> endPos;
            for (int i = 1; i < endPosStr.size() - 1; i++) {
                if (endPosStr[i] == ',' || endPosStr[i] == ' ') {
                    continue;
                }
                int curNum = 0;
                while (i < endPosStr.size() - 1 && endPosStr[i] != ',') {
                    curNum = curNum * 10 + endPosStr[i] - '0';
                    i++;
                }
                endPos.emplace_back(curNum);
            }
            cur.endPos = std::move(endPos);

            guid++;

            return cur;
        }
    }
}
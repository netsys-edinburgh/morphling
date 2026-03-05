//
// Created by Yuhao Chen on 2023/6/13.
//
#include "GPT2Embedding.h"
#include "MEmbedding.h"
#include "MLayerNorm.h"

namespace MNN {
    namespace Train {
        namespace Model {
            using namespace MNN::Express;

            /* position embedding */
            class Gpt2_PositionalEmbedding : public Module {
            public:
                Gpt2_PositionalEmbedding(int dModel, int maxLen = 1024);//512 -> 1024
                virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;
                VARP pe;
            };

            //add gpt_
            std::shared_ptr<Module> gpt_PositionalEmbedding(int dModel, int maxLen = 1024) {
                return std::shared_ptr<Module>(new Gpt2_PositionalEmbedding(dModel, maxLen));
            }

            Gpt2_PositionalEmbedding::Gpt2_PositionalEmbedding(int dModel, int maxLen) {
                auto position = _Unsqueeze(_Cast<float>(_Range(_Scalar<int>(0), _Scalar<int>(maxLen), _Scalar<int>(1))),
                                           {1});
                auto divTerm = _Exp(_Cast<float>(_Range(_Scalar<int>(0), _Scalar<int>(dModel), _Scalar<int>(2))) *
                                    _Scalar<float>(-log(10000.0) / dModel));

                auto sin = _Sin(position * divTerm);
                auto cos = _Cos(position * divTerm);

                this->pe = _Unsqueeze(_Reshape(_Stack({sin, cos}, -1), {maxLen, dModel}), {0});
                this->pe.fix(VARP::TRAINABLE);
            }

            std::vector<Express::VARP> Gpt2_PositionalEmbedding::onForward(const std::vector<Express::VARP> &inputs) {
                using namespace Express;
                VARP x = inputs[0];
                int shape = x->getInfo()->dim[1];
                auto output = _GatherV2(pe, _Range(_Scalar<int>(0), _Scalar<int>(shape), _Scalar<int>(1)), _Scalar<int>(1));

                return {output};
            }


/*segment embedding */
            class Gpt2__SegmentEmbedding : public Module {
            private:
                std::shared_ptr<Module> embedding;
            public:
                Gpt2__SegmentEmbedding(int embedSize = 768);//embedsize 512 改 768
                virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;

            };

            //add gpt_
            std::shared_ptr<Module> gpt_SegmentEmbedding(int embedSize = 768) {//embedsize 512 改 768
                return std::shared_ptr<Module>(new Gpt2__SegmentEmbedding(embedSize));
            }

            Gpt2__SegmentEmbedding::Gpt2__SegmentEmbedding(int embedSize) {
                embedding.reset(new MEmbedding(3, embedSize));
            }

            std::vector<Express::VARP> Gpt2__SegmentEmbedding::onForward(const std::vector<Express::VARP> &inputs) {
                using namespace Express;
                VARP x = inputs[0];
                auto out = embedding->forward(x);
                return {out};
            }

            /* GPT2Embedding */
            GPT2Embedding::GPT2Embedding(int vocabSize, int hiddenSize, int maxPositionEmbeddings, float dropoutProb) {
                this->wordEmbeddings = std::make_shared<MEmbedding>(vocabSize, hiddenSize);
                this->positionEmbeddings = std::make_shared<MEmbedding>(maxPositionEmbeddings, hiddenSize);
                this->hiddenSize = hiddenSize;
                this->dropout.reset(NN::Dropout(dropoutProb));

                registerModel({wordEmbeddings, positionEmbeddings, dropout});
            }

            std::vector<Express::VARP> GPT2Embedding::onForward(const std::vector<Express::VARP> &inputs) {
                using namespace Express;

                VARP inputIds = inputs[0];
                //VARP tokenTypeIds = inputs[1];

                int seqLength = inputIds->getInfo()->dim[1];
                auto positionIds = _Range(_Scalar<int>(0), _Scalar<int>(seqLength), _Scalar<int>(1));
                positionIds = _Unsqueeze(positionIds, {0});

                int inputIdsShapeSize = inputIds->getInfo()->dim.size();
                positionIds = _BroadcastTo(positionIds, _Const(inputIds->getInfo()->dim.data(), {inputIdsShapeSize}, NCHW, halide_type_of<int>()));

                auto wordEmbeddings = this->wordEmbeddings->forward(inputIds);
                auto positionEmbeddings = this->positionEmbeddings->forward(positionIds);

                auto embeddings = wordEmbeddings + positionEmbeddings ;

                embeddings = this->dropout->forward(embeddings);

                return {embeddings};
            }
        }
    }
}
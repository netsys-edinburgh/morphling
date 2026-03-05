//
// Created by Yuhao Chen on 2023/6/13.
//
#include "BERTEmbedding.h"
#include "MEmbedding.h"
#include "MLayerNorm.h"

namespace MNN {
    namespace Train {
        namespace Model {
            using namespace MNN::Express;

            class _TokenEmbedding : public Module {
            public:
                _TokenEmbedding(int vocabSize, int embedSize = 512);
                std::shared_ptr<Module> embedding;
                virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;

            };

            std::shared_ptr<Module> TokenEmbedding(int vocabSize, int embedSize = 512) {
                return std::shared_ptr<Module>(new _TokenEmbedding(vocabSize, embedSize));
            }

            _TokenEmbedding::_TokenEmbedding(int vocabSize, int embedSize) {
                embedding.reset(new MEmbedding(vocabSize, embedSize));
                registerModel({embedding});
            }

            std::vector<Express::VARP> _TokenEmbedding::onForward(const std::vector<Express::VARP> &inputs) {
                using namespace Express;
                VARP x = inputs[0];
                auto out = embedding->forward(x);
                return {out};
            }

            class _PositionalEmbedding : public Module {
            public:
                _PositionalEmbedding(int dModel, int maxLen = 512);
                virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;
                VARP pe;
            };

            std::shared_ptr<Module> PositionalEmbedding(int dModel, int maxLen = 512) {
                return std::shared_ptr<Module>(new _PositionalEmbedding(dModel, maxLen));
            }

            _PositionalEmbedding::_PositionalEmbedding(int dModel, int maxLen) {
                auto position = _Unsqueeze(_Cast<float>(_Range(_Scalar<int>(0), _Scalar<int>(maxLen), _Scalar<int>(1))),
                                           {1});
                auto divTerm = _Exp(_Cast<float>(_Range(_Scalar<int>(0), _Scalar<int>(dModel), _Scalar<int>(2))) *
                                    _Scalar<float>(-log(10000.0) / dModel));

                auto sin = _Sin(position * divTerm);
                auto cos = _Cos(position * divTerm);

                this->pe = _Unsqueeze(_Reshape(_Stack({sin, cos}, -1), {maxLen, dModel}), {0});
                this->pe.fix(VARP::TRAINABLE);
            }

            std::vector<Express::VARP> _PositionalEmbedding::onForward(const std::vector<Express::VARP> &inputs) {
                using namespace Express;
                VARP x = inputs[0];
                int shape = x->getInfo()->dim[1];
                auto output = _GatherV2(pe, _Range(_Scalar<int>(0), _Scalar<int>(shape), _Scalar<int>(1)), _Scalar<int>(1));

                return {output};
            }

            class _SegmentEmbedding : public Module {
            private:
                std::shared_ptr<Module> embedding;
            public:
                _SegmentEmbedding(int embedSize = 512);
                virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;

            };

            std::shared_ptr<Module> SegmentEmbedding(int embedSize = 512) {
                return std::shared_ptr<Module>(new _SegmentEmbedding(embedSize));
            }

            _SegmentEmbedding::_SegmentEmbedding(int embedSize) {
                embedding.reset(new MEmbedding(3, embedSize));
            }

            std::vector<Express::VARP> _SegmentEmbedding::onForward(const std::vector<Express::VARP> &inputs) {
                using namespace Express;
                VARP x = inputs[0];
                auto out = embedding->forward(x);
                return {out};
            }

            BERTEmbedding::BERTEmbedding(int vocabSize, int hiddenSize, int maxPositionEmbeddings, int typeVocabSize, float dropoutProb) {
                this->wordEmbeddings = std::make_shared<MEmbedding>(vocabSize, hiddenSize);
                this->positionEmbeddings = std::make_shared<MEmbedding>(maxPositionEmbeddings, hiddenSize);
                this->tokenTypeEmbeddings = std::make_shared<MEmbedding>(typeVocabSize, hiddenSize);
                this->layerNorm.reset(new MLayerNorm({hiddenSize}, true, 1e-12));
                this->hiddenSize = hiddenSize;
                this->dropout.reset(NN::Dropout(dropoutProb));

                registerModel({wordEmbeddings, positionEmbeddings, tokenTypeEmbeddings, layerNorm, dropout});
            }

            std::vector<Express::VARP> BERTEmbedding::onForward(const std::vector<Express::VARP> &inputs) {
                using namespace Express;

                VARP inputIds = inputs[0];
                VARP tokenTypeIds = inputs[1];

                int seqLength = inputIds->getInfo()->dim[1];
                auto positionIds = _Range(_Scalar<int>(0), _Scalar<int>(seqLength), _Scalar<int>(1));
                positionIds = _Unsqueeze(positionIds, {0});

                int inputIdsShapeSize = inputIds->getInfo()->dim.size();
                positionIds = _BroadcastTo(positionIds, _Const(inputIds->getInfo()->dim.data(), {inputIdsShapeSize}, NCHW, halide_type_of<int>()));

                auto wordEmbeddings = this->wordEmbeddings->forward(inputIds);
                auto positionEmbeddings = this->positionEmbeddings->forward(positionIds);
                auto tokenTypeEmbeddings = this->tokenTypeEmbeddings->forward(tokenTypeIds);

                auto embeddings = wordEmbeddings + positionEmbeddings + tokenTypeEmbeddings;

                embeddings = this->layerNorm->forward(embeddings);
                embeddings = this->dropout->forward(embeddings);

                return {embeddings};
            }
        }
    }
}
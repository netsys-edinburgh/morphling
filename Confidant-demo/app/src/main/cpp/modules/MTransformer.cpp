//
// Created by Yuhao Chen on 2023/5/17.
//

#include "MTransformer.h"
#include "MLayerNorm.h"
#include <utility>

using namespace MNN::Train::Model;

namespace MNN {

    using namespace MNN::Express;

    class _SiLU : public Module {
    public:
        virtual std::vector<Express::VARP>
        onForward(const std::vector<Express::VARP> &inputs) override;
    };

    std::shared_ptr<Module> SiLU() {
        return std::shared_ptr<Module>(new _SiLU());
    }

    std::vector<Express::VARP> _SiLU::onForward(const std::vector<Express::VARP> &inputs) {
        using namespace Express;
        VARP x = inputs[0];
        x = x * _Sigmoid(x);
        return {x};
    }

    class _Attention : public Module {
    private:
        int heads;
        VARP scale;

    public:
        _Attention(int dim, int heads = 8, int dimHead = 64, float dropout = 0.0);

        virtual std::vector<Express::VARP>
        onForward(const std::vector<Express::VARP> &inputs) override;

        std::shared_ptr<Module> qLinear, kLinear, vLinear;
        std::vector<std::shared_ptr<Module> > toOut;
    };

    std::shared_ptr<Module>
    Attention(int dim, int heads = 8, int dimHead = 64, float dropout = 0.0) {
        // 提供了抽象层级，提供一致的接口，也可以实现额外的逻辑
        return std::shared_ptr<Module>(new _Attention(dim, heads, dimHead, dropout));
    }

    _Attention::_Attention(int dim, int heads, int dimHead, float dropout) {
        int innerDim = dimHead * heads;
        bool projectOut = !(heads == 1 && dimHead == dim);

        this->heads = heads;

        float scaleFactor = pow(dimHead, -0.5);
        this->scale = _Const(scaleFactor, {}, NCHW);

        qLinear.reset(NN::Linear(dim, innerDim, false));
        kLinear.reset(NN::Linear(dim, innerDim, false));
        vLinear.reset(NN::Linear(dim, innerDim, false));

        if (projectOut) {
            toOut.emplace_back(std::shared_ptr<Module>(NN::Linear(innerDim, dim)));
            toOut.emplace_back(std::shared_ptr<Module>(NN::Dropout(dropout)));
        }

    }

    std::vector<Express::VARP> _Attention::onForward(const std::vector<Express::VARP> &inputs) {
        using namespace Express;
        VARP x = inputs[0];

        // Different from MobileViT, here Q K V is calculated separately
        auto q = qLinear->forward(x);
        auto k = kLinear->forward(x);
        auto v = vLinear->forward(x);
        // mvit[1] error
        auto shape = q->getInfo()->dim;

        q = _Reshape(q, {shape[0], shape[1], heads, shape[2], shape[3] / heads}, NCHW);
        k = _Reshape(k, {shape[0], shape[1], heads, shape[2], shape[3] / heads}, NCHW);
        v = _Reshape(v, {shape[0], shape[1], heads, shape[2], shape[3] / heads}, NCHW);

        auto dots = _MatMul(q, _Transpose(k, {0, 1, 2, 4, 3})) * scale;
        auto attn = _Softmax(dots); // self.attend
        auto out = _MatMul(attn, v);
        out = _Reshape(out, shape);

        if (!toOut.empty()) {
            for (int i = 0; i < toOut.size(); i++) {
                out = toOut[i]->forward(out);
            }
        }
        return {out};
    }

    class _FeedForward : public Module {
    public:
        _FeedForward(int dim, int hiddenDim, float dropout = 0.0);

        virtual std::vector<Express::VARP>
        onForward(const std::vector<Express::VARP> &inputs) override;

        std::vector<std::shared_ptr<Module> > net;
    };

    std::shared_ptr<Module> FeedForward(int dim, int hiddenDim, float dropout = 0.0) {
        return std::shared_ptr<Module>(new _FeedForward(dim, hiddenDim, dropout));
    }

    _FeedForward::_FeedForward(int dim, int hiddenDim, float dropout) {
        net.emplace_back(std::shared_ptr<Module>(NN::Linear(dim, hiddenDim)));
        net.emplace_back(SiLU());
        net.emplace_back(std::shared_ptr<Module>(NN::Dropout(dropout)));
        net.emplace_back(std::shared_ptr<Module>(NN::Linear(hiddenDim, dim)));
        net.emplace_back(std::shared_ptr<Module>(NN::Dropout(dropout)));
    }

    std::vector<Express::VARP> _FeedForward::onForward(const std::vector<Express::VARP> &inputs) {
        using namespace Express;
        VARP x = inputs[0];

        for (const auto &i: net) {
            x = i->forward(x);
        }

        return {x};
    }

    class _PreNorm : public Module {
    public:
        _PreNorm(int dim, std::shared_ptr<Module> fn);

        virtual std::vector<Express::VARP>
        onForward(const std::vector<Express::VARP> &inputs) override;

        std::shared_ptr<Module> norm;
        std::shared_ptr<Module> fn;
    };

    std::shared_ptr<Module> PreNorm(int dim, std::shared_ptr<Module> fn) {
        // 提供了抽象层级，提供一致的接口，也可以实现额外的逻辑
        return std::shared_ptr<Module>(new _PreNorm(dim, std::move(fn)));
    }

    _PreNorm::_PreNorm(int dim, std::shared_ptr<Module> fn) {
        this->norm.reset(new MLayerNorm({dim}));
        this->fn = std::move(fn);
    }

    std::vector<Express::VARP> _PreNorm::onForward(const std::vector<Express::VARP> &inputs) {
        using namespace Express;
        VARP x = inputs[0];

        x = norm->forward(x);
        x = fn->forward(x);

        return {x};
    }

    MTransformer::MTransformer(int dim, int depth, int heads, int dimHead, int mlpDim,
                               float dropout) {
        for (int i = 0; i < depth; i++) {
            layers.emplace_back(PreNorm(dim, Attention(dim, heads, dimHead, dropout)),
                                PreNorm(dim, FeedForward(dim, mlpDim, dropout)));
        }
    }

    std::vector<Express::VARP> MTransformer::onForward(const std::vector<Express::VARP> &inputs) {
        using namespace Express;
        VARP x = inputs[0];

        for (auto &layer: layers) {
            auto attn = layer.first;
            auto ff = layer.second;

            x = attn->forward(x) + x;
            x = ff->forward(x) + x;
        }

        return {x};
    }
}
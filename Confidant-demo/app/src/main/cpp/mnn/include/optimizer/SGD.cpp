//
//  SGD.cpp
//  MNN
//
//  Created by MNN on 2019/11/22.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "SGD.hpp"
#include "OpGrad.hpp"
using namespace MNN::Express;

namespace MNN {
namespace Train {
SGD::SGD(std::shared_ptr<Module> module) : ParameterOptimizer(module) {
    auto train = ParameterOptimizer::trainable();
    for (auto p : train) {
        mHistory[p] = _Const(0.0f, p->getInfo()->dim, p->getInfo()->order);
    }
}

void SGD::setLearningRate(float rate) {
    mLearningRate = rate;
}

void SGD::setMomentum(float momentum) {
    mMomentum = momentum;
}

void SGD::setWeightDecay(float decay) {
    mWeightDecay = decay;
}

void SGD::setRegularizationMethod(RegularizationMethod method) {
    mRegularizationMethod = method;
}

float SGD::currentLearningRate() {
    return mLearningRate;
}

float SGD::getMomentum() {
    return mMomentum;
}

float SGD::getWeightDecay() {
    return mWeightDecay;
}

SGD::RegularizationMethod SGD::getRegularizationMethod() {
    return mRegularizationMethod;
}

Express::VARP SGD::regularizeParameters(Express::VARP param, Express::VARP grad) {
    VARP addWeightDecayGrad;
    if (mRegularizationMethod == L1) {
        auto temp          = _Sign(param);
        addWeightDecayGrad = _Const(mWeightDecay, {}, NCHW) * temp + grad;
    } else if (mRegularizationMethod == L2) {
        addWeightDecayGrad = _Const(mWeightDecay, {}, NCHW) * param + grad;
    } else if (mRegularizationMethod == L1L2) {
        auto temp          = _Sign(param);
        auto L1 = _Const(mWeightDecay, {}, NCHW) * temp;
        auto L2 = _Const(mWeightDecay, {}, NCHW) * param;
        addWeightDecayGrad = L1 + L2 + grad;
    }

    return addWeightDecayGrad;
}

Express::VARP SGD::onComputeUpdateValue(Express::VARP param, Express::VARP grad) {
    auto lr         = _Const(mLearningRate, {}, NCHW);
    mHistory[param] = lr * (grad + _Const(mMomentum, {}, NCHW) * mHistory[param]);
    mHistory[param].fix(Express::VARP::CONSTANT);
    //FUNC_PRINT_ALL(_ReduceMax(grad)->readMap<float>()[0], f);
    return mHistory[param];
}

std::map<Express::VARP, Express::VARP> SGD::onGetNextParameter(Express::VARP loss) {
    auto grad = OpGrad::grad(loss, trainable(), mGradBlockExprName);
    auto parameters = module()->parameters();
    std::vector<VARP> prepareCompute;
    for (auto iter : parameters) {
        if (iter->expr().first->get() != nullptr) {
            prepareCompute.emplace_back(iter);
        }
    }
    for (auto& iter : grad) {
        prepareCompute.emplace_back(iter.second);
    }
    Variable::prepareCompute(prepareCompute);
    std::vector<VARP> replaceOp(prepareCompute.size());
    for (int i=0; i<prepareCompute.size(); ++i) {
        auto info = prepareCompute[i]->getInfo();
        auto ptr = prepareCompute[i]->readMap<void>();
        if (nullptr == ptr) {
            MNN_ERROR("Compute error in SGD\n");
            return {};
        }
        auto newVar = _Const(ptr, info->dim, info->order, info->type);
        replaceOp[i]= newVar;
    }
    for (int i=0; i<prepareCompute.size(); ++i) {
        Variable::replace(prepareCompute[i], replaceOp[i]);
    }

    for (auto& iter : grad) {
        // apply regularization
        auto addWeightDecayGrad = regularizeParameters(iter.first, iter.second);
        addWeightDecayGrad.fix(Express::VARP::CONSTANT);
        // apply momentum, etc.
        auto updateValue = this->onComputeUpdateValue(iter.first, addWeightDecayGrad);
        // apply update
        auto newParameter = iter.first - updateValue;
        iter.second       = newParameter;
    }
    return grad;
}

std::vector<Express::VARP> SGD::backward(Express::VARP loss, Express::VARP input, Express::VARP lossGrad, std::map<Express::VARP, Express::VARP>& grads) {
    // return the grad of the input tensor
    std::map<EXPRP, std::vector<VARP>> backwardMap;
    {
        // auto shape = loss->getInfo();
        // MNN_ASSERT(shape->size == 1);
        // auto init= _Const(1.0f, shape->dim, shape->order);
        // expr: pair<mFrom, mIndex>
        backwardMap[loss->expr().first] = std::vector<VARP>{lossGrad};
    }

    // gradCommon
    auto executeOrder = Variable::getExecuteOrder({loss});
    for (auto iter = executeOrder.rbegin(); iter != executeOrder.rend(); iter++) {
        auto expr    = *iter;
        auto& inputs = expr->inputs();
        if (backwardMap.find(expr) == backwardMap.end()) {
            continue;
        }
        if (nullptr == expr->get()) {
            continue;
        }

        auto grad = OpGrad::get(expr->get()->type()); // 传进去是 OpType_UnaryOp
        if (nullptr == grad) {
            // MNN_PRINT("Can't grad for %s, %s\n", expr->name().c_str(), MNN::EnumNameOpType(expr->get()->type()));
            continue;
        }
        // inputGrad 的 name 带一个 _Grad 的后缀
        auto inputGrad = grad->onGrad(expr, backwardMap[expr]); // 计算梯度，计算得到的梯度放在 backwardMap[expr]里面？
        auto empty     = true;
        for (auto grad : inputGrad) {
            if (nullptr != grad) {
                empty = false;
                break;
            }
        }
        if (empty) {
            // MNN_PRINT("Can't grad for %s, %d\n", expr->name().c_str(), expr->get()->type());
            continue;
        }
        if (!mGradBlockExprName.empty()) {
            if (std::find(mGradBlockExprName.begin(), mGradBlockExprName.end(), expr->name()) != mGradBlockExprName.end()) {
                for (int ii = 0; ii <inputGrad.size(); ii++) {
                    inputGrad[ii] = nullptr;
                }
                continue;
            }
        }

        MNN_ASSERT(inputGrad.size() <= inputs.size());
        for (int i = 0; i < inputGrad.size(); ++i) {
            auto inputExpr = inputs[i]->expr().first;
            auto index     = inputs[i]->expr().second;
            auto backward  = inputGrad[i];
            if (nullptr == backward) {
                continue;
            }
            if (backwardMap.find(inputExpr) == backwardMap.end()) {
                backwardMap.insert(std::make_pair(inputExpr, std::vector<VARP>(inputExpr->outputSize())));
            }
            auto& inputVarMap = backwardMap[inputExpr];
            if (nullptr == inputVarMap[index]) {
                inputVarMap[index] = backward;
            } else {
                inputVarMap[index] = _Add(inputVarMap[index], backward);
            }
        }
    }

    // std::map<Express::VARP, Express::VARP> grads;
    std::map<Expr*, VARP> parametersExpr;

    for (auto p : trainable()) {
        parametersExpr.insert(std::make_pair(p->expr().first.get(), p));
    }

    // 把 trainable() 相关的梯度放进去 grads 返回，其他expr不返回
    for (auto iter : backwardMap) {
        auto expr = iter.first.get();
        if (parametersExpr.find(expr) != parametersExpr.end()) {
            auto parameter   = parametersExpr[expr];
            grads[parameter] = iter.second[parameter->expr().second];
        }
    }

    // MNN_ASSERT(backwardMap.find(input->expr().first) != backwardMap.end());
    if (backwardMap.find(input->expr().first) == backwardMap.end()) {
        return {};
    }
    return backwardMap[input->expr().first];
}

void SGD::updateParameters(std::map<Express::VARP, Express::VARP>& grad) {
    auto parameters = module()->parameters(); // 包括模型的所有参数，不仅是可以训练的
    std::vector<VARP> prepareCompute;

    for (auto iter : parameters) {
        if (iter->expr().first->get() != nullptr) {
            prepareCompute.emplace_back(iter);
        }
    }

    for (auto& iter : grad) {
        prepareCompute.emplace_back(iter.second);
    }
    Variable::prepareCompute(prepareCompute); // 给 Executor 创建 cache
    std::vector<VARP> replaceOp(prepareCompute.size());
    for (int i=0; i<prepareCompute.size(); ++i) {
        auto info = prepareCompute[i]->getInfo();
        auto ptr = prepareCompute[i]->readMap<void>();  // 计算并且存入 cache？
        if (nullptr == ptr) {
            MNN_ERROR("Compute error in SGD\n");
            return ;
        }
        auto newVar = _Const(ptr, info->dim, info->order, info->type);
        replaceOp[i]= newVar;
    }
    for (int i=0; i<prepareCompute.size(); ++i) {
        // VARP 的地址不变，里面的内容变了
        Variable::replace(prepareCompute[i], replaceOp[i]);
    }

    for (auto& iter : grad) {
        // iter.second 的地址和 prepareCompute 里面的是一样的
        // apply regularization
        auto addWeightDecayGrad = regularizeParameters(iter.first, iter.second);
        addWeightDecayGrad.fix(Express::VARP::CONSTANT);
        // apply momentum, etc.
        auto updateValue = this->onComputeUpdateValue(iter.first, addWeightDecayGrad);
        // apply update
        auto newParameter = iter.first - updateValue;
        iter.second       = newParameter;  // 这个应该是更新好的参数 在上层函数进行赋值
    }
}

} // namespace Train
} // namespace MNN

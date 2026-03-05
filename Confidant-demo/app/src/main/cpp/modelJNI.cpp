//
// Created by Yuhao Chen on 2024/7/4.
//
#include <jni.h>
#include "RandomGenerator.hpp"

#include "jniUtils.h"
#include "model.h"
#include "commonStates.h"
#include "log.h"
#include "train.h"
#include "general.h"
#include "faultTolerance.h"

#include "multiProcessorScheduler.h"

#include "BERT.h"
#include "GPT2.h"


extern "C"
JNIEXPORT void JNICALL
Java_com_example_confidant_utils_Model_initModel(JNIEnv *env, jclass clazz, jstring name,
                                                 jobject args) {
    using namespace MNN;
    using namespace MNN::Train;
    using namespace Confidant;

    RandomGenerator::generator(17);
    std::string modelName = env->GetStringUTFChars(name, 0);

    std::map<std::string, double> modelArgs;
    Confidant::JavaHashMapToStlStringDoubleMap(env, args, modelArgs);

    Confidant::ModelZoo::createModel(modelName, modelArgs);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_confidant_utils_Model_loadModelWeights(JNIEnv *env, jclass clazz,
                                                        jstring model_name, jstring weights_path,
                                                        jint num_layers) {
    using namespace Confidant;
    const char *modelName = env->GetStringUTFChars(model_name, 0);
    const char *weightPath = env->GetStringUTFChars(weights_path, 0);
    int numLayers = num_layers;
    std::string modelNameStr = modelName;
    std::string weightsPathStr = weightPath;

    CommonStates::setModelWeightsPath(weightsPathStr);

    std::shared_ptr<SingleModel> model = ModelZoo::modelPtr;
    model->loadParam(weightsPathStr);
}

extern "C"
JNIEXPORT jobjectArray JNICALL
Java_com_example_confidant_utils_Model_profileModelTime(JNIEnv *env, jclass clazz,
                                                        jint total_layer, jint batch_size, jint seq_len) {
    using namespace Confidant;

    // here we put forward time and backwardtime into one vector
    vector<float> computeTime(total_layer * 2);
    std::shared_ptr<SingleModel> model = ModelZoo::modelPtr;
    // TODO: profile the time here
    for (int i = 0; i < total_layer; i++) {
        computeTime[i] = model->getModelTimeByIdx(i, batch_size, seq_len);
        LOGI("profileModelTime(): Layer %d forward time: %f", i, computeTime[i]);
    }

    // we assume the backward time is two times of the forward time
    for (int i = 0; i < total_layer; i++) {
        computeTime[i + total_layer] = computeTime[i] * 2;
    }

    // convert the float vector into jobjectArray
    jobjectArray result = env->NewObjectArray(computeTime.size(), env->FindClass("java/lang/Object"), NULL);
    for (int i = 0; i < computeTime.size(); i++) {
        jfloat cur = computeTime[i];
        jclass floatClass = env->FindClass("java/lang/Float");
        jmethodID floatConstructor = env->GetMethodID(floatClass, "<init>", "(F)V");
        jobject floatObj = env->NewObject(floatClass, floatConstructor, cur);
        env->SetObjectArrayElement(result, i, floatObj);
    }
    return result;
}

extern "C"
JNIEXPORT jobjectArray JNICALL
Java_com_example_confidant_utils_Model_profileDataSize(JNIEnv *env, jclass clazz,
                                                       jint total_layer, jint batch_size, jint seq_len) {
    using namespace Confidant;
    vector<float> dataSize(total_layer);
    std::shared_ptr<SingleModel> model = ModelZoo::modelPtr;
    // The unit of the data size is KB
    for (int i = 0; i < dataSize.size(); i++) {
        dataSize[i] = model->getOutputDataSizeByIdx(i, batch_size, seq_len);
    }

    // convert the float vector into jobjectArray
    jobjectArray result = env->NewObjectArray(dataSize.size(), env->FindClass("java/lang/Object"), NULL);
    for (int i = 0; i < dataSize.size(); i++) {
        jfloat cur = dataSize[i];
        jclass floatClass = env->FindClass("java/lang/Float");
        jmethodID floatConstructor = env->GetMethodID(floatClass, "<init>", "(F)V");
        jobject floatObj = env->NewObject(floatClass, floatConstructor, cur);
        env->SetObjectArrayElement(result, i, floatObj);
    }
    return result;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_confidant_utils_Model_initSubModel(JNIEnv *env, jclass clazz, jstring name,
                                                    jobject args, jint start_layer,
                                                    jint end_layer) {
    using namespace MNN;
    using namespace MNN::Train;
    using namespace Confidant;

    RandomGenerator::generator(17);
    std::string modelName = env->GetStringUTFChars(name, 0);

    std::map<std::string, double> modelArgs;
    JavaHashMapToStlStringDoubleMap(env, args, modelArgs);

    ModelZoo::createSubModel(modelName, modelArgs, start_layer, end_layer);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_confidant_utils_Model_loadSubModelWeights(JNIEnv *env, jclass clazz,
                                                           jstring weights_path, jint start,
                                                           jint end) {
    using namespace Confidant;
    const char *basePathTemp = env->GetStringUTFChars(weights_path, 0);
    std::string weightsBasePath = basePathTemp;
    std::shared_ptr<SubModel> subModel = ModelZoo::subModelPtr;

    CommonStates::setModelWeightsPath(weightsBasePath);

    for (int i = start; i <= end; i++) {
        subModel->loadParamByLayer(i, weightsBasePath, start);
    }
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_confidant_utils_Model_initTrainEpoch(JNIEnv *env, jclass clazz) {
    Confidant::initTrainEpoch();
}


extern "C"
JNIEXPORT jfloat JNICALL
Java_com_example_confidant_utils_General_profileTransformerBlock(JNIEnv *env, jclass clazz,
                                                                 jstring model_name, jobject args, jint num_blocks) {
    using namespace MNN;
    using namespace MNN::Train;
    using namespace Confidant;

    std::string modelName = env->GetStringUTFChars(model_name, 0);

    std::map<std::string, double> modelArgs;
    Confidant::JavaHashMapToStlStringDoubleMap(env, args, modelArgs);

    // TODO: phi2 & llama profile function
    if (modelName == "BERTForClassification") {
        return Model::BERTProfileBlock(modelArgs, num_blocks);
    } else if (modelName == "GPT2") {
        return Model::GPT2ProfileBlock(modelArgs, num_blocks);
    } else if (modelName == "Phi2Alpaca") {
        return Model::BERTProfileBlock(modelArgs, num_blocks);
    } else if (modelName == "LLaMALora") {
        return Model::BERTProfileBlock(modelArgs, num_blocks);
    } else {
        LOGI("profileTransformerBlock(): Unsupported model name\n");
    }

    return -1.0f;
}

extern "C"
JNIEXPORT jobjectArray JNICALL
Java_com_example_confidant_faultTolerance_ReplicationUtils_getSubModelWeightsByLayer(JNIEnv *env, jclass clazz, jint layer, jboolean is_trainable) {
    using namespace Confidant;
    auto subModel = ModelZoo::subModelPtr;
    int startLayer = subModel->getStartLayer();
    auto params = subModel->getParamsByLayer(layer - startLayer, is_trainable);
    return convertVARPArrIntoObjectArr(env, params);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_confidant_faultTolerance_ReplicationUtils_loadSubModelWeightsByLayer(JNIEnv *env, jclass clazz, jint layer, jobjectArray weights, jboolean is_trainable) {
    using namespace Confidant;
    auto subModel = ModelZoo::subModelPtr;
    auto params = convertObjectArrIntoVARPArr(env, weights);
    int startLayer = subModel->getStartLayer();
    auto originParams = subModel->getParamsByLayer(layer - startLayer, is_trainable);
    copyParameters(params, originParams);
}

extern "C"
JNIEXPORT jobjectArray JNICALL
Java_com_example_confidant_utils_Model_retrainBatchWithIterId(JNIEnv *env, jclass clazz,
                                                              jint iter_id) {
    using namespace Confidant;
    auto output = retrainBatchWithIterId(iter_id);
    return convertCentralForwardIntoObjectArr(env, output);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_confidant_utils_Model_initMultiProcessorScheduler(JNIEnv *env, jclass clazz,
                                                                   jstring name, jobject args) {
    using namespace Confidant;
    CommonStates::setBatchSize(1); // Batch size would be reset when calling initInit() afterward
    std::string modelName = env->GetStringUTFChars(name, 0);

    std::map<std::string, double> modelArgs;
    JavaHashMapToStlStringDoubleMap(env, args, modelArgs);

    MultiProcessorScheduler::mpsPtr = std::shared_ptr<MultiProcessorScheduler>(new MultiProcessorScheduler());
    MultiProcessorScheduler::mpsPtr->profileProcessors(modelName, modelArgs);
    MultiProcessorScheduler::mpsPtr->computeAllocationStrategy();

    std::vector<ProcessorInfo> allocationStrategy = MultiProcessorScheduler::mpsPtr->getAllocationStrategy().second;
    for (int i = 0; i < allocationStrategy.size(); i++) {
        LOGI("Processor %d: %d\n", i, allocationStrategy[i].numAttentionHead);
    }

    // Set processors for the model
    std::vector<pair<MNNForwardType, int> > types = {{MNN_FORWARD_OPENCL, 1}, {MNN_FORWARD_CPU, 1}};
    auto exe = Executor::getGlobalExecutor();
    MNN::BackendConfig config;
    config.precision = MNN::BackendConfig::Precision_High;

    for (auto& type : types) {
        exe->setGlobalExecutorConfig(type.first, config, type.second);
    }
}
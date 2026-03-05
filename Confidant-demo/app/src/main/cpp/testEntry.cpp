//
// Created by Yuhao Chen on 2024/6/20.
//

#include <jni.h>
#include "multiProcessorScheduler.h"
#include <jniUtils.h>
#include "commonStates.h"
#include "multiProcessorSchedulerTest.h"
#include "BERTLayer.h"
#include "RandomGenerator.hpp"
#include <MNN/AutoTime.hpp>
#include "log.h"
#include <random>

extern "C"
JNIEXPORT void JNICALL
Java_com_example_confidant_unitTest_MultiProcessorSchedulerTest_MPSProfileTest(JNIEnv *env, jclass clazz, jstring name, jobject args) {
    using namespace Confidant;

    CommonStates::setBatchSize(8);
    std::string modelName = env->GetStringUTFChars(name, 0);

    std::map<std::string, double> modelArgs;
    Confidant::JavaHashMapToStlStringDoubleMap(env, args, modelArgs);

    multiProcessorSchedulerTestEntry(modelName, modelArgs);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_confidant_unitTest_CrossFrameworkAdapterTest_convertObjectIntoVARPTest(JNIEnv *env, jclass clazz, jobjectArray data) {
    using namespace Confidant;

    MNN::Timer _100Time;
    auto varpData = convertObjectArrIntoVARPArr(env, data);
    auto ptr = varpData[0]->readMap<float>();
    auto time = (float)_100Time.durationInUs() / 1000.0f;
    LOGI("Convert ObjectArray to VARP time: %f ms\n", time);
}

extern "C"
JNIEXPORT jobjectArray JNICALL
Java_com_example_confidant_unitTest_CrossFrameworkAdapterTest_convertVARPIntoObjectTest(JNIEnv *env,
                                                                                        jclass clazz,
                                                                                        jint batch_size,
                                                                                        jint seq_len,
                                                                                        jint hidden_size) {
    using namespace Confidant;
    std::vector<int> shapeValue = {batch_size, seq_len, hidden_size};
    auto input = _Const(shapeValue.data(), {3}, NHWC, halide_type_of<int>());
    auto data = _RandomUnifom(input, halide_type_of<float>(), -1.0f, 1.0f);
    auto ptr = input->readMap<float>();
    std::vector<Express::VARP> dataArr = {data};

    MNN::Timer _100Time;
    jobjectArray dataObj = convertVARPArrIntoObjectArr(env, dataArr);
    auto time = (float)_100Time.durationInUs() / 1000.0f;
    LOGI("Convert VARP to ObjectArray time: %f ms\n", time);

    return dataObj;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_confidant_unitTest_CrossFrameworkAdapterTest_dataConversionOverheadTest(
        JNIEnv *env, jclass clazz, jint batch_size, jint seq_len, jint hidden_size) {
    using namespace Confidant;

    // VARP -> unified tensor
    std::vector<int> shapeValue = {batch_size, seq_len, hidden_size};
    auto input = _Const(shapeValue.data(), {3}, NHWC, halide_type_of<int>());
    auto data = _RandomUnifom(input, halide_type_of<float>(), -1.0f, 1.0f);
    auto ptr = data->readMap<float>();

    MNN::Timer _100Time;
    // put into std::vector<float>
    std::vector<float> varpVec(batch_size * seq_len * hidden_size);
    memcpy(varpVec.data(), ptr, varpVec.size() * sizeof(float));
    auto shape = data->getInfo()->dim;
    auto varp2TensorTime = (float)_100Time.durationInUs() / 1000.0f;
    LOGI("dataConversionOverheadTest(): VARP to Tensor time: %f ms\n", varp2TensorTime);

    // unified tensor -> VARP

    // generate random val of std::vector<float>
    std::vector<float> dataVec(batch_size * seq_len * hidden_size);
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<float> distribution(0, 2.0f);
    for(auto& elem : dataVec) {
        elem = distribution(generator);
    }

    _100Time.reset();
    auto tensor = _Input({batch_size, seq_len, hidden_size}, NHWC, halide_type_of<float>());
    auto tensorPtr = tensor->writeMap<float>();
    memcpy(tensorPtr, dataVec.data(), dataVec.size() * sizeof(float));
    auto tensor2VarpTime = (float)_100Time.durationInUs() / 1000.0f;
    LOGI("dataConversionOverheadTest(): Tensor to VARP time: %f ms\n", tensor2VarpTime);
}
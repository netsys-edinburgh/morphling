#include <jni.h>
#include <string>
#include "datasets.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include <android/bitmap.h>

#include "general.h"
#include "jniUtils.h"
#include "log.h"

#include "train.h"
#include "model.h"
#include "RandomGenerator.hpp"
#include "optimizer.h"

#include "singleTrain.h"
#include "BERT.h"
#include "GPT2.h"
#include "LLaMA.h"
#include "commonStates.h"

#include "faultTolerance.h"

#include "profiler.h"

#include "BERTLayer.h"
#include <MNN/AutoTime.hpp>
#include "ModelArgs.h"
#include "LLaMALayer.h"

#include "multiplicationTest.h"
#include "singleModelTest.h"

using namespace MNN;
using namespace MNN::Train::Model;

extern "C"
JNIEXPORT jobject JNICALL
Java_com_example_confidant_globalStates_Backend_getBackendsMap(JNIEnv *env, jclass clazz) {
    using namespace Confidant;
    auto exe = Executor::getGlobalExecutor();
    auto attr = exe->getAvailableBackends();

    // Create a new Java HashMap object
    jclass hashMapClass = env->FindClass("java/util/HashMap");
    jmethodID hashMapConstructor = env->GetMethodID(hashMapClass, "<init>", "()V");
    jobject hashMapObj = env->NewObject(hashMapClass, hashMapConstructor);

    // Get the method ID of HashMap.put() method
    jmethodID putMethod = env->GetMethodID(hashMapClass, "put", "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");

    // Get the method ID of Integer.valueOf() method
    jclass integerClass = env->FindClass("java/lang/Integer");
    jmethodID valueOfMethod = env->GetStaticMethodID(integerClass, "valueOf", "(I)Ljava/lang/Integer;");

    // Get the method ID of Integer.intValue() method
    jmethodID intValueMethod = env->GetMethodID(integerClass, "intValue", "()I");

    for (const auto& pair : attr) {
        jint key = static_cast<jint>(pair.first);
        jint value = static_cast<jint>(pair.second);

        jobject keyObj = env->CallStaticObjectMethod(integerClass, valueOfMethod, key);
        jobject valueObj = env->CallStaticObjectMethod(integerClass, valueOfMethod, value);

        env->CallObjectMethod(hashMapObj, putMethod, keyObj, valueObj);

        env->DeleteLocalRef(keyObj);
        env->DeleteLocalRef(valueObj);
    }

    env->DeleteLocalRef(hashMapClass);
    env->DeleteLocalRef(integerClass);

    return hashMapObj;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_confidant_utils_General_syncGlobalModelName(JNIEnv *env, jclass clazz, jint global_modal_name) {
    using namespace Confidant;
    CommonStates::setGlobalModelName(static_cast<CommonStates::ModelName>(global_modal_name));
}

//// For test only
//extern "C"
//JNIEXPORT void JNICALL
//Java_com_example_ftpipehd_1mnn_models_Model_selfAttentionHeadTest(JNIEnv *env, jclass clazz, jstring weights_path) {
//    using namespace FTPipeHD;
//    const char *basePathTemp = env->GetStringUTFChars(weights_path, 0);
//    std::string weightsBasePath = basePathTemp;
//    selfAttentionHeadTest(weightsBasePath);
//    // mobileNetV2Test();
//}
//
//extern "C"
//JNIEXPORT jobjectArray JNICALL
//Java_com_example_ftpipehd_1mnn_faultTolerance_Replication_getSubModelWeightsByLayer(JNIEnv *env,
//                                                                                    jclass clazz,
//                                                                                    jint layer,
//                                                                                    jboolean is_trainable) {
//    using namespace FTPipeHD;
//    auto subModel = ModelZoo::subModelPtr;
//    auto params = subModel->getParamsByLayer(layer, is_trainable);
//    return convertVARPArrIntoObjectArr(env, params);
//}
//extern "C"
//JNIEXPORT void JNICALL
//Java_com_example_ftpipehd_1mnn_faultTolerance_Replication_loadSubModelWeightsByLayer(JNIEnv *env,
//                                                                                     jclass clazz,
//                                                                                     jint layer,
//                                                                                     jobjectArray weights,
//                                                                                     jboolean is_trainable) {
//    using namespace FTPipeHD;
//    auto subModel = ModelZoo::subModelPtr;
//    auto params = convertObjectArrIntoVARPArr(env, weights);
//    auto originParams = subModel->getParamsByLayer(layer, is_trainable);
//    copyParameters(params, originParams);
//}
//
//extern "C"
//JNIEXPORT jobjectArray JNICALL
//Java_com_example_ftpipehd_1mnn_models_Model_retrainBatchWithIterId(JNIEnv *env, jclass clazz,
//                                                                   jint iter_id) {
//    // TODO: implement retrainBatchWithIterId()
//    using namespace FTPipeHD;
//    auto output = retrainBatchWithIterId(iter_id);
//    return convertVARPArrIntoObjectArr(env, output);
//}
//extern "C"
//JNIEXPORT jobjectArray JNICALL
//Java_com_example_ftpipehd_1mnn_models_Model_trainIntermediate__I_3Ljava_lang_Object_2(JNIEnv *env, jclass clazz,
//                                                                                      jint iter_id,
//                                                                                      jobjectArray input_data) {
//    using namespace FTPipeHD;
//    auto inputs = convertObjectArrIntoVARPArr(env, input_data);
//    auto output = trainIntermediate(iter_id, inputs);
//
//    return convertVARPArrIntoObjectArr(env, output);
//}

//extern "C"
//JNIEXPORT void JNICALL
//Java_com_example_ftpipehd_1mnn_models_Model_LLaMACreateTest(JNIEnv *env, jclass clazz) {
//    // TODO: implement LLaMACreateTest()
//    using namespace FTPipeHD;
//    LLaMACreateTest();
//}
//extern "C"
//JNIEXPORT void JNICALL
//Java_com_example_ftpipehd_1mnn_models_Model_GPT2CreateTest(JNIEnv *env, jclass clazz) {
//    // TODO: implement GPT2CreateTest()
//    using namespace FTPipeHD;
//    GPT2CreateTest();
//}
//extern "C"
//JNIEXPORT void JNICALL
//Java_com_example_ftpipehd_1mnn_utils_TrainSingle_LLaMASelfAttentionHeadTest(JNIEnv *env,
//                                                                            jclass clazz) {
//    // TODO: implement LLaMASelfAttentionHeadTest()
//    using namespace FTPipeHD;
//    LLaMASelfAttentionHeadTest();
//}
//
//extern "C"
//JNIEXPORT void JNICALL
//Java_com_example_ftpipehd_1mnn_unitTest_SingleTest_BERTSelfAttnHeadTest(JNIEnv *env, jclass clazz) {
//    using namespace FTPipeHD;
//    BERTSelfAttnTest();
//}
//
//extern "C"
//JNIEXPORT void JNICALL
//Java_com_example_ftpipehd_1mnn_unitTest_SingleTest_LLaMASelfAttnHeadTest(JNIEnv *env,
//                                                                         jclass clazz) {
//    using namespace FTPipeHD;
//    LLaMASelfAttnTest();
//}
//
//extern "C"
//JNIEXPORT jobjectArray JNICALL
//Java_com_example_ftpipehd_1mnn_models_Model_profileEncoderTime(JNIEnv *env, jclass clazz,
//                                                               jstring model_name,
//                                                               jobject model_args, jint num_encoders) {
//    using namespace FTPipeHD;
//    RandomGenerator::generator(17);
//    std::string modelName = env->GetStringUTFChars(model_name, 0);
//
//    std::map<std::string, double> modelArgs;
//    FTPipeHD::JavaHashMapToStlStringDoubleMap(env, model_args, modelArgs);
//
//    auto ccv = profileEncoders(modelName, modelArgs, num_encoders);
//
//    // conver the float vector into jobjectArray
//    jobjectArray result = env->NewObjectArray(ccv.size(), env->FindClass("java/lang/Object"), NULL);
//    for (int i = 0; i < ccv.size(); i++) {
//        jfloat cur = ccv[i];
//        jclass floatClass = env->FindClass("java/lang/Float");
//        jmethodID floatConstructor = env->GetMethodID(floatClass, "<init>", "(F)V");
//        jobject floatObj = env->NewObject(floatClass, floatConstructor, cur);
//        env->SetObjectArrayElement(result, i, floatObj);
//    }
//    return result;
//}
//extern "C"
//JNIEXPORT jobjectArray JNICALL
//Java_com_example_ftpipehd_1mnn_unitTest_SingleTest_getDataFromVARP(JNIEnv *env, jclass clazz,
//                                                                   jint batch_size, jint seq_length,
//                                                                   jint hidden_size) {
//    using namespace FTPipeHD;
//    RandomGenerator::generator(17);
//
//    std::vector<int> shapeValue = {batch_size, seq_length, hidden_size};
//    auto input = _Const(shapeValue.data(), {3}, NHWC, halide_type_of<int>());
//    auto testData = _RandomUnifom(input, halide_type_of<float>(), -1.0f, 1.0f);
//    // auto testData = _Const(1.0f, {batch_size, seq_length, hidden_size}, NCHW);
//    auto testPtr = input->readMap<float>();
//    std::vector<VARP> testDataArr = {testData};
//
//    Timer _100Time;
//    jobjectArray ret = convertVARPArrIntoObjectArr(env, testDataArr);
//    LOGI("Convert VARP into jobject Time: %f ms, shape: %d, %d, %d\n", (float)_100Time.durationInUs() / 1000.0f, batch_size, seq_length, hidden_size);
//    return ret;
//}
//extern "C"
//JNIEXPORT void JNICALL
//Java_com_example_ftpipehd_1mnn_unitTest_SingleTest_setDataToVARP(JNIEnv *env, jclass clazz,
//                                                                 jobjectArray data) {
//    using namespace FTPipeHD;
//    RandomGenerator::generator(17);
//
//    Timer _100Time;
//    auto testData = convertObjectArrIntoVARPArr(env, data);
//    auto testPtr = testData[0]->readMap<float>();
//    auto shape = testData[0]->getInfo()->dim;
//    LOGI("Convert jobject into VARP Time: %f ms, shape: %d, %d, %d\n", (float)_100Time.durationInUs() / 1000.0f, shape[0], shape[1], shape[2]);
//    return ;
//}
//
//extern "C"
//JNIEXPORT void JNICALL
//Java_com_example_ftpipehd_1mnn_unitTest_UnitTest_multiplicationTest(JNIEnv *env, jclass clazz) {
//    using namespace FTPipeHD;
//
//    // testMultiplication();
//    testIntermediateMultiplication();
//}
//
//extern "C"
//JNIEXPORT void JNICALL
//Java_com_example_ftpipehd_1mnn_unitTest_UnitTest_singleModelTest(JNIEnv *env, jclass clazz) {
//    using namespace FTPipeHD;
//    // BERTSelfAttnTest();
//    // smallChunkBERTSelfAttnTest();
//    // smallChunkBERTTest();
//    // testMatMulParallel();
//    // smallChunkBERTNewImplTest();
//    // singleBERTParallelTest();
//    // singleBERTMMPTest();
//    // singleBERTTest();
//    singleGPT2Test();
//}
//

//
//extern "C"
//JNIEXPORT jobjectArray JNICALL
//Java_com_example_ftpipehd_1mnn_models_Model_profileModelTime(JNIEnv *env, jclass clazz, jint total_layer) {
//    using namespace FTPipeHD;
//
//    // here we put forward time and backwardtime into one vector
//    vector<float> computeTime(total_layer * 2);
//
//    // TODO: profile the time here
//    for (int i = 0; i < computeTime.size(); i++) {
//        computeTime[i] = i;
//    }
//
//    // convert the float vector into jobjectArray
//    jobjectArray result = env->NewObjectArray(computeTime.size(), env->FindClass("java/lang/Object"), NULL);
//    for (int i = 0; i < computeTime.size(); i++) {
//        jfloat cur = computeTime[i];
//        jclass floatClass = env->FindClass("java/lang/Float");
//        jmethodID floatConstructor = env->GetMethodID(floatClass, "<init>", "(F)V");
//        jobject floatObj = env->NewObject(floatClass, floatConstructor, cur);
//        env->SetObjectArrayElement(result, i, floatObj);
//    }
//    return result;
//}
//
//extern "C"
//JNIEXPORT jobjectArray JNICALL
//Java_com_example_ftpipehd_1mnn_models_Model_profileDataSize(JNIEnv *env, jclass clazz,
//                                                            jint total_layer) {
//    vector<float> dataSize(total_layer);
//
//    // TODO: profile the time here
//    for (int i = 0; i < dataSize.size(); i++) {
//        dataSize[i] = i;
//    }
//
//    // convert the float vector into jobjectArray
//    jobjectArray result = env->NewObjectArray(dataSize.size(), env->FindClass("java/lang/Object"), NULL);
//    for (int i = 0; i < dataSize.size(); i++) {
//        jfloat cur = dataSize[i];
//        jclass floatClass = env->FindClass("java/lang/Float");
//        jmethodID floatConstructor = env->GetMethodID(floatClass, "<init>", "(F)V");
//        jobject floatObj = env->NewObject(floatClass, floatConstructor, cur);
//        env->SetObjectArrayElement(result, i, floatObj);
//    }
//    return result;
//}

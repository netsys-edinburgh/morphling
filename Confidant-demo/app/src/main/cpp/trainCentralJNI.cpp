//
// Created by Yuhao Chen on 2024/7/5.
//
#include <jni.h>
#include "jniUtils.h"
#include "train.h"
#include "datasets.h"
#include "log.h"

extern "C"
JNIEXPORT jobjectArray JNICALL
Java_com_example_confidant_utils_TrainCentral_forwardOneBatch(JNIEnv *env, jclass clazz,
                                                              jint iter_id) {
    using namespace Confidant;

    auto output = trainForward(iter_id);

    return convertCentralForwardIntoObjectArr(env, output);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_confidant_utils_TrainCentral_backwardIntermediateCentral(JNIEnv *env, jclass clazz,
                                                                          jint iter_id,
                                                                          jobjectArray output_grad) {
    using namespace Confidant;
    auto outputGrad = convertObjectArrIntoVARPArr(env, output_grad);
    trainBackwardCentral(iter_id, outputGrad[0]);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_confidant_utils_TrainCentral_skipOneBatch(JNIEnv *env, jclass clazz) {
    using namespace Confidant;
    auto dataLoader = Datasets::trainSetLoader;
    auto trainData  = dataLoader->next();

    LOGI("skipOneBatch(): %d", trainData[0].first[0]->readMap<int>()[0]);
}
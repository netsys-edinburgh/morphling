//
// Created by Yuhao Chen on 2024/7/5.
//
#include <jni.h>
#include "train.h"
#include "jniUtils.h"

extern "C"
JNIEXPORT jobjectArray JNICALL
Java_com_example_confidant_utils_TrainWorker_forwardIntermediate(JNIEnv *env, jclass clazz,
                                                                 jint iter_id,
                                                                 jobjectArray input_data) {
    using namespace Confidant;
    auto inputs = convertObjectArrIntoVARPArr(env, input_data);
    auto output = trainIntermediate(iter_id, inputs);

    return convertVARPArrIntoObjectArr(env, output);
}

extern "C"
JNIEXPORT jobjectArray JNICALL
Java_com_example_confidant_utils_TrainWorker_trainIntermediateLast(JNIEnv *env, jclass clazz,
                                                                   jint iter_id,
                                                                   jobjectArray input_data,
                                                                   jobjectArray labels_data) {
    using namespace Confidant;
    auto intermediates = convertObjectArrIntoVARPArr(env, input_data);
    auto labels = convertObjectArrIntoVARPArr(env, labels_data);

    auto output = trainIntermediateLast(iter_id, intermediates, labels);
    return convertVARPArrPairIntoObjectArr(env, output);
}

extern "C"
JNIEXPORT jobjectArray JNICALL
Java_com_example_confidant_utils_TrainWorker_backwardIntermediateWorker(JNIEnv *env, jclass clazz,
                                                                        jint iter_id,
                                                                        jobjectArray output_grad) {
    using namespace Confidant;
    auto outputGrad = convertObjectArrIntoVARPArr(env, output_grad);
    auto inputGrad = trainBackwardWorker(iter_id, outputGrad[0]);
    return convertVARPArrIntoObjectArr(env, inputGrad);
}
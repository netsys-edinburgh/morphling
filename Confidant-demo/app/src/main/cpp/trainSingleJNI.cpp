//
// Created by Yuhao Chen on 2024/7/4.
//
#include <jni.h>
#include "train.h"
#include "singleTrain.h"

extern "C"
JNIEXPORT void JNICALL
Java_com_example_confidant_utils_Model_initTrain(JNIEnv *env, jclass clazz, jint batch_size, jint deviceNum, jint deviceIdx) {
    using namespace Confidant;
    initTrain(batch_size, deviceNum, deviceIdx);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_confidant_utils_Model_singleTrainOneEpoch(JNIEnv *env, jclass clazz, jint epoch) {
    using namespace Confidant;
    singleTrainOneEpoch(epoch);
}
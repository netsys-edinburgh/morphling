//
// Created by Yuhao Chen on 2024/7/5.
//
#include <jni.h>
#include <iostream>
#include "optimizer.h"
#include "jniUtils.h"

extern "C"
JNIEXPORT void JNICALL
Java_com_example_confidant_utils_Optimizer_initSubOptimizer(JNIEnv *env, jclass clazz, jstring name,
                                                            jobject args) {
    using namespace Confidant;
    std::string optName = env->GetStringUTFChars(name, 0);
    std::map<std::string, double> optArgs;
    JavaHashMapToStlStringDoubleMap(env, args, optArgs);

    OptimizerWeightVersion::opt = std::shared_ptr<OptimizerWeightVersion>(new OptimizerWeightVersion(optName, optArgs));
}

extern "C"
JNIEXPORT jdouble JNICALL
Java_com_example_confidant_utils_Optimizer_getLearningRate(JNIEnv *env, jclass clazz) {
    return Confidant::OptimizerWeightVersion::opt->getLearningRate();
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_confidant_utils_Optimizer_setLearningRate(JNIEnv *env, jclass clazz, jdouble lr) {
    Confidant::OptimizerWeightVersion::opt->setLearningRate(lr);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_confidant_utils_Optimizer_resetOptimizer(JNIEnv *env, jclass clazz) {
    Confidant::OptimizerWeightVersion::opt->resetOptimizer();
}
//
// Created by Yuhao Chen on 2024/7/4.
//
#include <jni.h>
#include "datasets.h"

extern "C"
JNIEXPORT void JNICALL
Java_com_example_confidant_utils_Dataset_initDataset(JNIEnv *env, jclass clazz, jstring base_path,
                                                     jstring name, jstring path, jint batch_size) {
    using namespace MNN;
    using namespace MNN::Train;
    using namespace Confidant;

    const char *basePathTemp = env->GetStringUTFChars(base_path, 0);
    const char *nameTemp = env->GetStringUTFChars(name, 0);
    const char *pathTemp = env->GetStringUTFChars(path, 0);

    std::string datasetBasePath = basePathTemp;
    std::string datasetName = nameTemp;
    std::string datasetPath = pathTemp;

    Datasets::datasets = new Datasets(datasetBasePath, datasetName, datasetPath, batch_size);
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_example_confidant_utils_Dataset_getDataLen(JNIEnv *env, jclass clazz) {
    using namespace Confidant;
    return (int) Datasets::trainSetLoader->iterNumber();
}
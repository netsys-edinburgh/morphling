//
// Created by Yuhao Chen on 2023/2/21.
//

#ifndef CONFIDANT_JNIUTILS_H
#define CONFIDANT_JNIUTILS_H
#include <jni.h>
#include <string>
#include <map>
#include "MNN/expr/Module.hpp"

using namespace MNN;
using namespace MNN::Express;

namespace Confidant {
    void JavaHashMapToStlStringDoubleMap(JNIEnv *env, jobject hashMap, std::map<std::string, double>& mapOut);

    // general methods to convert a float type VARP into four jobject (data, dim, order, type)
    std::vector<jobject> convertVARPIntoObject(JNIEnv *env, VARP data);
    jobjectArray convertVARPArrIntoObjectArr(JNIEnv *env, std::vector<VARP>& varps);
    std::vector<VARP> convertObjectArrIntoVARPArr(JNIEnv *env, jobjectArray objArr);
    jobjectArray convertVARPArrPairIntoObjectArr(JNIEnv *env, std::pair<std::vector<VARP>, std::vector<VARP> >& varpPair);


    jobjectArray convertCentralForwardIntoObjectArr(JNIEnv *env, std::pair<std::vector<VARP>, std::vector<VARP> >& output);

//    jobjectArray convertVARPintoObjectArr(JNIEnv *env, VARP output);
//    jobjectArray convertVARPPairIntoObjectArr(JNIEnv *env, std::pair<std::vector<VARP>, VARP>& output);
//    jobjectArray convertFloatVARPPairIntoObjectArr(JNIEnv *env, std::pair<float, VARP>& output);
//
//    std::vector<Express::VARP> convertObjectArrIntoVARPs(JNIEnv *env, jobject input_datas, jobject dimArrs,
//                                                   jobject orders);
//    std::vector<Express::VARP> convertIntObjectArrIntoVARPs(JNIEnv *env, jobject input_datas, jobject dimArrs,
//                                                      jobject orders);
//    Express::VARP convertObjectIntoVARP(JNIEnv *env, jintArray input_data, jobject dimArr, jint order);
}

#endif //CONFIDANT_JNIUTILS_H

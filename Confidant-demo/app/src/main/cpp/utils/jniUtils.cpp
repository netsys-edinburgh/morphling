//
// Created by Yuhao Chen on 2023/2/21.
//
#include <jniUtils.h>
#include "log.h"
#include "MNN/expr/MathOp.hpp"
#include "MNN/expr/NeuralNetWorkOp.hpp"
#include "MNN/expr/Expr.hpp"
#include "commonStates.h"

using namespace MNN;

namespace Confidant {
    /*
     * Convert a Java HashMap into a C++ std::map<std::string, double>
     */
    void JavaHashMapToStlStringDoubleMap(JNIEnv *env, jobject hashMap, std::map<std::string, double>& mapOut) {
        // Get the Map's entry Set.
        jclass mapClass = env->FindClass("java/util/Map");
        if (mapClass == NULL) {
            return;
        }
        jmethodID entrySet =
                env->GetMethodID(mapClass, "entrySet", "()Ljava/util/Set;");
        if (entrySet == NULL) {
            return;
        }
        jobject set = env->CallObjectMethod(hashMap, entrySet);
        if (set == NULL) {
            return;
        }
        // Obtain an iterator over the Set
        jclass setClass = env->FindClass("java/util/Set");
        if (setClass == NULL) {
            return;
        }
        jmethodID iterator =
                env->GetMethodID(setClass, "iterator", "()Ljava/util/Iterator;");
        if (iterator == NULL) {
            return;
        }
        jobject iter = env->CallObjectMethod(set, iterator);
        if (iter == NULL) {
            return;
        }
        // Get the Iterator method IDs
        jclass iteratorClass = env->FindClass("java/util/Iterator");
        if (iteratorClass == NULL) {
            return;
        }
        jmethodID hasNext = env->GetMethodID(iteratorClass, "hasNext", "()Z");
        if (hasNext == NULL) {
            return;
        }
        jmethodID next =
                env->GetMethodID(iteratorClass, "next", "()Ljava/lang/Object;");
        if (next == NULL) {
            return;
        }
        // Get the Entry class method IDs
        jclass entryClass = env->FindClass("java/util/Map$Entry");
        if (entryClass == NULL) {
            return;
        }
        jmethodID getKey =
                env->GetMethodID(entryClass, "getKey", "()Ljava/lang/Object;");
        if (getKey == NULL) {
            return;
        }
        jmethodID getValue =
                env->GetMethodID(entryClass, "getValue", "()Ljava/lang/Object;");
        if (getValue == NULL) {
            return;
        }

        // Iterate over the entry Set
        while (env->CallBooleanMethod(iter, hasNext)) {
            jobject entry = env->CallObjectMethod(iter, next);
            jstring key = (jstring) env->CallObjectMethod(entry, getKey);
            jobject value = env->CallObjectMethod(entry, getValue);

            // convert the value into int type
            jclass intCls = env->FindClass("java/lang/Double");
            if (!intCls) {
                return ;
            }
            jmethodID doubleMethodID = env->GetMethodID(intCls, "doubleValue", "()D");
            if (doubleMethodID == NULL) {
                return ;
            }
            double doubleVal = env->CallDoubleMethod(value, doubleMethodID);

            const char* keyStr = env->GetStringUTFChars(key, NULL);
            if (!keyStr) {  // Out of memory
                return;
            }

            mapOut.insert(std::make_pair(std::string(keyStr), doubleVal));

            env->DeleteLocalRef(entry);
            env->ReleaseStringUTFChars(key, keyStr);
            env->DeleteLocalRef(key);
        }
    }

    /*
     *  Convert a float type VARP into a Object[] with four elements
     */
    std::vector<jobject> convertVARPIntoObject(JNIEnv *env, Express::VARP data) {
        auto dataInfo = data->getInfo();
        auto dataTotalSize = dataInfo->size;
        auto dataDims = dataInfo->dim;
        auto halideDataType = dataInfo->type;
        int dataType = CommonStates::MNNDataType::MNN_NOT_YET_SET;

        // dim vector
        jclass arrayListClass = env->FindClass("java/util/ArrayList");
        jmethodID arrayListConstructor = env->GetMethodID(arrayListClass, "<init>", "()V");
        jobject dimVector = env->NewObject(arrayListClass, arrayListConstructor);

        jmethodID arrayListAddMethod = env->GetMethodID(arrayListClass, "add", "(Ljava/lang/Object;)Z");
        jclass integerClass = env->FindClass("java/lang/Integer");
        jmethodID integerConstructor = env->GetMethodID(integerClass, "<init>", "(I)V");

        jclass floatClass = env->FindClass("java/lang/Float");
        jmethodID floatConstructor = env->GetMethodID(floatClass, "<init>", "(F)V");

        for (int i = 0; i < dataDims.size(); i++) {
            jint javaInt = dataDims[i];
            jobject javaIntObj = env->NewObject(integerClass, integerConstructor, javaInt);
            jboolean tmp = env->CallBooleanMethod(dimVector, arrayListAddMethod, javaIntObj);
            env->DeleteLocalRef(javaIntObj);
        }

        // order
        jint order = dataInfo->order;
        jobject integerObj = env->NewObject(integerClass, integerConstructor, order);

        if (halideDataType == halide_type_of<float>()) {
            auto dataPtr = data->readMap<float>();

            // Converting data array into List<Float> in JNI would cause significant performance overhead
            jfloatArray dataArr = env->NewFloatArray(dataTotalSize);

            env->SetFloatArrayRegion(dataArr, 0, dataTotalSize, dataPtr);

            // type
            dataType = CommonStates::MNNDataType::MNN_FLOAT;
            jint type = dataType;
            jobject typeObj = env->NewObject(integerClass, integerConstructor, type);

            return std::vector<jobject>{dataArr, dimVector, integerObj, typeObj};
        } else if (halideDataType == halide_type_of<int>()) {
            auto dataPtr = data->readMap<int>();
            dataType = CommonStates::MNNDataType::MNN_INT;

            // int[] impl
            jintArray dataArr = env->NewIntArray(dataTotalSize);
            env->SetIntArrayRegion(dataArr, 0, dataTotalSize, dataPtr);

            // type
            jint type = dataType;
            jobject typeObj = env->NewObject(integerClass, integerConstructor, type);

            return std::vector<jobject>{dataArr, dimVector, integerObj, typeObj};
        }

        return {};
    }

    /*
     * Convert a float type VARP array into Object[] array
     */
    jobjectArray convertVARPArrIntoObjectArr(JNIEnv *env, std::vector<Express::VARP>& varps) {
        int varpLen = varps.size();
        // each VARP has 4 jobject (data, dim, order, type)
        jobjectArray ret = env->NewObjectArray(varpLen * 4, env->FindClass("java/lang/Object"), NULL);

        for (int i = 0; i < varps.size(); i++) {
            auto obj = convertVARPIntoObject(env, varps[i]);
            env->SetObjectArrayElement(ret, i * 4, obj[0]);
            env->SetObjectArrayElement(ret, i * 4 + 1, obj[1]);
            env->SetObjectArrayElement(ret, i * 4 + 2, obj[2]);
            env->SetObjectArrayElement(ret, i * 4 + 3, obj[3]);

            // TODO: delete obj?
            for (int j = 0; j < obj.size(); j++) {
                env->DeleteLocalRef(obj[j]);
            }
            obj.clear();
        }

        return ret;
    }

    /*
     * Convert the Object[] array back to VARP array
     */
    std::vector<Express::VARP> convertObjectArrIntoVARPArr(JNIEnv *env, jobjectArray objArr) {
        // get the size of objArr
        jsize objArrSize = env->GetArrayLength(objArr);
        std::vector<Express::VARP> varps;

        // i is float[], i + 1 is dims, i + 2 is order, i + 3 is type
        for (int i = 0; i < objArrSize; i += 4) {
            // get type from types
            jclass jIntegerClass = env->FindClass("java/lang/Integer");
            jclass jFloatClass = env->FindClass("java/lang/Float");
            jmethodID jIntValueMethodID = env->GetMethodID(jIntegerClass, "intValue", "()I");
            jmethodID jFloatValueMethodID = env->GetMethodID(jFloatClass, "floatValue", "()F");

            jobject type = env->GetObjectArrayElement(objArr, i + 3);
            jint jType = (jint) env->CallIntMethod(type, jIntValueMethodID);
            int intType = (int) jType;

            // get the std::vector<int> from dimArrs
            jobject dimArr = env->GetObjectArrayElement(objArr, i + 1);
            jsize jArrayListSize = env->CallIntMethod(dimArr, env->GetMethodID(env->GetObjectClass(dimArr), "size", "()I"));
            jmethodID jArrayListGetMethodID = env->GetMethodID(env->GetObjectClass(dimArr), "get", "(I)Ljava/lang/Object;");

            std::vector<int> dim;
            int size = 1;
            for (int i = 0; i < jArrayListSize; i++) {
                jobject jIntElementObj = env->CallObjectMethod(dimArr, jArrayListGetMethodID, i);
                jint jIntElement = (jint) env->CallIntMethod(jIntElementObj, jIntValueMethodID);
                int intElement = (int) jIntElement;
                dim.push_back(intElement);
                size *= intElement;
            }

            // get order from orders
            jobject order = env->GetObjectArrayElement(objArr, i + 2);
            jint jOrder = (jint) env->CallIntMethod(order, jIntValueMethodID);
            int intOrder = (int) jOrder;

            if (intType == CommonStates::MNNDataType::MNN_INT) {
                // List<Integer> Impl
//                jobject dataArr = env->GetObjectArrayElement(objArr, i);
//                jsize jArrayListSize = env->CallIntMethod(dataArr, env->GetMethodID(env->GetObjectClass(dataArr), "size", "()I"));
//                jmethodID jArrayListGetMethodID = env->GetMethodID(env->GetObjectClass(dataArr), "get", "(I)Ljava/lang/Object;");
//                auto intArr = new int[jArrayListSize];
//                for (int i = 0; i < jArrayListSize; i++) {
//                    jobject jIntElementObj = env->CallObjectMethod(dataArr, jArrayListGetMethodID, i);
//                    jint jIntElement = (jint) env->CallIntMethod(jIntElementObj, jIntValueMethodID);
//                    int intElement = (int) jIntElement;
//                    intArr[i] = intElement;
//                    env->DeleteLocalRef(jIntElementObj);
//                }

                // int[] Impl
                jobject obj = env->GetObjectArrayElement(objArr, i);
                jintArray data = (jintArray) obj;
                jint* jIntArrayElements = env->GetIntArrayElements(data, nullptr);
                jsize jIntArrayLength = env->GetArrayLength(data);
                auto intArr = new int[jIntArrayLength];
                for (int i = 0; i < jIntArrayLength; i++) {
                    intArr[i] = jIntArrayElements[i];
                }

                env->ReleaseIntArrayElements(data, jIntArrayElements, 0);

                auto curInput = _Input(dim, static_cast<Express::Dimensionformat>(intOrder), halide_type_of<int>());
                auto inputPtr = curInput->writeMap<void>();
                ::memcpy(inputPtr, (void*) intArr, size * halide_type_of<int>().bytes());

                varps.push_back(curInput);
            } else if (intType == CommonStates::MNNDataType::MNN_FLOAT) {
                // List<Float> Impl
//                jobject dataArr = env->GetObjectArrayElement(objArr, i);
//                jsize jArrayListSize = env->CallIntMethod(dataArr, env->GetMethodID(env->GetObjectClass(dataArr), "size", "()I"));
//                jmethodID jArrayListGetMethodID = env->GetMethodID(env->GetObjectClass(dataArr), "get", "(I)Ljava/lang/Object;");
//
//                auto floatArr = new float[jArrayListSize];
//                for (int i = 0; i < jArrayListSize; i++) {
//                    jobject jFloatElementObj = env->CallObjectMethod(dataArr, jArrayListGetMethodID, i);
//                    jfloat jFloatElement = (jfloat) env->CallFloatMethod(jFloatElementObj, jFloatValueMethodID);
//                    float floatElement = (float) jFloatElement;
//                    floatArr[i] = floatElement;
//
//                    env->DeleteLocalRef(jFloatElementObj);
//                }

                // float[] Impl
                jobject obj = env->GetObjectArrayElement(objArr, i);
                jfloatArray data = (jfloatArray) obj;
                jfloat* jFloatArrayElements = env->GetFloatArrayElements(data, nullptr);
                jsize jFloatArrayLength = env->GetArrayLength(data);
                auto floatArr = new float[jFloatArrayLength];
                for (int i = 0; i < jFloatArrayLength; i++) {
                    floatArr[i] = jFloatArrayElements[i];
                }
                env->ReleaseFloatArrayElements(data, jFloatArrayElements, 0);

                auto curInput = _Input(dim, static_cast<Express::Dimensionformat>(intOrder), halide_type_of<float>());
                auto inputPtr = curInput->writeMap<void>();
                ::memcpy(inputPtr, (void*) floatArr, size * halide_type_of<float>().bytes());

                varps.push_back(curInput);
            } else {
                LOGE("Unknown type");
            }

        }
        return varps;
    }

    /*
     * Convert a float type VARP pair into Object[] array
     */
    jobjectArray convertVARPArrPairIntoObjectArr(JNIEnv *env, std::pair<std::vector<VARP>, std::vector<VARP> >& varpPair) {
        auto var1 = varpPair.first;
        auto var2 = varpPair.second;

        // We return an Object[] with 2 Object[] inside
        jobjectArray ret = env->NewObjectArray(2, env->FindClass("java/lang/Object"), NULL);

        // First put the output into an Object[]
        auto varObject1 = convertVARPArrIntoObjectArr(env, var1);
        auto varObject2 = convertVARPArrIntoObjectArr(env, var2);

        env->SetObjectArrayElement(ret, 0, varObject1);
        env->SetObjectArrayElement(ret, 1, varObject2);

        return ret;
    }

    /*
     *  Convert the intermediate output and labels pair into Object[]
     */
    jobjectArray convertCentralForwardIntoObjectArr(JNIEnv *env, std::pair<std::vector<Express::VARP>, std::vector<Express::VARP> >& output) {
        auto modelOutput = output.first;
        auto labels = output.second;

        // We return an Object[] with 2 Object[] inside
        jobjectArray ret = env->NewObjectArray(2, env->FindClass("java/lang/Object"), NULL);

        // First put the output into an Object[]
        auto outputObject = convertVARPArrIntoObjectArr(env, modelOutput);
        auto labelsObject = convertVARPArrIntoObjectArr(env, labels);

        env->SetObjectArrayElement(ret, 0, outputObject);
        env->SetObjectArrayElement(ret, 1, labelsObject);

        return ret;
    }
}
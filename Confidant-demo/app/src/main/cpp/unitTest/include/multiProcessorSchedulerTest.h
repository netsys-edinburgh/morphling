//
// Created by Yuhao Chen on 2024/7/18.
//

#ifndef CONFIDANT_MULTIPROCESSORSCHEDULERTEST_H
#define CONFIDANT_MULTIPROCESSORSCHEDULERTEST_H
#include <iostream>
#include <map>

namespace Confidant {
    void multiProcessorSchedulerTestEntry(std::string& modelName, std::map<std::string, double> &modelArgs);
    void BERTMPSTest(std::map<std::string, double> &modelArgs);
    void GPT2MPSTest(std::map<std::string, double> &modelArgs);
    void Phi2MPSTest(std::map<std::string, double> &modelArgs);
    void LLaMAMPSTest(std::map<std::string, double> &modelArgs);
}

#endif //CONFIDANT_MULTIPROCESSORSCHEDULERTEST_H

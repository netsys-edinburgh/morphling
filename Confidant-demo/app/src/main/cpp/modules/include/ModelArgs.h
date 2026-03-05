#include <utility>
//
// Created by Yuhao Chen on 2023/6/12.
//

#ifndef MNN_MODELARGS_H
#define MNN_MODELARGS_H

namespace MNN {
    namespace Train {
        namespace Model {
            struct LLaMAArgs {
                int vocabSize;
                int nLayers;
                int hiddenSize;
                float normEps;
                int nHeads;
                int nKVHeads = -1;
                int maxSeqLen;
                int maxBatchSize;
                int multipleOf;
                int ffnDimMultiplier;
                bool forParallel;
                std::string tokenizerPath;

                // LoRA params
                struct LoRAArgs {
                    int r;
                    std::vector<bool> enableLoRA;
                    float dropout;
                    float alpha; // for balancing the effect of lora
                    LoRAArgs() {
//                        r = 8;
//                        enableLoRA = {true, true, true};
                        r = -1;
                        enableLoRA = {false, false, false};
                        dropout = 0.0f;
                        alpha = 1.0f;
                    }
                    LoRAArgs(int _r, std::vector<bool> _enableLoRA, float _dropout, float _alpha): r(_r), enableLoRA(std::move(_enableLoRA)), dropout(_dropout), alpha(_alpha) {};
                } loraArgs;

                // LoRA
                LLaMAArgs() {
//                    vocabSize = -1;
                    vocabSize = 51200.0;
//                    nLayers = -1;
                    nLayers = 2.0;
                    hiddenSize = -1;
                    normEps = 0.0f;
                    nHeads = -1;
                    nKVHeads = -1;
//                    maxSeqLen = -1;
                    maxSeqLen = 256;
                    maxBatchSize = -1;
                    multipleOf = -1;
                    ffnDimMultiplier = -1;
                    tokenizerPath = "";
                    forParallel = false;
                    loraArgs = LoRAArgs(-1, {false, false, false}, 0.0f, 1.0);
//                    loraArgs = LoRAArgs(8, {true, true, true}, 0.0f, 1.0);
                }

                LLaMAArgs(int _vocabSize, int _nLayers, int _hiddenSize, float _normEps, int _nHeads, int _nKVHeads,
                          int _maxSeqLen, int _maxBatchSize, int _multipleOf, int _ffnDimMultiplier, std::string _tokenizerPath, bool _forParallel) :
                          vocabSize(_vocabSize), nLayers(_nLayers), hiddenSize(_hiddenSize), normEps(_normEps), nHeads(_nHeads), nKVHeads(_nKVHeads),
                          maxSeqLen(_maxSeqLen), maxBatchSize(_maxBatchSize), multipleOf(_multipleOf), ffnDimMultiplier(_ffnDimMultiplier), tokenizerPath(std::move(_tokenizerPath)),forParallel(_forParallel) {
                    loraArgs = LoRAArgs(-1, {false, false, false}, 0.0f, 1.0);
//                    loraArgs = LoRAArgs(8, {true, true, true}, 0.0f, 1.0);
                };

                LLaMAArgs(int _vocabSize, int _nLayers, int _hiddenSize, float _normEps, int _nHeads,
                          int _nKVHeads, int _maxSeqLen, int _maxBatchSize, int _multipleOf, int _ffnDimMultiplier,
                          std::string _tokenizerPath, bool _forParallel, LoRAArgs _loraArgs): vocabSize(_vocabSize), nLayers(_nLayers), hiddenSize(_hiddenSize), normEps(_normEps),
                          nHeads(_nHeads), nKVHeads(_nKVHeads), maxSeqLen(_maxSeqLen), maxBatchSize(_maxBatchSize), multipleOf(_multipleOf), ffnDimMultiplier(_ffnDimMultiplier),
                          tokenizerPath(std::move(_tokenizerPath)), forParallel(_forParallel), loraArgs(std::move(_loraArgs)) {};
            };

            struct GPT2Args {
                int vocabSize ;
                int maxPositionEmbeddings;
                int numHiddenLayers ;
                int numAttentionHeads ;
                int hiddenSize ;
                int intermediateSize ;
                float dropoutProb ;
                bool forParallel ;

                GPT2Args() {
                    vocabSize = 50257;
                    maxPositionEmbeddings =1024;
                    numHiddenLayers = 12;
                    numAttentionHeads = 12 ;
                    hiddenSize = 768;
                    intermediateSize = 768*4;
                    dropoutProb=0.0 ;
                    forParallel = false;
                }
                GPT2Args(int vocabSize, int maxPositionEmbeddings, int numHiddenLayers ,int numAttentionHeads ,
                         int hiddenSize , int intermediateSize, float dropoutPro ,bool forParallel ):
                        vocabSize(vocabSize), maxPositionEmbeddings(maxPositionEmbeddings), numHiddenLayers(numHiddenLayers),
                        numAttentionHeads(numAttentionHeads) , hiddenSize(hiddenSize), intermediateSize(intermediateSize),
                        dropoutProb(dropoutPro), forParallel(forParallel){};
            };

            struct PhiArgs {
                int vocabSize;
                int nLayers;
                int hiddenSize;
                float normEps;
                int nHeads;
                int nKVHeads = -1;
                int maxSeqLen;
                int maxBatchSize;
                int intermediateSize;
                float pdrop;
                std::string tokenizerPath;

                // LoRA params
                struct LoRAArgs {
                    int r;
                    std::vector<bool> enableLoRA;
                    float dropout;
                    float alpha; // for balancing the effect of lora
                    LoRAArgs() {
//                        r = 8;
//                        enableLoRA = {true, true, true};
                        r = -1;
                        enableLoRA = {false, false, false};
                        dropout = 0.0f;
                        alpha = 1.0f;
                    }
                    LoRAArgs(int _r, std::vector<bool> _enableLoRA, float _dropout, float _alpha): r(_r), enableLoRA(std::move(_enableLoRA)), dropout(_dropout), alpha(_alpha) {};
                } loraArgs;


                PhiArgs() {
                    vocabSize = -1;
                    nLayers = -1;
                    hiddenSize = -1;
                    normEps = 0.0f;
                    nHeads = -1;
                    nKVHeads = -1;
                    maxSeqLen = -1;
                    maxBatchSize = -1;
                    intermediateSize = 8192;
                    pdrop = 0.0;
                    tokenizerPath = "";
                    loraArgs = LoRAArgs(-1, {false, false, false}, 0.0f, 1.0);
//                    loraArgs = LoRAArgs(8, {true, true, true}, 0.0f, 1.0);

                }

                PhiArgs(int _vocabSize, int _nLayers, int _hiddenSize, float _normEps, int _nHeads, int _nKVHeads,
                        int _maxSeqLen, int _maxBatchSize, int intermediateSize,float pdrop,
                        std::string _tokenizerPath) :
                        vocabSize(_vocabSize), nLayers(_nLayers), hiddenSize(_hiddenSize), normEps(_normEps),
                        nHeads(_nHeads), nKVHeads(_nKVHeads),
                        maxSeqLen(_maxSeqLen), maxBatchSize(_maxBatchSize), intermediateSize(intermediateSize),pdrop(pdrop),
                        tokenizerPath(std::move(_tokenizerPath)) {
                };
            };
        }
    }
}

#endif // MNN_MODELARGS_H

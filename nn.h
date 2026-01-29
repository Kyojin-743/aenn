#pragma once

//===================================================================
//DEPENDANCIES
//===================================================================
#include <vector>
#include <cmath>
#include <cinttypes>
#include <fstream>
#include <unordered_map>
#include "tensor.h"


//===================================================================
//INTERNAL HELPER FUNCTIONS
//===================================================================
void writeWord(std::ofstream& ofs, uint32_t hex) {
    ofs.write((const char*)&hex, 4);
}

//===================================================================
//SUPPORTED ACTIVATION FUNCTIONS
//===================================================================
template <typename T>
using func = T(T);

template <typename T>
T sigmoid(T x) {
    return 1 / (1 + expf(-1 * (float)x));
}

template <typename T>
T dsigmoid(T x) {
    return  sigmoid(x) * (1 - sigmoid(x));
}

template <typename T>
T relu(T x) {
    return (x > 0)? x : 0;
}

template <typename T>
T drelu(T x) {
    return (x > 0)? 1: 0;
}

template <typename T>
T linear(T x) {
    return x;
}

template <typename T>
T dlinear(T x) {
    return (T) 1 + (x * (T)0);
}

enum class ActiVationType {
    Linear  = 0x0000,
    Sigmoid = 0x0001,
    ReLu    = 0x0002,
};

template <typename T>
func<T>* getActivation(ActiVationType type) {
    switch(type) {
        case ActiVationType::Linear     : return linear;
        case ActiVationType::Sigmoid    : return sigmoid;
        case ActiVationType::ReLu       : return relu;
        default: return nullptr;
    }
}

template <typename T>
func<T>* getActivationDerivative(ActiVationType type) {
    switch(type) {
        case ActiVationType::Linear     : return dlinear;
        case ActiVationType::Sigmoid    : return dsigmoid;
        case ActiVationType::ReLu       : return drelu;
        default: return nullptr;
    }
}

template <typename T>
std::array<size_t, 2> argMax(const Tensor<T>& arguments) {
    size_t nRows = arguments.dim()[0];
    size_t nCols = arguments.dim()[1];
    T max = arguments.at(0, 0);
    std::array<size_t, 2> maxIndex = {0,0};
    for(size_t r{0}; r < nRows; ++r) {
        for(size_t c{0}; c < nCols; ++c) {
            if (arguments.at(r, c) > max) {
                max = arguments.at(r, c);
                maxIndex = {r, c};
            }
        }
    }    
    return maxIndex;
}

template <typename T>
Tensor<T> softMax(const Tensor<T>& results) {
    size_t nRows = results.dim()[0];
    size_t nCols = results.dim()[1];

    T denom = (T) 0;
    T max = results.at(argMax(results));
    Tensor<T> normalized(results.dim());

    for(size_t r{0}; r < nRows; ++r) {
        for(size_t c{0}; c < nCols; ++c) {
            denom += exp(results.at(r, 0) - max);
        }
    }

    for(size_t r{0}; r < nRows; ++r) {
        for(size_t c{0}; c < nCols; ++c) {
            normalized.at(r, 0) = exp(results.at(r,0) - max) / denom;
        }
    }

    return normalized;
}

//===================================================================
//MAIN NEURAL NETWORK CLASS
//===================================================================
template <typename T>
class Model {
public:
    //=============================
    // CONSTRUCTORS
    //=============================
    Model(const std::vector<size_t>& desc, const std::vector<ActiVationType>& layerActivations){
        assert((desc.size() - 1) == layerActivations.size());

        size_t nLayerCount = desc.size();   //Layers + 1 (input)

        acts.reserve(nLayerCount);
        zs.reserve(nLayerCount - 1);

        wts.reserve(nLayerCount - 1);
        wtGrads.reserve(nLayerCount - 1);

        bs.reserve(nLayerCount - 1);
        bGrads.reserve(nLayerCount - 1);

        activations.reserve(nLayerCount - 1);

        nCountPropagated = 0;

        acts.push_back(Tensor<T>(desc.at(0), 1));
        //Iterate through Each Layer (get the nodes in that layer)
        for(size_t nLayer = 1; nLayer < nLayerCount; ++nLayer) {
            acts.push_back(Tensor<T>(desc.at(nLayer), 1));
            bs.push_back(Tensor<T>(desc.at(nLayer), 1).random(-1, 1));
            wts.push_back(Tensor<T>(desc.at(nLayer - 1),desc.at(nLayer)).random(0, 1));
            activations.push_back(layerActivations.at(nLayer - 1));
        }
    }    

    ~Model() {}

    //=============================
    // PUBLIC API
    //=============================
    void print(const char modelName[] = "Model") const {
        size_t nLayerCount = bs.size();
        char name[20];
    
        std::cout << modelName << " = [\n";
        for(size_t nLayer{0}; nLayer < nLayerCount; ++nLayer) {
            snprintf(name, sizeof(name), "Bias%lld", nLayer);
            bs.at(nLayer).print(name);
            
            snprintf(name, sizeof(name), "Weight%lld", nLayer);
            wts.at(nLayer).print(name);
        }
        std::cout << "]" << std::endl;
    }

    /*
        Word 1 = 0xDEADBEEF //(denotes my nn model savefile)
        Word 2 = number of layers (N)
        Word 3 = number of Bytes representing 1 value
        Word 4 = Configuration:
            Byte |  3   |   2   |   1   |   0   |
                    |       |       |       Cost Function 
                    |       |       Gradiant Type Used
                    |       Reserved (0)
                    Reserved (0)
            Cost Types:
                0 = MSE
            Gradient Types:
                0 = normal
                1 = stochastic
        Next N Words encode the activation functions per Layer:
            Byte |  3   |   2   |   1   |   0   |
                    |       |       |       Activation Type
                    \       |       /
                     Varies per type (TBD)
            Activation Types: (Determined by enumeration)
                0 = linear (none)
                1 = sigmoid
                2 = relu
        Rest is the Data:
            First N Tensors are the weights
            Last N  Tensors are the biases
                Each Tensor formatted like:
                    First Word = row
                    Second Word = col
                    Next row*col words (row*col*4 bytes) are the hex representations of floats
    */
    void saveModelParams(const char sFileName[]) const {
        std::ofstream ofs(sFileName);
        size_t nLayerCount = wts.size();
        constexpr size_t nNumBytesPerType = sizeof(T);

        //Intro
        writeWord(ofs, 0xDEADBEEF);
        writeWord(ofs, (uint32_t)nLayerCount);
        writeWord(ofs, (uint32_t)nNumBytesPerType);

        //Configuration
        {
            writeWord(ofs, 0);
        }

        //Activations
        for(size_t nLayer{0}; nLayer < nLayerCount; ++nLayer) {
            writeWord(ofs, (uint8_t) activations.at(nLayer));
        }

        //WeightTensors 
        for(size_t nLayer{0}; nLayer < nLayerCount; ++nLayer) {
            const Tensor<T>& t = wts.at(nLayer);
            size_t nRows = t.dim()[0];
            size_t nCols = t.dim()[1];
            writeWord(ofs, (uint32_t)nRows);
            writeWord(ofs, (uint32_t)nCols);
            for(size_t r{0}; r < nRows; ++r) {
                for(size_t c{0}; c < nCols; ++c) {
                    T tVal = t.at(r, c);
                    ofs.write((char*)&tVal, nNumBytesPerType);
                }
            }
        }
        //BiasTensors 
        for(size_t nLayer{0}; nLayer < nLayerCount; ++nLayer) {
            const Tensor<T>& t = bs.at(nLayer);
            size_t nRows = t.dim()[0];
            size_t nCols = t.dim()[1];
            writeWord(ofs, (uint32_t)nRows);
            writeWord(ofs, (uint32_t)nCols);
            for(size_t r{0}; r < nRows; ++r) {
                for(size_t c{0}; c < nCols; ++c) {
                    T tVal = t.at(r, c);
                    ofs.write((char*)&tVal, nNumBytesPerType);
                }
            }
        }
    }

    //Passes inputs through the model (Results are deposited at final activation layer)
    const Tensor<T>& forward(const Tensor<T>& inputs) {
        acts.at(0) = inputs;

        zs.clear();

        //iter through each layer and comput the outputs
        size_t nLayerCount = acts.size() - 1;
        for(size_t nLayer{0}; nLayer < nLayerCount; ++nLayer) {
            Tensor<T>& x = acts.at(nLayer);
            Tensor<T>& w = wts.at(nLayer);
            Tensor<T>& b = bs.at(nLayer);
            const Tensor<T>& y = w.transpose() * x + b;
            zs.push_back(y);
            acts.at(nLayer + 1) = y.apply(getActivation<T>(activations.at(nLayer)));
        }
        return getOutput();
    }

    //Forwards an input through the model and returns the loss Tensor
    Tensor<T> cost(const Tensor<T>& input, const Tensor<T>& expected) {
            Tensor<T> outp = forward(input);    //Outputs Stored at final Activation Layer
            Tensor<T> cost = outp - expected;
            return cost.squared() / ((T)2);
    }

    void train(const Tensor<T>& trainingData, const Tensor<T> labels, size_t epochs, size_t batchSize, float learnRate, bool seeCost = false) {
        (void) trainingData;
        (void) labels;
        (void) epochs;
        (void) batchSize;
        (void) learnRate;
        (void) seeCost;
        // size_t nTrainCount = trainingData.dim()[0];
        // size_t nOutputs = getOutput().dim()[0];

        // //create uniquely random batches

        // for(size_t i{0}; i < trainingData.dim()[1]; i++) {
            
        // }

        // for(size_t i{0}; i < epochs; i++) {
            
        // }


        // Tensor<T> totalLoss(nOutputs, 1);
        // for(size_t nIteration{0}; nIteration < epochs; ++nIteration) {
        //     totalLoss.fill((T)0);
        //     for(size_t nDataPoint{0}; nDataPoint < nTrainCount; ++nDataPoint) {
        //         const Tensor<T>& inp = trainingData.row(nDataPoint).transpose();
        //         const Tensor<T>& exp = labels.row(nDataPoint).transpose();
        //         //Returns the loss of the function (internally calls forward and deposits predictions at acts.end())
        //         totalLoss += this->cost(inp, exp);
        //         //Calculates the gradiants for this iteration and stores internally
        //         this->backward(exp, learnRate);
        //     }
        //     totalLoss = totalLoss / (T)nTrainCount;
        //     this->learn();
        //     if(seeCost) {
        //         T avgCost = totalLoss.average();
        //         std::cout << "epoch[" << nIteration << "]\t";
        //         std::cout << "cost: " << avgCost << '\n';
        //         std::cout << "\x1b[1A";
        //     }
        // }
    }

private:
    //=============================
    // Internal API
    //=============================
    Tensor<T> dCost(const Tensor<T>& expected) {
        return getOutput() - expected;   //Trust? Might not be correct
    }

    void backward(const Tensor<T>& expected, T learnRate) {
        //Number of layers (excluding input)
        size_t nLayerCount = acts.size() - 1;

        std::vector<Tensor<T>> deltas(nLayerCount);
        //calculate Delta for output Layer
        Tensor<T> d = dCost(expected);
        func<T>* derivative = getActivationDerivative<T>(activations.at(nLayerCount - 1));
        Tensor<T> z =  zs.at(nLayerCount - 1).apply(derivative);
        Tensor<T> layerDelta = hadamard(d, z);
        deltas.at(nLayerCount - 1) = layerDelta;

        //Starting from Last Hidden Layer, compute the Deltas for each layer
        for(long long nLayer = nLayerCount - 1 - 1; nLayer >= 0; --nLayer) {
            Tensor<T>& wNext = wts.at(nLayer + 1);
            Tensor<T>& deltaNext = deltas.at(nLayer + 1);

            derivative = getActivationDerivative<T>(activations.at(nLayer));
            Tensor<T> z =zs.at(nLayer).apply(derivative);

            layerDelta = hadamard((wNext * deltaNext), z);
            deltas.at(nLayer) = layerDelta;
        }
        //Calculate the Gradiants based on the Deltas for each layer
        for (size_t nLayer{0}; nLayer < nLayerCount; ++nLayer) {
            Tensor<T> wGrad = (acts.at(nLayer) * deltas.at(nLayer).transpose());
            Tensor<T> bGrad = deltas.at(nLayer);

            wGrad *= learnRate;
            bGrad *= learnRate;

            if(nCountPropagated == 0) wtGrads.push_back(wGrad);
            else wtGrads.at(nLayer) += wGrad;

            if(nCountPropagated == 0) bGrads.push_back(bGrad);
            else bGrads.at(nLayer) += bGrad;
        }
        nCountPropagated += 1;
    }

    void learn() {
        size_t nLayerCount = acts.size() - 1;
        for(size_t nLayer{0}; nLayer < nLayerCount; ++nLayer) {
            Tensor<T> bGrad = (bGrads.at(nLayer) / (T) nCountPropagated);
            Tensor<T> wtGrad = (wtGrads.at(nLayer) / (T) nCountPropagated);
            wts.at(nLayer) -= wtGrad;
            bs.at(nLayer) -= bGrad;
        }
        wtGrads.clear();
        bGrads.clear();
        nCountPropagated = 0;
    }

    //Returns RO reference to final activation layer (stores the result)
    const Tensor<T>& getOutput(void) const {
        return acts.back();
    }

    const Tensor<T>& getInput(void) const {
        return *acts.begin();
    }

    //=============================
    // Debug Functions
    //=============================
    void dumpGrads(const char modelName[] = "Model") {
        size_t nLayerCount = bs.size();
        char name[20];
    
        std::cout << modelName  << " Gradiants" << " = [\n";
        for(size_t nLayer{0}; nLayer < nLayerCount; ++nLayer) {
            snprintf(name, sizeof(name), "Bias%lld", nLayer);
            bGrads.at(nLayer).print(name);
            
            snprintf(name, sizeof(name), "Weight%lld", nLayer);
            wtGrads.at(nLayer).print(name);
        }
        std::cout << "]" << std::endl;
    }
    
    void dumpActivations(const char modelName[] = "Model") {
        size_t nLayerCount = bs.size();
        char name[20];
    
        std::cout << modelName  << " Activations" << " = [\n";
        for(size_t nLayer{0}; nLayer < nLayerCount + 1; ++nLayer) {
            snprintf(name, sizeof(name), "Act%lld", nLayer);
            acts.at(nLayer).print(name);
        }
        std::cout << "]" << std::endl;
    }

    void dumpZs(const char modelName[] = "Model") {
        size_t nLayerCount = bs.size();
        char name[20];
    
        std::cout << modelName  << " Gradiants" << " = [\n";
        for(size_t nLayer{0}; nLayer < nLayerCount; ++nLayer) {
            snprintf(name, sizeof(name), "Z%lld", nLayer);
            zs.at(nLayer).print(name);
        }
        std::cout << "]" << std::endl;
    }

private:
    std::vector<Tensor<T>> acts;        //Output of each Layer (activated)
    std::vector<Tensor<T>> zs;          //Temporary Storage of Pre-activated Outputs
    std::vector<Tensor<T>> bs;          //Biases of each Layer
    std::vector<Tensor<T>> bGrads;      //Temporary Storage of Gradiants for the Biases
    std::vector<Tensor<T>> wts;         //Incoming Weights of eachLayer
    std::vector<Tensor<T>> wtGrads;     //Temporary Storage of Gradiants for the Weights
    std::vector<ActiVationType> activations;
    size_t nCountPropagated = 0;
};

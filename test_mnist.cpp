#include <ctime>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <chrono>
#include <ctime>

#include "nn.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "thirdparty/stb_image_write.h"

#define READ_WORD(num, buf, idx)\
    num &= ~0xFFFFFFFF;\
    num |= (buf[idx++]<<24 & 0xFF000000);\
    num |= (buf[idx++]<<16 & 0x00FF0000);\
    num |= (buf[idx++]<<8  & 0x0000FF00);\
    num |= (buf[idx++]<<0  & 0x000000FF);

//Assumed float type since mnist
void loadIdx3ToVec(std::vector<std::vector<uint8_t>>& vecImages, const char filePath[]) {
    std::string sFileName(filePath);

    //Open fileStream
    std::ifstream ifs (filePath, std::ifstream::binary);
    char *buffer;
    if(!ifs) exit(-1);

    //Read file into char buffer
    ifs.seekg (0, ifs.end); 
    size_t nLength = ifs.tellg(); 
    ifs.seekg (0, ifs.beg); 
    buffer = new char[nLength];
    ifs.read(buffer, nLength);
    if(!ifs)  {
        ifs.close();
        exit(-1);
    }

    //Verify magic number
    uint32_t nMagicNumber = 0;
    size_t nIndex = 0;
    READ_WORD(nMagicNumber, buffer, nIndex);

    //Populate the dimension sizes
    constexpr size_t nNumDims = 3;
    uint32_t nDimSize[nNumDims] = {};
    for(size_t i = 0; i < nNumDims; ++i){
        READ_WORD(nDimSize[i], buffer, nIndex);
    }

    size_t nNumImages = nDimSize[0];
    size_t nNumValsInImage = nDimSize[1] * nDimSize[2];

    vecImages.clear();
    for(size_t image = 0; image < nNumImages; ++image) {
        std::vector<uint8_t> tempVec;
        for(size_t byte = 0; byte < nNumValsInImage; ++byte) {
            tempVec.push_back(buffer[nIndex++]);
        }
        vecImages.push_back(tempVec);
    }    
}

void loadIdx1ToVec(std::vector<uint8_t>& vecLabels, const char filePath[]) {
    std::string sFileName(filePath);

    //Open fileStream
    std::ifstream ifs (filePath, std::ifstream::binary);
    char *buffer;
    if(!ifs) exit(-1);

    //Read file into char buffer
    ifs.seekg (0, ifs.end); 
    size_t nLength = ifs.tellg(); 
    ifs.seekg (0, ifs.beg); 
    buffer = new char[nLength];
    ifs.read(buffer, nLength);
    if(!ifs)  {
        ifs.close();
        exit(-1);
    }

    //Verify magic number
    uint32_t nMagicNumber = 0;
    size_t nIndex = 0;
    READ_WORD(nMagicNumber, buffer, nIndex);

    //Populate the dimension sizes
    uint32_t nNumLabels;
    READ_WORD(nNumLabels, buffer, nIndex);

    vecLabels.clear();
    for(size_t image = 0; image < nNumLabels; ++image) {
        vecLabels.push_back(buffer[nIndex++]);
    }    
}

/*
    Converts the vectors to tensors
*/
void makeTensorSet(std::vector<std::vector<uint8_t>>& srcImages, std::vector<uint8_t>& srcLabels, Tensor<float>& destImages, Tensor<float>& destLabels) {
    constexpr size_t nImageSize = 784;
    constexpr size_t nLabelSize = 10;
    assert(srcImages.size() == srcLabels.size());
    const size_t nSetCount = srcImages.size();

    assert(destImages.dim()[0] == destLabels.dim()[0]);
    assert(destLabels.dim()[0] == nSetCount);
    assert(destImages.dim()[1] = nImageSize);
    assert(destLabels.dim()[1] = nLabelSize);

    for(size_t i{0}; i < nSetCount; i++ ) {
        for(size_t j{0}; j < srcImages.at(i).size(); j++) {
            destImages.at(i*nImageSize + j) = srcImages.at(i).at(j);
        }

    destLabels.clear(); //sets all to 0
    destLabels.at(i*10 + srcLabels.at(i)) = 1.0f;
    }

    destImages.transpose();
    destLabels.transpose();
}

int main(void) {

    srand(time(0));

    constexpr float fLearnRate = 1e-1;
    constexpr size_t nEpochs = 1000;
    constexpr size_t nBatchSize = 100;
    //Construct the Model Description
    /*
        MNIST Model:
            784 inputs
            Hidden Layer 1: 16 nodes 
                Activation sigmoid
            Hidden Layer 2: 16 nodes
                Activation sigmoid
            10 outputs
    */

    //Create Model from the Layer sizes and activations
    // Model<float> m(layerDesc, layerActivations);
    Model<float> m(
        std::vector<size_t>({784, 16, 16, 10}), 
        std::vector<ActiVationType>({ ActiVationType::Sigmoid, ActiVationType::Sigmoid, ActiVationType::Sigmoid}));

    
    std::vector<std::vector<uint8_t>> vecTrainImages;
    std::vector<uint8_t> vecTrainLabels;
    std::vector<std::vector<uint8_t>> vecTestImages;
    std::vector<uint8_t> vecTestLabels;
    size_t nTrainCount, nTestCount;
    std::cout << "Loading Images...\n";
    {
        auto start = std::chrono::system_clock::now();
        loadIdx3ToVec(vecTrainImages, "data/train-images.idx3-ubyte");
        loadIdx1ToVec(vecTrainLabels, "data/train-labels.idx1-ubyte");
        assert(vecTrainImages.size() == vecTrainLabels.size());
        nTrainCount = vecTrainImages.size();
        loadIdx3ToVec(vecTestImages, "data/t10k-images.idx3-ubyte");
        loadIdx1ToVec(vecTestLabels, "data/t10k-labels.idx1-ubyte");
        assert(vecTestImages.size() == vecTestLabels.size());
        nTestCount = vecTestImages.size();
        auto final = std::chrono::system_clock::now();
        std::cout << "Done! (" << (final - start).count() << ")" << std::endl;
    }

    Tensor<float> trainData(nTrainCount, 28*28);
    Tensor<float> trainLabels(nTrainCount, 10);
    Tensor<float> testData(nTestCount, 28*28);
    Tensor<float> testLabels(nTestCount, 10);
    std::cout << "Creating Tensor Sets...\n";
    {
        auto start = std::chrono::system_clock::now();
        makeTensorSet(vecTrainImages, vecTrainLabels, trainData, trainLabels);
        makeTensorSet(vecTestImages, vecTestLabels, testData, testLabels);
        auto final = std::chrono::system_clock::now();
        std::cout << "Done! (" << (final - start).count() << ")" << std::endl;
    }
    stbi_write_png("test.png",28, 28, 1, trainData.col(0,1).data(), 28);
    return 0;


    std::cout << "Training Model...\n";
    {
        auto start = std::chrono::system_clock::now();
        m.train(trainData, trainLabels, nEpochs, nBatchSize, fLearnRate, true);
        auto final = std::chrono::system_clock::now();
        std::cout << "Done! (" << (final - start).count() << ")" << std::endl;
    }


    size_t correct = 0;
    for(size_t nTestImage{0}; nTestImage < nTestCount; ++nTestImage) {
        const Tensor<float>& inp = testData.row(nTestImage).transpose();
        const Tensor<float>& exp = testLabels.row(nTestImage).transpose();

        Tensor<float> actual = m.forward(inp);

        actual = softMax(actual);
        std::array<size_t, 2> expectedIndex = argMax(exp);
        std::array<size_t, 2> predictionIndex = argMax(actual);
        
        std::cout << "Predicted Label[" << nTestImage << "]: " << predictionIndex[0] << '\n';
        std::cout << "Expected Label[" << nTestImage << "]: " << expectedIndex[0] << '\n';
        std::cout << '\n' << std::endl;
        correct += predictionIndex[0] == expectedIndex[0];
    }
    float accuracy = (float)correct / (float)nTestCount;
    std::cout << "Accuracy = " << std::setw(10) << std::fixed << std::setprecision(3) << accuracy << std::endl;

    // m.saveModelParams("TrainedModel.nn");

    return 0;
}
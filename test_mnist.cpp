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

int main(void) {
    return 0;
}
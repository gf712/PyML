//
// Created by gil on 21/11/17.
//

#ifndef PYML_DEV_EXCEPTIONCLASSES_H
#define PYML_DEV_EXCEPTIONCLASSES_H

#include <string>
#include <exception>
#include "flatArrays.h"

#ifdef __cplusplus
extern "C" {
#endif

class flatArrayException: public std::exception {
public:
    const char* what() const throw() override {
        return "flatArray Exception";
    }
};


class flatArrayDimensionMismatch: public flatArrayException {
    std::string errorMsg;
public:
    flatArrayDimensionMismatch(flatArray *A, flatArray* B) {
        std::string thisColumn = std::to_string(A->getCols());
        std::string thisRow = std::to_string(A->getRows());
        std::string otherColumn = std::to_string(B->getCols());
        std::string otherRow = std::to_string(B->getRows());

        std::string msg = "Shape mismatch! Got an array of shape {" + thisRow + ", " + thisColumn + "} and {" + otherRow + ", " + otherColumn + "}!";

        flatArrayDimensionMismatch::errorMsg = msg.c_str();
    };

    const char* what() const throw() override {
        return errorMsg.c_str();
    }
};


class flatArrayColumnMismatch: public flatArrayException {
    std::string errorMsg;
public:
    flatArrayColumnMismatch(flatArray *A, flatArray* B) {
        std::string thisColumn = std::to_string(A->getCols());
        std::string otherColumn = std::to_string(B->getCols());

        std::string msg = "Column number mismatch! Got an array with " + thisColumn + " columns " + " and an array with " + otherColumn + " columns!";

        flatArrayColumnMismatch::errorMsg = msg.c_str();
    };

    const char* what() const throw() override {
        return errorMsg.c_str();
    }
};


class flatArrayRowMismatch: public flatArrayException {
    std::string errorMsg;
public:
    flatArrayRowMismatch(flatArray *A, flatArray* B) {
        std::string thisRow = std::to_string(A->getRows());
        std::string otherRow = std::to_string(B->getRows());

        std::string msg = "Rows number mismatch! Got an array with " + thisRow + " rows " + " and an array with " + otherRow + " rows!";

        flatArrayRowMismatch::errorMsg = msg.c_str();
    };

    const char* what() const throw() override {
        return errorMsg.c_str();
    }
};


#ifdef __cplusplus
}
#endif


#endif //PYML_DEV_EXCEPTIONCLASSES_H

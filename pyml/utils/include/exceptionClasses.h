//
// Created by gil on 21/11/17.
//

#ifndef PYML_EXCEPTIONCLASSES_H
#define PYML_EXCEPTIONCLASSES_H

#include <string>
#include <exception>
#include "flatArrays.h"


class flatArrayException: public std::exception {
public:
    const char* what() const throw() override {
        return "flatArray Exception";
    }
};


template <class T>
class flatArrayDimensionMismatchException: public flatArrayException {
    std::string errorMsg;
public:
    flatArrayDimensionMismatchException(const flatArray<T> &A, const flatArray<T> &B) {
        std::string thisColumn = std::to_string(A.getCols());
        std::string thisRow = std::to_string(A.getRows());
        std::string otherColumn = std::to_string(B.getCols());
        std::string otherRow = std::to_string(B.getRows());

        std::string msg = "Shape mismatch! Got an array of shape {" + thisRow + ", " + thisColumn + "} and {" + otherRow + ", " + otherColumn + "}!";

        flatArrayDimensionMismatchException::errorMsg = msg.c_str();
    };

    const char* what() const throw() override {
        return errorMsg.c_str();
    }
};


template <class T>
class flatArrayColumnMismatchException: public flatArrayException {
    std::string errorMsg;
public:
    flatArrayColumnMismatchException(const flatArray<T> &A, const flatArray<T> &B) {
        std::string thisColumn = std::to_string(A.getCols());
        std::string otherColumn = std::to_string(B.getCols());

        std::string msg = "Column number mismatch! Got an array with " + thisColumn + " columns and an array with " + otherColumn + " columns!";

        flatArrayColumnMismatchException::errorMsg = msg.c_str();
    };

    const char* what() const throw() override {
        return errorMsg.c_str();
    }
};


template <class T>
class flatArrayRowMismatchException: public flatArrayException {
    std::string errorMsg;
public:
    flatArrayRowMismatchException(const flatArray<T> &A, const flatArray<T> &B) {
        std::string thisRow = std::to_string(A.getRows());
        std::string otherRow = std::to_string(B.getRows());

        std::string msg = "Row number mismatch! Got an array with " + thisRow + " rows and an array with " + otherRow + " rows!";

        flatArrayRowMismatchException::errorMsg = msg.c_str();
    };

    const char* what() const throw() override {
        return errorMsg.c_str();
    }
};


template <class T>
class flatArrayOutOfBoundsException: public flatArrayException {
    std::string errorMsg;
public:
    flatArrayOutOfBoundsException(const flatArray<T> &A, int n) {

        std::string nString = std::to_string(n);
        std::string sizeString = std::to_string(A.getSize());

        std::string msg = "Accessing an element that is out of bounds! Attempting to access element at position "
                          + nString + " in an array of size " + sizeString + " !";

        flatArrayOutOfBoundsException::errorMsg = msg.c_str();
    };

    const char* what() const throw() override {
        return errorMsg.c_str();
    }
};


template <class T>
class flatArrayOutOfBoundsRowException: public flatArrayException {
    std::string errorMsg;
public:
    flatArrayOutOfBoundsRowException(const flatArray<T> &A, int n) {

        std::string nString = std::to_string(n);
        std::string sizeString = std::to_string(A.getRows());

        std::string msg = "Accessing an element that is out of bounds! Attempting to access row "
                          + nString + " in an array with " + sizeString + " rows!";

        flatArrayOutOfBoundsRowException::errorMsg = msg.c_str();
    };

    const char* what() const throw() override {
        return errorMsg.c_str();
    }
};


template <class T>
class flatArrayOutOfBoundsColumnException: public flatArrayException {
    std::string errorMsg;
public:
    flatArrayOutOfBoundsColumnException(const flatArray<T> &A, int n) {

        std::string nString = std::to_string(n);
        std::string sizeString = std::to_string(A.getCols());

        std::string msg = "Accessing an element that is out of bounds! Attempting to access column "
                          + nString + " in an array with " + sizeString + " columns!";

        flatArrayOutOfBoundsColumnException::errorMsg = msg.c_str();
    };

    const char* what() const throw() override {
        return errorMsg.c_str();
    }
};


class flatArrayZeroDivisionError: public flatArrayException {
    std::string errorMsg;
public:
    flatArrayZeroDivisionError() {

        std::string msg = "Trying to divide array by zero!";

        flatArrayZeroDivisionError::errorMsg = msg.c_str();
    };

    const char* what() const throw() override {
        return errorMsg.c_str();
    }
};


class flatArrayUnknownAxis: public flatArrayException {
    std::string errorMsg;
public:
    flatArrayUnknownAxis(int axis) {

        std::string stringAxis = std::to_string(axis);

        std::string msg = "Got axis value of " + stringAxis + ", but expected value of 0 or 1!";

        flatArrayUnknownAxis::errorMsg = msg.c_str();
    };

    const char* what() const throw() override {
        return errorMsg.c_str();
    }
};


class arrayException: public std::exception {
public:
    const char* what() const throw() override {
        return "Array Exception";
    }
};


class arrayOutOfBoundsException: public arrayException {
    std::string errorMsg;
public:
    arrayOutOfBoundsException(int size, int n) {

        std::string nString = std::to_string(n);
        std::string sizeString = std::to_string(size);

        std::string msg = "Accessing an element that is out of bounds! Attempting to access element at position "
                          + nString + " in an array of size " + sizeString + " !";

        arrayOutOfBoundsException::errorMsg = msg.c_str();
    };

    const char* what() const throw() override {
        return errorMsg.c_str();
    }
};


class linearAlgebraException: public std::exception {
public:
    const char* what() const throw() override {
        return "Linear Algebra Exception";
    }
};

class singularMatrixException: public linearAlgebraException {
    std::string errorMsg;
public:
    singularMatrixException() {


        std::string msg = "Singular matrix!";

        singularMatrixException::errorMsg = msg.c_str();
    };

    const char* what() const throw() override {
        return errorMsg.c_str();
    }
};

#endif //PYML_EXCEPTIONCLASSES_H

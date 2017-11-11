//
// Created by gil on 10/11/17.
//

#ifndef PYML_DEV_FLATARRAYS_H
#define PYML_DEV_FLATARRAYS_H

#ifdef __cplusplus
extern "C" {
#endif

class flat2DArrays {
private:
    int rows;
    int cols;
    double* array;

public:
    flat2DArrays();
    ~flat2DArrays();
    void readFromPythonList(PyObject *pyList);
    void startEmptyArray(int rows_, int cols_);
    int getRows();
    void setRows(int rows);
    int getCols();
    void setCols(int cols);
    double* getArray();
    double getElement(int row, int col);
    double setElement(double value, int row, int col);

};

#ifdef __cplusplus
}
#endif

#endif //PYML_DEV_FLATARRAYS_H

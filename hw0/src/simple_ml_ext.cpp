#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

void exp_matrix(const float *X, float* res, size_t m, size_t n){

    for(int i = 0;i<=m;i++){
        for(int j = 0;j<n;j++){
            res[i*n + j] = std::exp(X[i*n + j]);
        }
    }
}

void vector_outer(const float *Left, const float *Right, float *res, size_t m, size_t n){
    for(int i = 0;i<m;i++){
        for(int j = 0;j<n;j++){
            res[i*n + j] = Left[i] * Right[j];
        }
    }
}

void matrix_multi(const float *Left, const float *Right, float *res, size_t m, size_t n, size_t k){

    for(int i = 0;i<m;i++){
        for(int j = 0;j<k;j++){
            float sum = 0.0;
            for(int t = 0;t<n;t++){
                sum += Left[i*n + t] * Right[t*k + j];
            }
            res[i*k + j] = sum;
        }
    }
}

matrix_sum(const float *gradient, float* gradient_sum, int n, int k){
    for(int i = 0;i<n;i++){
        for(int j = 0;j<k;j++){
            gradient_sum[i*k + j] +=  gradient[i*k + j];
        }
    }
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    int num_batches = m / batch;
    float *Z  = new float[100*20]();
    float *exp_Z = new float[100*20]();
    float *deri_Z = new float[20]();
    float *gradient = new float[1000*20]();
    float *gradient_sum = new float[1000*20]();
    for(int i = 0;i<num_batches;i++){

        matrix_multi(X + i*batch*n, theta, Z, batch, n, k);
        exp_matrix(Z, exp_Z, batch, k);
        std::fill(gradient_sum, gradient_sum + n*k, 0);
        for(int j = 0;j<batch;j++){
            float sum_Z = 0.0;
            for(int p = 0;p<k;p++){
                sum_Z += exp_Z[j*k + p];
            }
            for(int p = 0;p<k;p++){
                deri_Z[p] = exp_Z[j*k + p]/sum_Z;
            }
            deri_Z[y[i*batch + j]] -= 1;
            vector_outer(X+(i*batch+j)*n, deri_Z, gradient, n, k);
            matrix_sum(gradient, gradient_sum, n, k);
        }

        for(int i = 0;i<n;i++){
            for(int j = 0;j<k;j++){
                theta[i*k + j] -= lr * gradient_sum[i*k + j] / batch;
            }
        }

    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}

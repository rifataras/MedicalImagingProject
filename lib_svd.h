#ifndef LIB_SVD_H_INCLUDED
#define LIB_SVD_H_INCLUDED

#include <vector>

typedef std::vector<std::vector<float> > mat_t;
typedef std::vector<float> vec_t;
typedef std::vector<float>::iterator iter_t;

//! Compute the largest singular value by the power method
//! WARNING : tX is transposed!!!
double svd_trunc(mat_t &tX,
                 vec_t &U,
                 vec_t &V);

#endif // LIB_SVD_H_INCLUDED

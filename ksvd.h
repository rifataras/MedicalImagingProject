#ifndef KSVD_H
#define KSVD_H

#include <iostream>

#if defined(_WIN32) || defined(_WIN64)

#include <time.h>

#endif
#include "lib_ormp.h"
#include "lib_svd.h"

/***********KSVD***************************************/
typedef std::vector<std::vector<float> > matD_t;
typedef std::vector<std::vector<unsigned> > matU_t;
typedef std::vector<float> vecD_t;
typedef std::vector<unsigned> vecU_t;
typedef std::vector<float>::iterator iterD_t;
typedef std::vector<unsigned>::iterator iterU_t;

void randperm(vecU_t &);

void obtain_dict(matD_t&, matD_t const&);

void ksvd_process(matD_t        &,
                  matD_t        &,
                  matD_t		&,
                  const unsigned, // size of features (i.e. 324)
                  const unsigned, // size of the dictionary (i.e. 1000)
                  const unsigned, // i.e. 40
                  const double);

#endif
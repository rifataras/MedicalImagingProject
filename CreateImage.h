#ifndef CREATEIMAGE_H
#define CREATEIMAGE_H

#include "itkImage.h"

typedef float					KernelElementType;
typedef itk::Image<KernelElementType, 2>	KernelImageType;

void CreateImage(KernelImageType::Pointer, int, int);

#endif
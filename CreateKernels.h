#ifndef CREATEKERNELS_H
#define CREATEKERNELS_H

#include "itkImage.h"

typedef float					KernelElementType;
typedef itk::Image<KernelElementType, 2>	KernelImageType;

void CreateKernels(KernelImageType::Pointer kernel1,KernelImageType::Pointer kernel2,
		KernelImageType::Pointer kernel3,KernelImageType::Pointer kernel4,int scale);

#endif

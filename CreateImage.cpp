#include "CreateImage.h"



void CreateImage(KernelImageType::Pointer image, int width, int height)
{
	// Create an image with 2 connected components
	KernelImageType::IndexType start;
	start[0] = 0;
	start[1] = 0;

	KernelImageType::SizeType size;
	size[0] = width;
	size[1] = height;

	KernelImageType::RegionType region(start, size);

	image->SetRegions(region);
	image->Allocate();
	image->FillBuffer( itk::NumericTraits<KernelImageType::PixelType>::Zero);
}



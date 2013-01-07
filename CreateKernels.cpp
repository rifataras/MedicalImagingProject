#include "CreateKernels.h"
#include "itkImage.h"
#include "itkImageRegionIterator.h"
#include "itkNeighborhoodIterator.h"
#include "itkConstNeighborhoodIterator.h"

void CreateKernels(KernelImageType::Pointer kernel1,KernelImageType::Pointer kernel2,
		KernelImageType::Pointer kernel3,KernelImageType::Pointer kernel4,int scale)
{
	KernelImageType::SizeType size;
	size[0] = 2 + scale - 1;
	size[1] = 1;
	KernelImageType::IndexType index;
	index[0] = 0;
	index[1] = 0;
	KernelImageType::RegionType region1(index,size);
	kernel1->SetRegions(region1);
	kernel1->Allocate();

	std::vector<KernelElementType> krnlVals;
	krnlVals.reserve(size[0]);

	// Fill the values for the first and second kernels
	krnlVals.push_back(-1);
	for(int i = 0; i < scale - 1; i++)
		krnlVals.push_back(0);
	krnlVals.push_back(1);

	itk::ImageRegionIterator<KernelImageType> imageIterator(kernel1, region1);
	imageIterator.GoToBegin();
	while(!imageIterator.IsAtEnd())
	{
		KernelImageType::IndexType currntIndex = imageIterator.GetIndex();
		imageIterator.Set(krnlVals[currntIndex[0]]);
		++imageIterator;
	}

	// second kernel
	size[1] = 2 + scale - 1;
	size[0] = 1;
	index[0] = 0;
	index[1] = 0;
	KernelImageType::RegionType region2(index,size);
	kernel2->SetRegions(region2);
	kernel2->Allocate();

	itk::ImageRegionIterator<KernelImageType> imageIterator2(kernel2, region2);
	imageIterator2.GoToBegin();
	while(!imageIterator2.IsAtEnd())
	{
		KernelImageType::IndexType currntIndex = imageIterator2.GetIndex();
		imageIterator2.Set(krnlVals[currntIndex[1]]);
		++imageIterator2;
	}

	// now fill the values for the third and fourth kernels
	krnlVals.clear();
	krnlVals.push_back(0.5);
	for(int i = 0; i < scale - 1; i++)
		krnlVals.push_back(0);
	krnlVals.push_back(-1);
	for(int i = 0; i < scale - 1; i++)
		krnlVals.push_back(0);
	krnlVals.push_back(0.5);

	size[0] = 2 * scale + 1;
	size[1] = 1;
	index[0] = 0;
	index[1] = 0;
	KernelImageType::RegionType region3(index,size);
	kernel3->SetRegions(region3);
	kernel3->Allocate();

	itk::ImageRegionIterator<KernelImageType> imageIterator3(kernel3, region3);
	imageIterator3.GoToBegin();
	while(!imageIterator3.IsAtEnd())
	{
		KernelImageType::IndexType currntIndex = imageIterator3.GetIndex();
		imageIterator3.Set(krnlVals[currntIndex[0]]);
		++imageIterator3;
	}

	// fourth kernel
	size[1] = 2 * scale + 1;
	size[0] = 1;
	index[0] = 0;
	index[1] = 0;
	KernelImageType::RegionType region4(index,size);
	kernel4->SetRegions(region4);
	kernel4->Allocate();

	itk::ImageRegionIterator<KernelImageType> imageIterator4(kernel4, region4);
	imageIterator4.GoToBegin();
	while(!imageIterator4.IsAtEnd())
	{
		KernelImageType::IndexType currntIndex = imageIterator4.GetIndex();
		imageIterator4.Set(krnlVals[currntIndex[1]]);
		++imageIterator4;
	}
}

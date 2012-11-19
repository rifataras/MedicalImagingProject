#include "ImageToFeatureConverter.h"

template< typename TImageType >
ImageToFeatureConverter<TImageType>::ImageToFeatureConverter(int scale, int border, int overlap, int window)
{
	std::cout << "ImageToFeatureConverter constructor called" << std::endl;
	this->scale = scale;
	this->border = border;
	this->overlap = overlap;
	this->window = window;
}

template< typename TImageType >
ImageToFeatureConverter<TImageType>::~ImageToFeatureConverter()
{
	std::cout << "ImageToFeatureConverter destructor called" << std::endl;
}

template< typename TImageType >
void ImageToFeatureConverter<TImageType>::GetOutput(const ImageType * image, std::vector< std::vector< PixelType > > &featureVector)
{
	// A radius of 1 in all axial directions gives a 3x3x3x3x... neighborhood.
	typename NeighborhoodIterator::RadiusType radius;
	radius.Fill(scale * window / 2); // for a 9 by 9 kernel, set this parameter to 4

	typename ImageType::SizeType origSize = image->GetLargestPossibleRegion().GetSize();
	// define the region for the iterator
	// get the original image size and remove the border size
	typename ImageType::SizeType intrstSize = image->GetLargestPossibleRegion().GetSize();
	intrstSize[0] = intrstSize[0] - 2 * scale * border;
	intrstSize[1] = intrstSize[1] - 2 * scale * border;
	// Set the starting index of the region according to the
	// scale and border parameters
	typename ImageType::IndexType intrstIndex;
	intrstIndex.Fill(scale * border + radius[0]);

	typename ImageType::RegionType intrstRegion(intrstIndex,intrstSize);
	NeighborhoodIterator kernelIt(radius, image, intrstRegion);
	kernelIt.SetNeedToUseBoundaryCondition(true);
	// Now set the offset so that we jump the overlap amount
	typename NeighborhoodIterator::OffsetType offset;
	offset[0] = scale * (window - overlap);
	offset[1] = 0;
	kernelIt.GoToBegin();

	while ( ! kernelIt.IsAtEnd() )
	{
		std::vector< PixelType > windowPixels;

		for(int col = -1 * radius[0]; col <= (int)radius[0]; col++)
		{
			for(int row = -1 * radius[1]; row <= (int)radius[1]; row++)
			{
				typename NeighborhoodIterator::OffsetType nOffset;
				nOffset[0] = col;
				nOffset[1] = row;

				PixelType pp = kernelIt.GetPixel(nOffset);
				windowPixels.push_back(pp);
			}
		}

		featureVector.push_back(windowPixels);

		// We needed to implement the custom advancement scheme
		// to take care of the boundaries as we want
		itk::Index<2> curInd = kernelIt.GetIndex();

		/*std::cout << "[" << curInd[1] << "," << curInd[0] << "]: ";
		std::cout << "[rad: " << radius[0] << ", off: " << offset[0] << "], ";*/
		if(curInd[0] + radius[0] + offset[0] >= intrstSize[0] + scale * border)
		{
			if(curInd[1] + radius[0] + offset[0] >= intrstSize[1] + scale * border)
			{
				kernelIt.GoToEnd();
			}
			else
			{
				typename NeighborhoodIterator::IndexType newPos;
				newPos[0] = intrstIndex[0];
				newPos[1] = curInd[1] + offset[0];
				kernelIt.SetLocation(newPos);
			}
		}
		else
		{
			kernelIt += offset;
		}
	}
}

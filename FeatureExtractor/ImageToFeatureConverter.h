#ifndef IMAGETOFEATURECONVERTER_H
#define IMAGETOFEATURECONVERTER_H

#include <iostream>
#include "itkImage.h"
#include "itkImageRegionIterator.h"
#include "itkNeighborhoodIterator.h"
#include "itkConstNeighborhoodIterator.h"

template< typename TImageType >
class ImageToFeatureConverter
{
public:
	typedef TImageType ImageType;
	typedef typename ImageType::PixelType PixelType;
	typedef itk::ImageRegionIterator<ImageType> ImageIteratorType;
	typedef itk::NeighborhoodIterator<ImageType> NeighborhoodIteratorType;
	typedef itk::ConstNeighborhoodIterator<ImageType> ConstNeighborhoodIteratorType;

	ImageToFeatureConverter(int, int, int, int);
	void GetOutput(const ImageType * image, std::vector< std::vector< PixelType > > &featureVector);
	void GetImageBack(ImageType * image, ImageType *denominator, std::vector< std::vector< PixelType > > &featureVector);
	~ImageToFeatureConverter();
private:
	int scale;
	int window;
	int border;
	int overlap;
};

#endif

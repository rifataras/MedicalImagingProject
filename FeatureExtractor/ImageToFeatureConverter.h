#ifndef IMAGETOFEATURECONVERTER_H
#define IMAGETOFEATURECONVERTER_H

#include <iostream>
#include "itkImage.h"
#include "itkConstNeighborhoodIterator.h"

template< typename TImageType >
class ImageToFeatureConverter
{
public:
	typedef TImageType ImageType;
	typedef typename ImageType::PixelType PixelType;
	typedef itk::ConstNeighborhoodIterator<ImageType> NeighborhoodIterator;

	ImageToFeatureConverter(int, int, int, int);
	void GetOutput(const ImageType * image, std::vector< std::vector< PixelType > > &featureVector);
	~ImageToFeatureConverter();
private:
	int scale;
	int window;
	int border;
	int overlap;
};

#endif

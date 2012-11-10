#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include <itkExtractImageFilter.h>
#include <itkDirectory.h>

//headers for bicubic interpolation
#include "itkIdentityTransform.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkResampleImageFilter.h"

#include "itkConvolutionImageFilter.h"

#include "itkChangeInformationImageFilter.h"

// project specific preprocessor definitions
#define FUNCTEST // define this variable to test the functionality correctness


// typedefs
typedef unsigned char 				PixelType;
typedef itk::Image<PixelType, 2>	ImageType;
typedef itk::ExtractImageFilter< ImageType, ImageType > ExtractImageFilterType;
typedef itk::ImageFileReader< ImageType > ReaderType;
typedef itk::ImageFileWriter< ImageType > WriterType;


//typedefs for bicubic interpolation
typedef itk::IdentityTransform<double, 2>  IdentityTransformType;
typedef itk::BSplineInterpolateImageFunction<ImageType, double, double> InterpolatorType;
typedef itk::ResampleImageFilter<ImageType, ImageType>   ResampleFilterType;

// typedefs for feature extraction
typedef float						KernelElementType;
typedef itk::Image<KernelElementType, 2>	KernelImageType;
typedef itk::ConvolutionImageFilter<ImageType, KernelImageType, ImageType> ConvolutionFilterType;


void CreateKernels(KernelImageType::Pointer kernel1,KernelImageType::Pointer kernel2,
		KernelImageType::Pointer kernel3,KernelImageType::Pointer kernel4,int scale)
{
	KernelImageType::IndexType start;
	start.Fill(0);

	//const KernelImageType::SizeType &size1 = {2 + scale - 1, 1}; // gaussian x-dir

	KernelImageType::RegionType region1;
	region1.SetSize(2 + scale - 1, 1);
	region1.SetIndex(0, 0);
	region1.SetIndex(1, 0);

	kernel1->SetRegions(region1);
	kernel1->Allocate();

	/**/
	KernelImageType::IndexType pixelIndex;
	pixelIndex[0] = 0;
	pixelIndex[1] = 0;
	kernel1->SetPixel(pixelIndex, 3);
	std::cout << "bb " << kernel1->GetPixel(pixelIndex);
	for(unsigned int c = 0; c < scale - 1; c++)
	{
		pixelIndex[1] = c+1;

		kernel1->SetPixel(pixelIndex, 5);
		std::cout << "bv " << kernel1->GetPixel(pixelIndex);
	}
	pixelIndex[1] = scale;
	kernel1->SetPixel(pixelIndex, 9);
	std::cout << "bn " << kernel1->GetPixel(pixelIndex);

	/*itk::ImageRegionIterator<KernelImageType> imageIterator(kernel1, region1);

	   while(!imageIterator.IsAtEnd())
	    {
	    //imageIterator.Set(255);
	    imageIterator.Set(100);

	    ++imageIterator;
	    }*/

//  itk::ImageRegionIterator<ImageType> imageIterator(kernel, region);
//
//   while(!imageIterator.IsAtEnd())
//    {
//    //imageIterator.Set(255);
//    imageIterator.Set(1);

    //++imageIterator;
    }



int main( int argc, char *argv[] )
{
	// The parameters we are looking for are: 1) the directory of the training files,
	// 2) magnification scale to train for.
	if ( argc < 3 )
	{
		std::cerr << "Missing parameters. " << std::endl;
		std::cerr << "Usage: " << std::endl;
		std::cerr << argv[0]
		<< " directory scale"
		<< std::endl;
		return -1;
	}

	itksys::Directory trainDir;
	if(!trainDir.Load(argv[1]))
	{
		std::cerr << "Could not open the directory. " << std::endl;
		return -1;
	}

	int scale = ::atoi(argv[2]);

	int numberOfFiles = trainDir.GetNumberOfFiles();
	ReaderType::Pointer reader = ReaderType::New();

	// The big for loop in which the training images are processed
	for(int i = 0; i < numberOfFiles; i++)
	{
		const char *filename = trainDir.GetFile(i);

		// skip any directories
		if (itksys::SystemTools::FileIsDirectory(filename))
		{
			continue;
		}

		size_t arglen = strlen(argv[1]);
		char *fullpath = (char*)malloc(arglen + strlen(filename) + 2);
		if (fullpath == NULL)
		{
			std::cerr << "Could not allocate memory for the fullpath. " << std::endl;
			return -1;
		}

		sprintf(fullpath, "%s/%s", argv[1], filename);

		reader->SetFileName( fullpath );
		try
		{
			reader->UpdateLargestPossibleRegion();
		}
		catch ( itk::ExceptionObject &err)
		{
			std::cerr << "ExceptionObject caught? !" << std::endl;
			std::cerr << err << std::endl;
			free(fullpath);
			return -1;
		}
		free(fullpath);
		
		ImageType::Pointer image = reader->GetOutput();
		ImageType::RegionType region = image->GetLargestPossibleRegion();
		ImageType::SizeType size = region.GetSize();
		
		// First, we want to crop the file so that its horizontal
		// and vertical sizes are multiples of the scale value (modcrop)
		ImageType::IndexType desiredStart;
		desiredStart.Fill(0);
		ImageType::SizeType desiredSize;
		desiredSize[0] = size[0] - (size[0] % scale);
		desiredSize[1] = size[1] - (size[1] % scale);
		ImageType::RegionType desiredRegion(desiredStart, desiredSize);

#ifdef FUNCTEST
		std::cout << "TESTING MODCROP FUNC: i = " << i << std::endl;
		std::cout << "image size: " << size << std::endl;
		std::cout << "scale: " << scale << std::endl;
		std::cout << "desired size: " << desiredSize << std::endl;
		std::cout << "**************************************" << std::endl;
#endif

		ExtractImageFilterType::Pointer filter = ExtractImageFilterType::New();
		filter->SetExtractionRegion(desiredRegion);
		filter->SetInput(image);
		filter->Update();

		// we can override the original image as we are going
		// to work on this from now on. because of the smart pointers,
		// there should not be any memory leak.
		image = filter->GetOutput();
		// modcrop completed at this point. image has the cropped image data


		//downsampling to lores starts

		// Instantiate the b-spline interpolator and set it as the third order
		// for bicubic.
		InterpolatorType::Pointer _pInterpolator = InterpolatorType::New();
		_pInterpolator->SetSplineOrder(3);

		// Instantiate the resampler. Wire in the transform and the interpolator.
		ResampleFilterType::Pointer _pResizeFilter = ResampleFilterType::New();
		_pResizeFilter->SetInterpolator(_pInterpolator);

		const double vfOutputOrigin[2]  = { 0.0, 0.0 };
		_pResizeFilter->SetOutputOrigin(vfOutputOrigin);

		// Fetch original image size.
		const ImageType::RegionType& inputRegion = image->GetLargestPossibleRegion();
		const ImageType::SizeType& vnInputSize = inputRegion.GetSize();
		unsigned int nOldWidth = vnInputSize[0];
		unsigned int nOldHeight = vnInputSize[1];
		  
		unsigned int nNewWidth = vnInputSize[0]/scale;
		unsigned int nNewHeight = vnInputSize[1]/scale;

		// Fetch original image spacing.
		const ImageType::SpacingType& vfInputSpacing = image->GetSpacing();

		double vfOutputSpacing[2];
		vfOutputSpacing[0] = vfInputSpacing[0] * (double) nOldWidth / (double) nNewWidth;
		vfOutputSpacing[1] = vfInputSpacing[1] * (double) nOldHeight / (double) nNewHeight;
 
		_pResizeFilter->SetOutputSpacing(vfOutputSpacing);
		ResampleFilterType::SizeType vnOutputSize = { {nNewWidth, nNewHeight} };
		_pResizeFilter->SetSize(vnOutputSize);
		_pResizeFilter->SetInput(image);
		_pResizeFilter->Update();
		ImageType::Pointer lores = _pResizeFilter->GetOutput();
		lores->DisconnectPipeline();
		//downscaling interpolations ends

		const ImageType::RegionType& inputRegion2 = lores->GetLargestPossibleRegion();
		const ImageType::SizeType& vnInputSize2 = inputRegion2.GetSize();
		nOldWidth = vnInputSize2[0];
		nOldHeight = vnInputSize2[1];

		// now we want to upscale the image
		nNewWidth = vnInputSize2[0]*scale;
		nNewHeight = vnInputSize2[1]*scale;
		const ImageType::SpacingType& vfInputSpacingLow = lores->GetSpacing();
		vfOutputSpacing[0] = vfInputSpacingLow[0] * (double) nOldWidth / (double) nNewWidth;
		vfOutputSpacing[1] = vfInputSpacingLow[1] * (double) nOldHeight / (double) nNewHeight;
 
		_pResizeFilter->SetInterpolator(_pInterpolator);
		_pResizeFilter->SetOutputOrigin(vfOutputOrigin);
		_pResizeFilter->SetOutputSpacing(vfOutputSpacing);
		vnOutputSize[0] = nNewWidth;
		vnOutputSize[1] = nNewHeight;
		_pResizeFilter->SetSize(vnOutputSize);
		_pResizeFilter->UpdateLargestPossibleRegion();
		_pResizeFilter->SetInput(lores);
		_pResizeFilter->Update();
		ImageType::Pointer midres = _pResizeFilter->GetOutput();
		// at this point, we have the blurred versions in midres

		// defining the kernels to be used for the feature extraction
		KernelImageType::Pointer kernel1 = KernelImageType::New();
		KernelImageType::Pointer kernel2 = KernelImageType::New();
		KernelImageType::Pointer kernel3 = KernelImageType::New();
		KernelImageType::Pointer kernel4 = KernelImageType::New();

		CreateKernels(kernel1, kernel2, kernel3, kernel4, scale);
		ConvolutionFilterType::Pointer convolutionFilter = ConvolutionFilterType::New();
		convolutionFilter->SetInput(midres);
		convolutionFilter->SetKernelImage(kernel1);

//		KernelImageType::IndexType pixelIndex;
//			pixelIndex[0] = 0;
//
//			for(unsigned int c = 0; c < scale + 1; c++)
//			{
//				pixelIndex[1] = c;
//
//				std::cout << kernel1->GetPixel(pixelIndex) << " ind: ";
//				std::cout << pixelIndex[1] << " ";
//			}
			std::cout << std::endl;
		KernelImageType::RegionType reg = kernel1->GetLargestPossibleRegion();
		itk::ImageRegionIterator<KernelImageType> imageIterator(kernel1, reg);

		reg.SetIndex(0,0);
		reg.SetIndex(1,0);

		while(!imageIterator.IsAtEnd())
		    {
		    //imageIterator.Set(255);
		    std::cout << "pix: " << imageIterator.Get() << " ";

		    ++imageIterator;
		    }




		
#ifdef FUNCTEST
		std::cout << "TESTING SCALE DOWN FUNC: i = " << i << std::endl;
		std::cout << "image size: " << vnInputSize << std::endl;
		std::cout << "scale: " << scale << std::endl;		
		
		char *testpath = (char*)malloc(4 + strlen(filename) + 2);
		if (testpath == NULL)
		{
			std::cerr << "Could not allocate memory for the fullpath. " << std::endl;
			return -1;
		}

		sprintf(testpath, "test%s", filename);

		// Write the result
		WriterType::Pointer pWriter = WriterType::New();
		pWriter->SetFileName(testpath);
		pWriter->SetInput(convolutionFilter->GetOutput());
		pWriter->Update();
		free(testpath);
		std::cout << "**************************************" << std::endl;
#endif

	}

	return 0;
}

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include <itkExtractImageFilter.h>
#include <itkDirectory.h>

//headers for bicubic interpolation
#include "itkIdentityTransform.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkResampleImageFilter.h"

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
typedef itk::IdentityTransform<double, 2>  TransformType;
typedef itk::BSplineInterpolateImageFunction<ImageType, double, double> InterpolatorType;
typedef itk::ResampleImageFilter<ImageType, ImageType>   ResampleFilterType;

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
	//itk::Directory::Pointer trainDir = itk::Directory::New();
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

		// Instantiate the transform and specify it should be the id transform.
		 TransformType::Pointer _pTransform = TransformType::New();
 		_pTransform->SetIdentity();
		// Instantiate the b-spline interpolator and set it as the third order
		// for bicubic.
		InterpolatorType::Pointer _pInterpolator = InterpolatorType::New();
		_pInterpolator->SetSplineOrder(3);

		// Instantiate the resampler. Wire in the transform and the interpolator.
		ResampleFilterType::Pointer _pResizeFilter = ResampleFilterType::New();
		_pResizeFilter->SetTransform(_pTransform);
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
		itk::Size<2> vnOutputSize = { {nNewWidth, nNewHeight} };
		_pResizeFilter->SetSize(vnOutputSize);
		_pResizeFilter->SetInput(image);
		_pResizeFilter->Update();
		ImageType::Pointer lores = _pResizeFilter->GetOutput();
		_pResizeFilter->UpdateLargestPossibleRegion();

		//downscaling interpolations ends


		//upscaling with bicubic interpolation
		//// Instantiate the transform and specify it should be the id transform.
		////TransformType::Pointer _pTransform2 = TransformType::New();
 	//	//_pTransform2->SetIdentity();
		//// Instantiate the b-spline interpolator and set it as the third order
		//// for bicubic.
		////InterpolatorType::Pointer _pInterpolator2 = InterpolatorType::New();
		////_pInterpolator2->SetSplineOrder(3);

		//// Instantiate the resampler. Wire in the transform and the interpolator.
		//ResampleFilterType::Pointer _pResizeFilter2 = ResampleFilterType::New();
		//_pResizeFilter2->SetTransform(_pTransform);
		//_pResizeFilter2->SetInterpolator(_pInterpolator);

		//const double vfOutputOrigin2[2]  = { 0.0, 0.0 };
		//_pResizeFilter2->SetOutputOrigin(vfOutputOrigin2);

		//// Fetch original image size.
		//const ImageType::RegionType& inputRegion2 = lores->GetLargestPossibleRegion();
		//const ImageType::SizeType& vnInputSize2 = inputRegion2.GetSize();
		//unsigned int nOldWidth2 = vnInputSize2[0];
		//unsigned int nOldHeight2 = vnInputSize2[1];
		//  
		//unsigned int nNewWidth2 = vnInputSize2[0]*scale;
		//unsigned int nNewHeight2 = vnInputSize2[1]*scale;

		//// Fetch original image spacing.
		//const ImageType::SpacingType& vfInputSpacing2 = lores->GetSpacing();

		//double vfOutputSpacing2[2];
		//vfOutputSpacing2[0] = vfInputSpacing2[0] * (double) nOldWidth2 / (double) nNewWidth2;
		//vfOutputSpacing2[1] = vfInputSpacing2[1] * (double) nOldHeight2 / (double) nNewHeight2;
 
		//_pResizeFilter2->SetOutputSpacing(vfOutputSpacing2);
		//itk::Size<2> vnOutputSize2 = { {nNewWidth2, nNewHeight2} };
		//_pResizeFilter2->SetSize(vnOutputSize2);
		//_pResizeFilter2->SetInput(lores);
		//ImageType::Pointer midres = _pResizeFilter2->GetOutput();
		//_pResizeFilter2->UpdateLargestPossibleRegion();

		itk::ChangeInformationImageFilter<ImageType> ::Pointer somename = itk::ChangeInformationImageFilter<ImageType> :: New();
		somename->SetInput(lores);
		somename->Update();




		const ImageType::RegionType& inputRegion2 = lores->GetLargestPossibleRegion();
		const ImageType::SizeType& vnInputSize2 = inputRegion2.GetSize();
		nOldWidth = vnInputSize2[0];
		nOldHeight = vnInputSize2[1];
		//  
		nNewWidth = vnInputSize2[0]*scale;
		nNewHeight = vnInputSize2[1]*scale;
		const ImageType::SpacingType& vfInputSpacingLow = lores->GetSpacing();
		vfOutputSpacing[0] = vfInputSpacingLow[0] * (double) nOldWidth / (double) nNewWidth;
		vfOutputSpacing[1] = vfInputSpacingLow[1] * (double) nOldHeight / (double) nNewHeight;
 
		_pResizeFilter->SetTransform(_pTransform);
		_pResizeFilter->SetInterpolator(_pInterpolator);
		_pResizeFilter->SetOutputOrigin(vfOutputOrigin);
		_pResizeFilter->SetOutputSpacing(vfOutputSpacing);
		vnOutputSize[0] = nNewWidth;
		vnOutputSize[1] = nNewHeight;
		_pResizeFilter->SetSize(vnOutputSize);
		_pResizeFilter->UpdateLargestPossibleRegion();
		_pResizeFilter->SetInput(lores);
		_pResizeFilter->Modified();
		_pResizeFilter->Update();
		
		
		ImageType::Pointer midres = _pResizeFilter->GetOutput();
		



#ifdef FUNCTEST
		std::cout << "TESTING SCALE DOWN FUNC: i = " << i << std::endl;
		std::cout << "image size: " << vnInputSize << std::endl;
		std::cout << "scale: " << scale << std::endl;		
		std::cout << "Old Width: " << nOldWidth << std::endl;
		std::cout << "Old Height: " << nOldHeight << std::endl;
		std::cout << "New Width: " << nNewWidth << std::endl;
		std::cout << "New Height: " << nNewHeight << std::endl;
		
		
		std::cout << "Output image size: " << vnOutputSize << std::endl;
		std::cout << "Output spc size: " << vfOutputSpacing[0] << std::endl;
		
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
		pWriter->SetInput(midres);
		pWriter->UpdateLargestPossibleRegion();
		free(testpath);
		std::cout << "**************************************" << std::endl;
#endif

	}

	return 0;
}

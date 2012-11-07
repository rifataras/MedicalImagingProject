#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include <itkExtractImageFilter.h>
#include <itkDirectory.h>

// project specific preprocessor definitions
#define FUNCTEST // define this variable to test the functionality correctness


// typedefs
typedef unsigned char 				PixelType;
typedef itk::Image<PixelType, 2>	ImageType;
typedef itk::ExtractImageFilter< ImageType, ImageType > ExtractImageFilterType;
typedef itk::ImageFileReader< ImageType > ReaderType;

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
		free(fullpath); // if there was finally in c++, I would use that :(

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
	}

	return 0;
}

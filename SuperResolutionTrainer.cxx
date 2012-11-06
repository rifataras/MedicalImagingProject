#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

int main( int argc, char *argv[] )
{
	// The parameters we are looking for are: 1) the directory of the training files,
	// 2) the file extension of these files,  3) magnification scale to train for.
	if ( argc < 4 )
	{
		std::cerr << "Missing parameters. " << std::endl;
		std::cerr << "Usage: " << std::endl;
		std::cerr << argv[0]
		<< " directory fileExtension scale"
		<< std::endl;
		return -1;
	}
	return 0;
}

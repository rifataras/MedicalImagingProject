#if defined(_WIN32) || defined(_WIN64)

#include <time.h>

#endif

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include <itkExtractImageFilter.h>
#include <itkDirectory.h>

//headers for bicubic interpolation
#include "itkIdentityTransform.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkResampleImageFilter.h"

#include "CreateImage.h"


#include "itkSubtractImageFilter.h"

#include "itkConvolutionImageFilter.h"

#include "ImageToFeatureConverter.h"
#include "ImageToFeatureConverter.cxx" // !!! HACK! fix this later !!



#include "itkCastImageFilter.h"

#include "itkAddImageFilter.h"

#include "itkChangeInformationImageFilter.h"


#include "ksvd.h"

//#include "lib_ormp.h"
//#include "lib_svd.h"

#include "Eigen/Core"
#include "Eigen/Eigen"

// project specific preprocessor definitions
#define FUNCTEST // define this variable to test the functionality correctness


// typedefs
typedef unsigned char 				PixelType;
typedef itk::Image<PixelType, 2>	ImageType;
typedef float					KernelElementType;
typedef itk::Image<KernelElementType, 2>	KernelImageType;


typedef itk::ExtractImageFilter< ImageType, ImageType > ExtractImageFilterType;
typedef itk::ImageFileReader< ImageType > ReaderType;
typedef itk::ImageFileWriter< ImageType > WriterType;


//typedefs for bicubic interpolation
typedef itk::IdentityTransform<double, 2>  IdentityTransformType;
typedef itk::BSplineInterpolateImageFunction<ImageType, double, double> InterpolatorType;
typedef itk::ResampleImageFilter<ImageType, ImageType>   ResampleFilterType;
typedef itk::ConvolutionImageFilter<ImageType, KernelImageType, KernelImageType> ConvolutionFilterType;

// typedefs for image subtraction
typedef itk::SubtractImageFilter<ImageType,ImageType,KernelImageType> SubtractImageFilterType;

///***********KSVD***************************************/
//typedef std::vector<std::vector<float> > matD_t;
//typedef std::vector<std::vector<unsigned> > matU_t;
//typedef std::vector<float> vecD_t;
//typedef std::vector<unsigned> vecU_t;
//typedef std::vector<float>::iterator iterD_t;
//typedef std::vector<unsigned>::iterator iterU_t;


void save(const char *filename, const Eigen::MatrixXf& m)
{
	std::ofstream f(filename, std::ios::binary);
	Eigen::MatrixXf::Index rows, cols;
	rows = m.rows(); cols = m.cols();
	const float *mdata = m.data();
	f.write((char *)&(rows), sizeof(m.rows()));
	f.write((char *)&(cols), sizeof(m.cols()));
	f.write((char *)&(mdata), sizeof(float)*m.cols()*m.cols());
	f.close();
}

void load(const char *filename, Eigen::MatrixXf& m)
{
	Eigen::MatrixXf::Index rows, cols;
	std::ifstream f(filename, std::ios::binary);
	float *mdata = m.data();
	f.read((char *)&rows, sizeof(rows));
	f.read((char *)&cols, sizeof(cols));
	std::cout << "reading: " << rows << "," << cols << std::endl;
	m.resize(rows,cols);
	std::cout << "reading2";
	f.read((char *)&(mdata), sizeof(float)*rows*cols);
	//if (f.bad())
	//throw std::exception("Error reading matrix");
	f.close();
}

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



int main( int argc, char *argv[] )
{
	bool disableTraining = false;
	// The parameters we are looking for are:
	// 1) the directory of the training files,
	// 2) magnification scale to train for,
	// 3) the window size (will be scaled by scale)
	// 4) overlap amount (will be scaled by scale)
	// 5) border of the image to ignore (will be scaled)
	if ( argc < 6 )
	{
		std::cerr << "Missing parameters. " << std::endl;
		std::cerr << "Usage: " << std::endl;
		std::cerr << argv[0]
		<< " directory scale window overlap border"
		<< std::endl;
		return -1;
	}

	if(argc > 6)
	{
		// This means we are providing the dictionary information
		if(argc < 9)
		{
			// There should be 3 more parameters
			std::cerr << "Missing parameters. " << std::endl;
			std::cerr << "Usage: " << std::endl;
			std::cerr << argv[0]
			<< " directory scale window overlap border A_l.krmat A_h.krmat V_pca.krmat"
			<< std::endl;
			return -1;
		}
		disableTraining = true;
	}

	int scale = ::atoi(argv[2]);
	int window = ::atoi(argv[3]);
	int overlap = ::atoi(argv[4]);
	int border = ::atoi(argv[5]);
	const int dictionarySize = 500;
	const double C = 1.1139195378939404;
	const int numOfIterations = 20;
	double   eps;

	Eigen::MatrixXf V_pca;	// Dimension reduction matrix (B)
	Eigen::MatrixXf A_l;	// Lo-res dictionary
	Eigen::MatrixXf A_h;	// Hi-res dictionary

	matD_t dictionary; //(dictionarySize, vecD_t(features_pca.rows()));

	if(!disableTraining)
	{
		itksys::Directory trainDir;
		if(!trainDir.Load(argv[1]))
		{
			std::cerr << "Could not open the directory. " << std::endl;
			return -1;
		}

		int numberOfFiles = trainDir.GetNumberOfFiles();
		ReaderType::Pointer reader = ReaderType::New();

		// The feature vector from the convolved images
		std::vector<std::vector<KernelElementType> > globalFeatureMatrix;
		std::vector<std::vector<KernelElementType> > globalPatchMatrix;

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

			reader->Update();
			ImageType::Pointer image = reader->GetOutput();
			ImageType::RegionType region = image->GetLargestPossibleRegion();
			ImageType::SizeType size = region.GetSize();

	//		std::cout  << "*******" << std::endl << "Content of midres:" << std::endl;
	//		ImageType::RegionType midresReg = image->GetLargestPossibleRegion();
	//		itk::ImageRegionIterator<ImageType> imageIteratorMidres(image, midresReg);
	//		imageIteratorMidres.GoToBegin();
	//		int oldRow = 0;
	//		while(!imageIteratorMidres.IsAtEnd())
	//		{
	//			ImageType::IndexType curInd = imageIteratorMidres.GetIndex();
	//			if(curInd[1] != oldRow)
	//			{
	//				std::cout << std::endl;
	//				oldRow = curInd[1];
	//			}
	//			std::cout << "[" << curInd[1] << "," << curInd[0] << "]: ";
	//			std::cout << (unsigned int)imageIteratorMidres.Get() << " ";
	//			++imageIteratorMidres;
	//		}
	//		std::cout  << std::endl << "END Content of midres *******" << std::endl;

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

			// lets subtract the blurred image from the original
			// hires image to obtain the high frequencies
			SubtractImageFilterType::Pointer subtractFilter
				= SubtractImageFilterType::New ();
			subtractFilter->SetInput1(image);
			subtractFilter->SetInput2(midres);
			subtractFilter->Update();
			KernelImageType::Pointer differential = subtractFilter->GetOutput();

			// defining the kernels to be used for the feature extraction
			KernelImageType::Pointer kernel1 = KernelImageType::New();
			KernelImageType::Pointer kernel2 = KernelImageType::New();
			KernelImageType::Pointer kernel3 = KernelImageType::New();
			KernelImageType::Pointer kernel4 = KernelImageType::New();

			CreateKernels(kernel1, kernel2, kernel3, kernel4, scale);
	#ifdef FUNCTEST
			// print out the created kernels
			std::cout << "Printing the created kernels:";
			std::cout  << std::endl << "KERNEL 1:" << std::endl;
			KernelImageType::RegionType region1 = kernel1->GetLargestPossibleRegion();
			itk::ImageRegionIterator<KernelImageType> imageIterator1(kernel1, region1);
			imageIterator1.GoToBegin();
			while(!imageIterator1.IsAtEnd())
			{
				std::cout << imageIterator1.Get() << ", ";
				++imageIterator1;
			}

			std::cout  << std::endl << "KERNEL 2:" << std::endl;
			KernelImageType::RegionType region2 = kernel2->GetLargestPossibleRegion();
			itk::ImageRegionIterator<KernelImageType> imageIterator2(kernel2, region2);
			imageIterator2.GoToBegin();
			while(!imageIterator2.IsAtEnd())
			{
				std::cout << imageIterator2.Get() << ", ";
				++imageIterator2;
			}

			std::cout  << std::endl << "KERNEL 3:" << std::endl;
			KernelImageType::RegionType region3 = kernel3->GetLargestPossibleRegion();
			itk::ImageRegionIterator<KernelImageType> imageIterator3(kernel3, region3);
			imageIterator3.GoToBegin();
			while(!imageIterator3.IsAtEnd())
			{
				std::cout << imageIterator3.Get() << ", ";
				++imageIterator3;
			}

			std::cout  << std::endl << "KERNEL 4:" << std::endl;
			KernelImageType::RegionType region4 = kernel4->GetLargestPossibleRegion();
			itk::ImageRegionIterator<KernelImageType> imageIterator4(kernel4, region4);
			imageIterator4.GoToBegin();
			while(!imageIterator4.IsAtEnd())
			{
				std::cout << imageIterator4.Get() << ", ";
				++imageIterator4;
			}
	#endif



			ConvolutionFilterType::Pointer convolutionFilter1 = ConvolutionFilterType::New();
			convolutionFilter1->SetInput(midres);
			convolutionFilter1->SetKernelImage(kernel1);

			ConvolutionFilterType::Pointer convolutionFilter2 = ConvolutionFilterType::New();
			convolutionFilter2->SetInput(midres);
			convolutionFilter2->SetKernelImage(kernel2);

			ConvolutionFilterType::Pointer convolutionFilter3 = ConvolutionFilterType::New();
			convolutionFilter3->SetInput(midres);
			convolutionFilter3->SetKernelImage(kernel3);

			ConvolutionFilterType::Pointer convolutionFilter4 = ConvolutionFilterType::New();
			convolutionFilter4->SetInput(midres);
			convolutionFilter4->SetKernelImage(kernel4);

			// Now extract the features from the convolved images
			ImageToFeatureConverter<KernelImageType> im2feat(scale, border, overlap, window);
			std::vector<std::vector<KernelElementType> > featureMatrix1;
			std::vector<std::vector<KernelElementType> > featureMatrix2;
			std::vector<std::vector<KernelElementType> > featureMatrix3;
			std::vector<std::vector<KernelElementType> > featureMatrix4;
			std::vector<std::vector<KernelElementType> > patchesMatrix;

			// Get the individual feature matrices from the filtered images
			// Later on, these matrices will be aggregated to the global
			// feature matrix
			subtractFilter->Update();
			im2feat.GetOutput(subtractFilter->GetOutput(), patchesMatrix);

			convolutionFilter1->Update();
			im2feat.GetOutput(convolutionFilter1->GetOutput(), featureMatrix1);

			convolutionFilter2->Update();
			im2feat.GetOutput(convolutionFilter2->GetOutput(), featureMatrix2);

			convolutionFilter3->Update();
			im2feat.GetOutput(convolutionFilter3->GetOutput(), featureMatrix3);

			convolutionFilter4->Update();
			im2feat.GetOutput(convolutionFilter4->GetOutput(), featureMatrix4);


			/*
					{
					std::cout  << "*******" << std::endl << "Content of image:" << std::endl;
					ImageType::RegionType imageReg = image->GetLargestPossibleRegion();
					itk::ImageRegionIterator<ImageType> imageIteratorImage(image, imageReg);
					imageIteratorImage.GoToBegin();
					int oldRow = 0;
					while(!imageIteratorImage.IsAtEnd())
					{
						ImageType::IndexType curInd = imageIteratorImage.GetIndex();
						if(curInd[1] != oldRow)
						{
							std::cout << std::endl;
							oldRow = curInd[1];
						}
						std::cout << "[" << curInd[1] << "," << curInd[0] << "]: ";
						std::cout << (int)imageIteratorImage.Get() << " ";
						++imageIteratorImage;
					}
					std::cout  << std::endl << "END Content of image *******" << std::endl;
					}

					std::cout  << "*******" << std::endl << "Content of midres:" << std::endl;
					ImageType::RegionType midresReg = midres->GetLargestPossibleRegion();
					itk::ImageRegionIterator<ImageType> imageIteratorMidres(midres, midresReg);
					imageIteratorMidres.GoToBegin();
					int oldRow = 0;
					while(!imageIteratorMidres.IsAtEnd())
					{
						ImageType::IndexType curInd = imageIteratorMidres.GetIndex();
						if(curInd[1] != oldRow)
						{
							std::cout << std::endl;
							oldRow = curInd[1];
						}
						std::cout << "[" << curInd[1] << "," << curInd[0] << "]: ";
						std::cout << (int)imageIteratorMidres.Get() << " ";
						++imageIteratorMidres;
					}
					std::cout  << std::endl << "END Content of midres *******" << std::endl;

					{
					std::cout  << "*******" << std::endl << "Content of differential:" << std::endl;
					ImageType::RegionType midresReg = differential->GetLargestPossibleRegion();
					itk::ImageRegionIterator<KernelImageType> imageIteratorMidres(differential, midresReg);
					imageIteratorMidres.GoToBegin();
					int oldRow = 0;
					while(!imageIteratorMidres.IsAtEnd())
					{
						ImageType::IndexType curInd = imageIteratorMidres.GetIndex();
						if(curInd[1] != oldRow)
						{
							std::cout << std::endl;
							oldRow = curInd[1];
						}
						std::cout << "[" << curInd[1] << "," << curInd[0] << "]: ";
						std::cout << (float)imageIteratorMidres.Get() << " ";
						++imageIteratorMidres;
					}
					std::cout  << std::endl << "END Content of differential *******" << std::endl;
					}
					*/

					/*std::cout  << std::endl << "convolution 1 output:" << std::endl;
					KernelImageType::RegionType region4 = convolutionFilter1->GetOutput()->GetLargestPossibleRegion();
					itk::ImageRegionIterator<KernelImageType> imageIterator4(convolutionFilter1->GetOutput(), region4);
					imageIterator4.GoToBegin();
					oldRow = 0;
					while(!imageIterator4.IsAtEnd())
					{
						ImageType::IndexType curInd = imageIterator4.GetIndex();
						if(curInd[1] != oldRow)
						{
							std::cout << std::endl;
							oldRow = curInd[1];
						}
						std::cout << "[" << curInd[1] << "," << curInd[0] << "]: ";
						std::cout << imageIterator4.Get() << " ";
						++imageIterator4;
					}
					std::cout  << std::endl << "-0-0-0-0-0-0-0-0-" << std::endl;
					*/
					/*for(int hh = 0; hh < featureMatrix1.size(); hh++)
					{
						for(int tt = 0; tt < featureMatrix1[hh].size(); tt++)
						{
							std::cout << featureMatrix1[hh].at(tt) << ", ";
						}
						std::cout << "! "<<hh<< "! " <<std::endl;
					}*/






			for(int feat = 0; feat < featureMatrix1.size(); feat++)
			{
				std::vector<KernelElementType> aggregatedFeatures;
				aggregatedFeatures.insert( aggregatedFeatures.end(), featureMatrix1[feat].begin(), featureMatrix1[feat].end() );
				aggregatedFeatures.insert( aggregatedFeatures.end(), featureMatrix2[feat].begin(), featureMatrix2[feat].end() );
				aggregatedFeatures.insert( aggregatedFeatures.end(), featureMatrix3[feat].begin(), featureMatrix3[feat].end() );
				aggregatedFeatures.insert( aggregatedFeatures.end(), featureMatrix4[feat].begin(), featureMatrix4[feat].end() );
				globalFeatureMatrix.push_back(aggregatedFeatures);
			}

			for(int patc = 0; patc < patchesMatrix.size(); patc++)
			{
				globalPatchMatrix.push_back(patchesMatrix[patc]);
			}
		}

		// call ksvd train dataset
		std::cout << "global feature matrix size = " << globalFeatureMatrix.size() << std::endl;
		if(globalFeatureMatrix.size() > 0)
			std::cout << "feature  size = " << globalFeatureMatrix[0].size() << std::endl;

		unsigned int m = globalFeatureMatrix[0].size();   // dimension of each point
		unsigned int n = globalFeatureMatrix.size();  // number of points

		Eigen::MatrixXf DataPoints(m,n);
		for (int j=0; j<DataPoints.cols(); ++j) // loop over columns
			for (int i=0; i<DataPoints.rows(); ++i) // loop over rows
				DataPoints(i,j) = globalFeatureMatrix[j].at(i);

		std::cout << "kill 1 " << m << ", " << n <<  std::endl;
		float mean;
		Eigen::VectorXf meanVector;

		typedef std::pair<float, int> myPair;
		typedef std::vector<myPair> PermutationIndices;


		//
		// for each point
		//   center the poin with the mean among all the coordinates
		//
		/**/
		for (int i = 0; i < DataPoints.rows(); i++)
		{
		   mean = (DataPoints.row(i).sum())/n;		 //compute mean
		   meanVector  = Eigen::VectorXf::Constant(n,mean); // create a vector with constant value = mean
		   DataPoints.row(i) -= meanVector;
		}

		std::cout << "kill 2" <<  std::endl;

		// get the covariance matrix
		Eigen::MatrixXf Covariance = Eigen::MatrixXf::Zero(m, m);
		Covariance = (1 / (float) n) * DataPoints * DataPoints.transpose();
		//std::cout << Covariance ;

		std::cout << "kill 3" <<  std::endl;
		// compute the eigenvalue on the Cov Matrix
		Eigen::EigenSolver<Eigen::MatrixXf> m_solve(Covariance);
		std::cout << "PCA Done";
		Eigen::VectorXf eigenvalues = m_solve.eigenvalues().real();
		Eigen::MatrixXf eigenVectors = m_solve.eigenvectors().real();
		// std::cout << "sizeofeigenvectors: " << eigenVectors.rows() << ", " << eigenVectors.cols() << std::endl;


		// sort and get the permutation indices
		PermutationIndices pi;
		for (int i = 0 ; i < m; i++)
		{
			myPair mp;
			mp.first = eigenvalues(i);
			mp.second = i;
			pi.push_back(mp);
		}

		sort(pi.begin(), pi.end());

		/*for (unsigned int i = 0; i < m ; i++)
			std::cout << "eigen= " << pi[i].first << " pi= " << pi[i].second << std::endl;
		 */
		Eigen::VectorXf sortedEigenValues(m);
		Eigen::VectorXf eigenValuesCumSum(m);
		Eigen::VectorXf eigenValuesCumPerc(m);

		for (unsigned int i = 0; i < m ; i++)
		{
			sortedEigenValues(i) = pi[i].first < 0 ? 0 : pi[i].first;
			if(i == 0)
				eigenValuesCumSum(i) = sortedEigenValues(i);
			else
				eigenValuesCumSum(i) = eigenValuesCumSum(i-1) + sortedEigenValues(i);
		}

		float eigenValuesSum = sortedEigenValues.sum();

		std::cout << "print eigenvalues::" << std::endl;
		std::vector<Eigen::VectorXf> highestEigenVectors;
		for (unsigned int i = m-1; i > 0 ; i--)
		{
			eigenValuesCumPerc(i) = eigenValuesCumSum(i) / eigenValuesSum;

			if(eigenValuesCumPerc(i) > 0.05)
			{
				std::cout << eigenValuesCumPerc(i) << "\n";
				highestEigenVectors.push_back(eigenVectors.col(pi[i].second));
			}
			else
				break;
		}

		int numHighestEigenVectors = highestEigenVectors.size();
		V_pca.resize(m, numHighestEigenVectors);
		for(unsigned int i = 0; i < numHighestEigenVectors; i++)
		{
			V_pca.col(i) << highestEigenVectors[i];
		}

		Eigen::MatrixXf features_pca = V_pca.transpose() * DataPoints;
		std::cout << "features_pca: " << features_pca.rows() << "," << features_pca.cols() << "\n";

		/******************************************************/
		/***********KSVD***************************************/

		matD_t reducedFeatures(features_pca.cols(), vecD_t(features_pca.rows()));
		dictionary.resize(dictionarySize, vecD_t(features_pca.rows()));
		matD_t gamma(n, vecD_t (dictionarySize,0));

		// convert the features matrix from Eigen representation
		// to the matD_t representation.
		for(unsigned int i = 0; i < features_pca.cols(); i++)
		{
			for(unsigned int r = 0; r < features_pca.rows(); r++)
			{
				reducedFeatures[i].at(r) = features_pca(r,i);
			}
		}

		std::cout << "about to call obtain_dict" << std::endl;

		obtain_dict(dictionary, reducedFeatures);

		std::cout << "about to call ksvd_process" << std::endl;

		eps  = ((double) (features_pca.rows())) * C * C;
		ksvd_process(reducedFeatures, dictionary, gamma, features_pca.rows(), dictionarySize, numOfIterations, C);

		std::cout << std::endl << "dictionary size" << dictionary[0].size() << ", " << dictionary.size() << std::endl;
		std::cout << std::endl << "gamma matrix size: " << gamma[0].size() << ", " << gamma.size() << std::endl;
		std::cout << std::endl << "patches matrix size: " << globalPatchMatrix.size() << ", " << globalPatchMatrix[0].size() << std::endl;

		// Find hires  dictionary
		// convert the matrices to Eigen representation
		// from the matD_t representation for Eigen's operations.
		// !! we need to define Q matrix as follows because,
		// ksvd process may return incomplete alpha matrix.
		// the size should be like this. we will fill the missing
		// points with 0s.
		Eigen::MatrixXf Q_Matrix(dictionarySize, n);
		Eigen::MatrixXf P_Matrix(globalPatchMatrix[0].size(), globalPatchMatrix.size());
		A_l.resize(dictionary[0].size(), dictionary.size());


		std::cout << std::endl << "A_l size: " << A_l.rows() << ", " << A_l.cols() << std::endl;
		std::cout << std::endl << "P_Matrix size: " << P_Matrix.rows() << ", " << P_Matrix.cols() << std::endl;
		std::cout << std::endl << "Q_Matrix size: " << Q_Matrix.rows() << ", " << Q_Matrix.cols() << std::endl;

		std::cout << "w1" << std::endl;
		for(unsigned int i = 0; i < Q_Matrix.rows(); i++)
		{
			for(unsigned int j = 0; j < Q_Matrix.cols(); j++)
			{
				Q_Matrix(i,j) = gamma[j].at(i);
			}
		}
		std::cout << "w2" << std::endl;

		//std::cout << "here is Q:" << std::endl << Q_Matrix << std::endl;

		for(unsigned int i = 0; i < P_Matrix.rows(); i++)
		{
			for(unsigned int j = 0; j < P_Matrix.cols(); j++)
			{
				P_Matrix(i,j) = globalPatchMatrix[j].at(i);
			}
		}
		std::cout << "w3" << std::endl;

		for(unsigned int i = 0; i < A_l.rows(); i++)
		{
			for(unsigned int j = 0; j < A_l.cols(); j++)
			{
				A_l(i,j) = dictionary[j].at(i);
			}
		}
		std::cout << "w4" << std::endl;

		//std::cout << "here is P:" << std::endl << P_Matrix << std::endl;

		Eigen::MatrixXf QQT = Q_Matrix * Q_Matrix.transpose();
		A_h = P_Matrix * Q_Matrix.transpose() * QQT.inverse();

		//std::cout << "here is Ah:" << std::endl << A_h << std::endl;

		//Crashing upon saving for larger dictionary size
		// SAVE THE NECESSARY MATRICES TO USE LATER ON
		
		#if defined(_WIN32) || defined(_WIN64)
		
		#else if

		save("V_pca.krmat", V_pca);
		save("A_l.krmat", A_l);
		save("A_h.krmat", A_h);

		#endif
	}
	else	// training is not performed, read the matrices and reconstruct
	{
		load("V_pca.krmat", V_pca);
		load("A_l.krmat", A_l);
		load("A_h.krmat", A_h);

		std::cout << A_h;

		dictionary.resize(A_l.cols(), vecD_t(A_l.rows()));
		for(unsigned int i = 0; i < A_l.rows(); i++)
		{
			for(unsigned int j = 0; j < A_l.cols(); j++)
			{
				dictionary[j].at(i) = A_l(i,j);
			}
		}

		eps  = ((double) (A_l.rows())) * C * C;
	}

	std::cout << "r1" << std::endl;
	// NOW the reconstruction phase


	const char *filename = "test_reconstr.bmp";
	ReaderType::Pointer readerRecon = ReaderType::New();
	readerRecon->SetFileName( filename );
	try
	{
		readerRecon->UpdateLargestPossibleRegion();
	}
	catch ( itk::ExceptionObject &err)
	{
		std::cerr << "ExceptionObject caught? !" << std::endl;
		std::cerr << err << std::endl;
		//free(filename);
		return -1;
	}
	//free(filename);

	readerRecon->Update();
	ImageType::Pointer image = readerRecon->GetOutput();
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

	ExtractImageFilterType::Pointer filter = ExtractImageFilterType::New();
	filter->SetExtractionRegion(desiredRegion);
	filter->SetInput(image);
	filter->Update();

	// we can override the original image as we are going
	// to work on this from now on. because of the smart pointers,
	// there should not be any memory leak.
	image = filter->GetOutput();
	// modcrop completed at this point. image has the cropped image data
/*
	{

		typedef itk::CastImageFilter< ImageType, KernelImageType > CastFilterType;
		CastFilterType::Pointer castFilter = CastFilterType::New();
		castFilter->SetInput(image);
		castFilter->Update();
		ImageToFeatureConverter<KernelImageType> im2feat(scale, border, 0, 10);
		std::vector<std::vector<KernelElementType> > patchesMatrix;
		im2feat.GetOutput(castFilter->GetOutput(), patchesMatrix);

		std::cout << "!!!!\npatchesMatrix: " << patchesMatrix.size() << ", " << patchesMatrix[0].size() << std::endl;

		KernelImageType::Pointer reconstructedImage = KernelImageType::New();
		KernelImageType::Pointer denominatorImage = KernelImageType::New();
		CreateImage(reconstructedImage, desiredSize[0], desiredSize[1]);
		CreateImage(denominatorImage, desiredSize[0], desiredSize[1]);

		im2feat.GetImageBack(reconstructedImage,denominatorImage,patchesMatrix);
		typedef itk::CastImageFilter< KernelImageType, ImageType > CastFilterType2;
		CastFilterType2::Pointer castFilter2 = CastFilterType2::New();
		castFilter2->SetInput(reconstructedImage);
		castFilter2->Update();
		ImageType::Pointer reconstructedBMP = castFilter2->GetOutput();

		#ifdef FUNCTEST
			char *testpath = (char*)malloc(4 + strlen(filename) + 2);
			if (testpath == NULL)
			{
				std::cerr << "Could not allocate memory for the fullpath. " << std::endl;
				return -1;
			}

			sprintf(testpath, "recons.bmp", filename);

			// Write the result
			WriterType::Pointer pWriter = WriterType::New();
			pWriter->SetFileName(testpath);
			pWriter->SetInput(reconstructedBMP);
			pWriter->Update();
			free(testpath);
			std::cout << "**************************************" << std::endl;

#endif




	}

	return 0;

	*/

	// Now we need to upscale the image to feed it to feature extractor
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

	unsigned int nNewWidth = vnInputSize[0]*scale;
	unsigned int nNewHeight = vnInputSize[1]*scale;

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
	ImageType::Pointer midres = _pResizeFilter->GetOutput();


	typedef itk::ChangeInformationImageFilter <ImageType > SpacingFilterType;
	SpacingFilterType::Pointer SpacingFilter = SpacingFilterType::New();

	ImageType::SpacingType spacing;
	spacing[0] = 1;
	spacing[1] = 1;

	SpacingFilter->ChangeSpacingOn();
	SpacingFilter->SetOutputSpacing(spacing);

	SpacingFilter->SetInput( midres );
	SpacingFilter->Update();
	midres = SpacingFilter->GetOutput();


	// now apply the filters to obtain the patches
	// defining the kernels to be used for the feature extraction
	KernelImageType::Pointer kernel1 = KernelImageType::New();
	KernelImageType::Pointer kernel2 = KernelImageType::New();
	KernelImageType::Pointer kernel3 = KernelImageType::New();
	KernelImageType::Pointer kernel4 = KernelImageType::New();

	CreateKernels(kernel1, kernel2, kernel3, kernel4, scale);

	ConvolutionFilterType::Pointer convolutionFilter1 = ConvolutionFilterType::New();
	convolutionFilter1->SetInput(midres);
	convolutionFilter1->SetKernelImage(kernel1);

	ConvolutionFilterType::Pointer convolutionFilter2 = ConvolutionFilterType::New();
	convolutionFilter2->SetInput(midres);
	convolutionFilter2->SetKernelImage(kernel2);

	ConvolutionFilterType::Pointer convolutionFilter3 = ConvolutionFilterType::New();
	convolutionFilter3->SetInput(midres);
	convolutionFilter3->SetKernelImage(kernel3);

	ConvolutionFilterType::Pointer convolutionFilter4 = ConvolutionFilterType::New();
	convolutionFilter4->SetInput(midres);
	convolutionFilter4->SetKernelImage(kernel4);

	// Now extract the features from the convolved images
	ImageToFeatureConverter<KernelImageType> im2feat(scale, border, overlap, window);
	std::vector<std::vector<KernelElementType> > featureMatrix1;
	std::vector<std::vector<KernelElementType> > featureMatrix2;
	std::vector<std::vector<KernelElementType> > featureMatrix3;
	std::vector<std::vector<KernelElementType> > featureMatrix4;
	std::vector<std::vector<KernelElementType> > patchesMatrix;

	convolutionFilter1->Update();
	im2feat.GetOutput(convolutionFilter1->GetOutput(), featureMatrix1);

	convolutionFilter2->Update();
	im2feat.GetOutput(convolutionFilter2->GetOutput(), featureMatrix2);

	convolutionFilter3->Update();
	im2feat.GetOutput(convolutionFilter3->GetOutput(), featureMatrix3);

	convolutionFilter4->Update();
	im2feat.GetOutput(convolutionFilter4->GetOutput(), featureMatrix4);


	std::vector<std::vector<KernelElementType> > reconFeatureMatrix;

	for(int feat = 0; feat < featureMatrix1.size(); feat++)
	{
		std::vector<KernelElementType> aggregatedFeatures;
		aggregatedFeatures.insert( aggregatedFeatures.end(), featureMatrix1[feat].begin(), featureMatrix1[feat].end() );
		aggregatedFeatures.insert( aggregatedFeatures.end(), featureMatrix2[feat].begin(), featureMatrix2[feat].end() );
		aggregatedFeatures.insert( aggregatedFeatures.end(), featureMatrix3[feat].begin(), featureMatrix3[feat].end() );
		aggregatedFeatures.insert( aggregatedFeatures.end(), featureMatrix4[feat].begin(), featureMatrix4[feat].end() );
		reconFeatureMatrix.push_back(aggregatedFeatures);
	}

	std::cout << "r2" << std::endl;

	Eigen::MatrixXf featureMat(reconFeatureMatrix[0].size(), reconFeatureMatrix.size());
	for(unsigned int i = 0; i < featureMat.rows(); i++)
	{
		for(unsigned int j = 0; j < featureMat.cols(); j++)
		{
			featureMat(i,j) = reconFeatureMatrix[j].at(i);
		}
	}

	Eigen::MatrixXf reducedReconFeatures = V_pca.transpose() * featureMat;

	matD_t reducedReconFeat(reducedReconFeatures.cols(), vecD_t(reducedReconFeatures.rows()));

	// convert the features matrix from Eigen representation
	// to the matD_t representation.
	for(unsigned int i = 0; i < reducedReconFeatures.cols(); i++)
	{
		for(unsigned int r = 0; r < reducedReconFeatures.rows(); r++)
		{
			reducedReconFeat[i].at(r) = reducedReconFeatures(r,i);
		}
	}

	std::cout << "pre omp info:\n1): " << reducedReconFeat[0].size() << ", " << reducedReconFeat.size()
				<< " \n2): " << dictionary[0].size() << ", " << dictionary.size()
				<< std::endl;

	int N2 = dictionarySize;
	int w_p = reducedReconFeat.size();
	matD_t ormp_val (w_p, vecD_t ());
	matU_t ormp_ind (w_p, vecU_t ());
	//vecD_t normCol     (N2);
	matU_t omega_table     (N2, vecU_t ());
	vecU_t omega_size_table(N2, 0);
	matD_t alpha           (N2, vecD_t ());
	ormp_process(reducedReconFeat, dictionary, ormp_ind, ormp_val, dictionarySize, eps);

	/*for (unsigned i = 0; i < w_p; i++)
	{
		iterU_t it_ind = ormp_ind[i].begin();
		iterD_t it_val = ormp_val[i].begin();
		const unsigned size = ormp_val[i].size();
		for (unsigned j = 0; j < size; j++, it_ind++, it_val++)
			(*it_val) *= normCol[*it_ind];
	}*/

	//! Residus
	for (unsigned i = 0; i < N2; i++)
	{
		omega_size_table[i] = 0;
		omega_table[i].clear();
		alpha[i].clear();
	}

	for (unsigned i = 0; i < w_p; i++)
	{
		iterU_t it_ind = ormp_ind[i].begin();
		iterD_t it_val = ormp_val[i].begin();
		for (unsigned j = 0; j < ormp_val[i].size(); j++, it_ind++, it_val++)
		{
			omega_table[*it_ind].push_back(i);
			omega_size_table[*it_ind]++;
			alpha[*it_ind].push_back(*it_val);
		}
	}


	matD_t gamma(w_p, vecD_t (dictionarySize,0));
	// USE omega_table, omega_size_table, and alpha information
	// above to build the gamma matrix
	// the size of the gamma matrix should be (sizeofdict)x(numofpatches)
	for(unsigned i = 0; i < N2; i++)
	{
		for(unsigned j = 0; j < omega_size_table[i]; j++)
		{
			unsigned pI = omega_table[i].at(j);
			float alphaV = alpha[i].at(j);
			gamma[pI].at(i) = alphaV;
		}
	}

	Eigen::MatrixXf reconQMatrix(dictionarySize, w_p);
	for(unsigned int i = 0; i < reconQMatrix.rows(); i++)
	{
		for(unsigned int j = 0; j < reconQMatrix.cols(); j++)
		{
			reconQMatrix(i,j) = gamma[j].at(i);
		}
	}

	//std::cout << "12" << std::endl << reconQMatrix << std::endl;


	//reconQMatrıx ıs ıncorrect because reconstructedPatches
	//are full of NaN
	Eigen::MatrixXf reconstructedPatches = A_h * reconQMatrix;

	//Krzysztof added thıs because mın and max were NaN
	//std::cout << "reconQMatrıx" << std::endl;
	//std::cout << reconQMatrix << std::endl;   //<--Thıs ıs full of zeros so ıt means that what was added ıs always true

	float mmax = reconstructedPatches.maxCoeff();
	float mmin = reconstructedPatches.minCoeff();



	std::cout << mmax << " ! " << mmin << std::endl;

	std::vector<std::vector<KernelElementType> > reconPatchesVector(reconstructedPatches.cols(), std::vector<KernelElementType>(reconstructedPatches.rows()));
	// convert the reconstructed patches to vector<vector<PixelType>>
	for(unsigned int i = 0; i < reconstructedPatches.cols(); i++)
	{
		for(unsigned int r = 0; r < reconstructedPatches.rows(); r++)
		{
			reconPatchesVector[i].at(r) = reconstructedPatches(r,i);
		}
	}

	std::cout << "13" << std::endl;

	KernelImageType::Pointer reconstructedImage = KernelImageType::New();
	KernelImageType::Pointer denominatorImage = KernelImageType::New();
	CreateImage(reconstructedImage, nNewWidth, nNewHeight);
	CreateImage(denominatorImage, nNewWidth, nNewHeight);

	std::cout << "14" << std::endl;
	im2feat.GetImageBack(reconstructedImage, denominatorImage,reconPatchesVector);


	// Now add the reconstructed image to the previous interpolated
	// image (that is midres)
	typedef itk::CastImageFilter< ImageType, KernelImageType > Int2FloatFilterType;
	Int2FloatFilterType::Pointer i2fFilter = Int2FloatFilterType::New();
	i2fFilter->SetInput(midres);
	i2fFilter->Update();


	typedef itk::AddImageFilter <KernelImageType > AddImageFilterType;
	AddImageFilterType::Pointer addFilter = AddImageFilterType::New ();
	addFilter->SetInput1(i2fFilter->GetOutput());
	addFilter->SetInput2(reconstructedImage);
	addFilter->Update();

	typedef itk::CastImageFilter< KernelImageType, ImageType > CastFilterType;
	CastFilterType::Pointer castFilter = CastFilterType::New();
	castFilter->SetInput(addFilter->GetOutput());
	castFilter->Update();
	ImageType::Pointer reconstructedBMP = castFilter->GetOutput();

#ifdef FUNCTEST
	char *testpath = (char*)malloc(4 + strlen(filename) + 2);
	if (testpath == NULL)
	{
		std::cerr << "Could not allocate memory for the fullpath. " << std::endl;
		return -1;
	}

	sprintf(testpath, "recons.bmp", filename);

	// Write the result
	WriterType::Pointer pWriter = WriterType::New();
	pWriter->SetFileName(testpath);
	pWriter->SetInput(reconstructedBMP);
	pWriter->Update();
	free(testpath);
	std::cout << "**************************************" << std::endl;
#endif


	std::cout << "voila" << std::endl;
	return 0;
}

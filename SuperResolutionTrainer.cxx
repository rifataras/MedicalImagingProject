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

#include "ImageToFeatureConverter.h"
#include "ImageToFeatureConverter.cxx" // !!! HACK! fix this later !!

#include </home/rifat/workspace/CourseProjectFiles/MedicalImagingProject/Eigen/Core>
#include </home/rifat/workspace/CourseProjectFiles/MedicalImagingProject/Eigen/Eigen>

// project specific preprocessor definitions
//#define FUNCTEST // define this variable to test the functionality correctness


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
typedef float					KernelElementType;
typedef itk::Image<KernelElementType, 2>	KernelImageType;
typedef itk::ConvolutionImageFilter<ImageType, KernelImageType, KernelImageType> ConvolutionFilterType;


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

	itksys::Directory trainDir;
	if(!trainDir.Load(argv[1]))
	{
		std::cerr << "Could not open the directory. " << std::endl;
		return -1;
	}

	int scale = ::atoi(argv[2]);
	int window = ::atoi(argv[3]);
	int overlap = ::atoi(argv[4]);
	int border = ::atoi(argv[5]);

	int numberOfFiles = trainDir.GetNumberOfFiles();
	ReaderType::Pointer reader = ReaderType::New();

	// The feature vector from the convolved images
	std::vector<std::vector<KernelElementType> > globalFeatureMatrix;

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

		// Get the individual feature matrices from the filtered images
		// Later on, these matrices will be aggregated to the global
		// feature matrix
		convolutionFilter1->Update();
		im2feat.GetOutput(convolutionFilter1->GetOutput(), featureMatrix1);

		/**/
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
			std::cout << imageIteratorMidres.Get() << " ";
			++imageIteratorMidres;
		}
		std::cout  << std::endl << "END Content of midres *******" << std::endl;

		std::cout  << std::endl << "convolution 1 output:" << std::endl;
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

		/*for(int hh = 0; hh < featureMatrix1.size(); hh++)
		{
			for(int tt = 0; tt < featureMatrix1[hh].size(); tt++)
			{
				std::cout << featureMatrix1[hh].at(tt) << ", ";
			}
			std::cout << "! "<<hh<< "! " <<std::endl;
		}*/

		convolutionFilter2->Update();
		im2feat.GetOutput(convolutionFilter2->GetOutput(), featureMatrix2);

		convolutionFilter3->Update();
		im2feat.GetOutput(convolutionFilter3->GetOutput(), featureMatrix3);

		convolutionFilter4->Update();
		im2feat.GetOutput(convolutionFilter4->GetOutput(), featureMatrix4);


		for(int feat = 0; feat < featureMatrix1.size(); feat++)
		{
			std::vector<KernelElementType> aggregatedFeatures;
			aggregatedFeatures.insert( aggregatedFeatures.end(), featureMatrix1[feat].begin(), featureMatrix1[feat].end() );
			aggregatedFeatures.insert( aggregatedFeatures.end(), featureMatrix2[feat].begin(), featureMatrix2[feat].end() );
			aggregatedFeatures.insert( aggregatedFeatures.end(), featureMatrix3[feat].begin(), featureMatrix3[feat].end() );
			aggregatedFeatures.insert( aggregatedFeatures.end(), featureMatrix4[feat].begin(), featureMatrix4[feat].end() );
			globalFeatureMatrix.push_back(aggregatedFeatures);
		}


#ifdef FUNCTEST
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
		pWriter->SetInput(convolutionFilter2->GetOutput());
		pWriter->Update();
		free(testpath);
		std::cout << "**************************************" << std::endl;
#endif

	}

	// call ksvd train dataset
	std::cout << "global feature matrix size = " << globalFeatureMatrix.size() << std::endl;
	if(globalFeatureMatrix.size() > 0)
		std::cout << "feature  size = " << globalFeatureMatrix[0].size() << std::endl;

	unsigned int dimSpace = 10; // dimension space
	unsigned int m = globalFeatureMatrix[0].size();   // dimension of each point
	unsigned int n = globalFeatureMatrix.size();  // number of points

	Eigen::MatrixXf DataPoints(m,n);
	for (int j=0; j<DataPoints.cols(); ++j) // loop over columns
		for (int i=0; i<DataPoints.rows(); ++i) // loop over rows
			DataPoints(i,j) = globalFeatureMatrix[j].at(i);

	float mean;
	Eigen::VectorXf meanVector;

	typedef std::pair<float, int> myPair;
	typedef std::vector<myPair> PermutationIndices;


	//
	// for each point
	//   center the poin with the mean among all the coordinates
	//
	for (int i = 0; i < DataPoints.cols(); i++)
	{
	   mean = (DataPoints.col(i).sum())/m;		 //compute mean
	   meanVector  = Eigen::VectorXf::Constant(m,mean); // create a vector with constant value = mean
	   DataPoints.col(i) -= meanVector;
	   // std::cout << meanVector.transpose() << "\n" << DataPoints.col(i).transpose() << "\n\n";
	}

	// get the covariance matrix
	Eigen::MatrixXf Covariance = Eigen::MatrixXf::Zero(m, m);
	Covariance = (1 / (float) n) * DataPoints * DataPoints.transpose();
	//std::cout << Covariance ;

	// compute the eigenvalue on the Cov Matrix
	Eigen::EigenSolver<Eigen::MatrixXf> m_solve(Covariance);
	std::cout << "PCA Done";
	Eigen::VectorXf eigenvalues = Eigen::VectorXf::Zero(m);
	eigenvalues = m_solve.eigenvalues().real();

	Eigen::MatrixXf eigenVectors = Eigen::MatrixXf::Zero(n, m);  // matrix (n x m) (points, dims)
	eigenVectors = m_solve.eigenvectors().real();

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
		//std::cout << "eigen=" << pi[i].first << " pi=" << pi[i].second << std::endl;
	}

	float eigenValuesSum = sortedEigenValues.sum();
	//eigenValuesCumPerc = eigenValuesCumSum.cwiseQuotient(sortedEigenValues);
	for (unsigned int i = 0; i < m ; i++)
	{
		eigenValuesCumPerc(i) = eigenValuesCumSum(i) / eigenValuesSum;
		std::cout << "eigencumsum=" << eigenValuesCumPerc(i) << "\n";
	}


	// reconstruction:
	// Patch = meanvector + SIGMA(eigenvalues(i) * eigenvectors(i))

















	return 0;
}

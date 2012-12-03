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

#include "itkSubtractImageFilter.h"

#include "itkConvolutionImageFilter.h"

#include "ImageToFeatureConverter.h"
#include "ImageToFeatureConverter.cxx" // !!! HACK! fix this later !!

#include "itkCastImageFilter.h"

#include "itkAddImageFilter.h"

#include "itkChangeInformationImageFilter.h"

#include "lib_ormp.h"
#include "lib_svd.h"

#include </home/rifat/workspace/CourseProjectFiles/MedicalImagingProject/Eigen/Core>
#include </home/rifat/workspace/CourseProjectFiles/MedicalImagingProject/Eigen/Eigen>

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

/***********KSVD***************************************/
typedef std::vector<std::vector<float> > matD_t;
typedef std::vector<std::vector<unsigned> > matU_t;
typedef std::vector<float> vecD_t;
typedef std::vector<unsigned> vecU_t;
typedef std::vector<float>::iterator iterD_t;
typedef std::vector<unsigned>::iterator iterU_t;


void CreateImage(KernelImageType::Pointer image, int width, int height)
{
	std::cout << "21" << std::endl;
	// Create an image with 2 connected components
	KernelImageType::IndexType start;
	start[0] = 0;
	start[1] = 0;

	std::cout << "22" << std::endl;
	KernelImageType::SizeType size;
	size[0] = width;
	size[1] = height;

	std::cout << "23" << std::endl;
	KernelImageType::RegionType region(start, size);

	std::cout << "24" << std::endl;
	image->SetRegions(region);
	std::cout << "25" << std::endl;
	image->Allocate();
	std::cout << "26" << std::endl;
	image->FillBuffer( itk::NumericTraits<KernelImageType::PixelType>::Zero);
	std::cout << "27" << std::endl;
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

/**
 * @brief Obtain random permutation for a tabular
 *
 * @param perm : will contain a random sequence of [1, ..., N]
 *               where N is the size of perm.
 *
 * @return none.
 **/
void randperm(vecU_t &perm)
{
    //! Initializations
    const unsigned N = perm.size();
    vecU_t tmp(N + 1, 0);
    tmp[1] = 1;
    srand(unsigned(time(NULL)));

    for (unsigned i = 2; i < N + 1; i++)
    {
        unsigned j = rand() % i + 1;
        tmp[i] = tmp[j];
        tmp[j] = i;
    }

    iterU_t it_t = tmp.begin() + 1;
    for (iterU_t it_p = perm.begin(); it_p < perm.end(); it_p++, it_t++)
        (*it_p) = (*it_t) - 1;
}

/**
 * @brief Obtain the initial dictionary, which
 *        its columns are normalized
 *
 * @param dictionary : will contain random patches from patches,
 *                     with its columns normalized;
 * @param patches : contains all patches in the noisy image.
 *
 * @return none.
 **/
void obtain_dict(matD_t         &dictionary,
                 matD_t const&   patches)
{
    //! Declarations
    vecU_t perm(patches.size());

    std::cout << "1" << std::endl;

    //! Obtain random indices
    randperm(perm);

    std::cout << "2" << std::endl;

    //! Getting the initial random dictionary from patches
    iterU_t it_p = perm.begin();
    for (matD_t::iterator it_d = dictionary.begin(); it_d < dictionary.end();
                                                                    it_d++, it_p++)
        (*it_d) = patches[*it_p];

    std::cout << "3" << std::endl;
    //! Normalize column
    double norm;
    for (matD_t::iterator it = dictionary.begin(); it < dictionary.end(); it++)
    {
        norm = 0.0l;
        for (iterD_t it_d = (*it).begin(); it_d < (*it).end(); it_d++)
            norm += (*it_d) * (*it_d);

        norm = 1 / sqrtl(norm);
        for (iterD_t it_d = (*it).begin(); it_d < (*it).end(); it_d++)
            (*it_d) *= norm;
    }
}

/**
 * @brief Apply the whole algorithm of K-SVD
 *
 * @param img_noisy : pointer to an allocated array containing
 *                    the original noisy image;
 * @param img_denoised : pointer to an allocated array which
 *                       will contain the final denoised image;
 * @param patches : matrix containing all patches including in
 *                  img_noisy;
 * @param dictionary : initial random dictionary, which will be
 *                     updated in each iteration of the algo;
 * @param sigma : noise value;
 * @param N1 : size of patches (N1 x N1);
 * @param N2 : number of atoms in the dictionary;
 * @param N_iter : number of iteration;
 * @param gamma : value used in the correction matrix in the
 *                case of color image;
 * @param C : coefficient used for the stopping criteria of
 *            the ORMP;
 * @param width : width of both images;
 * @param height : height of both images;
 * @param chnls : number of channels of both images;
 * @param doReconstruction : if true, do the reconstruction of
 *                           the final denoised image from patches
 *                           (only in the case of the acceleration
 *                            trick).
 *
 * @return none.
 **/
void ksvd_process(matD_t        &patches,
                  matD_t        &dictionary,
                  matD_t		&alpha,
                  const unsigned N1, // size of features (i.e. 324)
                  const unsigned N2, // size of the dictionary (i.e. 1000)
                  const unsigned N_iter, // i.e. 40
                  const double   C)
{
	//! Declarations
	const unsigned N1_2 = N1;
	const double   corr = 0; //(sqrtl(1.0l + gamma) - 1.0l) / ((double) N1_2);
	const unsigned chnls = 1;
	const double   eps  = ((double) (N1_2)) * C * C;
	const unsigned h_p  = patches[0].size();
	const unsigned w_p  = patches.size();

	//! Mat & Vec initializations
	matD_t dict_ormp   (N2 , vecD_t(h_p, 0.0l));
	matD_t patches_ormp(w_p, vecD_t(h_p, 0.0l));
	matD_t tmp         (h_p, vecD_t(N2, 0.0l));
	vecD_t normCol     (N2);
	matD_t Corr        (h_p, vecD_t(h_p, 0.0l));
	vecD_t U           (h_p);
	vecD_t V;
	matD_t E           (w_p, vecD_t(h_p));

	//! Vector for ORMP
	matD_t ormp_val        (w_p, vecD_t ());
	matU_t ormp_ind        (w_p, vecU_t ());
	matD_t res_ormp        (N2, vecD_t (w_p));
	matU_t omega_table     (N2, vecU_t ());
	vecU_t omega_size_table(N2, 0);
	//matD_t alpha           (N2, vecD_t ()); // this is a function parameter

	//! To avoid reallocation of memory
	for (unsigned k = 0; k < w_p; k++)
	{
		ormp_val[k].reserve(N2);
		ormp_ind[k].reserve(N2);
	}

	for (matU_t::iterator it = omega_table.begin(); it < omega_table.end(); it++)
		it->reserve(w_p);

	V.reserve(w_p);

	//! Correcting matrix
	for (unsigned i = 0; i < h_p; i++)
		Corr[i][i] = 1.0l;

	for (unsigned c = 0; c < 1; c++)
	{
		matD_t::iterator it_Corr = Corr.begin() + N1_2 * c;
		for (unsigned i = 0; i < N1_2; i++, it_Corr++)
		{
			iterD_t it = it_Corr->begin() + N1_2 * c;
			for (unsigned j = 0; j < N1_2; j++, it++)
				(*it) += corr;
		}
	}

	#pragma omp parallel for
	for (unsigned j = 0; j < w_p; j++)
	{
		for (unsigned c = 0; c < chnls; c++)
		{
			iterD_t it_ormp = patches_ormp[j].begin() + c * N1_2;
			iterD_t it = patches[j].begin() + c * N1_2;
			for (unsigned i = 0; i < N1_2; i++, it++, it_ormp++)
			{
				double val = 0.0l;
				iterD_t it_tmp = patches[j].begin() + c * N1_2;
				for (unsigned k = 0; k < N1_2; k++, it_tmp++)
					val += corr * (*it_tmp);
				(*it_ormp) = val + (*it);
			}
		}
	}

	//! Big loop
	for (unsigned iter = 0; iter < N_iter; iter++)
	{
		std::cout << "Step " << iter + 1 << ":" << std::endl;
		std::cout << " - Sparse coding" << std::endl;

		for (unsigned i = 0; i < h_p; i++)
		{
			iterD_t it_tmp = tmp[i].begin();
			for (unsigned j = 0; j < N2; j++, it_tmp++)
			{
				double val = 0.0l;
				iterD_t it_corr_i = Corr[i].begin();
				iterD_t it_dict_j = dictionary[j].begin();
				for (unsigned k = 0; k < h_p; k++, it_corr_i++, it_dict_j++)
					val += (*it_corr_i) * (*it_dict_j);
				(*it_tmp) = val * val;
			}
		}

		iterD_t it_normCol = normCol.begin();
		for (unsigned j = 0; j < N2; j++, it_normCol++)
		{
			double val = 0.0l;
			for (unsigned i = 0; i < h_p; i++)
				val += tmp[i][j];
			(*it_normCol) = 1.0l / sqrtl(val);
		}

		for (unsigned i = 0; i < h_p; i++)
		{
			iterD_t it_normCol_j = normCol.begin();
			for (unsigned j = 0; j < N2; j++, it_normCol_j++)
			{
				double val = 0.0l;
				iterD_t it_corr_i  = Corr[i].begin();
				iterD_t it_dict_j = dictionary[j].begin();
				for (unsigned k = 0; k < h_p; k++, it_corr_i++, it_dict_j++)
					val += (*it_corr_i) * (*it_dict_j);
				dict_ormp[j][i] = val * (*it_normCol_j);
			}
		}

		//! ORMP process
		std::cout << " - ORMP process" << std::endl;
		ormp_process(patches_ormp, dict_ormp, ormp_ind, ormp_val, N2, eps);

		for (unsigned i = 0; i < w_p; i++)
		{
			iterU_t it_ind = ormp_ind[i].begin();
			iterD_t it_val = ormp_val[i].begin();
			const unsigned size = ormp_val[i].size();
			for (unsigned j = 0; j < size; j++, it_ind++, it_val++)
				(*it_val) *= normCol[*it_ind];
		}

		//! Residus
		for (unsigned i = 0; i < N2; i++)
		{
			omega_size_table[i] = 0;
			omega_table[i].clear();
			alpha[i].clear();
			for (iterD_t it = res_ormp[i].begin(); it < res_ormp[i].end(); it++)
				*it = 0.0l;
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
				res_ormp[*it_ind][i] = *it_val;
			}
		}

		//! Dictionary update
		std::cout << " - Dictionary update" << std::endl;
		for (unsigned l = 0; l < N2; l++)
		{
			//! Initializations
			const unsigned omega_size = omega_size_table[l];
			iterD_t it_dict_l = dictionary[l].begin();
			iterD_t it_alpha_l = alpha[l].begin();
			iterU_t it_omega_l = omega_table[l].begin();
			U.assign(U.size(), 0.0l);

			if (omega_size > 0)
			{
				iterD_t it_a = it_alpha_l;
				iterU_t it_o = it_omega_l;
				for (unsigned j = 0; j < omega_size; j++, it_a++, it_o++)
				{
					iterD_t it_d = it_dict_l;
					iterD_t it_e = E[j].begin();
					iterD_t it_p = patches[*it_o].begin();
					for (unsigned i = 0; i < h_p; i++, it_d++, it_e++, it_p++)
						(*it_e) = (*it_p) + (*it_d) * (*it_a);
				}

				matD_t::iterator it_res = res_ormp.begin();
				for (unsigned k = 0; k < N2; k++, it_res++)
				{
					iterU_t it_o = it_omega_l;
					iterD_t it_dict_k = dictionary[k].begin();
					for (unsigned j = 0; j < omega_size; j++, it_o++)
					{
						const double val = (*it_res)[*it_o];
						if (fabs(val) > 0.0l)
						{
							iterD_t it_d = it_dict_k;
							iterD_t it_e = E[j].begin();
							for (unsigned i = 0; i < h_p; i++, it_d++, it_e++)
								(*it_e) -= (*it_d) * val;
						}
					}
				}

				//! SVD truncated
				V.resize(omega_size);
				double S = svd_trunc(E, U, V);

				dictionary[l] = U;

				it_a = it_alpha_l;
				iterD_t it_v = V.begin();
				it_o = it_omega_l;
				for (unsigned j = 0; j < omega_size; j++, it_a++, it_v++, it_o++)
					res_ormp[l][*it_o] = (*it_a) = (*it_v) * S;
			}
		}
		std::cout << " - done." << std::endl;
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

	unsigned int dimSpace = 10; // dimension space
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
	Eigen::MatrixXf V_pca(m, numHighestEigenVectors);
	for(unsigned int i = 0; i < numHighestEigenVectors; i++)
	{
		V_pca.col(i) << highestEigenVectors[i];
	}

	Eigen::MatrixXf features_pca = V_pca.transpose() * DataPoints;
	std::cout << "features_pca: " << features_pca.rows() << "," << features_pca.cols() << "\n";

	for(unsigned int i = 0; i < features_pca.cols(); i++)
	{
		std::cout << "norm of " << i << ": "<<features_pca.col(i).norm() << std::endl;
	}

	/******************************************************/
	/***********KSVD***************************************/

	const int dictionarySize = 5;
	matD_t reducedFeatures(features_pca.cols(), vecD_t(features_pca.rows()));
	matD_t dictionary(dictionarySize, vecD_t(features_pca.rows()));
	matD_t alpha(dictionarySize, vecD_t ());

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

	const double C = 1.1139195378939404;
	const int numOfIterations = 10;
	const double   eps  = ((double) (features_pca.rows())) * C * C;
	ksvd_process(reducedFeatures, dictionary, alpha, features_pca.rows(), dictionarySize, numOfIterations, C);


//	std::cout << std::endl << "!!printing dictionary!!" << std::endl;
//	for(int i = 0; i < dictionary.size(); i++)
//	{
//		for(int j = 0; j < dictionary[0].size(); j++)
//		{
//			std::cout << dictionary[i].at(j) << ", ";
//		}
//		std::cout << std::endl;
//	}

	std::cout << std::endl << "alpha matrix size: " << alpha.size() << ", " << alpha[0].size() << std::endl;
	std::cout << std::endl << "patches matrix size: " << globalPatchMatrix.size() << ", " << globalPatchMatrix[0].size() << std::endl;

	// Find hires  dictionary
	// convert the matrices to Eigen representation
	// from the matD_t representation for Eigen's operations.
	Eigen::MatrixXf Q_Matrix(alpha.size(), alpha[0].size());
	Eigen::MatrixXf P_Matrix(globalPatchMatrix[0].size(), globalPatchMatrix.size());

	for(unsigned int i = 0; i < Q_Matrix.rows(); i++)
	{
		for(unsigned int j = 0; j < Q_Matrix.cols(); j++)
		{
			Q_Matrix(i,j) = alpha[i].at(j);
		}
	}

	//std::cout << "here is Q:" << std::endl << Q_Matrix << std::endl;

	for(unsigned int i = 0; i < P_Matrix.rows(); i++)
	{
		for(unsigned int j = 0; j < P_Matrix.cols(); j++)
		{
			P_Matrix(i,j) = globalPatchMatrix[j].at(i);
		}
	}

	//std::cout << "here is P:" << std::endl << P_Matrix << std::endl;

	Eigen::MatrixXf QQT = Q_Matrix * Q_Matrix.transpose();
	Eigen::MatrixXf A_h = P_Matrix * Q_Matrix.transpose() * QQT.inverse();

	//std::cout << "here is Ah:" << std::endl << A_h << std::endl;



	// NOW the reconstruction phase


	const char *filename = "Clipboard01.bmp";
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

	int w_p = reducedReconFeat.size();
	matD_t ormp_val (w_p, vecD_t ());
	matU_t ormp_ind (w_p, vecU_t ());
	std::cout << "pre omp info: 1): " << reducedReconFeat[0].size() << ", " << reducedReconFeat.size()
			<< " \n2): " << dictionary[0].size() << ", " << dictionary.size()
			<< std::endl;
	ormp_process(reducedReconFeat, dictionary, ormp_ind, ormp_val, dictionarySize, eps);

	std::cout << "\"q\" values size: " << ormp_val[0].size() << ", " << ormp_val.size() << std::endl;

	Eigen::MatrixXf reconQMatrix(ormp_val[0].size(), ormp_val.size());
	for(unsigned int i = 0; i < reconQMatrix.rows(); i++)
	{
		for(unsigned int j = 0; j < reconQMatrix.cols(); j++)
		{
			//std::cout << i << ", " << j << ": " << ormp_val[j].at(i) << std::endl;
			if(ormp_val[j].size() < ormp_val[0].size())
			{
				reconQMatrix(i,j) = 0;
			}
			else
			{
				reconQMatrix(i,j) = ormp_val[j].at(i);
			}
		}
	}

	std::cout << "12" << std::endl;

	Eigen::MatrixXf reconstructedPatches = A_h * reconQMatrix;

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

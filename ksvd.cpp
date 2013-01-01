#include "ksvd.h"
#include <stdlib.h>
#if defined(_WIN32) || defined(_WIN64)

#include <time.h>

#endif
#include <math.h>

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

    //! Obtain random indices
    randperm(perm);

    //! Getting the initial random dictionary from patches
    iterU_t it_p = perm.begin();
    for (matD_t::iterator it_d = dictionary.begin(); it_d < dictionary.end();
                                                                    it_d++, it_p++)
        (*it_d) = patches[*it_p];

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
                  matD_t		&gamma,
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
	matD_t alpha           (N2, vecD_t ()); // this is a function parameter

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
	for (int j = 0; j < w_p; j++)
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



}

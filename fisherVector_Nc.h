#ifndef _FISHER_VECTOR_ENCODING_H_
#define _FISHER_VECTOR_ENCODING_H_

#include "cxcore.h"
#include "cv.h"

#include "vl/gmm.h"
#include "vl/host.h"
#include "vl/kmeans.h"
#include "vl/fisher.h"
#include "vl/vlad.h"

#include "formatBinaryStream.h"
#include "waitKeySeconds.h"

using namespace cv;

const int _maxiterKM = 100;
const int _ntrees = 5;
const int _maxComp = 100;
const float _gmmSigmaLowerBound = 1e-5;
const int _maxGmmRep =  1;
const int _maxGmmIter = 50;

enum Init {KMeans, Rand, Custom};
class fvEncoding {

	VlKMeans			**_kmeans;
	VlGMM 				**_gmm;
	//vl_size				numData;
	int					_numClusters;
	int					_channels;  //number of channels
	Init				_init;  //fv initialization. 0 = KMeans(reading randData100kc.dat to do kmeans and gmm ), 
								//1 = Rand (random clustering ) 
								// 2 = Custom (default, load by reading pre-computed gmms )

	float				*_ft;
	int					*_fvStep;   //FV feature dimension for each channel,
	int					_fvDim;   //FV feature dimension for feature vector,

	int					*_dimension;   //feature dimension for each channel  (original non-FV feature),
	int					_dataDims;   //total dims of feature (original non-FV feature)

	
/*	
	BinClusterInStream	*_wordsFile;  //file to input bag of words
	Mat					*_bWords;	//bag of words
	
	
	int					*_wordNums;   //word beginning number  for each channel. if 3 channels, first is "0", 
									//	second is _bwords[0].col, third is _bwords[0].col+_bwords[1].col. Total number of words = _wordNums[_channels]
	int					*_ftStep0;   //feature dimension for each channel before pca projection,
	int					*_ftStep;   //feature dimension for each channel,

	int					*_step;
	DescriptorMatcher	**_matcher;   //opencv matcher for matching vectors to BoWs

	bagWordsFeature (const bagWordsFeature &q) {}  //fake copy 
	bagWordsFeature  &operator= (const bagWordsFeature &q) {return *this;}  //fake assignment

	PCA				    **_pca;  //using pca to reduce dimension
	int					*_maxComponents; // specify how many principal components to retain
	bool				_usePca;

	int					_colDims;   //total dims of _bWords[i].col, if pca, it equal to total reduced dim

	//bool				_sameWordDims;  // =1, if every channel has same number of words. For fast normalization
	int					_samples; // total number of samples per "getFeatures(hog3dFt0)". It equals to hog3dFt0.rows.
*/

public:
	fvEncoding (): _kmeans(NULL), _gmm(NULL),  
				_dimension(NULL), _fvStep(NULL),  _ft(NULL), _init(Custom)
	{
	}

	fvEncoding(const string *iFile, int channels, const int* dim = NULL, int numCluster = 256, int mTp = 2, string* gmmSaveName = NULL):_channels(channels)
    {
		_gmm = new VlGMM*[channels];
		_dimension = new int[channels];
		_kmeans = NULL;
		_numClusters = numCluster;
		_fvStep = new int[channels];
		_fvDim = 0;
		_dataDims = 0;
		//vl_set_num_threads(0) ; /* use the default number of threads */

		if(mTp==0 )
		{
			_init = KMeans;
			_kmeans = new VlKMeans*[channels];
			for (int i = 0; i < channels; i++)
			{
				_dimension[i] = dim[i];
				_gmm[i] = vl_gmm_new (VL_TYPE_FLOAT, dim[i], _numClusters);

				_kmeans[i] = vl_kmeans_new( VL_TYPE_FLOAT, VlDistanceL2);

				vl_kmeans_set_verbosity	(_kmeans[i],1);
				vl_kmeans_set_max_num_iterations (_kmeans[i], _maxiterKM) ;
				vl_kmeans_set_max_num_comparisons (_kmeans[i], _maxComp) ;
				vl_kmeans_set_num_trees (_kmeans[i], _ntrees);

				// Use Lloyd algorithm
				vl_kmeans_set_algorithm (_kmeans[i], VlKMeansLloyd) ;
				//vl_kmeans_set_algorithm (_kmeans[i], VlKMeansANN);

				// Initialize the cluster centers by randomly sampling the data
				vl_kmeans_set_initialization(_kmeans[i], VlKMeansPlusPlus);
				//vl_kmeans_set_initialization(kmeans, VlKMeansRandomSelection);

				// Run at most _maxiterKM iterations of cluster refinement using Lloyd algorithm
				vl_kmeans_set_max_num_iterations (_kmeans[i], _maxiterKM) ;
				//vl_kmeans_refine_centers (kmeans, data, numData) ;

				vl_gmm_set_initialization (_gmm[i],VlGMMKMeans);
				vl_gmm_set_kmeans_init_object(_gmm[i],_kmeans[i]);
			} 
		}
		else if (mTp==1)
		{
			_init = Rand;
			for (int i = 0; i < channels; i++)
			{
				_dimension[i] = dim[i];
				_gmm[i] = vl_gmm_new (VL_TYPE_FLOAT, dim[i], _numClusters);
				vl_gmm_set_initialization (_gmm[i],VlGMMRand);
			}
		}
		else  //default, loading pre-computed gmm from the saved files
		{
			_init = Custom;

			for (int i = 0; i < channels; i++)
			{
				loadGMM(iFile[i], &_gmm[i]);
				_dimension[i] = vl_gmm_get_dimension(_gmm[i]) ;
				_numClusters = vl_gmm_get_num_clusters(_gmm[i]) ;
				_fvStep[i] = 2*_dimension[i]*_numClusters;
				_fvDim += _fvStep[i];
				_dataDims += _dimension[i];
			}

		}

		if(_init != Custom)
		{
			float * data;
			Mat cluMat;
			//vl_size numData;
			//vl_size dimension;
		
			for (int i = 0; i < channels; i++)
			{
				BinClusterInStream *inFl;
				inFl = new BinClusterInStream (iFile[i] );
				inFl->read(cluMat);
				CV_Assert(_dimension[i] == cluMat.cols);
			
				data = (float*)vl_malloc(sizeof(float)*cluMat.rows*_dimension[i]);
				opencvMat2vl(data, cluMat, 0);
				delete inFl;

				vl_gmm_set_max_num_iterations (_gmm[i], _maxGmmIter) ;
				vl_gmm_set_num_repetitions(_gmm[i], _maxGmmRep);
				vl_gmm_set_verbosity(_gmm[i], 1);
				vl_gmm_set_covariance_lower_bound (_gmm[i], _gmmSigmaLowerBound);

				vl_gmm_cluster (_gmm[i], data, cluMat.rows);
				cluMat.release();
				vl_free(data);

				_fvStep[i] = 2*_dimension[i]*_numClusters;
				_fvDim += _fvStep[i];
				_dataDims += _dimension[i];

				if (gmmSaveName)
					saveGMM(gmmSaveName[i], _gmm[i]);  //save gmm results

				if(_kmeans)	
					vl_kmeans_delete(_kmeans[i]);

			}
		}
		delete []_kmeans;
		_kmeans = NULL;
		_ft = new float[_fvDim];
		//memset(_ft, 0, sizeof(int)*_fvDim);

	 }



	void getFeatures(Mat hog3dFt0)
	{
		Mat *buffer0 = new Mat[_channels];
		for(int ch = 0; ch < _channels; ch++)
			buffer0[ch] = Mat(hog3dFt0.rows, _dimension[ch], hog3dFt0.type());
		int col0 = 0;
		for(int ch = 0; ch < _channels;  ch++)
		{
			(hog3dFt0.colRange(col0, col0 + _dimension[ch])).copyTo(buffer0[ch]);
			col0 += _dimension[ch];
		}

		float *pt = _ft;
		float *data;
		
		for(int ch = 0; ch < _channels; ch++)
		{
			if(buffer0[ch].isContinuous() )
				data = buffer0[ch].ptr<float>(0);
			else
			{
				data = new float[buffer0[ch].cols*buffer0[ch].rows];
				for (int i = 0; i < buffer0[ch].rows; i++)
				{
					float *p0 = buffer0[ch].ptr<float>(i);
					memcpy(data, p0, buffer0[ch].cols*sizeof(float));
					data += buffer0[ch].cols;
				}

			}
			 
			vl_fisher_encode(pt, VL_TYPE_FLOAT, vl_gmm_get_means(_gmm[ch]), _dimension[ch], _numClusters, vl_gmm_get_covariances(_gmm[ch]),
							vl_gmm_get_priors(_gmm[ch]), data, buffer0[ch].rows, VL_FISHER_FLAG_IMPROVED );
			pt += _fvStep[ch];

			if(!buffer0[ch].isContinuous() )
				delete []data;
		}
		delete []buffer0;
	}

	void getFeatures(const Mat &hog3dFt, float *arr)
	{
		getFeatures(hog3dFt);
		memcpy(arr, _ft, sizeof(float)*_fvDim);

	}

	inline int getFVdim() const 
	 {
		 return _fvDim;
	 }


	inline int getFtDim() const  //non-FV feature dimension
	 {
		 int dim = _dimension[0];
		 for(int i = 1; i < _channels; i++)
			 dim += _dimension[i];
		 return dim;
	 }
	
	 ~fvEncoding()
	 {
		 if(_gmm)
		 {
			 for(int i = 0; i < _channels; i++)
				vl_gmm_delete(_gmm[i]);
			 delete []_gmm;
		 }

		 delete []_ft;
		 delete []_fvStep;
		 delete []_dimension;

		 if(_kmeans)
		 {
			 for(int i = 0; i < _channels; i++)
				vl_kmeans_delete(_kmeans[i]);
			delete[]_kmeans;
		 }
	 }

	 
//Write gmm to gmmfl.yml
void saveGMM(const std::string& gmmFl, VlGMM * gmm)
{
	int dimension = vl_gmm_get_dimension(gmm) ;
	int numClusters = vl_gmm_get_num_clusters(gmm) ;
	int dataType = vl_gmm_get_data_type(gmm) ;
	float const * sigmas = (float*)vl_gmm_get_covariances(gmm) ;
	float const * means = (float*)vl_gmm_get_means(gmm) ;
	float const * weights = (float*)vl_gmm_get_priors(gmm) ;
	float const * posteriors = (float*)vl_gmm_get_posteriors(gmm) ;

	CV_Assert(dataType != VL_TYPE_DOUBLE);
	cv::FileStorage fs(gmmFl, cv::FileStorage::WRITE);
	cv::Mat sigM, meanM, wM, posM;

std::cout<<"transfer float pointer to array...\n";

	vl2opencvMat(sigmas, sigM, numClusters, dimension);
	vl2opencvMat(means, meanM, numClusters, dimension);
	vl2opencvMat(weights, wM, numClusters, 1);
	//vl2opencvMat(posteriors, posM, numClusters, dimension);

	fs << "Dimension" << dimension;
	fs << "NumClusters" << numClusters;
	fs << "DataType" << dataType;
std::cout<<"writint Mat...\n"<<meanM.size()<<" "<<sigM.size()<<" "<<wM.size()<<" "<<posM.size();	
	fs << "Means" << meanM;
	fs << "Sigmas" << sigM;
	fs << "Weights" << wM;
	//fs << "Posteriors" << posM;
std::cout<<"done writing\n";
	fs.release();

}

//Load gmm from gmmfl.yml
void loadGMM(const std::string& gmmFl, VlGMM ** gmm, int maxiter = 50, int maxrep = 1, double sigmaLowerBound = 1e-5)
{
	
	float  * sigmas;
	float  * means;
	float * weights;
	float * posteriors;

	cv::FileStorage fs(gmmFl, cv::FileStorage::READ);
	cv::Mat sigM, meanM, wM, posM;


	vl_size dimension = (int)(fs["Dimension"]);
	vl_size numClusters = (int)fs["NumClusters"];
	vl_size dataType = (int)fs["DataType"];
	CV_Assert(dataType != VL_TYPE_DOUBLE);
		
	fs ["Means"] >> meanM;
	fs ["Sigmas"] >> sigM;
	fs ["Weights"] >> wM;
	//fs ["Posteriors"] >> posM;
	fs.release();

	VlGMM * tmpGmm = vl_gmm_new (dataType, dimension, numClusters);

	float * initSigmas;
    float * initMeans;
    float * initWeights;

    initSigmas = (float*)vl_malloc(sizeof(float) * numClusters * dimension);
    initWeights = (float*)vl_malloc(sizeof(float) * numClusters);
    initMeans = (float*)vl_malloc(sizeof(float) * numClusters * dimension);

    vl_gmm_set_initialization (tmpGmm, VlGMMCustom);
	
	opencvMat2vl(initSigmas, sigM, 0);
	opencvMat2vl(initMeans, meanM, 0);
	opencvMat2vl(initWeights, wM, 0);
	//opencvMat2vl(posteriors, posM, 0);

    vl_gmm_set_priors(tmpGmm,initWeights);
    vl_gmm_set_covariances(tmpGmm,initSigmas);
    vl_gmm_set_means(tmpGmm,initMeans);

	vl_gmm_set_max_num_iterations (tmpGmm, maxiter) ;
	vl_gmm_set_num_repetitions(tmpGmm, maxrep);
	vl_gmm_set_verbosity(tmpGmm,1);
	vl_gmm_set_covariance_lower_bound (tmpGmm,sigmaLowerBound);

	*gmm = tmpGmm;

}


void vl2opencvMat(const float* data, cv::Mat& rst, int rows, int cols)  //copy data to openCV Mat
{
	rst = cv::Mat(rows, cols, CV_32FC1);
	if(rst.isContinuous())
		memcpy(rst.ptr<float>(0), data, rows*cols*sizeof(float));
	else //copy one col each time
	{
		const float *pt0=data;
		int step = cols*sizeof(float);
		for(int i = 0; i < rst.rows; pt0 += cols, i++)
		{
			float *pt = rst.ptr<float>(i);
			memcpy(pt, pt0, step);
		}
	}

	//transpose(rst, rst);  //default vl data format is in column major order, while opencv matrix is in row major order
}


void opencvMat2vl(float* rst, cv::Mat src, int dim=0)  //copy opencv Mat to array, the array for data must already assigned 
{
	if(dim)
		CV_Assert(dim == src.rows * src.cols);

	//cv::Mat mt;
	//cv::transpose(src, mt);  //default vl data format is in column major order, while opencv matrix is in row major order
	int rows= src.rows, cols = src.cols;
	
	if(src.isContinuous())
	{
		cols = cols * rows;
		rows = 1;
	}

	float *pt0=rst;
	int step = cols*sizeof(float);


	for(int i = 0; i < rows; pt0+=cols, i++)
	{
		
		float *pt = src.ptr<float>(i);
		memcpy(pt0, pt, step);
	}

}


};  //end of class bagWordsFeature

#endif //_FISHER_VECTOR_ENCODING_H_

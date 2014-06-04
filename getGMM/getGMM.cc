#include "cxcore.h"
#include "cv.h"
#include "highgui.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include "stDescriptor.h"
#include "formatBinaryStream.h"
#include <windows.h> 
#include <string>
#include <cstdlib>
#include <atlbase.h>  //for A2W(lpr);
#include <string>
#include <time.h>
#include "waitKeySeconds.h"
#include "fisherVector_Nc.h"


const int classNum = 51; //number of total classes
const int dataNum = 150000; //number of total features for finding clusters by kmeans. 
							//due to the too many training features(over millions), we normally random-choose "dataNum" from all training features
const int _maxFrames = 160; //maxium frames per video. if the video has more frames, it needs to split the video to do multiple processing to avoid memory overflow
const int pSamples = 100;  //number of features chosen from each clip to train gmm

const int _rootSz = 64;   //root size
const int _partsSz = 64*8;  //part size 
const int _maxChnl = 4;  //2 channels = root channel + part channel; if 4 channels, MBHx(root + part) + MBHy(root + part)
const int _pcaRtSz = 32;  //reduce the root dim from 64 to 32
const int _pcaPsSz = 64;   //reduce the part dim from 64*8 to 64

const int const _Dim[4] = {32, 64, 32, 64};
const int _numClusters = 128;  //K = 128 
const string _pcaMat = "pca_Mat.yml";  //precomputed pca projective matrix
using namespace cv;

//This programs learn GMM and pca projection matrix for FV by kmeans from training videos

int main() 
{
	int chnl = 1;
	std::cout<<"How many channels(1-4)? \nInput: ";
	std::cin>>chnl;
	std::cout<<"\n";

	RNG seed(unsigned(time(NULL)));
	MBHparam * para = new MBHparam();
	if (!para->readParam("MBH_parameters_input.txt", 1))
	{
		std::cout<<"use default HOG3D parameters instead.\n";
		discoverUO::wait();
	}

	stDetector dscpt(para); 
	para->writeParam("MBH_parameters_clustering.txt");
	delete para;

	char tstr[5];
	int clsNum = classNum;
	while (clsNum <= 0)
	{
		std::cout<<"Pleas input total number of classes : \n";
		cin>>clsNum;
		std::cout<<std::endl;
	}

	int start, end, numClusters;

	int *rDataNum = new int[clsNum];
	//computing how many randomly chosen features from each training class
	rDataNum[0] = dataNum/classNum;
	for (int i = 1; i < classNum-1; i++)
		rDataNum[i] = rDataNum[0];
	rDataNum[classNum-1] = dataNum - rDataNum[0]*(classNum-1);

	//randomly choose 120000 features from the computed training features
	BinClusterInStream *iFile;
	int rowNo = 0;
	Mat roi;
	//int maxFrames = ((int)(para->rt2ps.z + 0.3)) ? _maxFrames : _maxFrames*1.5;
	int maxFrames = _maxFrames;

	bool flipIm4Training = 0;
	bool kmeanCluster = 0;
	
	Mat feature, cluMat;

	BinClusterOutStream<float> *ofile, *rawFile = NULL;

	//delete ofile;

	string fileName = "binOut.dat";
	string dirName = "c:\\Feng\\hmdb51\\";

	string fullName;
	string rFileNm = "randData100kc.dat";
	string dNm;

	int label;
	char sName[1024];
	string spFile = "split1.txt";
	ifstream spFileIn;

//compute the features from training videos
	for (int i = 0; i < clsNum; i++)
	{
		itoa(i+1, tstr, 10);
		dNm = dirName + (string)tstr + (string)"\\";
		fullName = dNm + spFile;
		ofile = new BinClusterOutStream<float> (fileName);

		spFileIn.open(fullName.c_str());
		if (!spFileIn.is_open())
		{
			std::cout<<"Unable to open split flie \""<<fullName<<"\" for reading the file name of training/testing split!\n";
			discoverUO::wait();
			exit(-1);
		}
//std::cout<<"folder name: "<<dNm<<"full split file name: "<<fullName<<endl;
		while (spFileIn>>sName>>label)
		{
			//cout<<sName<<"  label = "<<label<<endl;
			if (label == 1)
			{
				seed.next();
				fullName = dNm + sName;
				//cout<<fullName<<endl;
				if(!dscpt.preProcessing(fullName, maxFrames))
				{
					std::cout<<"Unable to process loaded video for computing training features!\n";
					discoverUO::wait();
					exit(-1);
				}

				dscpt.getRandomFeatures(feature, pSamples, seed);
				int height = feature.rows, width = feature.cols;
				float *data;
				for (int i0 = 0; i0 < height; i0++)
				{
					data = feature.ptr<float>(i0);
					ofile->write(data, width);
					//ofile->write((float*)feature.ptr(i), width);
				}

				int redoNum;
				if(redoNum = dscpt.reProcessNum())
				{
					for (int i0 = 1; i0 <= redoNum; i0++)
					{
						dscpt.re_Processing(fullName, maxFrames, i0);
						dscpt.getRandomFeatures(feature, pSamples, seed);
						height = feature.rows; 
						width = feature.cols;
						for (int j0 = 0; j0 < height; j0++)
						{
							data = feature.ptr<float>(j0);
							ofile->write(data, width);
							//ofile->write((float*)feature.ptr(i), width);
						}
					}
				}

				cout<<"Done video file: "<< fullName<<endl;
			}
		}
		spFileIn.clear();
		spFileIn.close();
		delete ofile;
		feature.release();
	//cout<<"done split file! i = " <<i<<endl;
//randomly choose 200000 features from the computed training features
		iFile = new BinClusterInStream (fileName);
		iFile->read(cluMat, rDataNum[i]);
		//std::cout<<cluMat.cols<<" "<<cluMat.rows<<" "<<cluMat.type()<<"\n";
		delete iFile;

		if (i == 0)
			rawFile = new BinClusterOutStream<float> (rFileNm);

		int height = cluMat.rows, width = cluMat.cols;
		for (int j = 0; j < height; j++)
		{
			float *data = cluMat.ptr<float>(j);
			rawFile->write(data, width);
		}
		cluMat.release();
	cout<<"done the folder! i = " <<i<<endl;
	}
	delete rawFile;
	delete []rDataNum;
	dscpt.clear();
//reading randomly chosen 200000 features from file
	iFile = new BinClusterInStream (rFileNm);
	iFile->read(cluMat);
	delete iFile;

/************************Now doing pca reduction****************/
	if(chnl==4)
		CV_Assert(2*_rootSz+2*_partsSz == cluMat.cols);
	else
		CV_Assert(_rootSz+_partsSz == cluMat.cols);  //chnl==2;
	Mat cMat[4], oMat[4], mCols;
	PCA pca0[4], pcaOut[4];
	Mat eigenval[4],eigenvec[4],mean[4];
	int maxComponent[4];

	Range colRg[4];
	if(chnl==2)
	{
		colRg[0] = Range(0, _rootSz);
		colRg[1] = Range(_rootSz, cluMat.cols);
	}
	else if(chnl==4)
	{
		colRg[0] = Range(0, _rootSz);
		colRg[1] = Range(_rootSz, _rootSz+_partsSz);
		colRg[2] = Range(_rootSz+_partsSz, 2*_rootSz+_partsSz);
		colRg[3] = Range(2*_rootSz+_partsSz, cluMat.cols);
	}
	else 
		exit(-1);

	maxComponent[0] = _pcaRtSz;
    maxComponent[1] = _pcaPsSz;
	maxComponent[2] = _pcaRtSz;
    maxComponent[3] = _pcaPsSz;

	for(int i = 0; i < chnl; i++)
	{
		cout<<"Now doing pca matrices training for channel "<<i<<"...\n";
		pca0[i](cluMat.colRange(colRg[i]), noArray(), CV_PCA_DATA_AS_ROW, maxComponent[i]);
	}


	cout<<"Now doing pca projection and writing the result data...\n";
	string fName[4] = {"randData100kc0.dat", "randData100kc1.dat", "randData100kc2.dat", "randData100kc3.dat"};
	BinClusterOutStream<float> *ofiles;
	float *data1;
	for(int i0 = 0; i0 < chnl; i0++)
	{
		pca0[i0].project(cluMat.colRange(colRg[i0]), oMat[i0]);

		ofiles= new BinClusterOutStream<float> (fName[i0]);
		for (int j = 0; j < oMat[i0].rows; j++)
		{
			data1= oMat[i0].ptr<float>(j);
			ofiles->write(data1, oMat[i0].cols);
		}
		delete ofiles;
	}

	cout<<"Now writing pca matrices...\n";
	//Write matrices to pca_mat.yml
	FileStorage fs("pca_Mat.yml", FileStorage::WRITE);
	fs << "number of channels" << chnl;
	fs << "Mean0" << pca0[0].mean;
	fs << "Eigenvalues0" << pca0[0].eigenvalues;
	fs << "Eigenvector0" << pca0[0].eigenvectors;
	fs << "Mean1" << pca0[1].mean;
	fs << "Eigenvalues1" << pca0[1].eigenvalues;
	fs << "Eigenvector1" << pca0[1].eigenvectors;
	if(chnl == 4)
	{
		fs << "Mean2" << pca0[2].mean;
		fs << "Eigenvalues2" << pca0[2].eigenvalues;
		fs << "Eigenvector2" << pca0[2].eigenvectors;
		fs << "Mean3" << pca0[3].mean;
		fs << "Eigenvalues3" << pca0[3].eigenvalues;
		fs << "Eigenvector3" << pca0[3].eigenvectors;
	}
	fs.release();
	cout<<"done writing pca matrices!";
	/************************Now doing GMM leaning****************/
	
	
cout<<"now performing GMM learning...\n";
	string  gmmFile[4]; 
	char tmpCs[10];
	for (int i = 0; i < chnl; i++)
	{
		//fName[i] = (string)"randData100kc"  + (string)itoa(i,tmpCs,10) + (string)".dat";
		gmmFile[i] = (string)"gmmResults"  + (string)itoa(i,tmpCs,10) + (string)".yml";

	}
	
	fvEncoding fvFt(fName, chnl, _Dim, _numClusters, 0, gmmFile);  //test reading leant gmm files

	discoverUO::wait();
	//std::cin.get();
	//cin.get();
	return 0;
}
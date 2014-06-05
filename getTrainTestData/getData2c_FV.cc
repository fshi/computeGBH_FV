#include "cxcore.h"
#include "cv.h"
#include "highgui.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <windows.h> 
#include <string>
//#include <atlbase.h>  //for A2W(lpr);
#include <time.h>
#include <direct.h>  // for _mkdir( "\\testtmp" );

#include "waitKeySeconds.h"
//#include "opencv.hpp"
#include <iostream>
#include <fstream>
#include "formatBinaryStream.h"
#include "stDescriptor.h"
#include "fisherVector_Nc.h"
 
//#include <opencv2/gpu/gpu.hpp>
//#define _GET_PROCESS_TIME_

using namespace cv;

const int _runNum = 1; 
//const int _runNum = 3; //for random feature seletion, run 3 times
const int _channels = 2;
const int classNum = 51; //number of total classes
//const int classNum = 101; //number of total classes
const int _maxFrames = 160; //maxium frames per video. if the video has more frames, it needs to split the video to do multiple processing to avoid memory overflow and vector overflow(patcher sampling from video)
const int const _Dim[4] = {32, 64, 32, 64};

const int _rootSz = 64;   //root size
const int _partsSz = 64*8;  //part size 

const int _numClusters = 128;  //K = 128 
const string _pcaMat = "pca_Mat.yml";  //precomputed pca projective matrix
//const int const *_pcaDim = NULL;
//const int _matchTp = 2;

bool loadPCA(PCA* pca0, const string& preComputedFl);

int main() {

	printf ("OpenCV version %s (%d.%d.%d)\n",
	    CV_VERSION,
	    CV_MAJOR_VERSION, CV_MINOR_VERSION, CV_SUBMINOR_VERSION);

	//USES_CONVERSION;  //for A2W(lpr) and W2A(...);

	RNG seed[3];
	seed[0] = RNG(unsigned(time(NULL)));
	seed[1] = RNG(seed[0].next());
	seed[2] = RNG(seed[0].next());

#ifdef _GET_PROCESS_TIME_
	clock_t starts, ends;
	clock_t start0, end0, start1, end1, start2, end2;
	double dura, dur0 = 0, dur1 = 0, dur2 = 0; 
#endif

	int samNum = 10000;
	if(samNum < 1000 || samNum > 100000)
	{
		std::cout<<"How many samples do you want to use?  \nInput: ";
		std::cin>>samNum;
		std::cout<<"\n";
		if (samNum<1000 || samNum>100000)
		{
			std::cout<<"Wrong number of samples input! please re-input number of samples again (100-10000): ";
			std::cin>>samNum;
			std::cout<<"\n";
		}
	}


	//input parameters
	MBHparam * para = new MBHparam();
	if (!para->readParam("MBH_parameters_input.txt", 1))
	{
		std::cout<<"use default HOG3D parameters instead.\n";
		discoverUO::wait();
	}
	stDetector dscpt(para); 
	
	para->writeParam("MBH_parameters.txt");
	delete para;

	int maxFrames = _maxFrames;
	int clsNum = classNum;
	if(clsNum <= 0)
	{
		std::cout<<"Pleas input total number of classes : \n";
		cin>>clsNum;
		std::cout<<std::endl;
	}

	int trainSt = 0, testSt = 0; //beginning point for train data and test data
	int trainEnd = clsNum, testEnd = clsNum;

	std::cout<<"Please input the start number of training dir (0 - number of classes)? \n";
	std::cout<<"If \"0\", means using all training classes. If \"number of classes\", means doing no training computing. \nInput: ";
	std::cin>>trainSt;
	std::cout<<"\n";
	if (trainSt < 0)
		trainSt = 0;
	if (trainSt < clsNum)
	{
		std::cout<<"Please input the end number of training dir (0 - number of classes)? \n";
		std::cout<<"If \"0\",  means doing no training computing. \nInput: ";
		std::cin>>trainEnd;
		std::cout<<"\n";
	}
	
	int rootSz = dscpt.toRootFtSz();
	int partsSz = dscpt.toPartsFtSz();

	std::fstream fRst;
	fRst.open("MBH_parameters.txt", ios::app | std::ios::out);
	if (!fRst.is_open())
	{
		std::cerr<<"Error opening file to write data dimensions!\n";
		discoverUO::wait();
	}
	else
	{
		fRst<<"\n\n********************************************************\n";
		fRst<<"\tRoot Size of descript feature is: "<<rootSz<<"\n";
		fRst<<"\tPart Size of descript feature is: "<<partsSz<<"\n";
		fRst<<"\tThe vector dimension of descript feature is: "<<rootSz+partsSz<<"\n";
	}

	char tstr[10], tstr1[10], tmpC[10], tmpCs[10];

	//const string dirName = "C:\\Feng\\ucf101\\";
	//const string ofDirNm = "H:\\Feng\\ucf101_of\\";  //oftical flow files
	const string dirName = "C:\\Feng\\hmdb51\\";
	//const string dirName = "C:\\Feng\\hmdb51_warp\\";
	
	string fName[_channels],  gmmFile[_channels];

	for (int i = 0; i < _channels; i++)
	{
		//fName[i] = (string)"randData100kc"  + (string)itoa(i,tmpCs,10) + (string)".dat";  //input training data for computing GMMs
		//gmmFile[i] = (string)"gmmResults"  + (string)itoa(i,tmpCs,10) + (string)".yml";
		fName[i] = (string)"gmmResults"  + (string)itoa(i,tmpCs,10) + (string)".yml";   //pre-computed GMMs
	}
	
	fvEncoding fvFt(fName, _channels, _Dim, _numClusters, 2, NULL);   //use pre-computed GMMs
	//fvEncoding fvFt(fName, _channels, _Dim, _numClusters, 0, gmmFile); // compute GMMs with input files(fName)
	int fvDim = fvFt.getFVdim();
	cout<<"Done initializing GMM!"<<endl;

	PCA pca0[_channels];
	if(!loadPCA(pca0, _pcaMat))
	{
		cout<<"The input precomputed pca file is wrong!"<<endl;
		discoverUO::wait();
		exit(-1);
	}
	int pcaCols[_channels+1], ftCols[_channels+1];
	pcaCols[0] = 0; 
	ftCols[0] = 0;

	for(int i = 0; i < _channels; i++)
	{
		pcaCols[i+1] = _Dim[i] + pcaCols[i];
		ftCols[i+1] = ((i%2) ?  partsSz : rootSz ) + ftCols[i]; // 0%2 == 0; 1%2 == 1
	}

	string dNm, dName;
	string inFileNm[2];

	float *arr[_runNum];
	Mat feature2[_runNum];

	string fileName[_runNum];
	string fileTp = ".avi",  dName2[_runNum];
	char sName[1024];
	int redoNum;

	HANDLE hFind;
	WIN32_FIND_DATA FileData;

	BinClusterOutStream<float> *ofile[_runNum];
	for (int i = 0; i < _runNum; i++)
	{
		ofile[i] = 0;
		arr[i] = new float[fvDim];
	}

	for (int i = 0; i < _runNum; i++)
	{
		itoa(i+1, tstr, 10);
		dName2[i] = (string)"run"+(string)tstr;
		_mkdir(dName2[i].c_str());
	}


#ifdef _GET_PROCESS_TIME_
	starts = clock();
#endif

	for (int i = trainSt; i < trainEnd; i++)
	{
		itoa(i+1, tstr, 10);
		for (int j = 0; j < _runNum; j++)
		{
			itoa(j+1, tstr1, 10);
			dName2[j] = (string)"run"+(string)tstr1+(string)"\\random"+(string)tstr;
			_mkdir(dName2[j].c_str());
		}
	
		
		dNm = dirName + (string)tstr + (string)"\\";
		cout<<"Now doing folder: "<<dNm<<endl;
		hFind = FindFirstFile((dNm+"*.avi").c_str(), &FileData);

		while (hFind != INVALID_HANDLE_VALUE)
		{
			strcpy(sName, dNm.c_str());
			strcat(sName, FileData.cFileName);
			std::cout<<"Processing file: "<<sName<<endl;

			if(!dscpt.preProcessing(sName, maxFrames))
			{
				std::cout<<"Unable to process loaded video for  computing training features 2!\n";
				discoverUO::wait();
				exit(-1);
			}
			for (int j = 0; j < _runNum; j++)
			{
				dscpt.getRandomFeatures(feature2[j], samNum, seed[j]);
			}
				
			if(redoNum = dscpt.reProcessNum())   //if redoNum("dscpt.reProcessNum()") is not equal to zero
			{
				int dSz0, dSz1;
				std::cout<<"redo...\n";
				for (int i0 = 1; i0 <= redoNum; i0++)
				{
					dSz0 = dscpt.getSamplingSz();
					dscpt.re_Processing(sName, maxFrames, i0);
					
					dSz1 = dscpt.getSamplingSz();
					Mat tmp0;
					for (int j = 0; j < _runNum; j++)
					{
						if (i0 < redoNum)
							dscpt.getRandomFeatures(tmp0, samNum, seed[j]);
						else
							dscpt.getRandomFeatures(tmp0, ((float)dSz1/(float)dSz0)*samNum, seed[j]);
						
						feature2[j].push_back(tmp0);
					}
				}
			}
			for (int j = 0; j < _runNum; j++)
			{
				Mat tmpPCA = Mat(feature2[j].rows, pcaCols[_channels], feature2[j].type());
				for(int i0 = 0; i0 < _channels; i0++)
				{
					Mat vec = feature2[j].colRange(ftCols[i0], ftCols[i0+1]);
					Mat coeffs = tmpPCA.colRange(pcaCols[i0], pcaCols[i0+1]);
					pca0[i0].project(vec, coeffs);

				}
				fvFt.getFeatures(tmpPCA, arr[j]);
				feature2[j].release();
			}
				
			for (int j = 0; j < _runNum; j++)
			{
				fileName[j] = dName2[j] + (string)"\\" + (string)(FileData.cFileName);
				fileName[j].replace(fileName[j].find(fileTp),fileTp.length(),".dat");
				ofile[j] = new BinClusterOutStream<float> (fileName[j]);
				ofile[j]->write(arr[j], fvDim);
				delete ofile[j];
			}
			if (FindNextFile(hFind, &FileData) == 0) break; // stop when none left
		}
	}
	
#ifdef _GET_PROCESS_TIME_
	ends = clock();
	dura = double((ends-starts))/ CLOCKS_PER_SEC;
	cout<<"Prcessed total frames : "<<frameCount<<" in "<<dura<<" seconds."<<endl;
	cout<<"It is equal to: "<<double(frameCount)/dura<<" frames per second."<<endl;
#endif
	for (int i = 0; i < _runNum; i++)
	{
		delete []arr[i];
	}

	discoverUO::wait();
	return 0;
}

bool loadPCA(PCA* pca0, const string& preComputedFl)
{
	cv::FileStorage fs(preComputedFl, cv::FileStorage::READ);
	if(!fs.isOpened())
	{
		cout<<"Can't open pca precomputed file: "<<preComputedFl<<endl;
		return false;
	}
	cv::Mat eigenval[_channels],eigenvec[_channels],mean[_channels];
	int chnl = (int)fs ["number of channels"];
	//std::cout<<chnl<<" "<<_channels;
	if(chnl != _channels)
	{
		cout<<"Wong! The precomputed pca file has different number of channels!"<<endl;
		return false;
	}

	{
		fs [ "Mean0"] >> pca0[0].mean;
		fs["Eigenvalues0"] >> pca0[0].eigenvalues;
		fs["Eigenvector0"] >> pca0[0].eigenvectors;
		fs [ "Mean1"] >> pca0[1].mean;
		fs["Eigenvalues1"] >> pca0[1].eigenvalues;
		fs["Eigenvector1"] >> pca0[1].eigenvectors;

	}
	if(chnl == 4)
	{
		fs [ "Mean2"] >> pca0[2].mean;
		fs["Eigenvalues2"] >> pca0[2].eigenvalues;
		fs["Eigenvector2"] >> pca0[2].eigenvectors;
		fs [ "Mean3"] >> pca0[3].mean;
		fs["Eigenvalues3"] >> pca0[3].eigenvalues;
		fs["Eigenvector3"] >> pca0[3].eigenvectors;

	}
	fs.release();
	return true;
}

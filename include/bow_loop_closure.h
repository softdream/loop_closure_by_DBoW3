#ifndef __BOW_LOOP_CLOSURE_H
#define __BOW_LOOP_CLOSURE_H

#include "DBoW3.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <vector>
#include <string>

#include <memory>

#define MINIMUM_FRAMES 5
#define MAX_RESULT_NUM 4

class Bow
{
public:
	static void training( const std::string &file_path, const int total_num ); 
	
	static bool loadVocabulary( const std::string &file_name );
	static const int loopClosureDetect( const cv::Mat& img );

private:
	static std::unique_ptr<DBoW3::Database> db;
	static long key_frame_count;
};

std::unique_ptr<DBoW3::Database> Bow::db;
long Bow::key_frame_count = 0;

bool Bow::loadVocabulary( const std::string &file_name )
{
	std::cout<<"Reading the database ..."<<std::endl;
	DBoW3::Vocabulary vocab( file_name );
	
	if ( vocab.empty() ) {
        	std::cerr << "Vocabulary does not exist." << std::endl;
        	return false;
    	}

	db = std::make_unique<DBoW3::Database>( vocab, false, 0 );

	std::cout<<"database info : "<< *db <<std::endl;

    	std::cout << "Loading a vacabulary ... " << std::endl;
	return true;
}

const int Bow::loopClosureDetect( const cv::Mat& img )
{
	key_frame_count ++;

	// 1. caculate the orb features
	cv::Ptr<cv::Feature2D> detector = cv::ORB::create();
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptor;
	detector->detectAndCompute(img, cv::Mat(), keypoints, descriptor);
	
	// 2. add to database
	db->add( descriptor );

	// 
	if( key_frame_count <= MINIMUM_FRAMES ){
		return -1;
	}

	DBoW3::QueryResults ret;
	db->query( descriptor, ret, MAX_RESULT_NUM ); // 

	std::cout << "searching for image returns " << ret << std::endl << std::endl;
	
	return 1;
}

void Bow::training( const std::string &file_path, const int total_num )
{
	std::vector<cv::Mat> descriptors;
	cv::Ptr<cv::Feature2D> detector = cv::ORB::create();

	cv::Mat img;
	for( int i = 0; i < total_num; i ++ ){
		// 1. get the image from the dataset 
		std::cout<<"reading the image ... "<<std::endl;
		std::string file_name = file_path + std::to_string( i + 1 ) + ".png";
		img = cv::imread( file_name );
		
		// 2. detect ORB features 
		std::cout<<"detecting the ORB features ... "<<std::endl;
		std::vector<cv::KeyPoint> keypoints;
		cv::Mat descriptor;
		
		detector->detectAndCompute( img, cv::Mat(), keypoints, descriptor );
		
		cv::drawKeypoints( img, keypoints, img, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );
		cv::imshow("ORB Features",img);
		cv::waitKey(30);
	
		descriptors.push_back( descriptor );
	}

	// 3. create the vocabulary
	std::cout<<"creating the vacabolary ... "<<std::endl;
	DBoW3::Vocabulary vocab;
	vocab.create( descriptors );
	std::cout<<"vocabulary info : "<<vocab<<std::endl;
	
	// 4. save the vocabulary
	vocab.save( "vocabulary.yml.gz" );
	std::cout<<"done ..."<<std::endl;
}

#endif

#include "bow_loop_closure.h"

int main()
{
	std::cout<<"----------------------- LOOP CLOSURE DETECT ----------------------"<<std::endl;
	
	Bow::loadVocabulary("/home/riki/Test/bow_loop_closure/database/vocab_larger.yml.gz");

	cv::Mat img;
        for( int i = 0; i < 10; i ++ ){
                // 1. get the image from the dataset 
                std::cout<<"reading the image ... "<<std::endl;
                std::string file_name = "/home/riki/Test/bow_loop_closure/data/" + std::to_string( i + 1 ) + ".png";
                img = cv::imread( file_name );

		Bow::loopClosureDetect( img );
        }

	
	return 0;
}

#include "bow_loop_closure.h"

int main()
{
	std::cout<<"---------------------------- BOW TRAINING -------------------------"<<std::endl;

	Bow::training("/home/riki/Test/bow_loop_closure/data/", 10);

	return 0;
}

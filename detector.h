#ifndef DETECTOR_H
#define DETECTOR_H

#include <vector>
#include <iostream>
#include <mutex>
#include <queue>
#include <opencv2/opencv.hpp>

#include "struct.h"


using namespace std;

struct Size {

	int w = 0;

	int h = 0;

	Size(int _w, int _h) : w(_w), h(_h) {};

};

struct bbox {

	public:

		bbox();
		
		bbox(vector <float> & fbbox, Size & size);

		bbox(float c, float s, float x, float y, float x1, float y1);

		int c = 0;

		float s = 0;

		int x = 0;

		int y = 0;

		int x1 = 0;

		int y1 = 0;

};

class detector {

	public:

		detector(string & _json, string & _params, string & _mean,  string & _device, vector <string> & _classes, Size & _size, bool switching);
		
		~ detector();

		void convert(vector <NDArray> & ndout, vector <vector <bbox>> & bboxes);

		int visualize(cv::Mat & in, vector <vector <bbox>> & bboxes);

		int detect(int tid);
		
		int capture(DetectType & dt, string & filename);

		int guard();

		void sw();

		int batch = 1;

		int cbatch = 1;

		bool switching = false;

		map <string, bool> alive;

		Size size;
		
		Symbol net;

		vector <Executor *> E;

		string json, params;

		NDArray mean;

		vector <string> classes;

		vector <cv::Mat> frames;

		vector <cv::Mat> input;
		
		vector <vector <cv::Mat>> tinput = vector <vector <cv::Mat>> (2);

		queue <cv::Mat> MatQueue;
		
		queue <vector <vector <bbox>>> BoxesQueue;

		/*

		map <string, NDArray> args;

		map <string, NDArray> args2;

		map <string, NDArray> auxs;
		
		map <string, NDArray> auxs2;
		
		map <string, NDArray> grad;
		
		map <string, OpReqType> req;

		*/
		
		vector <map <string, NDArray>> args = vector <map <string, NDArray>> (2);

		vector <map <string, NDArray>> auxs = vector <map <string, NDArray>> (2);
		
		map <string, NDArray> grad;
		
		map <string, OpReqType> req;
		
		unsigned int wait = 100;
		
		Context * ctx;

		//map <string, vector <mutex>> lock;
		
		vector <mutex> lock = vector <mutex> (2);

		mutex mMatQueue, mBoxesQueue;

	private:

		void unload(int no);
		void load(int no);
		int request();
		int post(bool v); // visualize

};



#endif

#ifndef FLT_MX_IMAGE_HH
#define FLT_MX_IMAGE_HH

#include <map>
#include <iostream>
#include <mxnet-cpp/MxNetCpp.h>
#include <opencv2/opencv.hpp>
#include "src/mx/shape.h"
#include "src/mx/image.h"
#include "src/debug.h"
#include "src/cv.hh"

using namespace std;
using namespace mxnet::cpp;


/* failed */

inline NDArray flt::mx::image::load(string filename, Context & ctx){
	
	cv::Mat image = imread(filename, cv::IMREAD_COLOR);
	
	return Mat_to_NDArray(image, ctx);
}


inline void flt::mx::image::save(string filename, NDArray &nd, float scale){
	
	cv::Mat mat = NDArray_to_Mat(nd, scale);
	
	cv::imwrite(filename, mat);
}

inline void flt::mx::image::save_1d(string filename, NDArray &nd, float scale){
	
	cv::Mat mat = NDArray_to_Mat_1d(nd, scale);
	
	cv::imwrite(filename, mat);
}
inline void flt::mx::image::load(string filename, NDArray & nd){
	
	cv::Mat image = imread(filename, cv::IMREAD_COLOR);
	
	Shape shape(image.size().height, image.size().width, image.channels());
	
	float * fimg = (float *)image.data;

	nd.SyncCopyFromCPU(fimg, shape.Size());

	NDArray::WaitAll();

	delete fimg;
}

inline Symbol flt::mx::image::decode(Symbol s){

	auto s1 = SwapAxis(s, 0, 1);

	auto s2 = SwapAxis(s1, 1, 2);

	return s2;
}

inline Symbol flt::mx::image::encode(Symbol s){

	auto s1 = SwapAxis(s, 1, 2);

	auto s2 = SwapAxis(s1, 0, 1);

	return s2;
}

inline Symbol flt::mx::image::decodeb(Symbol s){

	auto s1 = SwapAxis(s, 1, 2);

	auto s2 = SwapAxis(s1, 2, 3);

	return s2;
}

inline Symbol flt::mx::image::encodeb(Symbol s){

	auto s1 = SwapAxis(s, 2, 3);

	auto s2 = SwapAxis(s1, 1, 2);

	return s2;
}
/* not checked , but the convert scale need to be aware */

inline void flt::mx::image::saveb(int number, vector <NDArray> &vnd){

	for(int i = 0; i != vnd.size(); ++i)

		save(to_string(i + number) + ".jpg", vnd[i]);

}


inline void flt::mx::image::saveb(string prefix, int number, vector <NDArray> &vnd){

	for(int i = 0; i != vnd.size(); ++i)

		save(prefix + "_" + to_string(i + number) + ".jpg", vnd[i]);

}


inline void flt::mx::image::saveb(string prefix, int number, NDArray &nd, float scale){

	auto shape = Shape(nd.GetShape());

	auto batch = shape[0];

	auto height = shape[1];

	auto width = shape[2];

	auto channels = shape[3];
	
	//cout << "Batch : " << batch << endl;
	
	for(int i = 0; i != batch; ++i){
		
  		NDArray snd = nd.Slice(i, i + 1).Reshape(Shape(height, width, channels));

		save(prefix + "_" + to_string(i) + ".jpg", snd, scale);

	}

}
inline void flt::mx::image::saveb_1d(string prefix, int number, NDArray &nd, float scale){

	auto shape = Shape(nd.GetShape());

	auto batch = shape[0];

	auto height = shape[1];

	auto width = shape[2];

	//auto channels = shape[3];
	
	//cout << "Batch : " << batch << endl;
	
	for(int i = 0; i != batch; ++i){
		
  		NDArray snd = nd.Slice(i, i + 1).Reshape(Shape(height, width));

		save_1d(prefix + "_" + to_string(i) + ".jpg", snd, scale);

	}

}
inline void flt::mx::image::saveb(vector <NDArray> &vnd){

	for(int i = 0; i != vnd.size(); ++i)

		save(to_string(i) + ".jpg", vnd[i]);
}

inline NDArray flt::mx::image::MatVector_to_NDArray(vector <cv::Mat> &v, Context device){

	int h = v[0].size().height;

	int w = v[0].size().width;

	int channel = v[0].channels();

	int b = v.size();

	float * ftotal = new float [b * h * w * channel];

	int size = h * w * channel;

	Shape sim(h, w, channel);

	Shape stotal(b, h, w, channel);

	//cout << "float point : " << flt::fcv::getf(&v[0], 0, 1, 2) << endl;

	for(int i = 0; i != b; i++){

		cv::Mat mat(h, w, CV_32FC3);

		v[i].convertTo(mat, CV_32FC3, 1.0 / 255);

		float * fimg = (float *)mat.data;

		copy(fimg, fimg + size, ftotal + i * size);

	}

	//Shape stotal(b, h, w, channel);

	NDArray ndtotal(stotal, device);

	ndtotal.SyncCopyFromCPU(ftotal, b * size);

	NDArray::WaitAll();

	delete ftotal;
	
	return ndtotal;
}


// check ?

inline void flt::mx::image::MatVector_to_NDArray(NDArray &ndtotal, vector <cv::Mat> &v, Context device){

	int h = v[0].size().height;

	int w = v[0].size().width;

	int channel = v[0].channels();

	int b = v.size();

	float * ftotal = new float [b * h * w * channel];

	int size = h * w * channel;

	Shape sim(h, w, channel);

	Shape stotal(b, h, w, channel);

	//cout << "float point : " << flt::fcv::getf(&v[0], 0, 1, 2) << endl;

	for(int i = 0; i != b; i++){

		cv::Mat mat(h, w, CV_32FC3);

		v[i].convertTo(mat, CV_32FC3, 1.0 / 255);

		float * fimg = (float *)mat.data;

		copy(fimg, fimg + size, ftotal + i * size);

	}

	//shape stotal(b, h, w, channel);

	//NDArray ndtotal(stotal, device);

	ndtotal.SyncCopyFromCPU(ftotal, b * size);

	NDArray::WaitAll();

	delete ftotal;

}



inline void flt::mx::image::Mat_to_NDArray(cv::Mat &mat, NDArray &nd){
	
	int size = mat.size().height * mat.size().width * mat.channels();

	float *fmat = (float *)mat.data;

	float *fimg = new float[size];

	copy(fmat, fmat + size, fimg);
	
	nd.SyncCopyFromCPU(fimg, size);

	NDArray::WaitAll();

}

/* failed , return [] when cout <<  */

inline NDArray flt::mx::image::Mat_to_NDArray(cv::Mat &umat, Context &ctx){
	
	int height = umat.size().height;

	int width = umat.size().width;

	int channel = umat.channels();

	int size = height * width * channel;

	cout << "height : " << height << endl;

	cout << "width : " << width << endl;

	cout << "channels : " << channel << endl;

	cv::Mat fmat(height, width, CV_32FC3);
		
	cout << "uu" << endl;
	/*
	 *	only convert data type
	 *	
	 *	don't scale the pixel range from 0 - 255 to float 0.0 - 1.0
	 *
	 */
	
	umat.convertTo(fmat, CV_32FC3, 1.0 / 255);
	
	NDArray nd(Shape(height, width, channel), ctx);
	
	nd.SyncCopyFromCPU((float *)fmat.data, size);
	
	NDArray::WaitAll();

	return nd;

}

/* checked , it works properly */

inline cv::Mat flt::mx::image::NDArray_to_Mat(NDArray &nd, float scale){
	
	Shape shape = Shape(nd.GetShape());	
	
	int rank = nd.GetShape().size();
	
	int height, width, channel, size;

	if (rank == 3){
		
		height = shape[0];

		width = shape[1];

		channel = shape[2];

	}

	else if (rank == 4){

		height = shape[1];

		width = shape[2];

		channel = shape[3];
	
	}

	size = height * width * channel;
	
	float *fimg = new float[size];

	nd.SyncCopyToCPU(fimg, size);
	
	NDArray::WaitAll();

	/*	NDArray -> float
	 *	
	 *	NDArray -> CV_32FC3 -> CV_8UC3
	 *
	 *	although NDArray accept only float data
	 *
	 *	but we dont make pixel range from 0 - 255 => 0 - 1
	 *
	 * 	we still use 0.0 - 255.0
	 *
	 */
	cv::Mat mat(height, width, CV_32FC3, fimg);
	
	mat.convertTo(mat, CV_8UC3, scale);

	delete fimg;
	
	return mat;
}

inline cv::Mat flt::mx::image::NDArray_to_Mat_1d(NDArray &nd, float scale){

	Shape shape = Shape(nd.GetShape());	
	
	int height = shape[0];

	int width = shape[1];

	int size = height * width;

	float *fimg = new float[size];

	nd.SyncCopyToCPU(fimg, size);
	
	NDArray::WaitAll();

	/*	NDArray -> float
	 *	
	 *	NDArray -> CV_32FC3 -> CV_8UC3
	 *
	 *	although NDArray accept only float data
	 *
	 *	but we dont make pixel range from 0 - 255 => 0 - 1
	 *
	 * 	we still use 0.0 - 255.0
	 *
	 */

	cv::Mat mat(shape[0], shape[1], CV_32FC1, fimg);

	mat.convertTo(mat, CV_8UC1, scale);
	
	return mat;
}

/* not check yet  */

inline vector <cv::Mat> flt::mx::image::NDArray_to_MatVector(NDArray &n){

	vector <mx_uint> shape = n.GetShape();

	int b = shape[0];

	int h = shape[1];

	int w = shape[2];

	int channel = shape[3];

	vector <cv::Mat> v(b);

	float * ftotal = new float [b * h * w * channel];

	int size = h * w * channel;

	Shape sim(h, w, channel);

	Shape stotal(b, h, w, channel);

	n.SyncCopyToCPU(ftotal, b * size);

	NDArray::WaitAll();

	for(int i = 0; i != b; i++){

		float * fim = new float [size];

		copy(ftotal + i * size, ftotal + (i + 1) * size, fim);

		v[i] = cv::Mat(h, w, CV_32FC3, fim);

		delete fim;

	}
	
	delete ftotal; 

	return v;
}
#endif

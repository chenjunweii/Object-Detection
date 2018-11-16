#include <memory>
#include <assert.h>
#include "detector.h"
#include <mutex>
#include <unistd.h>


using namespace std;
using namespace flt::mx::image;

detector::detector(string & _json, string & _params, string & _mean, string & _device, vector <string> & _classes, Size & _size, bool _switching) : classes(_classes), size(_size), switching(_switching) {

	if (_device.compare("gpu") == 0)
	
		ctx = new Context(DeviceType::kGPU, 1);

	else
		
		ctx = new Context(DeviceType::kCPU, 0);

	int way;

	if (switching)

		way = 2;

	else

		way = 1;

	net = Symbol::Load(_json);

	map <string, NDArray> ndc = NDArray::LoadToMap(_params); // ndarray cpu
	map <string, NDArray> ndm = NDArray::Load(_mean); // ndarray mean
	mean = ndm["mean"];

	args[0]["data"] = NDArray(Shape(batch, 3, _size.h, _size.w), *ctx);
	if (switching)
		args[1]["data"] = NDArray(Shape(batch, 3, _size.h, _size.w), *ctx);

	req["data"] = OpReqType::kNullOp;

//	vector <string> nodes = net.ListArguments();
//	vector <string> aux_nodes = net.ListAuxiliaryStates();


	/*for (auto & k : ndc){
		auto type = k.first.substr(0, 3);
		auto node = k.first.substr(4);
		if (type.compare("arg") == 0){
			args[node] = NDArray(Shape(k.second.GetShape()), *ctx);
			//grad[node] = NDArray(Shape(1), ctx);
			//grad[node] = NDArray(Shape(k.second.GetShape()), ctx);
			k.second.CopyTo(&args[node]);
			req[node] = OpReqType::kNullOp;

			if (switching)

				args2[node] = args[node];
		}
		else if (type.compare("aux") == 0){
			auxs[node] = NDArray(Shape(k.second.GetShape()), *ctx);
			k.second.CopyTo(&auxs[node]);
			if (switching)
				auxs2[node] = auxs[node];
		}

	}*/

	for (auto & k : ndc){
		auto type = k.first.substr(0, 3);
		auto node = k.first.substr(4);
		if (type.compare("arg") == 0){
			args[0][node] = NDArray(Shape(k.second.GetShape()), *ctx);
			//grad[node] = NDArray(Shape(1), ctx);
			//grad[node] = NDArray(Shape(k.second.GetShape()), ctx);
			k.second.CopyTo(&args[0][node]);
			req[node] = OpReqType::kNullOp;

			if (switching)

				args[1][node] = args[0][node];
		}
		else if (type.compare("aux") == 0){
			auxs[0][node] = NDArray(Shape(k.second.GetShape()), *ctx);
			k.second.CopyTo(&auxs[0][node]);
			if (switching)
				auxs[1][node] = auxs[0][node];
		}

	}


	/*
	for (auto & k : nodes){
		auto node = string("arg:") + k;
		if (k.compare("data") == 0){
			for (int w = 0; w != way; ++w)
				args[w].push_back(NDArray(Shape(1, 3, _size.h, _size.w), *ctx));

		}

		else{

			args[0].push_back(NDArray(Shape(ndc[node].GetShape()), *ctx));
			grad.push_back(NDArray(Shape(1), *ctx));
			ndc[node].CopyTo(&args[0].back());
			if (switching)
				args[1].push_back(args[0].back());
		}
		req.push_back(OpReqType::kNullOp);
	}

	for (auto & k : aux_nodes){

		auto node = string("aux:") + k;
		
			auxs[0].push_back(NDArray(Shape(ndc[node].GetShape()), *ctx));
			ndc[node].CopyTo(&auxs[0].back());
		
		if (switching)
			auxs[1].push_back(auxs[0].back());
	}*/

	
	/*E = net.SimpleBind(*ctx, args, grad, req, auxs);
	if (switching)
		E2 = net.SimpleBind(*ctx, args2, grad, req, auxs);

	*/
		
	/*for (int i = 0; i != 2; i++)

		E.emplace_back(net.Bind(*ctx, args[i], grad, req, auxs[i]));

	*/
	
	for (int i = 0; i != 2; i++)

		E.emplace_back(net.SimpleBind(*ctx, args[i], grad, req, auxs[i]));

	cout << "Bind" << endl;
}

int detector::guard(){
	
	for (auto & i : alive)

		alive[i.first] = false;

}

inline int detector::capture(DetectType & dt, string & filename){

	cv::Mat resized;

	cv::Mat frame;

	cv::VideoCapture capture;

	if (dt == DetectType::video)
	
		capture = cv::VideoCapture(filename);

	//else if (dt == DetectType::camer)
    
	if (!capture.isOpened())
		
		return -1;

	bool get = false;

    while(true){

		capture >> frame;

		if(frame.empty())

			break;
		
		cv::resize(frame, resized, cv::Size(size.w, size.h));
		
		frames.emplace_back(frame);

		get = false;

		while (! get){

			get = mMatQueue.try_lock();

			if (not get)

				usleep(wait);
		}

		MatQueue.push(resized);

		mMatQueue.unlock();
		
		usleep(wait);

    }

	alive["capture"] = false;

    return 0;	
}

// one thread of caputre
//
// onw threaf for  load 0 + inference 0
//
// ine thread for load 1 + inference 1
//
// one threaf fo post process ; show detection plot figure
//

int detector::post(bool v){

	bool get = false;

	while (! get){
	
		while (BoxesQueue.empty()){

			usleep(wait);
			
		}
		
		get = mBoxesQueue.try_lock();

	}

	vector <vector <bbox>> boxes = BoxesQueue.front();

	BoxesQueue.pop();

	mBoxesQueue.unlock();
}


inline void detector::load(int no){

	cout << "in Load : " << no << endl;

	bool get = false;

	while (! get){
	
		while (MatQueue.empty()){

			usleep(wait);
			
		}

		get = mMatQueue.try_lock();

	}

	/*while (MatQueue.empty()){

		usleep(wait);
		
	}

	lock_guard <mutex> lg (mMatQueue);

	*/

	cv::Mat frame = MatQueue.front();

	MatQueue.pop();

	mMatQueue.unlock();

	tinput[no].emplace_back(frame);

	MatVector_to_NDArray(args[no]["data"], tinput[no], *ctx);

	args[no]["data"] -= args[no]["data"];
	
	tinput[no].clear();

	tinput[no].shrink_to_fit();

	usleep(wait);

}


inline void detector::unload(int no){

	cout << "In UnLoad" << endl;

	lock[no].unlock();

	cout << "out UnLoad" << endl;

}

inline int detector::request(){

	while (true)

		for (int j = 0; j != 2; ++j)
			
			if (lock[j].try_lock()){

				cout << "success get : " << j << endl;
			
				return j;
			}

		usleep(wait);
}

inline int detector::detect(int tid){

	int no;

	bool get = false;
	
	vector <vector <bbox>> boxes;
	
	//or (int i = 0; i != 10; ++i){
	//
	while (true){
		
		no = request();

		load(no);

		get = false;

		cout << "in Detect : " << no <<  endl;

		E[no]->Forward(false);

		cout << "after foward" << endl;

		convert(E[no]->outputs, boxes);

		while (! get){

			cout << "try lock boxes queue " << no << endl;
			
			get = mBoxesQueue.try_lock();

			if (! get)

				usleep(wait);

		}

		//lock_guard <mutex> lg (mBoxesQueue);
		
		BoxesQueue.push(boxes);

		mBoxesQueue.unlock();
		
		unload(no);
		
		cout << "out Detect : " << no << endl;

		boxes.clear();

		boxes.shrink_to_fit();

		usleep(wait);

	}

	cout << "End Detect " << no << endl;

	alive[to_string(tid)] = false;

}

detector::~detector(){

	delete ctx;

	for (auto & e : E)

		delete e;
}


inline void detector::convert(vector <NDArray> & ndout, vector <vector <bbox>> & bboxes){

	vector <mx_uint> shape = ndout[0].GetShape(); // batach, ndets, 6

	int fsize = shape[0] * shape[1] * shape[2];

	vector <float> fout (fsize);

	ndout[0].SyncCopyToCPU(fout.data(), fsize);

	NDArray::WaitAll();

	vector <float> slice;

	for (int i = 0; i != shape[0]; i++){

		(bboxes).emplace_back(vector <bbox> ());

		for (int j = 0; j != shape[1]; j++){

			int base = i * j * 6;

			slice = vector <float> (fout.begin() + base, fout.begin() + base + 6);

			if (slice[0] > 0)
		
				(bboxes)[i].emplace_back(bbox(slice, size));
		}
	}
}

int detector::visualize(cv::Mat & in, vector <vector <bbox>> & bboxes){

	assert(bboxes.size() == 1); // assert batch size = 1

	for (auto & box : bboxes[0]){
		cv::Point ul(box.x, box.y);
		cv::Point br(box.x1, box.y1);
		cv::Point tp(box.x1, box.y1 - 5);
		cv::rectangle(in, ul, br, cv::Scalar(0, 255, 0));
		char text [128];
		sprintf(text, "%s %f", classes[box.c], box.s);
		cv::putText(in, text, tp, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5,  cv::Scalar(0, 0, 255, 255));
	}
}



inline int clip(int x, int lower, int upper) {
  
	return max(lower, min(x, upper));
}

bbox::bbox(){};

bbox::bbox(vector <float> & fout, Size & size){

	c = int(fout[0]); s = fout[1];
	
	x = clip(int(fout[2] * size.w), 0, size.w); y = clip(int(fout[3] * size.h), 0, size.h);
	
	x1 = clip(int(fout[4] * size.w), 0, size.w); y1 = clip(int(fout[5] * size.h), 0, size.h);
}

bbox::bbox(float c, float s, float x, float y, float x1, float y1){

}


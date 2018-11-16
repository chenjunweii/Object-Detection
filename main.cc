#include <iostream>
#include "flt.h"
#include <map>
#include <thread>
#include <vector>
#include "detector.hh"


using namespace std;
using namespace flt::mx::image;
using namespace flt::mx;
using namespace mxnet::cpp;

int main(){

	string device = "gpu";
	
	Size size(512, 512);
	
	string json = "deploy_ssd_inceptionv3_512-symbol.json";
	
	string params = "deploy_ssd_inceptionv3_512-0002.params";

	string mean = "mean.nd";

	vector <string> classes = { "aeroplane", "bicycle", "bird", "boat", "bottle", \
								"bus", "car", "cat", "chair", "cow",
							  "diningtable", "dog", "horse", "motorbike",
                              "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor" };

	detector det(json, params, mean, device, classes, size, true);

	DetectType dt = DetectType::video;

	string video = "av10239720.mp4";

	thread capture (& detector::capture, & det, ref(dt), ref(video));

	int tid_0 = 0; int tid_1 = 1;

    thread det_0 (& detector::detect, & det, ref(tid_0));
    
	thread det_1 (& detector::detect, & det, ref(tid_1));

    capture.join();

    det_0.join();
    
	det_1.join();

	/*

	vector <string> nodes = symbol.ListArguments();

	for (auto & k : nodes)

		cout << k << endl;

	*/

	/*

	cout << Shape(ndc["data"].GetShape()) << endl;

	for (auto & nd : ndc)
	
		map <string, NDArray> ndg = NDArray()
	
	//
	*/

	/*

	Executor * E = symbol.SimpleBind(ctx, args, grad, req, auxs);

	cout << "dawd" << endl;

	E->Forward(false);

	vector <NDArray> ndout = (*E).outputs;

	Size size(512, 512);

	detection det(classes, size);

	cout << Shape(ndout[0].GetShape()) << endl;

	vector <vector <bbox>> bboxes;

	det.convert(ndout, bboxes);

	cout << "bboxes : " << bboxes[0].size() << endl;

	*/



	
	cout << "end" << endl;
	
	MXNotifyShutdown();

	return 0;

}


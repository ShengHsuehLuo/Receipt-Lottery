/*
	Text detection model: https://github.com/argman/EAST
	Download link: https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz?dl=1
	Text recognition models can be downloaded directly here:
	Download link: https://drive.google.com/drive/folders/1cTbQ3nuZG-EKWak6emD_s8_hHXWz7lAr?usp=sharing
	and doc/tutorials/dnn/dnn_text_spotting/dnn_text_spotting.markdown
	How to convert from pb to onnx:
	Using classes from here: https://github.com/meijieru/crnn.pytorch/blob/master/models/crnn.py
	import torch
	from models.crnn import CRNN
	model = CRNN(32, 1, 37, 256)
	model.load_state_dict(torch.load('crnn.pth'))
	dummy_input = torch.randn(1, 1, 32, 100)
	torch.onnx.export(model, dummy_input, "crnn.onnx", verbose=True)
	For more information, please refer to doc/tutorials/dnn/dnn_text_spotting/dnn_text_spotting.markdown and doc/tutorials/dnn/dnn_OCR/dnn_OCR.markdown
*/
/*
#include <iostream>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
using namespace cv;
using namespace cv::dnn;
const char* keys =
"{ help  h              | | Print help message. }"
"{ input i              | | Path to input image or video file. Skip this argument to capture frames from a camera.}"
"{ detModel dmp         | | Path to a binary .pb file contains trained detector network.}"
"{ thr                  | 0.5 | Confidence threshold. }"
"{ nms                  | 0.4 | Non-maximum suppression threshold. }"
"{ width                | 320 | Preprocess input image by resizing to a specific width. It should be multiple by 32. }"
"{ height               | 320 | Preprocess input image by resizing to a specific height. It should be multiple by 32. }"
"{ recModel rmp         | | Path to a binary .onnx file contains trained CRNN text recognition model. "
"Download links are provided in doc/tutorials/dnn/dnn_text_spotting/dnn_text_spotting.markdown}"
"{ RGBInput rgb         |0| 0: imread with flags=IMREAD_GRAYSCALE; 1: imread with flags=IMREAD_COLOR. }"
"{ vocabularyPath vp    | alphabet_36.txt | Path to benchmarks for evaluation. "
"Download links are provided in doc/tutorials/dnn/dnn_text_spotting/dnn_text_spotting.markdown}";
void fourPointsTransform(const Mat& frame, const Point2f vertices[], Mat& result);
int main(int argc, char** argv)
{
	float confThreshold = 0.5;
	float nmsThreshold = 0.0;
	int width = 732;
	int height = 732;
	int imreadRGB = 1;
	String detModelPath = "D:/Programming/Receipt Lottery/Receipt/x64/Release/DB_TD500_resnet50.onnx";
	String recModelPath = "D:/Programming/Receipt Lottery/Receipt/x64/Release/crnn_cs.onnx";
	String vocPath = "D:/Programming/Receipt Lottery/Receipt/x64/Release/alphabet_94.txt";
	String picPath = "D:/Programming/Receipt Lottery/Receipt/x64/Release/11.png";
	bool camera = 0;

	// Load networks.
	CV_Assert(!detModelPath.empty() && !recModelPath.empty());
	//TextDetectionModel_EAST detector(detModelPath);
	//detector.setConfidenceThreshold(confThreshold)
	//	.setNMSThreshold(nmsThreshold);
	TextDetectionModel_DB detector(detModelPath);
	detector.setBinaryThreshold(0.3);
	TextRecognitionModel recognizer(recModelPath);
	// Load vocabulary
	CV_Assert(!vocPath.empty());
	std::ifstream vocFile;
	vocFile.open(samples::findFile(vocPath));
	CV_Assert(vocFile.is_open());
	String vocLine;
	std::vector<String> vocabulary;
	while (std::getline(vocFile, vocLine)) {
		vocabulary.push_back(vocLine);
	}
	recognizer.setVocabulary(vocabulary);
	recognizer.setDecodeType("CTC-greedy");
	// Parameters for Recognition
	double recScale = 1.0 / 127.5;
	Scalar recMean = Scalar(127.5, 127.5, 127.5);
	Size recInputSize = Size(100, 32);
	recognizer.setInputParams(recScale, recInputSize, recMean,true,false);
	// Parameters for Detection
	double detScale = 1.0/255;
	Size detInputSize = Size(width, height);
	Scalar detMean = Scalar(123.68, 116.78, 103.94);
	//Scalar detMean = Scalar(127.5, 127.5, 127.5);
	bool swapRB = true;
	detector.setInputParams(detScale, detInputSize, detMean, swapRB);
	*/
	// Open a video file or an image file or a camera stream.
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <chrono>
#include <ctime>

using namespace cv;
using namespace cv::dnn;
using namespace std;
using namespace std::chrono;

void fourPointsTransform(const Mat& frame, const Point2f vertices[], Mat& result, const Size recInputSize)
{
	//const Size outputSize = Size(100, 32);
	const Size outputSize = recInputSize;
	Point2f targetVertices[4] = {
		Point(0, outputSize.height - 1),
		Point(0, 0), Point(outputSize.width - 1, 0),
		Point(outputSize.width - 1, outputSize.height - 1)
	};
	Mat rotationMatrix = getPerspectiveTransform(vertices, targetVertices);
	warpPerspective(frame, result, rotationMatrix, outputSize);
}
bool isNumber(char c) {
	return (c - '0' >= 0 && c - '9' <= 0);
}
bool isDate(const string& s) {
	if (s.size() != 10) return false;
	if(!isNumber(s[0])) return false;
	if (!isNumber(s[1])) return false;
	if (!isNumber(s[2])) return false;
	if (!isNumber(s[3])) return false;
	if (s[4] != '/' && s[4] != '-' && s[4] != '.') return false;
	if (!isNumber(s[5])) return false;
	if (!isNumber(s[6])) return false;
	if (s[7] != '/' && s[7] != '-' && s[7] != '.') return false;
	if (!isNumber(s[8])) return false;
	if (!isNumber(s[9])) return false;
	return true;
}
bool isReceipt(const string& s) {
	if (s.size() != 10 && s.size() != 11) return false;
	if (s[0] - 'A' < 0 || s[0] - 'Z' > 0) return false;
	if (s[1] - 'A' < 0 || s[1] - 'Z' > 0) return false;
	if (s.size() == 10) {
		for (int i = 2; i < 10; i++) {
			if (!isNumber(s[i])) return false;
		}
	} else {
		if (s[2] != '-' && s[2] != '.') return false;
		for (int i = 3; i < 11; i++) {
			if (!isNumber(s[i])) return false;
		}
	}
	return true;
}
class Receipt {
public:
	string num;
	string date;
	Receipt() :num(""), date("") {};
};
int main()
{
	//Parameter
	//parameter for detection
	double detScale = 1.0 / 255.0;
	float binThresh = 0.3;                                      //二值图的置信度阈值
	float polyThresh = 0.5;                                     //文本多边形阈值
	double unclipRatio = 2.0;                                   //检测到的文本区域的未压缩比率，gai比率确定输出大小
	uint maxCandidates = 200;                                   //输出结果的最大数量
	int height = 640;                                           //height of output image
	int width = 640;											//width of output image
	int imreadRGB = 1;
	String detModelPath = "../Input/DB_TD500_resnet50.onnx";

	// Parameters for Recognition
	double recScale = 1.0 / 255.0;
	Size recInputSize = Size(100, 32);
	Scalar recMean = Scalar(127.5, 127.5, 127.5);
	String recModelPath = "../Input/crnn_cs.onnx";
	String vocPath = "../Input/alphabet_94.txt";

	//other
	String picPath = "../Input/reanomCode.jpg";
	String outputPath = "../Output/";
	double frame_rate = 1.0;										//Times of detecting in 1 second
	vector<Receipt> _receipt;
	bool cameraSrc = true;										//If there is a camera, then true;

	// Load the network
	TextDetectionModel_DB detector(detModelPath);
	detector.setBinaryThreshold(binThresh)
		.setPolygonThreshold(polyThresh)
		.setUnclipRatio(unclipRatio)
		.setMaxCandidates(maxCandidates);
	Size inputSize = Size(width, height);
	Scalar mean = Scalar(122.67891434, 116.66876762, 104.00698793);
	detector.setInputParams(detScale, inputSize, mean);

	TextRecognitionModel recognizer(recModelPath);
	CV_Assert(!vocPath.empty());
	ifstream vocFile;
	vocFile.open(samples::findFile(vocPath));
	CV_Assert(vocFile.is_open());
	String vocLine;
	vector<String> vocabulary;
	while (getline(vocFile, vocLine)) {
		vocabulary.push_back(vocLine);
	}
	recognizer.setVocabulary(vocabulary);
	recognizer.setDecodeType("CTC-greedy");
	recognizer.setInputParams(recScale, recInputSize, recMean, true, false);


	// Create a window
	static const string winName = "TextDetectionModel";

	/*
	// Open an testing image file
	Mat frame2 = imread(picPath);
	CV_Assert(!frame2.empty());
	Mat frame1;
	cv::resize(frame2, frame1, Size(frame2.cols, frame2.rows), 0, 0, 1);

	vector<vector<Point>> detResults;
	detector.detect(frame1, detResults);

	polylines(frame1, detResults, true, Scalar(255, 0, 0), 2);
	imshow(winName, frame1);
	waitKey();
	*/
	VideoCapture cap;
	bool openSuccess = cameraSrc ? cap.open(0) : cap.open(picPath);
	CV_Assert(openSuccess);
	static const string kWinName = "Receipt detector";
	Mat frame;
	cout << frame.size <<endl;
	auto timeStart = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
	auto timeEnd = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
	bool receipt_flag = false;
	while (1) {
		timeEnd = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
		cap >> frame;
		if (frame.empty()) {
			waitKey();
			break;
		}
		if ((timeEnd - timeStart) >= 1000 / frame_rate) {
			//cout << (timeEnd) << "-"<< timeStart <<",";
			receipt_flag = false;
			timeStart = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
			time_t tt = system_clock::to_time_t(system_clock::now());
			struct tm tmpTime;
			localtime_s(&tmpTime, &tt);
			char MY_TIME[26];
			strftime(MY_TIME, sizeof(MY_TIME), "%F %T", &tmpTime);
			Receipt tmpRceipt;
			// Detection
			vector<vector<Point> > detResults;
			detector.detect(frame, detResults);
			putText(frame, MY_TIME, { 30, 30 }, FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
			//cout << MY_TIME << endl;
			if (detResults.size() > 0) {
				vector<vector<Point> > contours;
				for (uint i = 0; i < detResults.size(); i++)
				{
					const auto& quadrangle = detResults[i];
					CV_CheckEQ(quadrangle.size(), (size_t)4, "");
					contours.emplace_back(quadrangle);
					vector<Point2f> quadrangle_2f;
					for (int j = 0; j < 4; j++)
						quadrangle_2f.emplace_back(quadrangle[j]);
					Mat cropped;
					fourPointsTransform(frame, &quadrangle_2f[0], cropped, recInputSize);
					string recognitionResult = recognizer.recognize(cropped);
					if (isReceipt(recognitionResult)) {
						tmpRceipt.num = recognitionResult;
						putText(frame, recognitionResult, quadrangle[1], FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 1);
						cout << recognitionResult << endl;
						polylines(frame, quadrangle, true, Scalar(0, 255, 0), 2);
						receipt_flag = true;
					} else if (isDate(recognitionResult))
						tmpRceipt.date = recognitionResult;
					else
						continue;
					
				}
				if(receipt_flag)
					_receipt.push_back(tmpRceipt);
			}
		}
		imshow(kWinName, frame);
		if (waitKey(1) >= 0) {
			break;
		}
	}

	time_t tt = system_clock::to_time_t(system_clock::now());
	struct tm tmpTime;
	localtime_s(&tmpTime, &tt);
	char MY_TIME[26];
	strftime(MY_TIME, sizeof(MY_TIME), "%F-%H%M%S", &tmpTime);
	ofstream outfile(outputPath + "result_" + MY_TIME + ".txt");
	if (outfile.is_open()) {
		for (auto& r : _receipt) {
			outfile << r.date << "," << r.num << "\n";
		}
		outfile.close();
	} else 
		cout << "Unable to open file";
	system("pause");
	return 0;
}


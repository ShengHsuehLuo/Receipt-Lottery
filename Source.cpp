#include <iostream>
#include <fstream>
#include <string>
#include <thread>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <chrono>
#include <ctime>

using namespace cv;
using namespace cv::dnn;
using namespace std;
using namespace std::chrono;

mutex mu;

bool isNumber(char c) {
	return (c - '0' >= 0 && c - '9' <= 0);
}
bool isDate(const string& s) {
	if (s.size() != 10) return false;
	if (!isNumber(s[0])) return false;
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
void writeFile(ofstream& outputFile, const string& num, const string& date) {
	if (outputFile.is_open()) {
		outputFile << date << "," << num << "\n";
	} else
		cout << "Unable to open file";

}
void detectReceipt(TextDetectionModel_DB& detector, TextRecognitionModel& recognizer, Mat& frame, Size recInputSize, string& outputPath, ofstream& outputFile) {
	// Detection
	vector<vector<Point> > detResults;
	detector.detect(frame, detResults);
	bool receipt_flag = false;
	if (detResults.size() > 0) {
		vector<vector<Point> > contours;
		string num = "";
		string date = "";
		for (uint i = 0; i < detResults.size(); i++) {
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
				num = recognitionResult;
				putText(frame, recognitionResult, quadrangle[1], FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 1);
				cout << recognitionResult << endl;
				polylines(frame, quadrangle, true, Scalar(0, 255, 0), 2);
				receipt_flag = true;
			} else if (isDate(recognitionResult))
				date = recognitionResult;
			else
				continue;
		}
		if (receipt_flag) {
			mu.lock();
			writeFile(outputFile, num, date);
			mu.unlock();
		}

	}
}
int main()
{
	//Parameter
	//parameter for detection
	double detScale = 1.0 / 255.0;
	float binThresh = 0.3;                                      //二值?的置信度?值
	float polyThresh = 0.5;                                     //文本多?形?值
	double unclipRatio = 2.0;                                   //??到的文本?域的未??比率，gai比率确定?出大小
	uint maxCandidates = 200;                                   //?出?果的最大?量
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


	time_t tt = system_clock::to_time_t(system_clock::now());
	struct tm tmpTime;
	localtime_s(&tmpTime, &tt);
	char file_name[26];
	strftime(file_name, sizeof(file_name), "%F-%H%M%S", &tmpTime);
	outputPath = outputPath + "result_" + file_name + ".txt";
	ofstream outfile(outputPath);

	// Create a window
	static const string winName = "TextDetectionModel";
	VideoCapture cap;
	bool openSuccess = cameraSrc ? cap.open(0) : cap.open(picPath);
	CV_Assert(openSuccess);
	static const string kWinName = "Receipt detector";
	Mat frame;
	cout << frame.size << endl;
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
			thread t(detectReceipt, ref(detector), ref(recognizer), ref(frame), recInputSize, ref(outputPath), ref(outfile));
			t.detach();
			timeStart = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
		}
		time_t tt = system_clock::to_time_t(system_clock::now());
		struct tm tmpTime;
		localtime_s(&tmpTime, &tt);
		char MY_TIME[26];
		strftime(MY_TIME, sizeof(MY_TIME), "%F %T", &tmpTime);
		putText(frame, MY_TIME, { 30, 30 }, FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
		imshow(kWinName, frame);
		if (waitKey(1) >= 0) {
			break;
		}
	}
	outfile.close();
	system("pause");
	return 0;
}


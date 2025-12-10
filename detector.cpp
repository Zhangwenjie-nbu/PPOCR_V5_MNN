#include <stdlib.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <cmath>
#include <numeric>

// #include <onnxruntime_cxx_api.h>


#include <MNN/MNNDefine.h>
#include <MNN/AutoTime.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <MNN/ImageProcess.hpp>
#include <opencv2/opencv.hpp>
#include <clipper.hpp>

#include "ocr_detector.h"
#include "durian_error.h"
#include "durian_base.h"
#include "timer.h"

/* ========================= error mechanism =========================*/
static DURIAN_LOG_CALLBACK _logStub = NULL;
static void JuLog(int level, const char* format, ...)
{
    va_list argv;
    char    str[2048];
    memset(str, 0, sizeof(str));
    std::string moduleName("[OCR][Det]");
    snprintf(str, sizeof(str), "%s", moduleName.c_str());
    va_start(argv, format);
    vsnprintf(str + moduleName.size(), sizeof(str) - moduleName.size(), format, argv);
    va_end(argv);
    if (_logStub) _logStub(level, str);
}
#define JuLogDebug(_format, ...)    JuLog(DurianLogDebug, _format, ##__VA_ARGS__)

#define DURIAN_RIF_DBG(_states, _number, _format, ...) do { \
    if (_states) { \
        JuLogDebug("%s:%d,%d%s," _format, __FILE__, __LINE__, -(_number), DURIAN_DOMAIN, ##__VA_ARGS__); \
        return (_number); \
    } \
} while (0)
/* ========================= static method =========================*/
static std::vector<std::vector<float>> Mat2Vector(cv::Mat mat) {
  std::vector<std::vector<float>> img_vec;
  std::vector<float> tmp;

  for (int i = 0; i < mat.rows; ++i) {
    tmp.clear();
    for (int j = 0; j < mat.cols; ++j) {
      tmp.push_back(mat.at<float>(i, j));
    }
    img_vec.push_back(tmp);
  }
  return img_vec;
}

static float clamp(float x, float min, float max) {
    if (x > max)
        return max;
    if (x < min)
        return min;
    return x;
}

static void getContourLengthAndArea(std::vector<std::vector<float>> box,
    float &area, float &length) {

    int pts_num = 4;
    area = 0.0f;
    length = 0.0f;
    for (int i = 0; i < pts_num; i++) {
    area += box[i][0] * box[(i + 1) % pts_num][1] -
            box[i][1] * box[(i + 1) % pts_num][0];
    length += sqrtf((box[i][0] - box[(i + 1) % pts_num][0]) *
                        (box[i][0] - box[(i + 1) % pts_num][0]) +
                    (box[i][1] - box[(i + 1) % pts_num][1]) *
                        (box[i][1] - box[(i + 1) % pts_num][1]));
    }
    area = fabs(float(area / 2.0));
}

static bool compBoxesInAsd(std::vector<std::vector<int>> box1,
    std::vector<std::vector<int>> box2) {
    if (box1[0][1] < box2[0][1])
        return true;
    return false;
}

/*========================= OCRDetectorBase ========================= */
// LCOV_EXCL_START
OCRDetectorBase::OCRDetectorBase()
{
    _destHeight = 0;
    _destWidth = 0;
    _pred = NULL;
    _session = NULL;
}
OCRDetectorBase::~OCRDetectorBase() { ; } // LCOV_EXCL_STOP

/*========================= DBNet =========================*/
// LCOV_EXCL_START
OCRDetectorDBNet::OCRDetectorDBNet()
{
    _limitSideLength = 960;
    _mean[0] = 0.485; _mean[1] = 0.456; _mean[2] = 0.406;
    _stdv[0] = 0.229; _stdv[1] = 0.224; _stdv[2] = 0.225;
    _scale = 1.0 / 255.0f;
    _bitmapThresh = 0.3f;
    _maxCandidates = 1000;
    _boxThresh = 0.55f;
    _unclipRatio = 1.5f;
}

OCRDetectorDBNet::~OCRDetectorDBNet()
{
    release();
}
// LCOV_EXCL_STOP

int OCRDetectorDBNet::release()
{
    int status = DURIAN_OK;
    if (_net && _session) {
        bool ret = _net->releaseSession(_session);
        if (!ret)
            status = -DURIAN_ERROR_RUNTIME;
        _session = NULL;
    }
    if (_net) {
        _net->releaseModel();
        _net = NULL;
    }
    if (_pred) {
        delete _pred;
        _pred = NULL;
    }
    return status;
}

int OCRDetectorDBNet::init(unsigned char *model, long mSize, int thread)
{
    /* check input */
    if (!model || mSize == 0) {
        return -DURIAN_ERROR_PARAM_INVALID;
    }
    /* create net */
    _net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromBuffer(model, mSize));
    DURIAN_RIF_DBG(_net == nullptr, -DURIAN_ERROR_MODEL_INVALID,
        "failed to create detector model _net %p _session %p", _net.get(), _session);
    _net->setSessionMode(MNN::Interpreter::Session_Release);

    /* create session */
    MNNForwardType type = MNN_FORWARD_CPU;
    MNN::ScheduleConfig config;
    config.type = type;
    config.numThread = thread;
    MNN::BackendConfig backendConfig;
    backendConfig.precision = static_cast<MNN::BackendConfig::PrecisionMode>(MNN::BackendConfig::Precision_Low);
    config.backendConfig = &backendConfig;

    _session = _net->createSession(config);
    DURIAN_RIF_DBG(_session == nullptr, -DURIAN_ERROR_RUNTIME,
        "failed to create detector model _net %p _session %p", _net.get(), _session);

    return DURIAN_OK;
}

/* ============= preprocess =============*/
void OCRDetectorDBNet::normalizeImage(cv::Mat &inImage, cv::Mat &outImage)
{
    inImage.convertTo(inImage, CV_32FC3);

    std::vector<cv::Mat> vecMat(3);
    cv::split(inImage, vecMat);
    for (int i = 0; i < 3; i++) {
        vecMat.at(i) = (vecMat.at(i).mul(cv::Scalar::all(_scale))) - (cv::Scalar::all(_mean[i]));
        vecMat.at(i) = vecMat.at(i).mul(cv::Scalar::all(1.0 / _stdv[i]));
    }
    cv::merge(vecMat, outImage);
}

void OCRDetectorDBNet::resizeImage(cv::Mat &image)
{
    /* read image */
    int width = image.cols;
    int height = image.rows;
    float ratio = 0.0f;
    /* resize */
    if (std::max(width, height) > _limitSideLength) {
        if (height > width) {
            ratio = _limitSideLength / (float)height;
        } else {
            ratio = _limitSideLength / (float)width;
        }
    } else {
        ratio = 1.0f;
    }
    int resizeH = (int)(height * ratio);
    int resizeW = (int)(width * ratio);

    resizeH = std::max((int)(std::round(resizeH / 32.0f) * 32.0f), 32);
    resizeW = std::max((int)(std::round(resizeW / 32.0f) * 32.0f), 32);
    cv::resize(image, image, cv::Size(resizeW, resizeH));
}

void OCRDetectorDBNet::toCHWImage(cv::Mat &inImage, float *&data)
{
    int width = inImage.cols;
    int height = inImage.rows;
    int skipIdx = width * height;
    int bpp = inImage.channels();
    std::vector<cv::Mat> vecMat(bpp);
    cv::split(inImage, vecMat);
    memcpy(data, vecMat.at(0).data, skipIdx*sizeof(float));//b
    memcpy(data+skipIdx, vecMat.at(1).data, skipIdx*sizeof(float));//g
    memcpy(data+skipIdx*2, vecMat.at(2).data, skipIdx*sizeof(float));//r
}

int OCRDetectorDBNet::preprocess(const cv::Mat &image)
{
    // timer_begin(2);
    DURIAN_RIF_DBG(!_net || !_session, -DURIAN_ERROR_RUNTIME, "_net %p _session %p", _net.get(), _session);
    // NOTE: CV_8UC3 format image
    /* create inputTensor */
    // resize
    _destHeight = image.rows;
    _destWidth = image.cols;
    cv::Mat tmpImage = image.clone();
    resizeImage(tmpImage);
    // normalized
    cv::Mat normalizedImage;
    normalizeImage(tmpImage, normalizedImage);
    // transpose
    int width = normalizedImage.cols;
    int height = normalizedImage.rows;
    int bpp = normalizedImage.channels();
    float *data = (float*)malloc(width * height * bpp * sizeof(float));
    if (!data)
        return -DURIAN_ERROR_MEM_ALLOC;
    memset(data, 0, width * height * bpp * sizeof(float));
    toCHWImage(normalizedImage, data);
    // timer_end(2);
    // resize intputTensor and session
    std::vector<int> inputDims;
    inputDims.push_back(1);
    inputDims.push_back(bpp);
    inputDims.push_back(height);
    inputDims.push_back(width);
    auto inputTensor = _net->getSessionInput(_session, NULL);
    DURIAN_RIF_DBG(!inputTensor, -DURIAN_ERROR_RUNTIME, "detector create inputTensor failed");
    DURIAN_RIF_DBG(bpp <= 0 || height <= 0 || width <= 0, -DURIAN_ERROR_RUNTIME,
        "input dimision invalid bpp %d h %d w %d", bpp, height, width);
    // timer_begin(3);
    _net->resizeTensor(inputTensor, inputDims);
    _net->resizeSession(_session);
    // timer_end(3);

    MNN::Tensor *inputUser = MNN::Tensor::create(inputDims, halide_type_of<float>(), data, MNN::Tensor::CAFFE);
    bool ret = inputTensor->copyFromHostTensor(inputUser);
    if (data) {
        free(data);
        data = NULL;
    }
    if (inputUser) {
        delete inputUser;
        inputUser = NULL;
    }
    if (!ret)
        return -DURIAN_ERROR_RUNTIME;
    return DURIAN_OK;
}

int OCRDetectorDBNet::run()
{
    DURIAN_RIF_DBG(!_net || !_session, -DURIAN_ERROR_RUNTIME, "_net %p _session %p", _net.get(), _session);
    /* infer */
    // timer_begin(4);
    int status = _net->runSession(_session);
    // timer_end(4);
    DURIAN_RIF_DBG(status != 0, -DURIAN_ERROR_RUNTIME, "MNN runSession error with code %d", status);
    /* get output */
    auto output = _net->getSessionOutput(_session, NULL);
    DURIAN_RIF_DBG(_pred != nullptr, -DURIAN_ERROR_RUNTIME, "detector release intermediate vars failed");
    _pred = new MNN::Tensor(output, MNN::Tensor::CAFFE);
    DURIAN_RIF_DBG(_pred == nullptr, -DURIAN_ERROR_MEM_ALLOC, "detector failed to alloc _pred");
    output->copyToHostTensor(_pred);
    return DURIAN_OK;
}
/* ============= postprocess =============*/
void OCRDetectorDBNet::getMiniBoxes(const std::vector<cv::Point> &contour,
    std::vector<std::vector<float>> &box,float &sside)
{
    cv::RotatedRect bBox = cv::minAreaRect(contour); //外接四边形
    cv::Mat points;
    cv::boxPoints(bBox, points); // get 4 vertexes
    std::vector<std::vector<float>> pointsArray = Mat2Vector(points);
    std::sort(pointsArray.begin(), pointsArray.end(), std::less<std::vector<float>>()); // ascending order

    int index1, index2, index3, index4;
    if (pointsArray[1][1] > pointsArray[0][1]) {
        index1 = 0;
        index4 = 1;
    } else {
        index1 = 1;
        index4 = 0;
    }
    if (pointsArray[3][1] > pointsArray[2][1]) {
        index2 = 2;
        index3 = 3;
    } else {
        index2 = 3;
        index3 = 2;
    }

    box.push_back(pointsArray[index1]);
    box.push_back(pointsArray[index2]);
    box.push_back(pointsArray[index3]);
    box.push_back(pointsArray[index4]);
    sside = std::min(bBox.size.width, bBox.size.height);
}

void OCRDetectorDBNet::getBoxScore(const cv::Mat &pred, const std::vector<std::vector<float>> &box,
    const std::vector<cv::Point> &contour, float &score)
{
    int height = pred.rows;
    int width = pred.cols;
    // clip box
    float boxX[4] = {box[0][0], box[1][0], box[2][0], box[3][0]};
    float boxY[4] = {box[0][1], box[1][1], box[2][1], box[3][1]};

    int xmin = (int)(clamp(
        (std::floor(*(std::min_element(boxX, boxX + 4)))), 0.0,
        (float)(width - 1)));
    int xmax = (int)(clamp(
        (std::ceil(*(std::max_element(boxX, boxX + 4)))), 0.0,
        (float)(width - 1)));
    int ymin = (int)(clamp(
        (std::floor(*(std::min_element(boxY, boxY + 4)))), 0.0,
        (float)(height - 1)));
    int ymax = (int)(clamp(
        (std::ceil(*(std::max_element(boxY, boxY + 4)))), 0.0,
        (float)(height - 1)));
    cv::Mat mask = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8UC1);
    std::vector<cv::Point> rootPoints(contour.size());
    for(int i = 0;i < static_cast<int>(contour.size()); i++){
        rootPoints[i] = cv::Point(contour[i].x - xmin, contour[i].y - ymin);
    }
    const cv::Point *ppt[1] = {rootPoints.data()};
    int npt[] = {static_cast<int>(rootPoints.size())};
    cv::fillPoly(mask, ppt, npt, 1, cv::Scalar(1));

    cv::Mat croppedImage;
    pred(cv::Rect(xmin, ymin, xmax-xmin+1, ymax-ymin+1)).copyTo(croppedImage);
    score = cv::mean(croppedImage, mask)[0];

#ifdef CARDOCR_DEBUG
    static int imageIndex = 0;
    std::string imageName = std::string("getBoxScore_mask") + std::to_string(imageIndex++) + ".jpg";
    cv::imwrite(imageName.c_str(), mask * 255);
#endif
}

void OCRDetectorDBNet::unclip(const std::vector<std::vector<float>> &box,
    std::vector<cv::Point> &points)
{
    float area = 0.0f;
    float length = 0.0f;
    getContourLengthAndArea(box, area, length);
    float distance = area * _unclipRatio / length;
    // apply Vatti clipping algorithm to dilate Gd(Gd -> G)
    ClipperLib::ClipperOffset offset;
    ClipperLib::Path p;
    p << ClipperLib::IntPoint(static_cast<int>(box[0][0]),
                            static_cast<int>(box[0][1]))
    << ClipperLib::IntPoint(static_cast<int>(box[1][0]),
                            static_cast<int>(box[1][1]))
    << ClipperLib::IntPoint(static_cast<int>(box[2][0]),
                            static_cast<int>(box[2][1]))
    << ClipperLib::IntPoint(static_cast<int>(box[3][0]),
                            static_cast<int>(box[3][1]));
    offset.AddPath(p, ClipperLib::jtRound, ClipperLib::etClosedPolygon);
    ClipperLib::Paths soln;
    offset.Execute(soln, distance);
    for (int j = 0; j < (int)soln.size(); j++) {
        for (int i = 0; i < (int)(soln[soln.size() - 1].size()); i++) {
        points.emplace_back(soln[j][i].X, soln[j][i].Y);
        }
    }
}

int OCRDetectorDBNet::boxesFromBitmap(cv::Mat &pred, cv::Mat &bitmap,
    std::vector<std::vector<std::vector<int>>> &boxes)
{
    /* find countours */
    int width = bitmap.cols;
    int height = bitmap.rows;
    std::vector<std::vector<cv::Point>> contours;
    int minSize = 3;

    std::vector<cv::Vec4i> hierarchy;
    bitmap.convertTo(bitmap, CV_8UC1);
    cv::findContours(bitmap, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    int numContours = std::min((int)(contours.size()), _maxCandidates);
    /* process countours */
    for (int i = 0; i < numContours; i++) {
        // 1. get box vertexes (外接四边形) self.get_mini_boxes
        std::vector<cv::Point> contour = contours[i];
        std::vector<std::vector<float>> box;
        float sside = 0.0f;
        getMiniBoxes(contour, box, sside);
        if (sside < minSize)
            continue;
        // 2. get box score
        float score = 0.0f;
        getBoxScore(pred, box, contour, score);
        // JuLogDebug("contour %d, score %f, (%f, %f), (%f, %f), (%f, %f), (%f, %f)",
        //     i, score, box[0][0], box[0][1], box[1][0], box[1][1], box[2][0], box[2][1], box[3][0], box[3][1]);
        if (score < _boxThresh)
            continue;
        // 3. get G: dilate Gd to get G
        std::vector<cv::Point> points;
        unclip(box, points);
        // 4. get box vertexes from G (外接四边形) self.get_mini_boxes
        std::vector<std::vector<float>> unclippedBox;
        getMiniBoxes(points, unclippedBox, sside);
        // 5. filter out small boxes
        if (sside < minSize + 2)
            continue;
        // 6. rescale and clip box
        std::vector<std::vector<int>> intUnclippedBox;
        for (int j = 0; j < 4; j++) {
            int px = (int)(clamp(std::round(unclippedBox[j][0] / (float)width * (float)_destWidth),
                0.0f, (float)_destWidth));
            int py = (int)(clamp(std::round(unclippedBox[j][1] / (float)height * (float)_destHeight),
                0.0f, (float)_destHeight));
            std::vector<int>a{px, py};
            intUnclippedBox.push_back(a);
        }
        boxes.push_back(intUnclippedBox);
    }
    return DURIAN_OK;
}

void OCRDetectorDBNet::sortBoxes(const std::vector<std::vector<std::vector<int>>> &boxes,
    std::vector<std::vector<std::vector<int>>> &sortedBoxes)
{
    int boxNums = (int)(boxes.size());
    sortedBoxes = boxes;
    std::sort(sortedBoxes.begin(), sortedBoxes.end(), compBoxesInAsd);
    for (int i = 0; i < boxNums-1; i++) {
        if (std::abs(sortedBoxes[i+1][0][1] - sortedBoxes[i][0][1]) < 10 &&
            sortedBoxes[i+1][0][0] < sortedBoxes[i][0][0]) {
            // switch box
            std::vector<std::vector<int>> tmp = sortedBoxes[i];
            sortedBoxes[i] = sortedBoxes[i+1];
            sortedBoxes[i+1] = tmp;
        }
    }
}

int OCRDetectorDBNet::postprocess()
{
    DURIAN_RIF_DBG(!_pred, -DURIAN_ERROR_RUNTIME, "detector _pred is nullptr");

    float *outData = _pred->host<float>();
    int dim2 = _pred->shape()[2];
    int dim3 = _pred->shape()[3];
    /* get binarymap */
    cv::Mat prediction(dim2, dim3, CV_32FC1);
    memcpy(prediction.data, outData, dim2*dim3*sizeof(float));
#ifdef CARDOCR_DEBUG
    cv::imwrite("detector_pred.jpg", prediction * 255);
#endif
    cv::Mat mask(dim2, dim3, CV_8UC1);
    cv::threshold(prediction, mask, _bitmapThresh, 1, cv::THRESH_BINARY);
    std::vector<std::vector<std::vector<int>>> boxes;
    /* get boxes from bitmap */
    int status = boxesFromBitmap(prediction, mask, boxes);
    DURIAN_RIF_DBG(status != DURIAN_OK, status, "detector boxesFromBitmap failed");
    /* sort boxes from left to right, top to bottom */
    sortBoxes(boxes, _boxes);
    /* clear itermediate vars */
    if (_pred) {
        delete _pred;
        _pred = NULL;
    }

    return DURIAN_OK;
}
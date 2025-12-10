#ifndef OCR_DETECTOR_H
#define OCR_DETECTOR_H

#include <stdlib.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include <MNN/Interpreter.hpp>

/**
 * @brief ancestor to OCRDetector class
 * @details It's an abstract class. It should not be instantiated directly.
 *
 */
class OCRDetectorBase
{
public:
    OCRDetectorBase();
    virtual ~OCRDetectorBase();
    virtual int init(unsigned char *model, long mSize, int thread) = 0;
    virtual int preprocess(const cv::Mat &image) = 0;
    virtual int run() = 0;
    virtual int postprocess() = 0;
    virtual int release() = 0;
    std::vector<std::vector<std::vector<int>>> _boxes;
protected:
    int _destHeight;
    int _destWidth;
    MNN::Tensor *_pred;
    std::shared_ptr<MNN::Interpreter> _net;
    MNN::Session *_session;
};

/**
 * @brief DBNet. Derived class of OCRDetectorBase
 * reference from https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/ppocr/postprocess/db_postprocess.py
 *
 */
class OCRDetectorDBNet : public virtual OCRDetectorBase
{
public:
    OCRDetectorDBNet();
    ~OCRDetectorDBNet();
    int init(unsigned char *model, long mSize, int thread);
    int preprocess(const cv::Mat &image);
    int run();
    int postprocess();
    int release();
private:
    /* preprocess method */
    void resizeImage(cv::Mat &image);
    void normalizeImage(cv::Mat &inImage, cv::Mat &outImage);
    void toCHWImage(cv::Mat &inImage, float *&data);

    /* postprocess method */
    void getMiniBoxes(const std::vector<cv::Point> &contour, std::vector<std::vector<float>> &box,float &sside);
    void unclip(const std::vector<std::vector<float>> &box, std::vector<cv::Point> &points);
    void getBoxScore(const cv::Mat &pred, const std::vector<std::vector<float>> &box, const std::vector<cv::Point> &contour, float &score);
    int boxesFromBitmap(cv::Mat &pred, cv::Mat &bitmap, std::vector<std::vector<std::vector<int>>> &boxes);
    void sortBoxes(const std::vector<std::vector<std::vector<int>>> &boxes, std::vector<std::vector<std::vector<int>>> &sortedBoxes);

    /* class member */
    int _limitSideLength;
    float _mean[3];
    float _stdv[3];
    float _scale;
    float _bitmapThresh;
    int _maxCandidates;
    float _boxThresh;
    float _unclipRatio;
};

#endif
#ifndef OCR_RECOGNIZER_H_
#define OCR_RECOGNIZER_H_

#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <MNN/Interpreter.hpp>

class OCRRecognizerBase {
public:
    OCRRecognizerBase();
    virtual ~OCRRecognizerBase();

    /// @param model   MNN 模型 buffer
    /// @param mSize   buffer 大小
    /// @param threads 线程数
    virtual int init(unsigned char* model, long mSize, int threads) = 0;

    /// 对一张图像 + 多个检测框做识别
    /// @param image   原图（BGR）
    /// @param boxes   检测框，格式与 OCRDetectorDBNet::_boxes 相同：
    ///                [num_box][4][2]，4 个点顺时针，元素是 (x,y)
    /// @param dict    字典（0 号必须是 ""，用于 CTC blank）
    /// @param texts   输出，每个框对应一段文本
    /// @param recScores 输出，每个框对应一个平均置信度
    virtual int recognize(const cv::Mat& image,
                          const std::vector<std::vector<std::vector<int>>>& boxes,
                          const std::vector<std::string>& dict,
                          std::vector<std::string>& texts,
                          std::vector<float>& recScores) = 0;

    virtual void release() = 0;

protected:
    int   _imgH;              // 识别网络输入高度（默认 48）
    int   _imgW;              // 识别网络输入宽度上限（默认 320）
    float _mean[3];
    float _stdv[3];

    std::shared_ptr<MNN::Interpreter> _net;
    MNN::Session*                     _session;
};


/// CRNN / PPOCRv5 默认识别模型的 MNN 版本
class OCRRecognizerCRNN : public OCRRecognizerBase {
public:
    OCRRecognizerCRNN();
    ~OCRRecognizerCRNN() override;

    int init(unsigned char* model, long mSize, int threads) override;

    int recognize(const cv::Mat& image,
                  const std::vector<std::vector<std::vector<int>>>& boxes,
                  const std::vector<std::string>& dict,
                  std::vector<std::string>& texts,
                  std::vector<float>& recScores) override;

    void release() override;

    /// 可选：设置“软约束字符增强”的参数（对应 ONNX 版本中的 sc_boost）
    /// @param indices   需要增强的字符在字典中的索引
    /// @param strength  增强强度（例如 0.8f）
    void setBoost(const std::vector<int>& indices, float strength);

    /// 可选：在 logits 上对某些类别做加权增强
    void setBoostByString(const std::string& chars,
                          const std::vector<std::string>& dict,
                          float strength);

private:
    /// 从原图中裁剪出一个检测框对应的 ROI（旋转拉正）
    cv::Mat getRotateCropImage(const cv::Mat& image,
                               const std::vector<std::vector<int>>& box) const;

    /// 将裁剪后的 ROI 变成 [1,3,H,W] 的 CHW float 向量，同时做 resize + 归一化
    std::vector<float> prepareInput(const cv::Mat& crop,
                                    std::vector<int>& shapeOut) const;

    /// 在一个 time-step 上做 softmax + argmax，返回最大概率，输出 max_idx
    float softmaxMaxProb(const float* logits, int classes, int& maxIdx) const;

    /// 简单 CTC 解码（去掉 blank 和重复）
    std::string ctcDecode(const std::vector<int>& indices,
                          const std::vector<float>& probs,
                          const std::vector<std::string>& dict,
                          float& avgScore) const;

    void applyScBoost(float* logits, int seqLen, int numClasses) const;

private:
    std::vector<int> _scBoostIndices;
    float            _scBoostStrength;
};

#endif  // OCR_RECOGNIZER_H_

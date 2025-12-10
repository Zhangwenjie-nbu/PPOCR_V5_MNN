#include "ocr_recognizer.h"
#include <MNN/Tensor.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

OCRRecognizerBase::OCRRecognizerBase()
    : _imgH(48),
      _imgW(320),
      _session(nullptr) {
    _mean[0] = _mean[1] = _mean[2] = 0.5f;
    _stdv[0] = _stdv[1] = _stdv[2] = 0.5f;
}

OCRRecognizerBase::~OCRRecognizerBase() = default;


// --------- OCRRecognizerCRNN ---------

OCRRecognizerCRNN::OCRRecognizerCRNN()
    : _scBoostStrength(0.0f) {}

OCRRecognizerCRNN::~OCRRecognizerCRNN() {
    release();
}

int OCRRecognizerCRNN::init(unsigned char* model, long mSize, int threads) {
    if (!model || mSize <= 0) {
        std::cerr << "[OCR][Rec] invalid model buffer" << std::endl;
        return -1;
    }

    _net.reset(MNN::Interpreter::createFromBuffer(model, static_cast<size_t>(mSize)));
    if (!_net) {
        std::cerr << "[OCR][Rec] createFromBuffer failed" << std::endl;
        return -2;
    }

    MNN::ScheduleConfig config;
    config.numThread = threads > 0 ? threads : 1;
    config.type      = MNN_FORWARD_CPU;

    _session = _net->createSession(config);
    if (!_session) {
        std::cerr << "[OCR][Rec] createSession failed" << std::endl;
        _net.reset();
        return -3;
    }

    return 0;
}

void OCRRecognizerCRNN::release() {
    if (_net && _session) {
        _net->releaseSession(_session);
    }
    _session = nullptr;
    _net.reset();
}

void OCRRecognizerCRNN::setBoost(const std::vector<int>& indices, float strength) {
    _scBoostIndices = indices;
    _scBoostStrength = strength;

}

cv::Mat OCRRecognizerCRNN::getRotateCropImage(const cv::Mat& image,
                                              const std::vector<std::vector<int>>& box) const {
    if (box.size() != 4) {
        return cv::Mat();
    }

    std::array<cv::Point2f, 4> pts;
    for (int i = 0; i < 4; ++i) {
        if (box[i].size() < 2) {
            return cv::Mat();
        }
        pts[i] = cv::Point2f(static_cast<float>(box[i][0]),
                             static_cast<float>(box[i][1]));
    }

    float width_top    = cv::norm(pts[0] - pts[1]);
    float width_bottom = cv::norm(pts[2] - pts[3]);
    float height_left  = cv::norm(pts[0] - pts[3]);
    float height_right = cv::norm(pts[1] - pts[2]);

    float width  = std::max(width_top, width_bottom);
    float height = std::max(height_left, height_right);
    width  = std::max(width, 1.0f);
    height = std::max(height, 1.0f);

    std::array<cv::Point2f, 4> dst_pts = {
        cv::Point2f(0.f,     0.f),
        cv::Point2f(width,   0.f),
        cv::Point2f(width,   height),
        cv::Point2f(0.f,     height)
    };

    cv::Mat M = cv::getPerspectiveTransform(pts.data(), dst_pts.data());
    cv::Mat warped;
    cv::warpPerspective(image, warped, M,
                        cv::Size(static_cast<int>(std::round(width)),
                                 static_cast<int>(std::round(height))),
                        cv::INTER_CUBIC, cv::BORDER_REPLICATE);

    // 如果过“竖”，旋转成横条
    if (!warped.empty() && warped.rows >= warped.cols * 1.5f) {
        cv::rotate(warped, warped, cv::ROTATE_90_CLOCKWISE);
    }
    return warped;
}

std::vector<float> OCRRecognizerCRNN::prepareInput(const cv::Mat& crop,
                                                   std::vector<int>& shapeOut) const {
    if (crop.empty()) {
        return {};
    }

    const int img_h = _imgH;
    const int img_w = _imgW;
    shapeOut = {1, 3, img_h, img_w};

    float ratio = static_cast<float>(crop.cols) / static_cast<float>(crop.rows);
    int target_w = static_cast<int>(std::ceil(img_h * ratio));
    if (target_w < 1) {
        target_w = 1;
    } else if (target_w > img_w) {
        target_w = img_w;
    }

    cv::Mat resized;
    cv::resize(crop, resized, cv::Size(target_w, img_h), 0, 0, cv::INTER_LINEAR);

    cv::Mat resized_float;
    resized.convertTo(resized_float, CV_32FC3, 1.0f / 255.0f);
    cv::Scalar mean(_mean[0], _mean[1], _mean[2]);
    cv::Scalar std_dev(_stdv[0], _stdv[1], _stdv[2]);
    resized_float = (resized_float - mean) / std_dev;

    std::vector<float> chw(3 * img_h * img_w, 0.0f);
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < img_h; ++y) {
            const float* row_ptr = resized_float.ptr<float>(y);
            for (int x = 0; x < target_w; ++x) {
                float value = row_ptr[x * 3 + c];
                chw[c * img_h * img_w + y * img_w + x] = value;
            }
        }
    }
    return chw;
}

float OCRRecognizerCRNN::softmaxMaxProb(const float* logits, int classes, int& maxIdx) const {
    float max_logit = logits[0];
    for (int i = 1; i < classes; ++i) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }

    float sum = 0.f;
    for (int i = 0; i < classes; ++i) {
        sum += std::exp(logits[i] - max_logit);
    }

    float max_prob = 0.f;
    maxIdx = 0;
    for (int i = 0; i < classes; ++i) {
        float p = std::exp(logits[i] - max_logit) / sum;
        if (p > max_prob) {
            max_prob = p;
            maxIdx = i;
        }
    }
    return max_prob;
}

std::string OCRRecognizerCRNN::ctcDecode(const std::vector<int>& indices,
                                         const std::vector<float>& probs,
                                         const std::vector<std::string>& dict,
                                         float& avgScore) const {
    std::string text;
    avgScore = 0.f;
    int prev_idx = -1;
    int valid = 0;

    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        if (idx > 0 && idx < static_cast<int>(dict.size()) && idx != prev_idx) {
            text += dict[idx];
            avgScore += probs[i];
            ++valid;
        }
        prev_idx = idx;
    }
    if (valid > 0) {
        avgScore /= static_cast<float>(valid);
    }
    return text;
}


void OCRRecognizerCRNN::setBoostByString(const std::string& chars,
                                         const std::vector<std::string>& dict,
                                         float strength) {
    std::vector<int> indices;
    if (chars.empty() || dict.empty()) {
        _scBoostIndices.clear();
        _scBoostStrength = 0.0f;
        return;
    }

    // 从 1 开始跳过 CTC blank
    for (int i = 1; i < static_cast<int>(dict.size()); ++i) {
        const std::string& token = dict[i];
        if (token.empty()) continue;

        // 简单做法：只要 dict[i] 作为子串出现在 chars 中，就认为要增强
        // 这样 UTF-8 多字节字符也能正常匹配（前提是字典里的 token 与输入字符串编码一致）
        if (chars.find(token) != std::string::npos) {
            indices.push_back(i);
        }
    }

    setBoost(indices, strength);

    std::cout << "[OCR][Rec] setBoostByString chars=\"" << chars
              << "\" -> " << _scBoostIndices.size()
              << " classes boosted, strength=" << _scBoostStrength << std::endl;
}


void OCRRecognizerCRNN::applyScBoost(float* logits, int seqLen, int numClasses) const {
    if (_scBoostIndices.empty() || _scBoostStrength == 0.0f) {
        return;
    }

    for (int t = 0; t < seqLen; ++t) {
        float* row = logits + t * numClasses;
        for (int idx : _scBoostIndices) {
            if (idx >= 0 && idx < numClasses) {
                row[idx] += _scBoostStrength;
            }
        }
        // 对 blank 类（0）稍微加一点
        if (numClasses > 0) {
            row[0] += _scBoostStrength * 0.5f;
        }
    }
}


int OCRRecognizerCRNN::recognize(const cv::Mat& image,
                                 const std::vector<std::vector<std::vector<int>>>& boxes,
                                 const std::vector<std::string>& dict,
                                 std::vector<std::string>& texts,
                                 std::vector<float>& recScores) {
    texts.clear();
    recScores.clear();

    if (image.empty()) {
        std::cerr << "[OCR][Rec] input image is empty" << std::endl;
        return -1;
    }
    if (!_net || !_session) {
        std::cerr << "[OCR][Rec] recognizer not initialized" << std::endl;
        return -2;
    }
    if (dict.empty()) {
        std::cerr << "[OCR][Rec] dictionary is empty" << std::endl;
        return -3;
    }

    texts.reserve(boxes.size());
    recScores.reserve(boxes.size());

    auto* inputTensor = _net->getSessionInput(_session, nullptr);
    if (!inputTensor) {
        std::cerr << "[OCR][Rec] getSessionInput failed" << std::endl;
        return -4;
    }

    for (size_t i = 0; i < boxes.size(); ++i) {
        const auto& box = boxes[i];

        cv::Mat crop = getRotateCropImage(image, box);
        std::vector<int> shape;
        std::vector<float> chw = prepareInput(crop, shape);
        if (chw.empty()) {
            texts.emplace_back("");
            recScores.emplace_back(0.f);
            continue;
        }

        // 调整输入 tensor 尺寸并拷贝数据
        _net->resizeTensor(inputTensor, shape);
        _net->resizeSession(_session);

        MNN::Tensor inputHost(inputTensor, MNN::Tensor::CAFFE);
        float* hostPtr = inputHost.host<float>();
        std::memcpy(hostPtr, chw.data(),
                    std::min<size_t>(chw.size(), inputHost.elementSize()) * sizeof(float));
        inputTensor->copyFromHostTensor(&inputHost);

        // 前向推理
        _net->runSession(_session);

        auto* outputTensor = _net->getSessionOutput(_session, nullptr);
        MNN::Tensor outputHost(outputTensor, MNN::Tensor::CAFFE);
        outputTensor->copyToHostTensor(&outputHost);

        const auto& outShape = outputHost.shape();
        if (outShape.size() != 3) {
            std::cerr << "[OCR][Rec] unexpected output shape, size="
                      << outShape.size() << std::endl;
            texts.emplace_back("");
            recScores.emplace_back(0.f);
            continue;
        }

        int seqLen     = outShape[1];
        int numClasses = outShape[2];

        float* logits = outputHost.host<float>();

        applyScBoost(logits, seqLen, numClasses);

        std::vector<int>   indices(seqLen);
        std::vector<float> probs(seqLen);
        for (int t = 0; t < seqLen; ++t) {
            int   maxIdx   = 0;
            float maxProb  = softmaxMaxProb(logits + t * numClasses, numClasses, maxIdx);
            indices[t]     = maxIdx;
            probs[t]       = maxProb;
        }

        float avgScore = 0.f;
        std::string text = ctcDecode(indices, probs, dict, avgScore);
        texts.push_back(std::move(text));
        recScores.push_back(avgScore);

        // std::cout << "[OCR][Rec] box " << i
        //           << " text=\"" << texts.back()
        //           << "\" score=" << recScores.back() << std::endl;
    }

    return 0;
}

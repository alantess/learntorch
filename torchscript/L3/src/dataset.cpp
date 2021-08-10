#include "dataset.h"

namespace {
constexpr int kTrainSize = 8007;
constexpr int kTestSize = 2027;

constexpr int kRows = 300;
constexpr int kCols = 300;

torch::Tensor CVtoTensor(cv::Mat img) {
  cv::resize(img, img, cv::Size{kRows, kCols}, 0, 0, cv::INTER_LINEAR);
  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
  auto img_tensor = torch::from_blob(img.data, {kRows, kCols, 3}, torch::kByte);
  img_tensor = img_tensor.permute({2, 0, 1}).toType(torch::kFloat).div_(255);
  return img_tensor;
}

std::pair<torch::Tensor, torch::Tensor> read_data(const std::string& root,
                                                  bool train) {
  int i = 0;
  std::string ext(".jpg");
  const auto num_samples = train ? kTrainSize : kTestSize;
  const auto folder = train ? root + "/train" : root + "/test";
  auto targets = torch::empty(num_samples, torch::kInt64);
  auto images = torch::empty({num_samples, 3, kRows, kCols}, torch::kFloat);

  std::string cats_folder = folder + "/cats";
  std::string dogs_folder = folder + "/dogs";
  std::vector<std::string> folders = {cats_folder, dogs_folder};

  for (auto& f : folders) {
    int64_t label = 0;
    for (const auto& p : fs::directory_iterator(f)) {
      if (p.path().extension() == ext) {
        cv::Mat img = cv::imread(p.path());
        auto img_tensor = CVtoTensor(img);
        images[i] = img_tensor;
        targets[i] = torch::tensor(label, torch::kInt64);
      }

      if (i >= num_samples) break;
      i++;
    }
    label++;
  }

  return {images, targets};
}
}  // namespace

CatDog::CatDog(const std::string& root, Mode mode) : mode_(mode) {
  auto data = read_data(root, mode == Mode::kTrain);

  images_ = std::move(data.first);
  targets_ = std::move(data.second);
}

torch::data::Example<> CatDog::get(size_t index) {
  return {images_[index], targets_[index]};
}

torch::optional<size_t> CatDog::size() const { return images_.size(0); }

bool CatDog::is_train() const noexcept { return mode_ == Mode::kTrain; }

const torch::Tensor& CatDog::images() const { return images_; }

const torch::Tensor& CatDog::targets() const { return targets_; }


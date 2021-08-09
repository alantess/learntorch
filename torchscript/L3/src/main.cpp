#include <torch/torch.h>

#include <iostream>

#include "dataset.h"

cv::Mat TensortoCv(torch::Tensor x);

int main() {
  std::string root = "../dataset/";
  auto train_dataset = CatDog(root)
                           .map(torch::data::transforms::Normalize<>(
                               {0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}))
                           .map(torch::data::transforms::Stack<>());

  auto train_loader =
      torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
          std::move(train_dataset), 4);

  auto test_dataset = CatDog(root, CatDog::Mode::kTest)
                          .map(torch::data::transforms::Normalize<>(
                              {0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}))
                          .map(torch::data::transforms::Stack<>());
  auto test_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(test_dataset), 4);

  for (auto& batch : *train_loader) {
    auto img = batch.data;
    auto labels = batch.target;
    auto out = TensortoCv(img[0]);
    cv::imshow("Display window", out);
    int k = cv::waitKey(0);  // Wait for a keystroke in the window

    break;
  }
}

cv::Mat TensortoCv(torch::Tensor x) {
  x = x.permute({1, 2, 0});
  x = x.mul(0.5).add(0.5).mul(255).clamp(0, 255).to(torch::kByte);
  x = x.contiguous();
  int height = x.size(0);
  int width = x.size(1);
  cv::Mat output(cv::Size{width, height}, CV_8UC3);
  std::memcpy((void*)output.data, x.data_ptr(), sizeof(torch::kU8) * x.numel());

  return output.clone();
}

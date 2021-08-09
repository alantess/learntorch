#pragma once

#include <iostream>
#include <filesystem>
#include<opencv2/opencv.hpp>
#include <torch/torch.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
namespace fs = std::filesystem;

struct CatDog : torch::data::datasets::Dataset<CatDog> {
 public:
    // The mode in which the dataset is loaded
    enum Mode { kTrain, kTest };

    explicit CatDog(const std::string& root, Mode mode = Mode::kTrain);

    // Returns the `Example` at the given `index`.
    torch::data::Example<> get(size_t index) override;

    // Returns the size of the dataset.
    torch::optional<size_t> size() const override;

    // Returns true if this is the training subset of CatDog.
    bool is_train() const noexcept;

    // Returns all images stacked into a single tensor.
    const torch::Tensor& images() const;

    // Returns all targets stacked into a single tensor.
    const torch::Tensor& targets() const;



 private:
    torch::Tensor images_;
    torch::Tensor targets_;
    Mode mode_;
};


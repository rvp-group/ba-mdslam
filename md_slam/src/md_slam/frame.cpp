// Copyright 2022 Luca Di Giammarino
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "frame.h"

namespace md_slam_closures {

  Frame::Frame(const size_t& id_,
               const double& timestamp_,
               const size_t& num_framepoints_,
               const cv::Mat& cv_intensity_image_) :
    _id(id_) {
    if (_framepoints) {
      throw std::runtime_error("Frame::Frame|ERROR, framepoints already there");
    }
    _timestamp   = timestamp_;
    _framepoints = new FramePointVector();
    _framepoints->reserve(num_framepoints_);
    _cv_intensity_image = cv_intensity_image_;
  }

  Frame::Frame(const size_t& id_,
               const double& timestamp_,
               const size_t& num_framepoints_,
               const cv::Mat& cv_intensity_image_,
               MDImagePyramid* pyramid_) :
    _id(id_) {
    if (_framepoints) {
      throw std::runtime_error("Frame::Frame|ERROR, framepoints already there");
    }
    _timestamp   = timestamp_;
    _framepoints = new FramePointVector();
    _framepoints->reserve(num_framepoints_);
    _cv_intensity_image = cv_intensity_image_;
    _pyramid            = pyramid_;
  }

  Frame::~Frame() {
    if (_framepoints) {
      for (FramePoint* fp : *_framepoints) {
        delete fp;
      }
      _framepoints->clear();
      delete _framepoints;
    }
  }

  size_t FramesContainer::_frame_id_generator = 0;
  void FramesContainer::resetIDGenerator() {
    _frame_id_generator = 0;
  }

  FramesContainer::FramesContainer() {
  }

  FramesContainer::~FramesContainer() {
    if (_frames_pull.size()) {
      for (Frame* f : _frames_pull) {
        delete f;
      }
      _frames_pull.clear();
    }
    FramesContainer::resetIDGenerator();
  }

  Frame* FramesContainer::createFrame(const double& timestamp_,
                                      const size_t& num_framepoints_,
                                      const cv::Mat& cv_intensity_image_) {
    Frame* f = new Frame(_frame_id_generator++, timestamp_, num_framepoints_, cv_intensity_image_);
    _frames_pull.push_back(f);
    _frame_map.insert(std::make_pair(f->id(), f));
    return f;
  }

  Frame* FramesContainer::createFrame(const double& timestamp_,
                                      const size_t& num_framepoints_,
                                      const cv::Mat& cv_intensity_image_,
                                      MDImagePyramid* pyramid_) {
    Frame* f =
      new Frame(_frame_id_generator++, timestamp_, num_framepoints_, cv_intensity_image_, pyramid_);
    _frames_pull.push_back(f);
    _frame_map.insert(std::make_pair(f->id(), f));
    return f;
  }

} // namespace md_slam_closures

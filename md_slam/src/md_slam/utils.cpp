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

#include "utils.cuh"
#include <srrg_system_utils/shell_colors.h>

namespace md_slam {
  using namespace srrg2_core;

  void prepareImage(ImageFloat& dest,
                    const ImageFloat& src,
                    int max_scale,
                    int row_scale,
                    int col_scale,
                    bool suppress_zero) {
    int d_rows = src.rows() / row_scale;
    int d_cols = src.cols() / col_scale;
    d_rows     = (d_rows / max_scale) * max_scale;
    d_cols     = (d_cols / max_scale) * max_scale;

    int s_rows = d_rows * row_scale;
    int s_cols = d_cols * col_scale;
    // cerr << "d_rows: " << d_rows << " d_cols: " << d_cols << endl;
    // cerr << "s_rows: " << s_rows << " s_cols: " << s_cols << endl;

    dest.resize(d_rows, d_cols);
    dest.fill(0);
    ImageInt counts(d_rows, d_cols);
    counts.fill(0);
    for (int r = 0; r < s_rows; ++r) {
      int dr = r / row_scale;
      for (int c = 0; c < s_cols; ++c) {
        int dc  = c / col_scale;
        float v = src.at(r, c);
        if (suppress_zero && !v)
          continue;
        dest.at(dr, dc) += src.at(r, c);
        ++counts.at(dr, dc);
      }
    }
    for (int r = 0; r < d_rows; ++r) {
      for (int c = 0; c < d_cols; ++c) {
        int cnt = counts.at(r, c);
        if (cnt)
          dest.at(r, c) *= (1. / cnt);
      }
    }
  }

  bool loadImage(BaseImage& img, const std::string filename) {
    // try for depth
    ImageUInt16* depth = dynamic_cast<ImageUInt16*>(&img);
    if (depth) {
      // cerr << "loading image from file [" << filename << "] (depth, uint16)" << endl;
      cv::Mat depth_cv = cv::imread(filename, CV_LOAD_IMAGE_ANYDEPTH);
      assert(!depth_cv.empty() &&
             std::string("utils::loadImage|unable to load depth [ " + filename + " ]").c_str());
      depth->fromCv(depth_cv);
      size_t accumulator = 0;
      size_t num_pixels  = 0;
      for (size_t r = 0; r < depth->rows(); ++r)
        for (size_t c = 0; c < depth->cols(); ++c) {
          uint16_t v = depth->at(r, c);
          if (v) {
            accumulator += v;
            ++num_pixels;
          }
        }
      // cerr << "loaded, rows: " << depth->rows()
      //      << " cols: " << depth->cols()
      //      << " pix: " << depth->cols() * depth->rows()
      //      << " avg: " << (float)accumulator/float(num_pixels)
      //      << " null: " << depth->cols() * depth->rows() - num_pixels << endl;

      return true;
    }

    // try for rgb
    ImageVector3uc* rgb = dynamic_cast<ImageVector3uc*>(&img);
    if (rgb) {
      // cerr << "loading image from file [" << filename << "] (intensity, rgb)" << endl;
      cv::Mat rgb_cv = cv::imread(filename, CV_LOAD_IMAGE_UNCHANGED);
      assert(!rgb_cv.empty() &&
             std::string("utils::loadImage|unable to load rgb [ " + filename + " ]").c_str());
      rgb->fromCv(rgb_cv);
      return true;
    }

    // try for monochrome
    ImageUInt8* intensity = dynamic_cast<ImageUInt8*>(&img);
    if (intensity) {
      // cerr << "loading image from file [" << filename << "] (intensity, uint8)" << endl;
      cv::Mat intensity_cv = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
      assert(
        !intensity_cv.empty() &&
        std::string("utils::loadImage | unable to load intensity [ " + filename + " ]").c_str());
      intensity->fromCv(intensity_cv);
      return true;
    }
    return false;
  }

  void showImage(const BaseImage& img, const std::string& title, int wait_time) {
    cv::Mat m;
    img.toCv(m);
    cv::imshow(title, m);
    if (wait_time > -1)
      cv::waitKey(wait_time);
  }

  void sparseProjection(Point2fVectorCloud& dest_,
                        const Point3fVectorCloud& source_,
                        const Matrix3f& cam_matrix_,
                        const float& min_depth_,
                        const float& max_depth_,
                        const int& n_rows_,
                        const int& n_cols_,
                        const CameraType& camera_type_) {
    assert((n_rows_ != 0 || n_cols_ != 0) && "sparseProjection|num rows and cols of "
                                             "image not set");
    assert(camera_type_ != CameraType::Unknown && "sparseProjection|camera type not set");

    size_t i = 0;
    dest_.resize(source_.size());
    for (auto it = source_.begin(); it != source_.end(); ++it, ++i) {
      const auto& point_full = *it;
      const Vector3f& point  = point_full.coordinates();

      float depth           = 0.f;
      Vector3f camera_point = Vector3f::Zero();
      // initialize
      dest_[i].status        = POINT_STATUS::Invalid;
      dest_[i].coordinates() = Vector2f::Zero();
      Vector2f image_point   = Vector2f::Zero();

      const bool is_good = project(
        image_point, camera_point, depth, point, camera_type_, cam_matrix_, min_depth_, max_depth_);
      if (!is_good)
        continue;

      const int irow = cvRound(image_point.y());
      const int icol = cvRound(image_point.x());

      if (irow > n_rows_ || icol > n_cols_)
        continue;

      dest_[i].status        = POINT_STATUS::Valid;
      dest_[i].coordinates() = image_point;
    }
  }

} // namespace md_slam

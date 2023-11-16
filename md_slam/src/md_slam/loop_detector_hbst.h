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

#pragma once
#include <srrg_hbst/types/binary_matchable.hpp>
#include <srrg_hbst/types/binary_tree.hpp>

#include "loop_detector_base.h"

namespace md_slam_closures {

  class LoopDetectorHBST : public md_slam_closures::LoopDetectorBase {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // clang-format off
    //! @brief a bit of usings
    using ThisType = LoopDetectorHBST;
    using BaseType = LoopDetectorBase;

    static constexpr size_t DESCRIPTOR_SIZE_BITS = BaseType::DESCRIPTOR_SIZE_ORB;
    using BinaryEntry       = srrg_hbst::BinaryMatchable<FramePoint*, DESCRIPTOR_SIZE_BITS>;
    using BinaryEntryNode   = srrg_hbst::BinaryNode<BinaryEntry>;
    using BinaryEntryVector = srrg_hbst::BinaryNode<BinaryEntry>::MatchableVector;
    using HSBTTree          = srrg_hbst::BinaryTree<BinaryEntryNode>;

    PARAM(srrg2_core::PropertyUnsignedInt, max_descriptor_distance, "maximum distance (Hamming) between descriptors", 25, nullptr);
    PARAM(srrg2_core::PropertyFloat, minimum_matching_ratio, "minimum ratio between the number of framepoints in the query and in the match", 0.1f, nullptr);
    PARAM(srrg2_core::PropertyBool, enable_matches_filtering, "enables matches container filtering based on the match score", true, nullptr);
    // clang-format on

    LoopDetectorHBST();
    virtual ~LoopDetectorHBST() = default;

    //! @brief resets the tree
    void reset() override;

    //! @brief adds the current frame to the tree and performs also a query
    void compute() override;

  protected:
    //! @brief hbst tree
    HSBTTree _tree;
  };

  using LoopDetectorHBSTPtr = std::shared_ptr<LoopDetectorHBST>;
} // namespace md_slam_closures

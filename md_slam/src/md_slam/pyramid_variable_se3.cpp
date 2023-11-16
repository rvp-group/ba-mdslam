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

#include "pyramid_variable_se3.h"

namespace md_slam {
  using namespace srrg2_solver;
  void MDVariableSE3::serialize(srrg2_core::ObjectData& odata, srrg2_core::IdContext& context) {
    VariableSE3QuaternionRight::serialize(odata, context);
    ObjectData* reference_data = new ObjectData;
    _pyramid.serialize(*reference_data, context);
    odata.setDouble("timestamp", pyramid()->timestamp());
    odata.setField("pyramid", reference_data);
  }

  void MDVariableSE3::deserialize(srrg2_core::ObjectData& odata, srrg2_core::IdContext& context) {
    VariableSE3QuaternionRight::deserialize(odata, context);
    _timestamp                 = odata.getDouble("timestamp");
    ObjectData* reference_data = dynamic_cast<ObjectData*>(odata.getField("pyramid"));
    if (!reference_data) {
      throw std::runtime_error("MDVariableSE3::deserialize | no zero level");
    }
    _pyramid.deserialize(*reference_data, context);
  }
} // namespace md_slam

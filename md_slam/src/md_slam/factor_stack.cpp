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

#include "factor_stack.h"
#include <srrg_solver/solver_core/solver.h>

namespace md_slam {

  using namespace srrg2_core;

  // stack of factor representing pyramid
  // order -> tail, level with lower resolution (top pyramid)
  // order -> front, level with higher resolution (bottom pyramid)

  void MDFactorStack::setFixed(MDImagePyramid& pyramid) {
    _fixed = &pyramid;
  }

  void MDFactorStack::setMoving(MDImagePyramid& pyramid) {
    _moving = &pyramid;
  }

  void MDFactorStack::makeFactors() {
    // sanity check: pyramids should have same depth
    assert(_fixed && "MDFactorStack::makeFactors|not fixed set");
    size_t levels = _fixed->numLevels();
    resize(levels);
    for (size_t l = 0; l < levels; ++l) {
      MDFactorPtr& factor = at(l);
      factor.reset(new MDFactor);
      factor->setLevel(l);
    }
  }

  void MDFactorStack::_fixedPyramidToDevice() {
    const uint8_t levels = _fixed->numLevels();
    for (uint8_t l = 0; l < levels; ++l) {
      _fixed->at(l).get()->matrix.toDevice();
    }
  }

  void MDFactorStack::assignPyramids() {
    // sanity check: pyramids should have same number of levels
    assert(_moving && "MDFactorStack::assignPyramids|not moving set");
    assert(_fixed && "MDFactorStack::assignPyramids|not fixed set");
    assert(_fixed->numLevels() == _moving->numLevels() &&
           "MDFactorStack::assignPyramids|fixed and moving num levels differ");

    // this allows to copy keyframe pyramid just once
    if (_fixed != _prev_fixed) {
      _fixedPyramidToDevice();
      _prev_fixed = _fixed;
    }

    const uint8_t levels = _fixed->numLevels();
    for (uint8_t l = 0; l < levels; ++l) {
      MDPyramidLevel* l_fix = _fixed->at(l).get(); // retrieve image ptr

      MDPyramidLevel& l_mov = *_moving->at(l); // retrieve image ptr
      l_mov.matrix.toDevice();                 // copy image on device

      MDFactorPtr& factor = at(l);
      assert(factor && "MDFactorStack::assignPyramids|not factor, forgot to call makeFactors()?");
      factor->setFixed(l_fix);
      factor->setMoving(l_mov);
    }
  }

  void MDFactorStack::setVariableId(srrg2_solver::VariableBase::Id id) {
    for (auto& f : (*this)) {
      f->setVariableId(0, id);
    }
  }

  void MDFactorStack::addFactors(srrg2_solver::FactorGraph& graph) {
    for (auto& f : (*this)) {
      graph.addFactor(f);
    }
  }

  void MDFactorShowAction::doAction() {
    if (!_md_factors) {
      return;
    }
    if (_md_factors->empty())
      return;
    _need_redraw = true;
    draw();
  }

  void MDFactorShowAction::_drawImpl(ViewerCanvasPtr gl_canvas_) const {
    if (_md_factors->empty())
      return;
    if (_solver_ptr->currentLevel() < 0 || _solver_ptr->currentLevel() >= (int) _md_factors->size())
      return;
    for (auto& f : *_md_factors) {
      ImageVector3f canvas;
      f->toTiledImage(canvas);
      cv::Mat dest;
      canvas.toCv(dest);
      gl_canvas_->putImage(dest);
    }
    gl_canvas_->flush();
  }

} // namespace md_slam

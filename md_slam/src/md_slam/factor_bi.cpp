#include "factor_bi.h"
#include "utils.h"
#include <srrg_solver/solver_core/factor_impl.cpp>

namespace md_slam {
  using namespace srrg2_core;

  inline Matrix3_6f Jicp(const Vector3f& p) {
    Matrix3_6f J_icp;
    J_icp.block<3, 3>(0, 0).setIdentity();
    J_icp.block<3, 3>(0, 3) = -2 * geometry3d::skew(p);
    return J_icp;
  }

  PointStatusFlag MDFactorBivariable::errorAndJacobian(srrg2_core::Vector5f& e_,
                                                       srrg2_core::Matrix5_6f& Ji_,
                                                       srrg2_core::Matrix5_6f& Jj_,
                                                       WorkspaceEntry& entry_,
                                                       bool chi_only) {
    // initialization and aliasing
    PointStatusFlag status            = Good;
    const float z                     = entry_.depth();
    const float iz                    = 1.f / z;
    const Vector3f& point             = entry_._point;
    const Vector3f& normal            = entry_._normal;
    const Vector3f& transformed_point = entry_._transformed_point;
    const Vector3f& camera_point      = entry_._camera_point;
    const Vector2f& image_point       = entry_._image_point;
    e_.setZero();
    Jj_.setZero();
    Ji_.setZero();

    Vector5f measurement;
    Matrix5_2f image_derivatives;

    const bool ok = _level_ptr->getSubPixel(measurement, image_derivatives, image_point);
    if (!ok) {
      return Masked;
    }

    // error calculation
    e_            = entry_._prediction - measurement;
    entry_._error = e_;
    e_(0) *= _omega_intensity_sqrt;
    e_(1) *= _omega_depth_sqrt;
    e_.tail(3) *= _omega_normals_sqrt;

    // if the distance between a point and the corresponding one is too big, we drop
    if (std::pow(e_(1), 2) > _depth_error_rejection_threshold)
      return DepthError;
    if (chi_only)
      return status;

    Matrix3_6f A_j = -_sensor_offset_inverse.linear() * Jicp((const Vector3f)(_X_ji * point));
    Matrix3_6f A_i = _SX.linear() * Jicp(point);

    // extract values from hom for readability
    const float iz2 = iz * iz;

    // extract the valiues from camera matrix
    const float& fx = _camera_matrix(0, 0);
    const float& fy = _camera_matrix(1, 1);
    const float& cx = _camera_matrix(0, 2);
    const float& cy = _camera_matrix(1, 2);

    // J proj unchanged wrt to md factor monolita
    // computes J_hom*K explicitly to avoid matrix multiplication and stores it in J_proj
    Matrix2_3f J_proj = Matrix2_3f::Zero();

    switch (_camera_type) {
      case CameraType::Pinhole:
        // fill the left  and the right 2x3 blocks of J_proj with J_hom*K
        J_proj(0, 0) = fx * iz;
        J_proj(0, 2) = cx * iz - camera_point.x() * iz2;
        J_proj(1, 1) = fy * iz;
        J_proj(1, 2) = cy * iz - camera_point.y() * iz2;

        // add the jacobian of depth prediction to row 1.
        Jj_.row(1) = A_j.row(2);
        Ji_.row(1) = A_i.row(2);

        break;
      case CameraType::Spherical: {
        const float ir    = iz;
        const float ir2   = iz2;
        const float rxy2  = transformed_point.head<2>().squaredNorm();
        const float irxy2 = 1. / rxy2;
        const float rxy   = sqrt(rxy2);
        const float irxy  = 1. / rxy;

        J_proj << -fx * transformed_point.y() * irxy2, // 1st row
          fx * transformed_point.x() * irxy2, 0,
          -fy * transformed_point.x() * transformed_point.z() * irxy * ir2, // 2nd row
          -fy * transformed_point.y() * transformed_point.z() * irxy * ir2, fy * rxy * ir2;

        Matrix1_3f J_range; // jacobian of range(x,y,z)
        J_range << transformed_point.x() * ir, transformed_point.y() * ir,
          transformed_point.z() * ir;

        // add the jacobian of range prediction to row 1.
        Jj_.row(1) = J_range * A_j;
        Ji_.row(1) = J_range * A_i;
      } break;
      default:
        throw std::runtime_error("MDFactorBivariable::errorAndJacobian|unknown camera model");
    }

    Jj_.noalias() -= image_derivatives * J_proj * A_j;
    Ji_.noalias() -= image_derivatives * J_proj * A_i;

    Jj_.block<3, 3>(2, 3).noalias() += _sensor_offset_inverse.linear() * 2.f *
                                       geometry3d::skew((const Vector3f)(_X_ji.linear() * normal));

    Ji_.block<3, 3>(2, 3).noalias() +=
      _SX.linear() * -2.f * geometry3d::skew((const Vector3f) normal);

    // omega is diagonal matrix
    // to avoid multiplications we premultiply the rows of J by sqrt of diag
    // elements
    Jj_.row(0) *= _omega_intensity_sqrt;
    Jj_.row(1) *= _omega_depth_sqrt;
    Jj_.block<3, 2>(2, 0) *= _omega_normals_sqrt;

    Ji_.row(0) *= _omega_intensity_sqrt;
    Ji_.row(1) *= _omega_depth_sqrt;
    Ji_.block<3, 2>(2, 0) *= _omega_normals_sqrt;

    return status;
  }

  // https : // en.wikipedia.org/wiki/Kahan_summation_algorithm
  template <typename MatType>
  inline MatType kahanSum(const std::vector<MatType>& vec_) {
    MatType sum = vec_[0];
    MatType c   = MatType::Zero(); // a running compensation for lost low-order bits
    for (size_t i = 1; i < vec_.size(); ++i) {
      const MatType y = vec_[i] + c; // so far so good, c is 0
      const MatType t = sum + y;     // sum is big, y small, so low-order digits of y are lost
      c = (t - sum) - y; // recovers the high-order part of y; subtract y recovers -(low part of y)
      sum = t; // next time around, the lost low part will be added to y in a fresh attempt.
    }
    return sum;
  }

  void MDFactorBivariable::linearize(bool chi_only) {
    Chrono t_lin("linearize_bivariable", &timings, false);
    _omega_intensity_sqrt = std::sqrt(_omega_intensity);
    _omega_depth_sqrt     = std::sqrt(_omega_depth);
    _omega_normals_sqrt   = std::sqrt(_omega_normals);
    _neg2rotSX            = -2.f * _SX.linear();

    float total_chi    = 0;
    size_t num_inliers = 0;

    const int& system_size = _workspace->size();
    const double scaling   = 1.0 / system_size;

    // init system matrices Hii, Hij, Hjj, bi, bj
    // allocate only once
    static std::vector<Matrix6d> Hii_container;
    static std::vector<Matrix6d> Hij_container;
    static std::vector<Matrix6d> Hjj_container;
    static std::vector<Vector6d> bi_container;
    static std::vector<Vector6d> bj_container;

    Hii_container.reserve(system_size);
    Hij_container.reserve(system_size);
    Hjj_container.reserve(system_size);
    bi_container.reserve(system_size);
    bj_container.reserve(system_size);

    int num_good = 0;
    for (size_t r = 0; r < _workspace->rows(); ++r) {
      WorkspaceEntry* entry_ptr = _workspace->rowPtr(r);
      for (size_t c = 0; c < _workspace->cols(); ++c, ++entry_ptr) {
        WorkspaceEntry& entry = *entry_ptr;
        const int idx         = entry._index;
        Vector5f e;
        Matrix5_6f J_i, J_j;

        if (idx < 0)
          continue;

        PointStatusFlag& status = entry._status;
        if (status != Good)
          continue;

        status = errorAndJacobian(e, J_i, J_j, entry, chi_only);
        if (status != Good)
          continue;

        ++num_good;
        float chi    = e.dot(e);
        entry._chi   = chi;
        float lambda = 1.f;
        if (chi > _kernel_chi_threshold) {
          lambda = sqrt(_kernel_chi_threshold / chi);
        } else {
          ++num_inliers;
        }
        total_chi += chi;
        if (!chi_only) {
          // TODO don't compute - copy Hii and Hjj lower block
          const Matrix6d Hii_contrib = (J_i.transpose() * J_i * lambda).cast<double>();
          const Matrix6d Hij_contrib = (J_i.transpose() * J_j * lambda).cast<double>();
          const Matrix6d Hjj_contrib = (J_j.transpose() * J_j * lambda).cast<double>();
          const Vector6d bi_contrib  = (J_i.transpose() * e * lambda).cast<double>();
          const Vector6d bj_contrib  = (J_j.transpose() * e * lambda).cast<double>();
          // insert and scale
          Hii_container.push_back(Hii_contrib * scaling);
          Hij_container.push_back(Hij_contrib * scaling);
          Hjj_container.push_back(Hjj_contrib * scaling);
          bi_container.push_back(bi_contrib * scaling);
          bj_container.push_back(bj_contrib * scaling);
        }
      }
    }

    // TODO
    // if num good is 0 fucks chi up, if we don't have any inliers is
    // likely num good is very low, if this happen better not moving variable
    if (!num_good || !num_inliers) {
      _stats.status = srrg2_solver::FactorStats::Suppressed;
      return;
    }

    _stats.chi = total_chi / num_good;

    // initially container are empty
    if (bi_container.empty()) {
      return;
    }

    // use compensated sum to avoid loss of precision and clear containers
    Matrix6d H_tmp_blocks[3];
    Vector6d b_tmp_blocks[2];

    H_tmp_blocks[0] = kahanSum<Matrix6d>(Hii_container);
    H_tmp_blocks[1] = kahanSum<Matrix6d>(Hij_container);
    if (_H_transpose[1]) // transpose if indices in solver are flipped
      H_tmp_blocks[1].transposeInPlace();
    H_tmp_blocks[2] = kahanSum<Matrix6d>(Hjj_container);
    b_tmp_blocks[0] = kahanSum<Vector6d>(bi_container);
    b_tmp_blocks[1] = kahanSum<Vector6d>(bj_container);

    Hii_container.clear();
    Hij_container.clear();
    Hjj_container.clear();
    bi_container.clear();
    bj_container.clear();

    if (!chi_only) {
      // retrieve the blocks of H and b for writing (+=, noalias)
      for (int r = 0; r < 2; ++r) {
        if (!this->_b_blocks[r])
          continue;
        Eigen::Map<Vector6f> _b(this->_b_blocks[r]->storage());
        _b.noalias() -= b_tmp_blocks[r].cast<float>();
        for (int c = r; c < 2; ++c) {
          int linear_index = blockOffset(r, c);
          if (!this->_H_blocks[linear_index])
            continue;
          Eigen::Map<Matrix6f> _H(this->_H_blocks[linear_index]->storage());
          _H.noalias() += H_tmp_blocks[linear_index].cast<float>();
        }
      }
    }
  }

  MDFactorBivariable::CloudMap MDFactorBivariable::_cloud_map;

  void MDFactorBivariable::setMoving(MDMatrixCloud& cloud_,
                                     MDPyramidLevelPtr pyr_level_,
                                     const size_t& var_id_,
                                     const size_t& level_) {
    // TODO once level is over, remove all clouds belonging to that level
    const auto key = std::pair<size_t, size_t>(var_id_, level_);
    if (auto it{_cloud_map.find(key)}; it != std::end(_cloud_map)) {
      // if cloud exists in map, retrieve
      cloud_ = *(it->second);
    } else {
      // if does not exist create one from pyramid level
      std::shared_ptr<MDMatrixCloud> cloud = std::make_shared<MDMatrixCloud>();
      pyr_level_->toCloud(*cloud);
      _cloud_map.insert({key, cloud});
      cloud_ = *cloud;
    }
  }

  void MDFactorBivariable::compute(bool chi_only, bool force) {
    if (!this->isActive() && !force)
      return;

    if (level() != this->currentLevel()) {
      _stats.status = srrg2_solver::FactorStats::Suppressed;
      return;
    }
    _stats.status = srrg2_solver::FactorStats::Inlier;

    // retrieve the variables
    auto& v_i = _variables.at<0>();
    auto& v_j = _variables.at<1>();

    // retrieve the pyramid level from the variables
    auto pyr_i = v_i->pyramid()->at(this->level());
    auto pyr_j = v_j->pyramid()->at(this->level());

    // retrieve level to inverse project from factory, do this operation only once
    setMoving(_cloud, pyr_i, v_i->graphId(), this->level());
    // retrieve other pyr level
    setFixed(*pyr_j);

    // store current estimates
    _X_i     = v_i->estimate();
    _X_j     = v_j->estimate();
    _inv_X_j = _X_j.inverse();
    _X_ji    = _inv_X_j * _X_i;
    setMovingInFixedEstimate(_X_ji);

    // project cloud using current estimate
    computeProjections();

    // for each point, compute the jacobians, check the state of the point
    // fill H and b to get new estimate
    linearize(chi_only);
    _workspace.reset();
  }

  void MDFactorBivariable::serialize(ObjectData& odata, IdContext& context) {
    Identifiable::serialize(odata, context);
    odata.setInt("graph_id", graphId());
    odata.setBool("enabled", enabled());
    odata.setInt("level", level());
    ArrayData* adata = new ArrayData;
    for (int pos = 0; pos < NumVariables; ++pos) {
      adata->add((int) variableId(pos));
    }
    odata.setField("variables", adata);
    odata.setFloat("omega_intensity", _omega_intensity);
    odata.setFloat("omega_depth", _omega_depth);
    odata.setFloat("omega_normals", _omega_normals);
    odata.setFloat("dept_rejection", _depth_error_rejection_threshold);
    odata.setFloat("kernel_chi", _kernel_chi_threshold);
  }

  void MDFactorBivariable::deserialize(ObjectData& odata, IdContext& context) {
    Identifiable::deserialize(odata, context);
    _graph_id = odata.getInt("graph_id");
    if (odata.getField("enabled")) {
      FactorBase::_enabled = odata.getBool("enabled");
    }
    if (odata.getField("level")) {
      FactorBase::setLevel(odata.getInt("level"));
    }
    ArrayData* adata = dynamic_cast<ArrayData*>(odata.getField("variables"));
    int pos          = 0;
    for (auto it = adata->begin(); it != adata->end(); ++it) {
      ThisType::_variables.setGraphId(pos, (*it)->getInt());
      ++pos;
    }
    _omega_intensity                 = odata.getFloat("omega_intensity");
    _omega_depth                     = odata.getFloat("omega_depth");
    _omega_normals                   = odata.getFloat("omega_normals");
    _depth_error_rejection_threshold = odata.getFloat("dept_rejection");
    _kernel_chi_threshold            = odata.getFloat("kernel_chi");
  }

} // namespace md_slam

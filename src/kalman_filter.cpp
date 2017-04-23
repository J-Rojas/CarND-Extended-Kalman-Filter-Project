#include "kalman_filter.h"

using Eigen::Matrix;
using Eigen::MatrixXd;
using Eigen::VectorXd;

static MatrixXd I_ = Matrix<double, 4, 4>::Identity();


KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
    * predict the state
  */
    x_ = F_ * x_;
    P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::UpdateEKF(const Eigen::VectorXd &z, const Eigen::VectorXd& xTrans, NormalizeFunc normalizeFunc) {
  /**
    * update the state by using Extended Kalman Filter equations
  */
    VectorXd y = z - xTrans;
    if (normalizeFunc != nullptr)
        y = normalizeFunc(y);
    MatrixXd Ht = H_.transpose();
    MatrixXd S = H_ * P_ * Ht + R_;
    MatrixXd K = P_ * Ht * S.inverse();

    x_ = x_ + K * y;
    P_ = (I_ - K * H_) * P_;
}

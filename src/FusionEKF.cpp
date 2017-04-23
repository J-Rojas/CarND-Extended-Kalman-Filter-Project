#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>
#include <math.h>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  H_radar_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
        0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
        0, 0.0009, 0,
        0, 0, 0.09;

    /**
    * Finish initializing the FusionEKF.
    * Set the process and measurement noises
    */
    // state vector
    VectorXd x(4);
    // state covariance matrix
    MatrixXd P(4, 4);
    // state transition matrix
    MatrixXd F(4, 4);
    // process covariance matrix
    MatrixXd Q(4, 4);

    x << 0, 0, 0, 0;

    //These values were empirically determined to reduce the RMSE within the necessary thresholds defined by the project rubric.
    P << 0.5, -2.0, -0.5, 0,
         0, 0, 0, 0,
         0.5, -2.0, 0.5, 0,
         0, 0, 0, 0;

    F << 1, 0, 0, 0,
         0, 1, 0, 0,
         0, 0, 1, 0,
         0, 0, 0, 1;
    Q << 0, 0, 0, 0,
         0, 0, 0, 0,
         0, 0, 0, 0,
         0, 0, 0, 0;
    H_laser_ <<
         1, 0, 0, 0,
         0, 1, 0, 0;

    ekf_.Init(
        x, P, F, H_laser_, R_laser_, Q
    );

}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

VectorXd polar_to_cartesian(const VectorXd& in) {
    VectorXd cartesian(4);

    double cosangle = cos(in(1));
    double sinangle = sin(in(1));
    double range = in(0);
    double range_rate = in(2);
    double x = cosangle * range;
    double y = sinangle * range;
    double rate_x = cosangle * range_rate;
    double rate_y = sinangle * range_rate;

    cartesian << x, y, rate_x, rate_y;

    return cartesian;
}

VectorXd cartesian_to_polar(const VectorXd& in) {
    VectorXd polar(3);

    double px = in(0), py = in(1);
    double vx = in(2), vy = in(3);

    double norm = sqrt(px * px + py * py);

    //normalize angle
    polar << sqrt(px * px + py * py),
             atan2(py,px),
            (px * vx + py * vy) / norm;

    return polar;
}

VectorXd normalizePhi(VectorXd& vec) {

    double phi = vec(1);

    while (phi < -M_PI)
        phi += M_PI * 2;
    while (phi > M_PI)
        phi -= M_PI * 2;

    vec(1) = phi;

    return vec;
}

void update_prediction_time_dependencies(KalmanFilter& kf, double dt, double noise_ax, double noise_ay) {

    /*
     * The State Transition Matrix needs the time delta in 2 of the matrix components
     * related to velocity
     */

    kf.F_(0, 2) = kf.F_(1, 3) = dt;

    /*
     * The Process Noise Covariance Matrix needs a full calculation since most of the components are time dependent.
     *
     * Noise Vector
     * v = (Vpx, Vpy, Vvx, Vvy) = ((ax * dt)^2 / 2, (ay * dt)^2 / 2, ax * dt, ay * dt)
     * Q = E[vv^T]
     *
     */

    double t4 = pow(dt, 4);
    double t3 = pow(dt, 3);
    double t2 = pow(dt, 2);
    kf.Q_ << (t4 * noise_ax) / 4, 0, (t3 * noise_ax) / 2, 0,
            0, (t4 * noise_ay) / 4, 0, (t3 * noise_ay) / 2,
            (t3 * noise_ax) / 2, 0, (t2 * noise_ax), 0,
            0, (t3 * noise_ay) / 2, 0, (t2 * noise_ay);
}


void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
    */
    // first measurement
    cout << "EKF: " << endl;

    previous_timestamp_ = measurement_pack.timestamp_;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      ekf_.x_ = polar_to_cartesian(measurement_pack.raw_measurements_);
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
      ekf_.x_ << measurement_pack.raw_measurements_(0), measurement_pack.raw_measurements_(1), 0, 0;
    }

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  /**
     * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
     * Update the process noise covariance matrix.
     * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */
  double dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;
  update_prediction_time_dependencies(ekf_, dt, 9, 9);

  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
     *
     * NOTE: Instead of using Update(), I rely on the generalized Extended Kalman Filter equations for both lidar/radar.
     * The specifics for sensor and model non-linearity are handled in this module instead of the generic KalmanFilter class.
     * Notice that the H_ and R_ matrices are updated and the predicted measurements are transformed into the proper format
     * to match the measurement state.
     *
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {

    // Radar updates
    VectorXd predictedMeasurement = cartesian_to_polar(ekf_.x_);

    //update the Hj matrix
    ekf_.H_ = Tools::CalculateJacobian(ekf_.x_);
    ekf_.R_ = R_radar_;

    //The radar data needs normalization, specifically the phi angle error
    ekf_.UpdateEKF(measurement_pack.raw_measurements_, predictedMeasurement, &normalizePhi);

  } else {

    // Laser update
    VectorXd predictedMeasurement(2);
    predictedMeasurement << ekf_.x_(0), ekf_.x_(1);

    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;

    //The Lidar data does not need normalization since it is a linear model
    ekf_.UpdateEKF(measurement_pack.raw_measurements_, predictedMeasurement, nullptr);

  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}

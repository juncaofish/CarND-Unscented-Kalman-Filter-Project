#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
    * Calculate the RMSE here.
  */
    VectorXd rmse = VectorXd(4);
    rmse << 0,0,0,0;

    size_t estimations_size = estimations.size();
    size_t groud_truth_size = ground_truth.size();
    if(estimations_size == 0 || estimations_size != groud_truth_size){
        cout << "Wrong size of estimations or ground truth vector." << endl;
        return rmse;
    }
    for (auto i=0; i<estimations_size;i++){

        VectorXd residuals = estimations[i] - ground_truth[i];
        residuals = residuals.array() * residuals.array();
        rmse += residuals;

    }
    rmse /= estimations_size;
    rmse = rmse.array().sqrt();
    return rmse;

}
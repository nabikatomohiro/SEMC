#ifndef BASE_SPECTRAL_DECONVOLUTION_HPP
#define BASE_SPECTRAL_DECONVOLUTION_HPP

#include "../../core/base_emc.hpp"
#include "../../Eigen/Dense"

#define PI 3.14159265358979323846


struct HighDimensionalData{
    double sigma_beta;
    double sigma_eps;
    int p;
    vector<vector<double> > x;
    vector<double> y;
};

inline double ErrorExhaustiveSearch(const HighDimensionalData data, const vector<int>& parameters, int data_num, int K){
    double output = 0;
    //後で保存型にしておく
    output += K*log(data.sigma_beta);
    output += data_num*log(2*PI)/2;
    output += data_num*log(data.sigma_eps);
    for(int i=0; i<data_num; i++){
        output += data.y[i]*data.y[i]/(2*data.sigma_eps*data.sigma_eps);
    }
    Eigen::MatrixXd X_I = Eigen::MatrixXd::Zero(K, data_num);
    for(int i=0; i<K; i++){
        for(int j=0; j<data_num; j++){
            X_I(i,j) = data.x[parameters[i]][j];
        }
    }
    Eigen::MatrixXd y_mat = Eigen::MatrixXd::Zero(1, data_num);
    for(int i=0; i<data_num; i++){
        y_mat(0,i) = data.y[i];
    }
    Eigen::MatrixXd Delta_inv = X_I*X_I.transpose()/data.sigma_eps/data.sigma_eps + Eigen::MatrixXd::Identity(K, K)/data.sigma_beta/data.sigma_beta;
    Eigen::MatrixXd Delta = Delta_inv.inverse();
    Eigen::MatrixXd mu = Delta*X_I*y_mat.transpose()/data.sigma_eps/data.sigma_eps;
    output -= (mu.transpose()*Delta_inv*mu/2).value();
    output -= log(Delta.determinant())/2;
    return output/data_num;
};

inline vector<int> n_choose_k(int n, int k, mt19937& engine){
    vector<int> output;
    for(int i=0; i<k; i++){
        uniform_int_distribution<> dist(0, n-i-1);
        int r = dist(engine);
        for(int j=0; j<output.size(); j++){
            if(output[j] <= r){
                r += 1;
            }
        }
        output.push_back(r);
        sort(output.begin(), output.end());
    }
    return output;
};

#endif
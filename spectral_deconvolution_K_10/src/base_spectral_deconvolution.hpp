#ifndef BASE_SPECTRAL_DECONVOLUTION_HPP
#define BASE_SPECTRAL_DECONVOLUTION_HPP

#include "../../core/base_emc.hpp"

struct GaussData{
    vector<double> x;
    vector<double> y;
    double sigma;
};

inline double ErrorSpectralDeconvolution(const vector<vector<double> >& parameters, GaussData data, int data_num, int model_dim){
    double error = 0;
    for(int i=0; i<data_num; i++){
        // 関数値の計算
        double f_x = 0;
        for(int j=0; j<model_dim; j++){
            f_x += exp(parameters[j][0])*exp(-exp(parameters[j][2])*(data.x[i]-parameters[j][1])*(data.x[i]-parameters[j][1])/2);
        }
        error += (data.y[i] - f_x)*(data.y[i] - f_x);
    }
    return error/(2*data.sigma*data.sigma)/data_num;
};

// double ErrorSpectralDeconvolutionSlow_2(const vector<vector<double> >& parameters, GaussData data, int data_num, int model_dim){
//     double error = 0;
//     for(int i=0; i<data_num; i++){
//         // 関数値の計算
//         double f_x = 0;
//         for(int j=0; j<model_dim; j++){
//             f_x += (parameters[j][0])*exp(-(parameters[j][2])*(data.x[i]-parameters[j][1])*(data.x[i]-parameters[j][1])/2);
//         }
//         error += (data.y[i] - f_x)*(data.y[i] - f_x);
//     }
//     return error/(2*data.sigma*data.sigma)/data_num;
// };

// double ErrorSpectralDeconvolution(const vector<vector<double> >& subparameter, GaussData data, int data_num, int model_dim){
//     double error = 0;
//     vector<double> f_x(data_num,0);
//     for(int i=0; i < model_dim; i++){
//         for(int j=0; j<data_num; j++){
//             f_x[j] += subparameter[i][j];
//         }
//     }
//     for(int i=0; i<data_num; i++){
//         // 関数値の計算
//         error += (data.y[i] - f_x[i])*(data.y[i] - f_x[i]);
//     }
//     return error/(2*data.sigma*data.sigma)/data_num;
// };

// vector<double> FunctionSpectralDeconvolution(const vector<double>& parameter, GaussData data, int data_num, int model_dim){
//     vector<double> f_x(data_num,0);
//     for(int i=0; i<data_num; i++){
//         f_x[i] += exp(parameter[0])*exp(-exp(parameter[2])*(data.x[i]-parameter[1])*(data.x[i]-parameter[1])/2);
//     }
//     return f_x;
// };

// double EnergyCalculationPartialUpdate(const vector<vector<double> > & subparameters, const vector<double>& subparameter, int model_id, int model_dim, int data_num, GaussData data){
//     double error = 0;
//     vector<double> f_x(data_num,0);
//     for(int i=0; i<model_dim; i++){
//         if(i == model_id){
//             for(int j=0; j<data_num; j++){
//                 f_x[j] += subparameter[j];
//             }
//         }else{
//             for(int j=0; j<data_num; j++){
//                 f_x[j] += subparameters[i][j];
//             }
//         }
//     }
//     for(int i=0; i<data_num; i++){
//         error += (data.y[i] - f_x[i])*(data.y[i] - f_x[i]);
//     }
//     return error/(2*data.sigma*data.sigma)/data_num;
// };

#endif
#ifndef BASE_SPECTRAL_DECONVOLUTION_HPP
#define BASE_SPECTRAL_DECONVOLUTION_HPP

#include "../../core/base_emc.hpp"

inline double ErrorMultimodalSampling(const vector<vector<double> >& parameters, int data_num, int model_dim){

    double output = 0;
    double ratio = 1.001;
    // if(parameters[0][0] < 0.5 && parameters[0][1] < 0.5){
    //     output += ratio*(parameters[0][0] - 0.25)*(parameters[0][0] - 0.25);
    //     output += ratio*(parameters[0][1] - 0.25)*(parameters[0][1] - 0.25);
    // }else if(parameters[0][0] < 0.5 && parameters[0][1] > 0.5){
    //     output += (parameters[0][0] - 0.25)*(parameters[0][0] - 0.25);
    //     output += (parameters[0][1] - 0.75)*(parameters[0][1] - 0.75);
    //     output += (ratio-1)/8;
    // }else if(parameters[0][0] > 0.5 && parameters[0][1] < 0.5){
    //     output += (parameters[0][0] - 0.75)*(parameters[0][0] - 0.75);
    //     output += (parameters[0][1] - 0.25)*(parameters[0][1] - 0.25);
    //     output += (ratio-1)/8;
    // }else{
    //     output += (parameters[0][0] - 0.75)*(parameters[0][0] - 0.75);
    //     output += (parameters[0][1] - 0.75)*(parameters[0][1] - 0.75);
    //     output += (ratio-1)/8;
    // }
    if(parameters[0][0] < 0.5){
        for(int i=0; i<data_num; i++){
            output += ratio*100*(parameters[0][0] - 0.25)*(parameters[0][0] - 0.25);
            output += 100*(parameters[0][1] - 0.5)*(parameters[0][1] - 0.5);
        }
    }else{
        for(int i=0; i<data_num; i++){
            output += 100*(parameters[0][0] - 0.75)*(parameters[0][0] - 0.75);
            output += 100*(parameters[0][1] - 0.5)*(parameters[0][1] - 0.5);
            output += (ratio-1)/16*100;
        }
    }
    return output/data_num;
};

#endif
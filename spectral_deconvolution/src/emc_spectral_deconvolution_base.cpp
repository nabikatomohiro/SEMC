#include "base_spectral_deconvolution.hpp"
#include "../../core/base_emc.hpp"

#define PI 3.14159265358979323846

class SpectralDeconvolutionEMC : public BaseEMC {
    /** @class SpectralDeconvolutionEMC
     * @brief Nagata et al. 2012のモデルに対する交換モンテカルロ法
     */
    protected:
        const GaussData data_;
    public:
        SpectralDeconvolutionEMC(ModelOptions op, vector<vector<PriorParameter> > prior, GaussData data): BaseEMC(op, prior), data_(data), BaseSampling(op, prior){};

        inline double ErrorCalculation(const vector<vector<double> >& parameters) const;
        inline void FreeEnergyCalculation();
};

double SpectralDeconvolutionEMC::ErrorCalculation(const vector<vector<double> >& parameters) const{
    return ErrorSpectralDeconvolution(parameters, data_, op_.data_num, op_.model_dim);
}

void SpectralDeconvolutionEMC::FreeEnergyCalculation() {
    double free_energy = 0;
    vector<double> Z(op_.replica_num-1);
    vector<double> constant_for_logsumexp(op_.replica_num-1);
    for(int i=0; i<op_.replica_num-1; i++){
        constant_for_logsumexp[i] = op_.data_num*(settings_.inverse_temperatures[i+1] - settings_.inverse_temperatures[i])*energies_[op_.burn_in][i];
       for(int j=op_.burn_in; j<op_.sample_num; j++){
           Z[i] += exp(-op_.data_num*(settings_.inverse_temperatures[i+1] - settings_.inverse_temperatures[i])*energies_[j][i] + constant_for_logsumexp[i]);
       }
       free_energy -= log(Z[i]/(op_.sample_num - op_.burn_in));
       free_energy += constant_for_logsumexp[i];
    }
    free_energy_ = free_energy;
}

int main(){
ModelOptions op;
    op.replica_num = 50;
    op.gamma_decision = 0.3;
    op.model_dim = 3;
    op.parameter_nums = {3,3,3};
    op.robbins_parameter.M = 20;
    op.robbins_parameter.p_goal = 0.5;
    op.robbins_parameter.N_0 = 15;
    op.robbins_parameter.c = 4.0;
    op.sample_num = 2000000;
    op.burn_in = 1000000;
    op.data_num = 301;
    op.stepsize_parameter = {{{0.5, 301}, {1.2, 301}, {0.3, 301}},{{0.5, 301}, {1.2, 301}, {0.3, 301}},{{0.5, 301}, {1.2, 301}, {0.3, 301}}};

    vector<vector<PriorParameter> > prior(op.model_dim);
    for(int i=0; i<op.model_dim; i++){
        prior[i].resize(op.parameter_nums[i]);
        for(int j = 0; j<op.parameter_nums[i]; j++){
            if(j == 0){
                prior[i][j].type = 2;
                prior[i][j].parameter.first = 5.0;
                prior[i][j].parameter.second = 5.0;
            }else if(j == 1){
                prior[i][j].type = 0;
                prior[i][j].parameter.first = 1.5;
                prior[i][j].parameter.second = sqrt(0.2);
            }else if(j == 2){
                prior[i][j].type = 2;
                prior[i][j].parameter.first = 5.0;
                prior[i][j].parameter.second = 0.04;
            }
        }
    }

    GaussData data;
    data.x = vector<double>(op.data_num);
    data.y = vector<double>(op.data_num);
    data.sigma = 0.1;
    string input_name = "../input/";
    ifstream y_file(input_name + "data_0.txt");
    for(int i=0; i<op.data_num; i++){
        data.x[i] = 0 + 0.01*i;
        y_file >> data.y[i];
    }
    timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    SpectralDeconvolutionEMC emc(op, prior, data);
    emc.Execution();
    emc.FreeEnergyCalculation();
    clock_gettime(CLOCK_MONOTONIC, &end);
    emc.SetCalculationTime(end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec)/1e9);
    emc.Save("../output/baseline/");
    return 0;
}

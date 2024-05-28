#include "base_multimodal_sampling.hpp"
#include "../../core/base_semc.hpp"

#define PI 3.14159265358979323846

class MultimodalSamplingSEMC : public BaseSEMC {
    /** @class MultimodalSamplingEMC
     * @brief 簡単な多峰性分布に対する交換モンテカルロ法
     */
    public:
        MultimodalSamplingSEMC(ModelOptions op, vector<vector<PriorParameter> > prior): BaseSEMC(op, prior), BaseSampling(op, prior){};

        inline double ErrorCalculation(const vector<vector<double> >& parameters) const;
        inline void FreeEnergyCalculation();

};

double MultimodalSamplingSEMC::ErrorCalculation(const vector<vector<double> >& parameters) const{
    return ErrorMultimodalSampling(parameters, op_.data_num, op_.model_dim);
}

void MultimodalSamplingSEMC::FreeEnergyCalculation() {
    double free_energy = 0;
    vector<double> Z(op_.replica_num-1);
    vector<double> constant_for_logsumexp(op_.replica_num-1);
    for(int i=0; i<op_.replica_num-1; i++){
        constant_for_logsumexp[i] = op_.data_num*(settings_.inverse_temperatures[i+1] - settings_.inverse_temperatures[i])*energies_[i][op_.burn_in];
       for(int j=op_.burn_in; j<op_.sample_num; j++){
           Z[i] += exp(-op_.data_num*(settings_.inverse_temperatures[i+1] - settings_.inverse_temperatures[i])*energies_[i][j] + constant_for_logsumexp[i]);
       }
       free_energy -= log(Z[i]/(op_.sample_num - op_.burn_in));
       free_energy += constant_for_logsumexp[i];
    }
    free_energy_ = free_energy;
    cout << "free_energy: " << free_energy_ << endl;
}

int main(){
ModelOptions op;
    op.replica_num = 0;
    vector<double> gamma_decisions{0.1,0.3,0.5,0.7,0.9};
    op.model_dim = 1;
    op.parameter_nums = {2};
    op.robbins_parameter.M = 50;
    op.robbins_parameter.p_goal = 0.5;
    op.robbins_parameter.N_0 = 15;
    op.robbins_parameter.c = 4.0;
    op.data_num = 300;
    op.parallel_num = 50;

    vector<vector<PriorParameter> > prior(op.model_dim);
    for(int i=0; i<op.model_dim; i++){
        prior[i].resize(op.parameter_nums[i]);
        for(int j = 0; j<op.parameter_nums[i]; j++){
            prior[i][j].type = 1;
            prior[i][j].parameter.first = 0;
            prior[i][j].parameter.second = 1;
        }
    }
    
    int repeat = 10;
    vector<int> burn_in_array = {25000,15000,10000,6000,2000};
    vector<int> times = {1,3,10,30};
    for(int j=0;j<gamma_decisions.size();j++){
        op.gamma_decision = gamma_decisions[j];
        for(int i=0; i<times.size(); i++){
            op.sample_num = burn_in_array[j]*2*times[i];
            op.burn_in = burn_in_array[j]*times[i];
            string folder_name_base = "../output/SEMC/burn_in_" + to_string(burn_in_array[j]*times[i]) + "_gamma_" + to_string(gamma_decisions[j]) + "/";
            mkdir(folder_name_base.c_str(), 0777);
            for(int iter = 0; iter < repeat; iter++){
                timespec start, end;
                clock_gettime(CLOCK_MONOTONIC, &start);
                MultimodalSamplingSEMC semc(op, prior);
                semc.Execution();
                semc.FreeEnergyCalculation();
                clock_gettime(CLOCK_MONOTONIC, &end);
                semc.SetCalculationTime(end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec)/1e9);
                string folder_name = folder_name_base + "trial_" + to_string(iter) + "/";
                semc.Save(folder_name);
            }
        }
    }
    return 0;
}

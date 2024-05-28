#include "base_multimodal_sampling.hpp"
#include "../../core/base_emc.hpp"

#define PI 3.14159265358979323846

class MultimodalSamplingEMC : public BaseEMC {
    /** @class MultimodalSamplingEMC
     * @brief 簡単な多峰性分布に対する交換モンテカルロ法
     */
    public:
        MultimodalSamplingEMC(ModelOptions op, vector<vector<PriorParameter> > prior): BaseEMC(op, prior), BaseSampling(op, prior){};

        inline double ErrorCalculation(const vector<vector<double> >& parameters) const;
        inline void FreeEnergyCalculation();
};

double MultimodalSamplingEMC::ErrorCalculation(const vector<vector<double> >& parameters) const{
    return ErrorMultimodalSampling(parameters, op_.data_num, op_.model_dim);
}

void MultimodalSamplingEMC::FreeEnergyCalculation() {
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
    cout << "free_energy: " << free_energy_ << endl;
}

int main(){
    ModelOptions op;
    op.gamma_decision = 0.5;
    op.model_dim = 1;
    op.parameter_nums = {2};
    op.robbins_parameter.M = 20;
    op.robbins_parameter.p_goal = 0.5;
    op.robbins_parameter.N_0 = 15;
    op.robbins_parameter.c = 4.0;
    op.data_num = 300;
    op.stepsize_parameter = {{{0.4,3000},{0.4,3000}}};

    vector<vector<PriorParameter> > prior(op.model_dim);
    for(int i=0; i<op.model_dim; i++){
        prior[i].resize(op.parameter_nums[i]);
        for(int j = 0; j<op.parameter_nums[i]; j++){
            prior[i][j].type = 1;
            prior[i][j].parameter.first = 0;
            prior[i][j].parameter.second = 1;
        }
    }

    int repeat = 1;
    vector<int> burn_in_array{10000};
    vector<int> L_array{10,30,100,300};
    for(int j=0; j<L_array.size(); j++){
        for(int i=0; i<1; i++){
            string folder_name_base = "../output/EMC_hyperparameter/burn_in_" + to_string(burn_in_array[i]) + "_L_" + to_string(L_array[j]) + "/";
            mkdir(folder_name_base.c_str(), 0777);
            for(int iter = 0; iter < repeat; iter++){
                op.sample_num = burn_in_array[i]*2;
                op.burn_in = burn_in_array[i];
                op.replica_num = L_array[j];
                timespec start, end;
                clock_gettime(CLOCK_MONOTONIC, &start);
                MultimodalSamplingEMC emc(op, prior);
                emc.Execution();
                emc.FreeEnergyCalculation();
                clock_gettime(CLOCK_MONOTONIC, &end);
                emc.SetCalculationTime(end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec)/1e9);
                string folder_name = folder_name_base + "trial_" + to_string(iter) + "/";
                emc.Save(folder_name);
            }
        }
    }
    return 0;
}

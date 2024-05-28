#include "base_multimodal_sampling.hpp"
#include "../../core/base_tmcmc.hpp"

#define PI 3.14159265358979323846

class MultimodalSamplingTMCMC : public BaseTMCMC {
    /** @class MultimodalSamplingTMCMC
     * @brief 簡単な多峰性分布に対する交換モンテカルロ法
     */
    public:
        MultimodalSamplingTMCMC(ModelOptions op, vector<vector<PriorParameter> > prior): BaseTMCMC(op, prior), BaseSampling(op, prior){};

        inline double ErrorCalculation(const vector<vector<double> >& parameters) const;
        inline void FreeEnergyCalculation();

};

double MultimodalSamplingTMCMC::ErrorCalculation(const vector<vector<double> >& parameters) const{
    return ErrorMultimodalSampling(parameters, op_.data_num, op_.model_dim);
}

void MultimodalSamplingTMCMC::FreeEnergyCalculation() {
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
    op.gamma_decision = 0.3;
    op.model_dim = 1;
    op.parameter_nums = {2};
    op.control_parameter.G = 2.1;
    op.control_parameter.p_goal = 0.234;
    op.control_parameter.c_0 = 0.1;
    op.data_num = 300;
    op.burn_in = 0;
    vector<int> n_steps_array{1,10,100,1000};
    vector<int> sample_array{10,30,100,300,1000,3000,10000,30000,100000,300000,1000000,3000000,10000000};

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
    for(int i=0; i<n_steps_array.size(); i++){
        for(int j=(4-i)*2; j<(4-i)*2 + 4; j++){
            cout << i << " " << j << endl;
            string folder_name_base = "../output/TMCMC/sample_" + to_string(sample_array[j]) + "_nsteps_" + to_string(n_steps_array[i]) + "/";
            mkdir(folder_name_base.c_str(), 0777);
            op.n_steps = n_steps_array[i];
            op.sample_num = sample_array[j];
            for(int iter = 0; iter < repeat; iter++){
                timespec start, end;
                clock_gettime(CLOCK_MONOTONIC, &start);
                MultimodalSamplingTMCMC tmcmc(op, prior);
                tmcmc.Execution();
                tmcmc.FreeEnergyCalculation();
                clock_gettime(CLOCK_MONOTONIC, &end);
                tmcmc.SetCalculationTime(end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec)/1e9);
                string folder_name = folder_name_base + "/trial_" + to_string(iter) + "/";
                tmcmc.Save(folder_name);
            }
        }
    }
    return 0;
}

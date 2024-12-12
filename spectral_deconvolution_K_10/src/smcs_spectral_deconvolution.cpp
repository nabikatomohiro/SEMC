#include "base_spectral_deconvolution.hpp"
#include "../../core/base_smcs.hpp"

#define PI 3.14159265358979323846

class SpectralDeconvolutionSEMC : public BaseSMCS {
    /** @class SpectralDeconvolutionEMC
     * @brief Nagata et al. 2012のモデルに対する交換モンテカルロ法
     */
    protected:
        const GaussData data_;
    public:
        SpectralDeconvolutionSEMC(ModelOptions op, vector<vector<PriorParameter> > prior, GaussData data): BaseSMCS(op, prior), BaseSampling(op, prior), data_(data){};

        inline double ErrorCalculation(const vector<vector<double> >& parameters) const;
        inline void FreeEnergyCalculation();

};

double SpectralDeconvolutionSEMC::ErrorCalculation(const vector<vector<double> >& parameters) const{
    return ErrorSpectralDeconvolution(parameters, data_, op_.data_num, op_.model_dim);
}

void SpectralDeconvolutionSEMC::FreeEnergyCalculation() {
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
    vector<double> gamma_decisions{0.5};
    op.model_dim = 10;
    op.parameter_nums = {3,3,3,3,3,3,3,3,3,3};
    op.robbins_parameter.M = 50;
    op.robbins_parameter.p_goal = 0.5;
    op.robbins_parameter.N_0 = 15;
    op.robbins_parameter.c = 4.0;
    op.data_num = 301;
    op.parallel_num = 50;
    vector<int> M_pops = {1,10,100};

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
                prior[i][j].parameter.second = sqrt(1.0);
            }else if(j == 2){
                prior[i][j].type = 2;
                prior[i][j].parameter.first = 5.0;
                prior[i][j].parameter.second = 0.004;
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
    
    int repeat = 10;
    vector<int> burn_in_array = {10000};
    vector<int> times = {1,3,10};
    for(int j=0;j<M_pops.size();j++){
        op.gamma_decision = gamma_decisions[0];
        for(int i=0; i<times.size(); i++){
            op.M_pop = burn_in_array[0]*times[i]/M_pops[j];
            op.sample_num = burn_in_array[0]*2*times[i];
            op.burn_in = burn_in_array[0]*times[i];
            string folder_name_base = "../output/SMCS/burn_in_" + to_string(burn_in_array[0]*times[i]) + "_M_pop_" + to_string(M_pops[j]) + "/";
            mkdir(folder_name_base.c_str(), 0777);
            for(int iter = 0; iter < repeat; iter++){
                timespec start, end;
                clock_gettime(CLOCK_MONOTONIC, &start);
                SpectralDeconvolutionSEMC semc(op, prior, data);
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
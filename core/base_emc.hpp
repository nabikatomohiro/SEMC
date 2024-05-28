#ifndef BASE_EMC_HPP
#define BASE_EMC_HPP

#include "base_sampling.hpp"

#include <omp.h>

using namespace std;

class BaseEMC : public BaseMonteCarlo{
    /** @class EMC基底クラス */
    public:
        BaseEMC(ModelOptions op, vector<vector<PriorParameter> > prior): BaseMonteCarlo(op, prior){
            parameters_ = vector<vector<vector<vector<double> > > > (op.sample_num, vector<vector<vector<double> > > (op.replica_num, vector<vector<double> >(op.model_dim)));
            for(int i=0; i<op.sample_num; i++){
                for(int j=0; j<op.replica_num; j++){
                    for(int k=0; k<op.model_dim; k++){
                        parameters_[i][j][k].resize(op.parameter_nums[k]);
                    }
                }
            }
            energies_ = vector<vector<double> > (op.sample_num, vector<double>(op.replica_num));
            accept_counts_ = vector<vector<vector<int> > > (op.replica_num, vector<vector<int> >(op.model_dim));
            accept_counts_for_Robbins_ = vector<vector<vector<int> > >(op.replica_num, vector<vector<int> >(op.model_dim));
            exchange_counts_ = vector<int>(op.replica_num-1, 0);
            for(int i=0; i<op.replica_num; i++){
                for(int j=0; j<op.model_dim; j++){
                    accept_counts_[i][j].resize(op.parameter_nums[j], 0);
                    accept_counts_for_Robbins_[i][j].resize(op.parameter_nums[j], 0);
                }
            }    
        }


    inline void InitStates();
    /** @fn パラメータ, エネルギーを事前分布から初期化 */
    inline void InitializeInverseTemperatures();
    /** @fn 逆温度の初期化 */
    inline void InitializeStepSizes();
    /** @fn ステップサイズの初期化 */
    inline void Metropolis(int sample_id);
    /** @fn メトロポリス法によって, パラメータ更新 */
    inline void Exchange(int sample_id);
    /** @fn 交換モンテカルロ法によって, レプリカ層間のパラメータ交換 */
    inline void Execution();
    /** @fn 交換モンテカルロ法の実行 */
    inline void UpdateStepSizes(int sample_id);
    /** @fn ロビンスモンロー法によって, ステップサイズ更新 */
    inline void SaveParameters();
    /** @fn パラメータの保存 */
    inline void SaveEnergies();
    /** @fn エネルギーの保存 */
    inline void Save(string folder_name);
    /** @fn 全てをファイルに保存 */

    inline virtual double ErrorCalculation(const vector<vector<double> >& parameters) const = 0;
    /** @fn 誤差関数の計算 */
    inline virtual void FreeEnergyCalculation() {cout << "Error: FreeEnergyCalculation is not defined" << endl;};
    /** @fn 自由エネルギーの計算 */

};

void BaseEMC::InitStates(){
    /** @fn パラメータ, エネルギーを事前分布から初期化 */
    #pragma omp parallel for schedule (dynamic, 1)
    for(int i=0; i<op_.replica_num; i++){
        int thread = omp_get_thread_num();
        parameters_[0][i] = SamplingFromPrior(engines_thread_[thread]);
        energies_[0][i] = ErrorCalculation(parameters_[0][i]);
    }
    #pragma omp parallel for schedule (dynamic, 1)
    for(int i=1; i<op_.sample_num; i++){
        int thread = omp_get_thread_num();
        parameters_[i][0] = SamplingFromPrior(engines_thread_[thread]);
        energies_[i][0] = ErrorCalculation(parameters_[i][0]);
    }
};

void BaseEMC::InitializeInverseTemperatures(){
    vector<double> inverse_temperatures(op_.replica_num);
    double energy_mean_ = 0;
    double energy_min_ = energies_[0][0];
    for(int i=0; i<op_.sample_num; i++){
        energy_mean_ += energies_[i][0];
        if(energy_min_ > energies_[i][0]){
            energy_min_ = energies_[i][0];
        }
    }
    energy_mean_ /= op_.sample_num;
    vector<double> energy_diff(op_.sample_num);
    for(int i=0; i<op_.sample_num; i++){
        energy_diff[i] = energies_[i][0] - energy_min_;
    }
    sort(energy_diff.begin(), energy_diff.end());
    double low_beta = 0;
    double old_beta = 0;
    double high_beta = 1.1;
    double threshold = op_.gamma_decision;
    double new_beta;
    double value_1, value_2;
    while(high_beta - low_beta > 1e-3*high_beta){
        new_beta = (high_beta + low_beta)/2;
        value_1 = 0;
        for(int i=0; i<op_.sample_num; i++){
            value_1 += exp(-op_.data_num*energy_diff[i]*(new_beta - old_beta));
        }
        value_1 /= op_.sample_num;
        value_2 = 0;
        for(int i=0; i<op_.sample_num; i++){
            value_2 += i*exp(-op_.data_num*energy_diff[i]*(new_beta - old_beta))/op_.sample_num;
        }
        value_2 /= op_.sample_num - 1;
        if(2*value_2/value_1 < threshold){
            high_beta = new_beta;
        }else{
            low_beta = new_beta;
        }
        if(low_beta > 1){
            new_beta = 1;
            break;
        }
    } 

    inverse_temperatures[0] = 0;
    inverse_temperatures[1] = new_beta;
    double gamma = pow(inverse_temperatures[1],1.0/(1 - op_.replica_num));
    for(int i=2; i<op_.replica_num; i++){
        inverse_temperatures[i] = pow(gamma, i-op_.replica_num+1);
    }
    settings_.inverse_temperatures = inverse_temperatures;
    
}

void BaseEMC::InitializeStepSizes(){
    vector<vector<vector<double> > > step_sizes(op_.replica_num, vector<vector<double> > (op_.model_dim));
    vector<vector<double> > step_sizes_C(op_.model_dim);
    for(int i=0; i<op_.model_dim; i++){
        step_sizes_C[i].resize(op_.parameter_nums[i]);
        for(int j=0; j<op_.parameter_nums[i]; j++){
            if(prior_[i][j].type == 0){
                step_sizes_C[i][j] = prior_[i][j].parameter.second * 2.94;
            }else if(prior_[i][j].type == 1){
                step_sizes_C[i][j] = (prior_[i][j].parameter.second - prior_[i][j].parameter.first);
            }else if(prior_[i][j].type == 2){
                double mean = 0;
                for(int k=0; k<op_.sample_num; k++){
                    mean += parameters_[k][0][i][j];
                }
                mean /= op_.sample_num;
                double var = 0;
                for(int k=0; k<op_.sample_num; k++){
                    var += (parameters_[k][0][i][j] - mean)*(parameters_[k][0][i][j] - mean);
                }
                var /= op_.sample_num;
                step_sizes_C[i][j] = sqrt(var) * 2.94;
            }else{
                printf("Error: Invalid prior type\n");
            }
        }
    }
    for(int i=0; i<op_.replica_num; i++){
        for(int j=0; j<op_.model_dim; j++){
            step_sizes[i][j].resize(op_.parameter_nums[j]);
            for(int k=0; k<op_.parameter_nums[j]; k++){
                double nbeta = settings_.inverse_temperatures[i]*op_.stepsize_parameter[j][k][1];
                if(nbeta < 1){
                    step_sizes[i][j][k] = step_sizes_C[j][k];
                }else{
                    step_sizes[i][j][k] = step_sizes_C[j][k] / pow(nbeta, op_.stepsize_parameter[j][k][0]);
                }
            }
        }
    }
    settings_.step_sizes = step_sizes;
}

void BaseEMC::Metropolis(int sample_id){
    uniform_real_distribution<> dist_delta(-1, 1);
    uniform_real_distribution<> dist(0, 1);
    #pragma omp parallel for schedule (dynamic, 1)
    for(int i=1; i<op_.replica_num; i++){
        int thread = omp_get_thread_num();
        for(int j=0; j<op_.model_dim; j++){
            for(int k=0; k<op_.parameter_nums[j]; k++){
                vector<vector<double> > new_parameters = parameters_[sample_id][i];
                double delta = dist_delta(engines_thread_[thread])*settings_.step_sizes[i][j][k];
                double prior_prob_before = ProbCalculationOnPrior(parameters_[sample_id][i][j][k], prior_[j][k].parameter, prior_[j][k].type);
                double new_parameter = parameters_[sample_id][i][j][k] + delta;
                new_parameters[j][k] = new_parameter;
                double new_energy = ErrorCalculation(new_parameters);
                double prior_prob = ProbCalculationOnPrior(new_parameter, prior_[j][k].parameter, prior_[j][k].type);
                double energy_diff = new_energy - energies_[sample_id][i];
                double prob = -op_.data_num*settings_.inverse_temperatures[i]*energy_diff + prior_prob - prior_prob_before;
                if(log(dist(engines_thread_[thread])) < prob){
                    parameters_[sample_id][i][j][k] = new_parameter;
                    energies_[sample_id][i] = new_energy;
                    if(sample_id > op_.burn_in){
                        accept_counts_[i][j][k]++;
                    }else{
                        accept_counts_for_Robbins_[i][j][k]++;
                    }
                }
            }
        }
    }
}

void BaseEMC::Exchange(int sample){
    uniform_real_distribution<> dist(0, 1);
    for(int i=0; i<op_.replica_num-1; i++){
        double prob = op_.data_num*(settings_.inverse_temperatures[i+1] - settings_.inverse_temperatures[i])*(energies_[sample][i+1] - energies_[sample][i]);
        if(log(dist(engines_thread_[0])) < prob){
            if(sample > op_.burn_in){
                exchange_counts_[i]++;
            }
            swap(energies_[sample][i], energies_[sample][i+1]);
            swap(parameters_[sample][i], parameters_[sample][i+1]);
        }
    }
}

void BaseEMC::UpdateStepSizes(int sample_id){
    int M = op_.robbins_parameter.M;
    double p_goal = op_.robbins_parameter.p_goal;
    int N_0 = op_.robbins_parameter.N_0;
    double c = op_.robbins_parameter.c;
    for(int i=0; i<op_.replica_num; i++){
        for(int j=0; j<op_.model_dim; j++){
            for(int k=0; k<op_.parameter_nums[j]; k++){
                double accept_rate = (double)accept_counts_for_Robbins_[i][j][k]/M;
                settings_.step_sizes[i][j][k] += c*settings_.step_sizes[i][j][k] * (accept_rate - p_goal)/(N_0 + sample_id/M);
                accept_counts_for_Robbins_[i][j][k] = 0;
            }
        }
    }
}

void BaseEMC::Execution(){
    InitStates();
    InitializeInverseTemperatures();
    InitializeStepSizes();
    for(int i=1; i<op_.sample_num; i++){
        for(int j=1; j<op_.replica_num; j++){
            parameters_[i][j] = parameters_[i-1][j];
            energies_[i][j] = energies_[i-1][j];
        }
        
        if(i < op_.burn_in && i%op_.robbins_parameter.M == 0){
            UpdateStepSizes(i);
        }
        Metropolis(i);
        Exchange(i);
    }
}

void BaseEMC::SaveParameters(){
    ofstream ofs(folder_name_ + "parameters.txt");
    for(int i=0; i<op_.sample_num; i++){
        for(int k=0; k<op_.model_dim; k++){
            for(int l=0; l<op_.parameter_nums[k]; l++){
                ofs << parameters_[i][op_.replica_num-1][k][l] << " ";
            }
        }
        ofs << endl;
    }
    ofs.close();
}

void BaseEMC::SaveEnergies(){
    ofstream ofs(folder_name_ + "energies.txt");
    for(int i=0; i<op_.sample_num; i++){
        ofs << energies_[i][op_.replica_num-1] << endl;
    }
    ofs.close();
}

void BaseEMC::Save(string folder_name){
    SetFolderName(folder_name);
    SaveParameters();
    SaveEnergies();
    SaveAcceptanceRate();
    SaveExchangeRate();
    SaveFreeEnergy();
    SaveCalculationTime();
    SaveInverseTemperatures();
}

#endif
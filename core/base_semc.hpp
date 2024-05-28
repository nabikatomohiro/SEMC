#ifndef BASE_SEMC_HPP
#define BASE_SEMC_HPP

#include "base_sampling.hpp"

#include <omp.h>

using namespace std;

class BaseSEMC : public BasePopulation, public BaseMonteCarlo{
    /** @class SEMC基底クラス */
    public:
        BaseSEMC(ModelOptions op, vector<vector<PriorParameter> > prior): BasePopulation(op, prior), BaseMonteCarlo(op, prior){
            accept_counts_ = vector<vector<vector<int> > > (1, vector<vector<int> >(op.model_dim));
            accept_counts_for_Robbins_ = vector<vector<vector<int> > >(1, vector<vector<int> >(op.model_dim));
            for(int j=0; j<op.model_dim; j++){
                accept_counts_[0][j].resize(op.parameter_nums[j], 0);
                accept_counts_for_Robbins_[0][j].resize(op.parameter_nums[j], 0);
            }
            exchange_counts_ = vector<int>(0);
        };

        inline void NextInverseTemperatures(int l);
        /** @fn 逆温度の初期化 */
        inline void NextStepSizes(int l);
        /** @fn ステップサイズの初期化 */
        inline void UpdateStepSizes(int l, int sample_id);
        /** @fn ロビンスモンロー法によって, ステップサイズ更新 */
        inline void ExchangeMetropolis(int l);
        /** @fn メトロポリス法と交換によって, パラメータ更新 */
        inline void Execution();
        /** @fn 焼きなまし交換モンテカルロ法の実行 */
        inline void Save(string folder_name);
        /** @fn 全てをファイルに保存 */

        inline virtual double ErrorCalculation(const vector<vector<double> >& parameters) const = 0;
        /** @fn 誤差関数の計算 */

        int print = 0;

};

// void BaseSEMC::NextInverseTemperatures(int l){
//     if(l == 0){
//         settings_.inverse_temperatures.push_back(0);
//     }else{
//         double energy_mean_ = 0;
//         double energy_min_ = energies_[l-1][0];
//         for(int i=0; i<op_.sample_num; i++){
//             energy_mean_ += energies_[l-1][i];
//             if(energy_min_ > energies_[l-1][i]){
//                 energy_min_ = energies_[l-1][i];
//             }
//         }
//         energy_mean_ /= op_.sample_num;
//         double next_beta = -log(op_.gamma_decision)/(op_.data_num*(energy_mean_ - energy_min_)) + settings_.inverse_temperatures[l-1];
//         if(next_beta > 1.0){
//             next_beta = 1.0;
//         }
//         settings_.inverse_temperatures.push_back(next_beta);
//     }
// }

// void BaseSEMC::NextInverseTemperatures(int l){
//     if(l == 0){
//         settings_.inverse_temperatures.push_back(0);
//     }else{
//         double energy_mean_ = 0;
//         double energy_min_ = energies_[l-1][0];
//         for(int i=0; i<op_.sample_num; i++){
//             energy_mean_ += energies_[l-1][i];
//             if(energy_min_ > energies_[l-1][i]){
//                 energy_min_ = energies_[l-1][i];
//             }
//         }
//         energy_mean_ /= op_.sample_num;
//         vector<double> energy_diff(op_.sample_num);
//         for(int i=0; i<op_.sample_num; i++){
//             energy_diff[i] = energies_[l-1][i] - energy_min_;
//         }
//         double low_beta = settings_.inverse_temperatures[l-1];
//         double old_beta = settings_.inverse_temperatures[l-1];
//         double high_beta = 1.1;
//         double threshold = op_.gamma_decision;
//         double new_beta;
//         double value_1, value_2;
//         while(high_beta - low_beta > 1e-8){
//             new_beta = (high_beta + low_beta)/2;
//             value_1 = 0;
//             for(int i=0; i<op_.sample_num; i++){
//                 value_1 += exp(-op_.data_num*energy_diff[i]*(new_beta - old_beta));
//             }
//             value_1 /= op_.sample_num;
//             value_1 = log(value_1);
//             value_2 = op_.data_num*(new_beta - old_beta)*(energy_mean_ - energy_min_);
//             if(value_1 + value_2 > threshold){
//                 high_beta = new_beta;
//             }else{
//                 low_beta = new_beta;
//             }
//             if(low_beta > 1){
//                 new_beta = 1;
//                 break;
//             }
//         } 
//         cout << "value_1: " << value_1 << " value_2: " << value_2 << " threshold: " << threshold << " new_beta: " << new_beta << endl;
//         if(new_beta > 1){
//             settings_.inverse_temperatures.push_back(1);
//         }else{
//             settings_.inverse_temperatures.push_back(new_beta); 
//         }
//     }
// }

void BaseSEMC::NextInverseTemperatures(int l){
    if(l == 0){
        settings_.inverse_temperatures.push_back(0);
    }else{
        double energy_min_ = energies_[l-1][0];
        for(int i=0; i<op_.sample_num; i++){
            if(energy_min_ > energies_[l-1][i]){
                energy_min_ = energies_[l-1][i];
            }
        }
        vector<double> energy_diff(op_.sample_num);
        for(int i=0; i<op_.sample_num; i++){
            energy_diff[i] = energies_[l-1][i] - energy_min_;
        }
        sort(energy_diff.begin(), energy_diff.end());
        double low_beta = settings_.inverse_temperatures[l-1];
        double old_beta = settings_.inverse_temperatures[l-1];
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
        if(new_beta > 1){
            settings_.inverse_temperatures.push_back(1);
        }else{
            settings_.inverse_temperatures.push_back(new_beta); 
        }
    }
}

void BaseSEMC::NextStepSizes(int l){
    vector<vector<double> > next_stepsize(op_.model_dim);
    for(int i=0; i<op_.model_dim; i++){
        next_stepsize[i].resize(op_.parameter_nums[i]);
    }
    if(l == 0){
        for(int i=0; i<op_.model_dim; i++){
            for(int j=0; j<op_.parameter_nums[i]; j++){
                if(prior_[i][j].type == 0){
                    next_stepsize[i][j] = prior_[i][j].parameter.second * 2.94;
                }else if(prior_[i][j].type == 1){
                    next_stepsize[i][j] = (prior_[i][j].parameter.second - prior_[i][j].parameter.first);
                }else if(prior_[i][j].type == 2){
                    double mean = 0;
                    for(int k=0; k<op_.sample_num; k++){
                        mean += parameters_[0][k][i][j];
                    }
                    mean /= op_.sample_num;
                    double var = 0;
                    for(int k=0; k<op_.sample_num; k++){
                        var += (parameters_[0][k][i][j] - mean)*(parameters_[0][k][i][j] - mean);
                    }
                    var /= op_.sample_num;
                    next_stepsize[i][j] = sqrt(var) * 2.94;
                }else{
                    printf("Error: Invalid prior type\n");
                }
            }
        }
        settings_.step_sizes.push_back(next_stepsize);
    }else if(l == 1){
        for(int i=0; i<op_.model_dim; i++){
            for(int j=0; j<op_.parameter_nums[i]; j++){
                next_stepsize[i][j] = settings_.step_sizes[l-1][i][j];
            }
        }
        settings_.step_sizes.push_back(next_stepsize);
    }else{
        for(int i=0; i<op_.model_dim; i++){
            for(int j=0; j<op_.parameter_nums[i]; j++){
                double d = (log(settings_.step_sizes[l-2][i][j]) - log(settings_.step_sizes[l-1][i][j]))/(log(settings_.inverse_temperatures[l-1]) - log(settings_.inverse_temperatures[l-2]));
                next_stepsize[i][j] = settings_.step_sizes[l-1][i][j]*pow(settings_.inverse_temperatures[l-1]/settings_.inverse_temperatures[l],d);
            }
        }
        settings_.step_sizes.push_back(next_stepsize);
    }
}

void BaseSEMC::UpdateStepSizes(int l, int sample_id){
    int M = op_.robbins_parameter.M;
    double p_goal = op_.robbins_parameter.p_goal;
    int N_0 = op_.robbins_parameter.N_0;
    double c = op_.robbins_parameter.c;
    for(int j=0; j<op_.model_dim; j++){
        for(int k=0; k<op_.parameter_nums[j]; k++){
            double accept_rate = (double)accept_counts_for_Robbins_[l][j][k]/M;
            // ステップサイズが更新される様子を確認するためにファイル
            if(print == 1){
                string filename = "../output/SEMC_hyperparameter/step_sizes_" + to_string(l) + ".txt";
                ofstream file(filename, ios::app);
                file << "step_sizes[" << l << "][" << j << "][" << k << "]: " << settings_.step_sizes[l][j][k] << " accept_rate: " << accept_rate << endl;
                file.close();
            }
            settings_.step_sizes[l][j][k] += c*settings_.step_sizes[l][j][k] * (accept_rate - p_goal)/(N_0 + sample_id/M);
            accept_counts_for_Robbins_[l][j][k] = 0;
        }
    }
}

void BaseSEMC::ExchangeMetropolis(int l){
    uniform_real_distribution<> dist(0, 1);
    uniform_real_distribution<> dist_delta(-1, 1);

    weight_for_resampling(l);
    discrete_distribution<> dist_index(weight_.begin(), weight_.end());
    vector<int> initial_index;
    int s = 0;
    while(s < op_.parallel_num){
        int candidate = dist_index(engines_thread_[0]);
        if(find(initial_index.begin(), initial_index.end(), candidate) == initial_index.end()){
            initial_index.push_back(candidate + op_.burn_in);
            s += 1;
        }
    } 

    vector<vector<vector<double> > > new_parameters = vector<vector<vector<double> > >(op_.sample_num, vector<vector<double> >(op_.model_dim));
    vector<double> new_energies = vector<double> (op_.sample_num);
    for(int i = 0; i < op_.sample_num; i++){
        for(int j=0; j<op_.model_dim; j++){
            new_parameters[i][j].resize(op_.parameter_nums[j]);
        }
    }
    parameters_.push_back(new_parameters);
    energies_.push_back(new_energies);

    for(int thread = 0; thread < op_.parallel_num; thread++){
        parameters_[l][thread] = parameters_[l-1][initial_index[thread]];
        energies_[l][thread] = energies_[l-1][initial_index[thread]];
    }

    vector<vector<vector<int> > > accept_counts_temp = vector<vector<vector<int> > >(op_.parallel_num, vector<vector<int> >(op_.model_dim));
    vector<vector<vector<int> > > accept_counts_for_Robbins_temp = vector<vector<vector<int> > >(op_.parallel_num, vector<vector<int> >(op_.model_dim));
    vector<int> exchange_counts_temp = vector<int>(op_.parallel_num, 0);
    for(int i=0; i<op_.parallel_num; i++){
        for(int j=0; j<op_.model_dim; j++){
            accept_counts_temp[i][j].resize(op_.parameter_nums[j], 0);
            accept_counts_for_Robbins_temp[i][j].resize(op_.parameter_nums[j], 0);
        }
    }

    vector<vector<int> > accept_counts_new = vector<vector<int> >(op_.model_dim);
    vector<vector<int> > accept_counts_for_Robbins_new = vector<vector<int> >(op_.model_dim);
    for(int j=0; j<op_.model_dim; j++){
        accept_counts_new[j].resize(op_.parameter_nums[j], 0);
        accept_counts_for_Robbins_new[j].resize(op_.parameter_nums[j], 0);
    }
    accept_counts_.push_back(accept_counts_new);
    accept_counts_for_Robbins_.push_back(accept_counts_for_Robbins_new);
    exchange_counts_.push_back(0);

    vector<int> index(op_.burn_in);
    iota(index.begin(), index.end(), 0);
    shuffle(index.begin(), index.end(), engines_thread_[0]);
    vector<int> index_2(op_.burn_in);
    iota(index_2.begin(), index_2.end(), 0);
    shuffle(index_2.begin(), index_2.end(), engines_thread_[0]);
    index.insert(index.end(), index_2.begin(), index_2.end());

    for(int iter = 1; iter < op_.sample_num/op_.parallel_num; iter++){
        if(iter*op_.parallel_num < op_.burn_in && iter*op_.parallel_num%op_.robbins_parameter.M == 0){
            UpdateStepSizes(l, iter*op_.parallel_num);
        }
        vector<int> change_array(op_.parallel_num);
        for(int thread = 0; thread < op_.parallel_num; thread++){
            change_array[thread] = index[thread + iter*op_.parallel_num];
        }
        #pragma omp parallel for schedule (dynamic, 1)
        for(int thread = 0; thread < op_.parallel_num; thread++){
            int thread_num = omp_get_thread_num();
            int sample_id = iter*op_.parallel_num + thread;
            parameters_[l][sample_id] = parameters_[l][sample_id - op_.parallel_num];
            energies_[l][sample_id] = energies_[l][sample_id - op_.parallel_num];
            for(int j=0; j<op_.model_dim; j++){
                for(int k=0; k<op_.parameter_nums[j]; k++){
                    vector<vector<double> > new_parameters = parameters_[l][sample_id];
                    double delta = dist_delta(engines_thread_[thread_num])*settings_.step_sizes[l][j][k];
                    double prior_prob_before = ProbCalculationOnPrior(parameters_[l][sample_id][j][k], prior_[j][k].parameter, prior_[j][k].type);
                    double new_parameter = parameters_[l][sample_id][j][k] + delta;
                    new_parameters[j][k] = new_parameter;
                    double new_energy = ErrorCalculation(new_parameters);
                    double prior_prob = ProbCalculationOnPrior(new_parameter, prior_[j][k].parameter, prior_[j][k].type);
                    double energy_diff = new_energy - energies_[l][sample_id];
                    double prob = -op_.data_num*settings_.inverse_temperatures[l]*energy_diff + prior_prob - prior_prob_before;
                    if(log(dist(engines_thread_[thread_num])) < prob){
                        parameters_[l][sample_id][j][k] = new_parameter;
                        energies_[l][sample_id] = new_energy;
                        if(sample_id > op_.burn_in){
                            accept_counts_temp[thread][j][k]++;
                        }else{
                            accept_counts_for_Robbins_temp[thread][j][k]++;
                        }
                    }
                }
            }
            int change_index = change_array[thread] + op_.burn_in;
            double prob = op_.data_num*(settings_.inverse_temperatures[l] - settings_.inverse_temperatures[l-1])*(energies_[l][sample_id] - energies_[l-1][change_index]);
            if(log(dist(engines_thread_[thread_num])) < prob){
                if(sample_id > op_.burn_in){
                    exchange_counts_temp[thread]++;
                }
                swap(energies_[l][sample_id], energies_[l-1][change_index]);
                swap(parameters_[l][sample_id], parameters_[l-1][change_index]);
            }
        }

        for(int thread = 0; thread < op_.parallel_num; thread++){
            for(int j=0; j<op_.model_dim; j++){
                for(int k=0; k<op_.parameter_nums[j]; k++){
                    accept_counts_[l][j][k] += accept_counts_temp[thread][j][k];
                    accept_counts_for_Robbins_[l][j][k] += accept_counts_for_Robbins_temp[thread][j][k];
                    accept_counts_temp[thread][j][k] = 0;
                    accept_counts_for_Robbins_temp[thread][j][k] = 0;
                }
            }
            exchange_counts_[l-1] += exchange_counts_temp[thread];
            exchange_counts_temp[thread] = 0;
        }
    }
}

void BaseSEMC::Execution(){
    op_.replica_num = 1;
    InitStates();
    NextInverseTemperatures(op_.replica_num-1);
    NextStepSizes(op_.replica_num-1);
    while(true){
        op_.replica_num += 1;
        NextInverseTemperatures(op_.replica_num-1);
        NextStepSizes(op_.replica_num-1);
        ExchangeMetropolis(op_.replica_num-1);
        if(settings_.inverse_temperatures[op_.replica_num-1] == 1.0){
            break;
        }
    }
}

void BaseSEMC::Save(string folder_name){
    SetFolderName(folder_name);
    SaveParameters();
    SaveEnergies();
    SaveAcceptanceRate();
    SaveExchangeRate();
    SaveFreeEnergy();
    SaveCalculationTime();
    SaveInverseTemperatures();
};

#endif // BASE_SEMC_HPP
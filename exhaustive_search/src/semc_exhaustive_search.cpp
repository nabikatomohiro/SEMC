#include "base_exhaustive_search.hpp"
#include "../../core/base_semc.hpp"
#include <omp.h>

class ExhaustiveSearchSEMC : public BaseSEMC {
    /** @class ExhaustiveSearchSEMC
     * @brief Igarashi et al. 2018のモデルに対する焼きなまし交換モンテカルロ法
     */
    protected:
        const HighDimensionalData data_;

    public:
        ExhaustiveSearchSEMC(ModelOptions op, vector<vector<PriorParameter> > prior, HighDimensionalData data): BaseSEMC(op, prior), data_(data), BaseSampling(op, prior){
        };
        inline double ErrorCalculation(const vector<vector<double> >& parameter) const;
        inline void FreeEnergyCalculation();
        inline void InitStates();
        inline void ExchangeMetropolis(int l);
        inline void Execution();
};

double ExhaustiveSearchSEMC::ErrorCalculation(const vector<vector<double> >& parameters) const{
    vector<int> param(op_.model_dim);
    for(int i=0; i<op_.model_dim; i++){
        param[i] = parameters[i][0];
    }
    return ErrorExhaustiveSearch(data_, param, op_.data_num, op_.model_dim);
}

void ExhaustiveSearchSEMC::FreeEnergyCalculation() {
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
};

void ExhaustiveSearchSEMC::InitStates(){
    /** @fn パラメータ, エネルギーを事前分布から初期化 */
    #pragma omp parallel for schedule (dynamic, 1)
    for(int i=0; i<op_.sample_num; i++){
        vector<int> index = n_choose_k(data_.p, op_.model_dim, engines_thread_[omp_get_thread_num()]);
        for(int j = 0; j < op_.model_dim; j++){
            parameters_[0][i][j][0] = index[j];
        }
        energies_[0][i] = ErrorCalculation(parameters_[0][i]);
    }
};

void ExhaustiveSearchSEMC::ExchangeMetropolis(int l){
    uniform_int_distribution<> dist_remove(0, op_.model_dim-1);
    uniform_int_distribution<> dist_add(0, data_.p-op_.model_dim-1);
    uniform_real_distribution<> dist(0, 1);

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
        vector<int> change_array(op_.parallel_num);
        for(int thread = 0; thread < op_.parallel_num; thread++){
            change_array[thread] = index[thread + iter*op_.parallel_num];
        }
        #pragma omp parallel for schedule (dynamic, 1)
        for(int thread = 0; thread < op_.parallel_num; thread++){
            int sample_id = iter*op_.parallel_num + thread;
            parameters_[l][sample_id] = parameters_[l][sample_id - op_.parallel_num];
            energies_[l][sample_id] = energies_[l][sample_id - op_.parallel_num];
            vector<vector<double> > new_parameters = parameters_[l][sample_id];
            int thread_id = omp_get_thread_num();
            int index_remove = dist_remove(engines_thread_[thread_id]);
            int index_add = dist_add(engines_thread_[thread_id]);
            for(int j = 0; j < op_.model_dim; j++){
                if(index_add >= new_parameters[j][0]){
                    index_add += 1;
                }
            }
            new_parameters[index_remove][0] = index_add;
            sort(new_parameters.begin(), new_parameters.end());
            double new_energy = ErrorCalculation(new_parameters);
            double energy_diff = new_energy - energies_[l][sample_id];
            double prob = -op_.data_num*settings_.inverse_temperatures[l]*energy_diff;
            if(log(dist(engines_thread_[thread_id])) < prob){
                parameters_[l][sample_id] = new_parameters;
                energies_[l][sample_id] = new_energy;
                if(sample_id > op_.burn_in){
                    accept_counts_temp[thread][0][0]++;
                }else{
                    accept_counts_for_Robbins_temp[thread][0][0]++;
                }
            }
            int change_index = change_array[thread] + op_.burn_in;
            double prob_2 = op_.data_num*(settings_.inverse_temperatures[l] - settings_.inverse_temperatures[l-1])*(energies_[l][sample_id] - energies_[l-1][change_index]);
            if(log(dist(engines_thread_[thread_id])) < prob_2){
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

void ExhaustiveSearchSEMC::Execution(){
    op_.replica_num = 1;
    InitStates();
    NextInverseTemperatures(op_.replica_num-1);
    while(true){
        op_.replica_num += 1;
        NextInverseTemperatures(op_.replica_num-1);
        ExchangeMetropolis(op_.replica_num-1);
        if(settings_.inverse_temperatures[op_.replica_num-1] == 1.0){
            break;
        }
    }
}

int main(){
    int data_num, parameter_num;
    HighDimensionalData Data;
    Data.sigma_beta = 1;
    Data.sigma_eps = sqrt(0.1);
    string file_x = "../input/x.txt";
    string file_y = "../input/y.txt";
    ifstream x_file(file_x);
    ifstream y_file(file_y);
    x_file >> data_num >> parameter_num;
    y_file >> data_num;
    Data.x = vector<vector<double> >(parameter_num, vector<double>(data_num, 0));
    Data.y = vector<double>(data_num, 0);
    for(int i=0; i<data_num; i++){
        for(int j=0; j<parameter_num; j++){
            x_file >> Data.x[j][i];
        }
        y_file >> Data.y[i];
    }
    Data.p = parameter_num;
    ModelOptions op;
    op.replica_num = 0;
    vector<double> gamma_decisions{0.1, 0.3, 0.5, 0.7, 0.9};
    op.model_dim = 4;
    for(int i=0; i<op.model_dim; i++){
        op.parameter_nums.push_back(1);
    }
    op.robbins_parameter.M = 50;
    op.robbins_parameter.p_goal = 0.5;
    op.robbins_parameter.N_0 = 15;
    op.robbins_parameter.c = 4.0;
    op.data_num = data_num;
    op.parallel_num = 50;
    // vector<double> d_w{1.0,1.0,0.6,1.0,1.0,1.0,1.0,1.0,0.8};
    // vector<double> d_b{0.8,0.8,0.6,0.4};

    // op.stepsize_parameter = vector<vector<vector<double> > > (op.model_dim);
    // for(int i = 0; i < op.model_dim; i++){
    //     op.stepsize_parameter[i] = vector<vector<double> > (op.parameter_nums[i]);
    //     for(int j = 0; j < op.parameter_nums[i]; j++){
    //         if(i < op.model_dim-1){
    //             op.stepsize_parameter[i][j] = {d_w[j], (double) op.data_num};
    //         }else{
    //             op.stepsize_parameter[i][j] = {d_b[j], (double) op.data_num};
    //         }
    //     }
    // }

    vector<vector<PriorParameter> > prior;

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
                ExhaustiveSearchSEMC semc(op, prior, Data);
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

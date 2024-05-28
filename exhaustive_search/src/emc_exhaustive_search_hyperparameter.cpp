#include "base_exhaustive_search.hpp"
#include "../../core/base_emc.hpp"
#include <omp.h>

class ExhaustiveSearchEMC : public BaseEMC {
    /** @class ExhaustiveSearchEMC
     * @brief Igarashi et al. 2018のモデルに対する交換モンテカルロ法
     */
    protected:
        const HighDimensionalData data_;

    public:
        ExhaustiveSearchEMC(ModelOptions op, vector<vector<PriorParameter> > prior, HighDimensionalData data): BaseEMC(op, prior), data_(data), BaseSampling(op, prior){
        };
        inline double ErrorCalculation(const vector<vector<double> >& parameter) const;
        inline void FreeEnergyCalculation();
        inline void InitStates();
        inline void Metropolis(int sample_id);
        inline void Execution();
        inline void CalculationForDos(string folder_name);
};

double ExhaustiveSearchEMC::ErrorCalculation(const vector<vector<double> >& parameters) const{
    vector<int> param(op_.model_dim);
    for(int i=0; i<op_.model_dim; i++){
        param[i] = parameters[i][0];
    }
    return ErrorExhaustiveSearch(data_, param, op_.data_num, op_.model_dim);
}

void ExhaustiveSearchEMC::FreeEnergyCalculation() {
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
};

void ExhaustiveSearchEMC::CalculationForDos(string folder_name) {
    vector<double> Z(op_.replica_num-1);
    vector<double> constant_for_logsumexp(op_.replica_num-1);
    for(int i=0; i<op_.replica_num-1; i++){
        if(i > 0){
            Z[i] = Z[i-1];
        }
        constant_for_logsumexp[i] = op_.data_num*(settings_.inverse_temperatures[i+1] - settings_.inverse_temperatures[i])*energies_[op_.burn_in][i];
        double temp = 0;
       for(int j=op_.burn_in; j<op_.sample_num; j++){
            temp += exp(-op_.data_num*(settings_.inverse_temperatures[i+1] - settings_.inverse_temperatures[i])*energies_[j][i] + constant_for_logsumexp[i]);
       }
       Z[i] -= log(temp/(op_.sample_num - op_.burn_in));
       Z[i] += constant_for_logsumexp[i];
    }
    string file_name = folder_name + "Z.txt";
    ofstream ofs(file_name);
    for(int i=0; i<op_.replica_num-1; i++){
        ofs << Z[i] << endl;
    }
    ofs.close();
    // エネルギーを全て出力
    file_name = folder_name + "energies_all.txt";
    ofstream ofs_energies(file_name);
    for(int i=0; i<op_.replica_num; i++){
        for(int j=op_.burn_in; j<op_.sample_num; j++){
            ofs_energies << energies_[j][i] << endl;
        }
    }
    ofs_energies.close();
};

void ExhaustiveSearchEMC::InitStates(){
    /** @fn パラメータ, エネルギーを事前分布から初期化 */
    cout << "InitStates" << endl;
    #pragma omp parallel for schedule (dynamic, 1)
    for(int i=0; i<op_.replica_num; i++){
        vector<int> index = n_choose_k(data_.p, op_.model_dim, engines_thread_[omp_get_thread_num()]);
        for(int j = 0; j < op_.model_dim; j++){
            parameters_[0][i][j][0] = index[j];
        }
        energies_[0][i] = ErrorCalculation(parameters_[0][i]);
    }
    #pragma omp parallel for schedule (dynamic, 1)
    for(int i=1; i<op_.sample_num; i++){
        vector<int> index = n_choose_k(data_.p, op_.model_dim, engines_thread_[omp_get_thread_num()]);
        for(int j = 0; j < op_.model_dim; j++){
            parameters_[i][0][j][0] = index[j];
        }
        energies_[i][0] = ErrorCalculation(parameters_[i][0]);
    }
};

void ExhaustiveSearchEMC::Metropolis(int sample_id){
    uniform_int_distribution<> dist_remove(0, op_.model_dim-1);
    uniform_int_distribution<> dist_add(0, data_.p-op_.model_dim-1);
    uniform_real_distribution<> dist(0, 1);
    #pragma omp parallel for schedule (dynamic, 1)
    for(int i=1; i<op_.replica_num; i++){
        vector<vector<double> > new_parameters = parameters_[sample_id][i];
        int thread = omp_get_thread_num();
        int index_remove = dist_remove(engines_thread_[thread]);
        int index_add = dist_add(engines_thread_[thread]);
        for(int j = 0; j < op_.model_dim; j++){
            if(index_add >= new_parameters[j][0]){
                index_add += 1;
            }
        }
        new_parameters[index_remove][0] = index_add;
        sort(new_parameters.begin(), new_parameters.end());
        double new_energy = ErrorCalculation(new_parameters);
        double energy_diff = new_energy - energies_[sample_id][i];
        double prob = -op_.data_num*settings_.inverse_temperatures[i]*energy_diff;
        if(log(dist(engines_thread_[thread])) < prob){
            parameters_[sample_id][i] = new_parameters;
            energies_[sample_id][i] = new_energy;
            if(sample_id > op_.burn_in){
                accept_counts_[i][0][0]++;
            }else{
                accept_counts_for_Robbins_[i][0][0]++;
            }
        }
    }
}

void ExhaustiveSearchEMC::Execution(){
    InitStates();
    InitializeInverseTemperatures();
    for(int i=1; i<op_.sample_num; i++){
        for(int j=1; j<op_.replica_num; j++){
            parameters_[i][j] = parameters_[i-1][j];
            energies_[i][j] = energies_[i-1][j];
        }
        Metropolis(i);
        Exchange(i);
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
    op.replica_num = 100;
    op.gamma_decision = 0.5;
    op.model_dim = 4;
    for(int i=0; i<op.model_dim; i++){
        op.parameter_nums.push_back(1);
    }
    op.robbins_parameter.M = 20;
    op.robbins_parameter.p_goal = 0.5;
    op.robbins_parameter.N_0 = 15;
    op.robbins_parameter.c = 4.0;
    op.sample_num = 200000;
    op.burn_in = 100000;
    op.data_num = data_num;
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

    int repeat = 1;
    vector<int> L_array{10,30,100,300};
    vector<int> burn_in_array{10000};
    for(int j=0; j<L_array.size(); j++){
        for(int i = 0; i<1; i++){
            string folder_name_base = "../output/EMC_hyperparameter/burn_in_" + to_string(burn_in_array[i]) + "_L_" + to_string(L_array[j]) + "/";
            mkdir(folder_name_base.c_str(), 0777);
            for(int iter = 0; iter < repeat; iter++){
                op.sample_num = burn_in_array[i]*2;
                op.burn_in = burn_in_array[i];
                op.replica_num = L_array[j];
                timespec start, end;
                clock_gettime(CLOCK_MONOTONIC, &start);
                ExhaustiveSearchEMC emc(op, prior, Data);
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

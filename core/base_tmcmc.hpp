#ifndef BASE_TMCMC_HPP
#define BASE_TMCMC_HPP

#include "base_sampling.hpp"

#include <omp.h>

using namespace std;

class BaseTMCMC : public BasePopulation{
    /** @class TMCMC基底クラス */
    protected:
        vector<int> accept_counts_ = vector<int> (1,0);
        int all_parameters_num_ = (int) 0;
        vector<double> c_;

    public:
        BaseTMCMC(ModelOptions op, vector<vector<PriorParameter> > prior): BasePopulation(op, prior){
            for(int i = 0; i < op_.model_dim; i++){
                all_parameters_num_ += op_.parameter_nums[i];
            }
        };


    inline void NextInverseTemperatures(int l);
    /** @fn 逆温度の設定 */
    inline vector<vector<double> > ParameterCovariance(int l);
    /** @fn パラメータの共分散行列の計算 */
    inline vector<vector<double> > CholeskyDecomposition(vector<vector<double> > matrix);
    /** @fn コレスキー分解 */
    inline vector<double> MultivariateNormalRamdom(vector<vector<double> > lower);
    /** @fn 多変量正規分布の乱数生成 */
    inline void Resampling(int l);
    /** @fn パラメータのリサンプリング */
    inline void Metropolis(int l, vector<vector<double> > lower);
    /** @fn メトロポリス法によって, パラメータ更新 */
    inline void NextStepSizes(int l);
    /** @fn ステップサイズの更新 */
    inline void Execution();
    /** @fn 焼きなまし交換モンテカルロ法の実行 */
    inline void SaveAcceptanceRate();
    /** @fn 受容率の保存 */
    inline void Save(string folder_name);
    /** @fn 全てをファイルに保存 */

    inline virtual double ErrorCalculation(const vector<vector<double> >& parameters) const = 0;
    /** @fn 誤差関数の計算 */
};

// void BaseTMCMC::NextInverseTemperatures(int l){
//     if(l == 0){
//         settings_.inverse_temperatures.push_back(0);
//     }else{
//         double low_beta = settings_.inverse_temperatures[l-1];
//         double old_beta = settings_.inverse_temperatures[l-1];
//         double high_beta = 2.0;
//         double threshold = 0.5;
//         int rN = int(op_.sample_num*threshold);
//         double new_beta;
//         while(high_beta - low_beta > 1e-6){
//             new_beta = (high_beta + low_beta)/2;
//             vector<double> log_weights_un(op_.sample_num);
//             for(int i = 0; i < op_.sample_num; i++){
//                 log_weights_un[i] = -op_.data_num*(new_beta - old_beta)*energies_[l-1][i];
//             }
//             double log_weights_sum = 0;
//             double exclude = log_weights_un[0];
//             for(int i = 1; i < op_.sample_num; i++){
//                 log_weights_sum += exp(log_weights_un[i] - exclude);
//             }
//             log_weights_sum = log(log_weights_sum) + exclude;
//             vector<double> log_weights(op_.sample_num);
//             for(int i = 0; i < op_.sample_num; i++){
//                 log_weights[i] = log_weights_un[i] - log_weights_sum;
//             }
//             double log_weights_2_sum = 0;
//             exclude = log_weights[0]*2;
//             for(int i = 0; i < op_.sample_num; i++){
//                 log_weights_2_sum += exp(log_weights[i]*2 - exclude);
//             }
//             log_weights_2_sum = log(log_weights_2_sum) + exclude;
//             int ESS = int(exp(-log_weights_2_sum));
//             if (ESS == rN){
//                 break;
//             }else if(ESS < rN){
//                 high_beta = new_beta;
//             }else{
//                 low_beta = new_beta;
//             }
//         } 
//         if(new_beta > 1){
//             settings_.inverse_temperatures.push_back(1);
//         }else{
//             settings_.inverse_temperatures.push_back(new_beta); 
//         }
//         cout << "inverse_temperatures: " << settings_.inverse_temperatures[l] << endl;
//     }
// }

void BaseTMCMC::NextInverseTemperatures(int l){
    if(l == 0){
        settings_.inverse_temperatures.push_back(0);
    }else{
        double low_beta = settings_.inverse_temperatures[l-1];
        double old_beta = settings_.inverse_temperatures[l-1];
        double high_beta = 2.0;
        double new_beta;
        while(high_beta - low_beta > 1e-6){
            new_beta = (high_beta + low_beta)/2;
            double mean = 0;
            double exclude = exp(-op_.data_num*(new_beta - old_beta)*energies_[l-1][0]);
            for(int i = 0; i < op_.sample_num; i++){
                mean += exp(-op_.data_num*(new_beta - old_beta)*energies_[l-1][i] - exclude);
            }
            mean /= op_.sample_num;
            double CV = 0;
            for (int i = 0; i < op_.sample_num; i++){
                CV += pow(exp(-op_.data_num*(new_beta - old_beta)*energies_[l-1][i] - exclude) - mean, 2);
            }
            CV = sqrt(CV/(op_.sample_num))/mean;
            if (CV == 1){
                break;
            }else if(CV > 1){
                high_beta = new_beta;
            }else{
                low_beta = new_beta;
            }
        } 
        if(new_beta > 1){
            settings_.inverse_temperatures.push_back(1);
        }else{
            settings_.inverse_temperatures.push_back(new_beta); 
        }
        cout << "inverse_temperatures: " << settings_.inverse_temperatures[l] << endl;
    }
}


vector<vector<double> > BaseTMCMC::ParameterCovariance(int l){
    // 重みつきのパラメータの共分散行列の計算
    vector<vector<double> > covariance(all_parameters_num_, vector<double>(all_parameters_num_));
    vector<double> mean(all_parameters_num_);
     int index = 0;
    for(int i = 0; i < op_.model_dim; i++){
        for(int j = 0; j < op_.parameter_nums[i]; j++){
            mean[index] = 0;
            for(int k = 0; k < op_.sample_num; k++){
                mean[index] += parameters_[l-1][k][i][j]*weight_[k];
            }
            index += 1;
        }
    }
    int left_index = 0;
    for(int i_1 = 0; i_1 < op_.model_dim; i_1++){
        for(int j_1 = 0; j_1 < op_.parameter_nums[i_1]; j_1++){
            int right_index = 0;
            for(int i_2 = 0; i_2 < op_.model_dim; i_2++){
                for(int j_2 = 0; j_2 < op_.parameter_nums[i_2]; j_2++){
                    covariance[left_index][right_index] = 0;
                    for(int k = 0; k < op_.sample_num; k++){
                        covariance[left_index][right_index] += (parameters_[l-1][k][i_1][j_1] - mean[left_index])*(parameters_[l-1][k][i_2][j_2] - mean[right_index])*weight_[k];
                    }
                    right_index += 1;
                }
            }
            left_index += 1;
        }
    }
    return covariance;
}

vector<vector<double> > BaseTMCMC::CholeskyDecomposition(vector<vector<double> > matrix){
    // コレスキー分解
    vector<vector<double> > lower(all_parameters_num_, vector<double>(all_parameters_num_));
    for(int i = 0; i < all_parameters_num_; i++){
        for(int j = 0; j < all_parameters_num_; j++){
            lower[i][j] = 0;
        }
    }
    for(int i = 0; i < all_parameters_num_; i++){
        for(int j = 0; j < i+1; j++){
            double sum = 0;
            for(int k = 0; k < j; k++){
                sum += lower[i][k]*lower[j][k];
            }
            if(i == j){
                if(matrix[i][i] - sum < 0){
                    cout << "i: " << i << " j: " << j << " matrix[i][i]: " << matrix[i][i] << " sum: " << sum << " matrix[i][i] - sum: " << matrix[i][i] - sum << " lower[i][j]: " << lower[i][j] << endl;
                    lower[i][j] = 0;
                    cout << "Error: Matrix is not positive definite" << endl;
                }else{
                    lower[i][j] = sqrt(matrix[i][i] - sum);
                }
            }else{
                if(lower[j][j] == 0){
                    cout << "i: " << i << " j: " << j << " matrix[i][i]: " << matrix[i][i] << " sum: " << sum << " matrix[i][i] - sum: " << matrix[i][i] - sum << " lower[i][j]: " << lower[i][j] << endl;
                    lower[i][j] = 0;
                    cout << "Error: Matrix is not positive definite" << endl;
                }else{
                    lower[i][j] = (matrix[i][j] - sum)/lower[j][j];
                }
            }
        }
    }
    return lower;
}

vector<double> BaseTMCMC::MultivariateNormalRamdom(vector<vector<double> > lower) {
    // 平均0、分散1の正規乱数を生成
    normal_distribution<> dist(0, 1);   
    vector<double> x;
    int thread = omp_get_thread_num();
    for (int i = 0; i < all_parameters_num_; i++) {
        x.push_back(dist(engines_thread_[thread]));
    }
    // matrixを使って乱数を変換
    vector<double> y;
    for (int i = 0; i < all_parameters_num_; i++) {
        double sum = 0;
        for (int j = 0; j < all_parameters_num_; j++) {
            sum += lower[i][j] * x[j];
        }
        y.push_back(sum);
    }
    return y;
}

void BaseTMCMC::NextStepSizes(int l){
    // ステップサイズの更新
    double p_accept = 0;
    p_accept = (double) accept_counts_[l]/(op_.sample_num*op_.n_steps);
    c_.push_back(c_[l-1]*exp(-op_.control_parameter.G*(op_.control_parameter.p_goal - p_accept)));
}

void BaseTMCMC::Resampling(int l){
    // パラメータのリサンプリング
    vector<int> resample(op_.sample_num, 0);
    discrete_distribution<> dist_index(weight_.begin(), weight_.end());
    for(int iter = 0; iter < op_.sample_num; iter++){
        int result = dist_index(engines_thread_[0]);
        resample[result] += 1;
    }
    vector<vector<vector<double> > > new_parameters = vector<vector<vector<double> > >(op_.sample_num);
    vector<double> new_energies = vector<double> (op_.sample_num);
    int index = 0;
    for(int i = 0; i < op_.sample_num; i++){
        int s = resample[i];
        for(int j = 0; j < s; j++){
            new_parameters[index] = parameters_[l-1][i];
            new_energies[index] = energies_[l-1][i];
            index += 1;
        }
    }
    parameters_.push_back(new_parameters);
    energies_.push_back(new_energies);
}

void BaseTMCMC::Metropolis(int l, vector<vector<double> > lower){
    // メトロポリス法によって, パラメータ更新
    uniform_real_distribution<> dist(0, 1);
    vector<int> accept_temp(op_.sample_num, 0);
    for(int i = 0; i < op_.n_steps; i++){
        #pragma omp parallel for schedule (dynamic, 1)
        for(int j = 0; j < op_.sample_num; j++){
            int thread = omp_get_thread_num();
            vector<vector<double> > new_parameter = parameters_[l][j];
            vector<double> y = MultivariateNormalRamdom(lower);
            double prob = 0;
            int index = 0;
            for(int k = 0; k < op_.model_dim; k++){
                for(int m = 0; m < op_.parameter_nums[k]; m++){
                    new_parameter[k][m] += c_[l-1]*y[index];
                    prob -= ProbCalculationOnPrior(parameters_[l][j][k][m], prior_[k][m].parameter, prior_[k][m].type);
                    prob += ProbCalculationOnPrior(new_parameter[k][m], prior_[k][m].parameter, prior_[k][m].type);
                    index += 1;
                }
            }
            double Energy = ErrorCalculation(new_parameter);
            double energy_diff = Energy - energies_[l][j];
            prob += -op_.data_num*settings_.inverse_temperatures[l]*energy_diff;
            if(log(dist(engines_thread_[thread])) < prob){
                parameters_[l][j] = new_parameter;
                energies_[l][j] = Energy;
                accept_temp[j] += 1;
            }
        }
    }
    int sum = 0;
    for(int i = 0; i < op_.sample_num; i++){
        sum += accept_temp[i];
    }
    accept_counts_.push_back(sum);
}

void BaseTMCMC::Execution(){
    InitStates();
    NextInverseTemperatures(0);
    op_.replica_num = 1;   
    c_.push_back(op_.control_parameter.c_0);
    while(true){
        NextInverseTemperatures(op_.replica_num);
        weight_for_resampling(op_.replica_num);
        vector<vector<double> > covariance = ParameterCovariance(op_.replica_num);
        vector<vector<double> > lower = CholeskyDecomposition(covariance);
        Resampling(op_.replica_num);
        Metropolis(op_.replica_num, lower);
        NextStepSizes(op_.replica_num);
        op_.replica_num += 1;
        if(settings_.inverse_temperatures[op_.replica_num-1] == 1){
            break;
        }
    }
}

void BaseTMCMC::SaveAcceptanceRate(){
    /** @fn 受容率の保存 */
    string file_name = "acceptance_rate.txt";
    ofstream file(folder_name_ + file_name);
    for(int i=1; i<op_.replica_num; i++){
        file <<  (double) accept_counts_[i]/(op_.sample_num*op_.n_steps) << " ";
    }
    file.close();
};

void BaseTMCMC::Save(string folder_name){
    SetFolderName(folder_name);
    SaveParameters();
    SaveEnergies();
    SaveAcceptanceRate();
    SaveFreeEnergy();
    SaveCalculationTime();
    SaveInverseTemperatures();
}

#endif

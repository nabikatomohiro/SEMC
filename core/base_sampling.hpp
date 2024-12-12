#ifndef BASIC_SAMPLING_HPP
#define BASIC_SAMPLING_HPP

#include <iostream>
#include <cmath>
#include <random>
#include <fstream>
#include <vector>
#include <sys/stat.h>
#include <time.h>
#include <algorithm>
#include <omp.h>

using namespace std;

struct RobbinsParameter{
    /** @struct ロビンスモンロー法のパラメータ 
     * @param M       : 更新頻度パラメータ
     * @param p_goal  : 目標採択率
     * @param N_0     : 更新係数
     * @param c       : 更新係数
     */

    int M;
    double p_goal;
    int N_0;
    double c;
};

struct ControlParameter{
    /** @struct ロビンスモンロー法のパラメータ 
     * @param G       : 更新係数
     * @param p_goal  : 目標採択率
     * @param c_0       : 初期値
     */

    double G;
    double p_goal;
    double c_0;
};


struct PriorParameter{
    /** @struct 事前分布のパラメータ
     * @param type      : 事前分布の種類(0:ガウス分布, 1:一様分布)
     * @param parameter : 事前分布のパラメータ(ガウス分布: 平均, 分散, 一様分布: 下限, 上限)
     */
    int type;
    pair<double, double> parameter;
};

struct ModelOptions {
    /** @struct 交換モンテカルロ法の設定
     * @param replica_num           : レプリカ層数
     * @param gamma_decision        : 逆温度設定用パラメータ
     * @param model_dim             : モデルの次元
     * @param parameter_nums        : 各モデルのパラメータの数
     * @param stepsize_parameter    : ステップサイズの初期値調整用パラメータ
     * @param robbins_parameter     : ロビンスモンロー法のパラメータ
     * @param control_parameter     : 制御論のパラメータ
     * @param sample_num            : サンプル数
     * @param burn_in               : バーンイン回数
     * @param data_num              : データ数
     * @param parallel_num          : 並列数
     * @param n_steps               : ステップ数
     * @param M_pop               : Waste freeのぽピュレーション数
     */

    int replica_num;
    double gamma_decision;
    int model_dim;
    vector<int> parameter_nums;
    vector<vector<vector<double> > > stepsize_parameter;
    RobbinsParameter robbins_parameter;
    ControlParameter control_parameter;
    int sample_num;
    int burn_in;
    int data_num;
    int parallel_num;
    int n_steps;
    int M_pop;
};

struct ModelSettings{
    /** @struct 温度層等のパラメータ
     * @param inverse_temperatures  : 各レプリカ層の逆温度
     * @param step_sizes            : 全パラメータのステップサイズ
    */

   vector<double> inverse_temperatures;
   vector<vector<vector<double> > > step_sizes; // (温度層数, モデル数, パラメータ数)
};

class BaseSampling{
    protected:
        ModelOptions op_;
        const vector<vector<PriorParameter> > prior_;
        /** パラメータサンプル [サンプルID][レプリカ層][モデル数][パラメータ数] または [レプリカ層][サンプルID][モデル数][パラメータ数]*/
        vector<vector<vector<vector<double> > > > parameters_; 
        /** エネルギーサンプル [サンプルID][レプリカ層] または [レプリカ層][サンプルID]*/
        vector<vector<double> > energies_; 
        /** 温度層等の設定 */
        ModelSettings settings_;
        /** 自由エネルギー */
        double free_energy_;
        /** 計算時間 */
        double calculation_time_;
        /** スレッドに対する乱数エンジン*/
        vector<mt19937> engines_thread_;

    public:
        BaseSampling(ModelOptions op, vector<vector<PriorParameter> > prior): op_(op), prior_(prior){
            random_device seed_gen;
            mt19937 engine(seed_gen());
            //ランダムな整数を生成
            int thread_num = omp_get_max_threads();
            vector<int> random_index(thread_num);
            for(int i=0; i<thread_num; i++){
                engines_thread_.push_back(mt19937(random_index[i]));
            }
        };
        /** データ保存先 */
        string folder_name_; 

    inline double ProbCalculationOnPrior(const double parameter, const pair<double, double>& prior_parameter, const int type);
    /** @fn 事前分布の確率計算 */
    inline vector<vector<double> > SamplingFromPrior(mt19937& engine);
    /** @fn 事前分布からのサンプリング */
    inline void SetCalculationTime(const double calculation_time);
    /** @fn 計算時間のセット */
    inline void SetFolderName(const string& folder_name);
    /** @fn データ保存先の設定 */
    inline void SaveInverseTemperatures();
    /** @fn 逆温度の保存 */
    inline void SaveFreeEnergy();
    /** @fn 自由エネルギーの保存 */
    inline void SaveCalculationTime();
    /** @fn 計算時間の保存 */
};

double BaseSampling::ProbCalculationOnPrior(const double parameter, const pair<double, double>& prior_parameter, const int type){
    if(type == 0){
        double temp = (parameter-prior_parameter.first)/prior_parameter.second;
        double prob = -0.5*temp*temp;
        return prob;
    }else if(type == 1){
        if(prior_parameter.first <= parameter && parameter <= prior_parameter.second){
            return 0;
        }else{
            return -INFINITY;
        }
    }else if(type == 2){
        return parameter*prior_parameter.first - prior_parameter.second*exp(parameter);
    }else{
        printf("Error: Invalid prior type\n");
        return -INFINITY;
    }
}

vector<vector<double> > BaseSampling::SamplingFromPrior(mt19937& engine){
    vector<vector<double> > parameter((op_.model_dim));
    for(int j=0; j<op_.model_dim; j++){
        parameter[j] = vector<double>(op_.parameter_nums[j]);
        for(int k=0; k<op_.parameter_nums[j]; k++){
            if(prior_[j][k].type == 0){
                normal_distribution<> dist(prior_[j][k].parameter.first, prior_[j][k].parameter.second);
                parameter[j][k] = dist(engine);
            }else if(prior_[j][k].type == 1){
                uniform_real_distribution<> dist(prior_[j][k].parameter.first, prior_[j][k].parameter.second);
                parameter[j][k] = dist(engine);
            }else if(prior_[j][k].type == 2){
                gamma_distribution<> dist(prior_[j][k].parameter.first, 1.0/prior_[j][k].parameter.second);
                parameter[j][k] = log(dist(engine));
            }else{
                printf("Error: Invalid prior type\n");
            }
        }
    }
    return parameter;
}

void BaseSampling::SetCalculationTime(const double calculation_time){
    calculation_time_ = calculation_time;
}

void BaseSampling::SetFolderName(const string& folder_name){
    folder_name_ = folder_name;
    // フォルダが存在しない場合は作成
    mkdir(folder_name_.c_str(), 0777);
}

void BaseSampling::SaveInverseTemperatures(){
    ofstream file(folder_name_ + "inverse_temperatures.txt");
    file << op_.replica_num << endl;
    for(int i=0; i<op_.replica_num; i++){
        file << settings_.inverse_temperatures[i] << " ";
    }
    file << endl;
    file.close();
}

void BaseSampling::SaveFreeEnergy(){
    ofstream file(folder_name_ + "free_energy.txt");
    file << free_energy_ << endl;
    file.close();
}

void BaseSampling::SaveCalculationTime(){
    ofstream file(folder_name_ + "calculation_time.txt");
    file << calculation_time_ << endl;
    file.close();
}

class BaseMonteCarlo : virtual public BaseSampling{
    protected:
        /** 受容回数 [レプリカ層][モデル数][パラメータ数] */
        vector<vector<vector<int> > > accept_counts_; 
        /** 受容回数 [レプリカ層][モデル数][パラメータ数] */
        vector<vector<vector<int> > > accept_counts_for_Robbins_;
        /** 交換回数 [レプリカ層] */
        vector<int> exchange_counts_; 

    public:
        BaseMonteCarlo(ModelOptions op, vector<vector<PriorParameter> > prior): BaseSampling(op, prior){
        };

    inline void SaveAcceptanceRate();
    /** @fn 受容率の保存 */
    inline void SaveExchangeRate();
    /** @fn 交換率の保存 */
};

void BaseMonteCarlo::SaveAcceptanceRate(){
    /** @fn 受容率の保存 */
    string file_name = "acceptance_rate.txt";
    ofstream file(folder_name_ + file_name);
    for(int i=1; i<op_.replica_num; i++){
        for(int j=0; j<op_.model_dim; j++){
            for(int k=0; k<op_.parameter_nums[j]; k++){
                file << (double) accept_counts_[i][j][k]/(op_.sample_num - op_.burn_in) << " ";
            }
        }
        file << endl;
    }
    file.close();
};

void BaseMonteCarlo::SaveExchangeRate(){
    /** @fn 交換率の保存 */
    string file_name = "exchange_rate.txt";
    ofstream file(folder_name_ + file_name);
    for(int i=0; i<op_.replica_num-1; i++){
        file << (double) exchange_counts_[i]/(op_.sample_num - op_.burn_in) << " ";
    }
    file << endl;
    file.close();
};

class BasePopulation : virtual public BaseSampling{
    protected:
        vector<double> weight_;
    public:
        BasePopulation(ModelOptions op, vector<vector<PriorParameter> > prior): BaseSampling(op, prior){
            parameters_ = vector<vector<vector<vector<double> > > > (1, vector<vector<vector<double> > > (op.sample_num, vector<vector<double> >(op.model_dim)));
            for(int i=0; i<1; i++){
                for(int j=0; j<op.sample_num; j++){
                    for(int k=0; k<op.model_dim; k++){
                        parameters_[i][j][k].resize(op.parameter_nums[k]);
                    }
                }
            }
            energies_ = vector<vector<double> > (1, vector<double>(op.sample_num));
            weight_ = vector<double>(op.sample_num - op.burn_in);
        };
    
    inline virtual double ErrorCalculation(const vector<vector<double> >& parameters) const = 0;
    /** @fn 誤差関数の計算 */
    inline void InitStates();
    /** @fn パラメータ, エネルギーを事前分布から初期化 */
    inline void weight_for_resampling(int l);
    /** @fn リサンプリング用の重みの計算 */
    inline void SaveParameters();
    /** @fn パラメータの保存 */
    inline void SaveEnergies();
    /** @fn エネルギーの保存 */
};

void BasePopulation::InitStates(){
    /** @fn パラメータ, エネルギーを事前分布から初期化 */
    #pragma omp parallel for schedule (dynamic, 1)
    for(int i=0; i<op_.sample_num; i++){
        int thread = omp_get_thread_num();
        parameters_[0][i] = SamplingFromPrior(engines_thread_[thread]);
        energies_[0][i] = ErrorCalculation(parameters_[0][i]);
    }
};

void BasePopulation::weight_for_resampling(int l){
    double exclude = energies_[l-1][op_.burn_in];
    double sum = 0;
    for(int i=op_.burn_in; i<op_.sample_num; i++){
        weight_[i-op_.burn_in] = exp(-op_.data_num*(settings_.inverse_temperatures[l] - settings_.inverse_temperatures[l-1])*(energies_[l-1][i] - exclude));
        sum += weight_[i-op_.burn_in];
    }
    for(int i=op_.burn_in; i<op_.sample_num; i++){
        weight_[i-op_.burn_in] /= sum;
    }
}

void BasePopulation::SaveParameters(){
    /** @fn パラメータの保存 */
    string file_name = "parameters.txt";
    ofstream file(folder_name_ + file_name);
    for(int i=0; i<op_.sample_num; i++){
        for(int j=0; j<op_.model_dim; j++){
            for(int k=0; k<op_.parameter_nums[j]; k++){
                file << parameters_[op_.replica_num - 1][i][j][k] << " ";
            }
        }
        file << endl;
    }
    file.close();
};

void BasePopulation::SaveEnergies(){
    /** @fn エネルギーの保存 */
    string file_name = "energies.txt";
    ofstream file(folder_name_ + file_name);
    for(int i=0; i<op_.sample_num; i++){
        file << energies_[op_.replica_num - 1][i] << endl;
    }
    file.close();
};



#endif
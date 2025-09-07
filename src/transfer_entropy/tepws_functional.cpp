/*
09/2025

Transfer entropy calculation vis-a-vis functional programming

Compilation:
------------
- Optimized:
    g++ -std=c++11 -O3 -Wall -march=native -flto -DNDEBUG -ffast-math -funroll-loops -o tepws_functional.x tepws_functional.cpp
- Debug:
    g++ -std=c++11 -g -O0 -Wall -Wextra -fsanitize=address -o tepws_functional_debug.x tepws_functional.cpp

References:
-----------
- Schreiber, PRL, 85, 2 (2000)
    https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.85.461
- Das and Wolde, PRL, 135, 107404 (2025):
    https://journals.aps.org/prl/abstract/10.1103/t8z9-ylvg

Author:
--------
    Sam Dawley
*/

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <chrono>

// Global random number generator for efficiency
static std::mt19937 g_rng;
static std::uniform_real_distribution<double> g_uniform_dist(0.0, 1.0);
static std::normal_distribution<double> g_normal_dist(0.0, 1.0);

struct TEPWSParams {
    double dt;    // Timestep
    int N;        // Number of timesteps
    int I;        // Total time of trajectory
    int M1, M2;   // Number of trajectories for Monte-Carlo averages
    bool process; // Controls whether the process is continuous or discrete
};

struct TrajectoryData {
    std::vector<double> X1_nu, X2_nu, X3_nu;  // nu-labelled trajectories
    std::vector<double> X1_mu, X2_mu, X3_mu;  // mu-labelled trajectories
    std::vector<double> Tnu_a, Tnu_b;         // cumulative transfer entropy
    std::vector<double> weights;              // weights of individual trajectories
    std::vector<int> resample_indices;        // when resampling, randomly choose trajectories
};


void initialize_rng(unsigned int seed = 0) {
    if (seed == 0) {
        seed = std::chrono::steady_clock::now().time_since_epoch().count();
    }
    g_rng.seed(seed);
}

// Fast access to trajectory data using flat indexing
inline double& getX1(std::vector<double>& X1_nu, int M, int mu, int k) { 
    return X1_nu[mu * M + k]; 
}

inline double& getX2(std::vector<double>& X2_nu, int M, int mu, int k) { 
    return X2_nu[mu * M + k]; 
}

inline double& getX3(std::vector<double>& X3_nu, int M, int mu, int k) { 
    return X3_nu[mu * M + k]; 
}

inline const double& getX1_const(const std::vector<double>& X1_nu, int M, int mu, int k) { 
    return X1_nu[mu * M + k]; 
}

inline const double& getX2_const(const std::vector<double>& X2_nu, int M, int mu, int k) { 
    return X2_nu[mu * M + k]; 
}

inline const double& getX3_const(const std::vector<double>& X3_nu, int M, int mu, int k) { 
    return X3_nu[mu * M + k]; 
}

// Initialize trajectory data structure
void initialize_trajectory_data(TrajectoryData& data, const TEPWSParams& params) {
    // Pre-allocate all vectors for efficiency
    data.X1_nu.reserve(params.M1 * params.I);
    data.X2_nu.reserve(params.M1 * params.I);
    data.X3_nu.reserve(params.M1 * params.I);
    data.weights.resize(params.M1, 0.0);
    
    data.Tnu_a.resize(params.N + 1, 0.0);
    data.Tnu_b.resize(params.N + 1, 0.0);
    
    // Temporary storage
    data.X1_mu.reserve(params.M2 * params.I);
    data.X2_mu.reserve(params.M2 * params.I);
    data.X3_mu.reserve(params.M2 * params.I);
    data.resample_indices.reserve(std::max(params.M1, params.M2));

    return;
}

// Ulam map for testing initial conditions
inline double tent_map(double x) {
    if (x < 0.5) {
        return 2 * x;
    } else {
        return 2 - 2 * x; 
    }
}

// Generate initial conditions from steady-state trajectory
void generate_initial_conditions(
    std::vector<double>& x1_init,
    std::vector<double>& x2_init,
    std::vector<double>& x3_init,
    int num_samples
) {
    // Placeholder: In practice, this would sample from the steady-state distribution
    // For demonstration, using Gaussian initial conditions
    x1_init.resize(num_samples);
    x2_init.resize(num_samples);
    x3_init.resize(num_samples);
    
    // for (int i = 0; i < num_samples; ++i) {
    //     x1_init[i] = g_normal_dist(g_rng);
    //     x2_init[i] = g_normal_dist(g_rng);
    //     x3_init[i] = g_normal_dist(g_rng);
    // }

    // parameters for using the tent, ulam maps
    double epsilon = 0.03; 
    x1_init[0] = g_normal_dist(g_rng);
    x2_init[0] = g_normal_dist(g_rng);
    x3_init[0] = g_normal_dist(g_rng);

    for (int i = 1; i < num_samples; ++i) {
        x1_init[i] = tent_map(epsilon * x3_init[i - 1] + (1 - epsilon) * x1_init[i]);
        x2_init[i] = tent_map(epsilon * x1_init[i - 1] + (1 - epsilon) * x2_init[i]);
        x3_init[i] = tent_map(epsilon * x2_init[i - 1] + (1 - epsilon) * x3_init[i]);
    }

    return;
}

/**
 * @brief Propagate reference dynamics (placeholder for actual dynamics)
 * @param data
 * @param M
 * @param mu
 * @param k_start
 * @param k_end
 */
void propagate_reference_dynamics(TrajectoryData& data, int M, int mu, int k_start, int k_end) {
    // Placeholder: Implement actual stochastic dynamics here
    // This should propagate X1_{k,k+1}, X2_{k,k+1}, X3_{k,k+1}
    // using the reference dynamics P0(X1_{k,k+1}, X2_{k,k+1} | X1_{0,k}, X2_{0,k})
    
    for (int k = k_start; k < k_end; ++k) {
        // Simple Ornstein-Uhlenbeck process as example
        double dt = 0.01;
        double noise_strength = 0.1;
        
        getX1(data.X1_nu, M, mu, k + 1) = ( 
            getX1(data.X1_nu, M, mu, k) - 
            0.1 * getX1(data.X1_nu, M, mu, k) * dt + 
            noise_strength * g_normal_dist(g_rng) * std::sqrt(dt)    
        );

        getX2(data.X2_nu, M, mu, k + 1) = ( 
            getX2(data.X2_nu, M, mu, k) - 
            0.1 * getX2(data.X2_nu, M, mu, k) * dt + 
            0.05 * getX1(data.X1_nu, M, mu, k) * dt +  // coupling from X1
            noise_strength * g_normal_dist(g_rng) * std::sqrt(dt)
        );
            
        getX3(data.X3_nu, M, mu, k + 1) = (
            getX3(data.X3_nu, M, mu, k) -
            0.1 * getX3(data.X3_nu, M, mu, k) * dt + 
            noise_strength * g_normal_dist(g_rng) * std::sqrt(dt)
        );
    }
}

/**
 * @brief Compute logarithmic probability ratios for weight updates
 * @param data
 * @param M
 * @param mu
 * @param k
 * @return Log-probability ratio
 */
double compute_log_probability_ratio(const TrajectoryData& data, int M, int mu, int k) {
    // Placeholder: Implement actual probability ratio computation
    // This should compute ln P(X1_{k,k+1}, X2_{k,k+1} | X1_{0,k}, X2_{0,k}, X3_{0,k}) - 
    //                     ln P(X1_{k,k+1}, X2_{k,k+1} | X1_{0,k}, X2_{0,k})
    
    // Example: Simple coupling strength difference
    double coupling_effect = 0.01 * getX1_const(data.X1_nu, M, mu, k) * getX2_const(data.X2_nu, M, mu, k + 1);
    return coupling_effect;
}

// Efficient resampling using systematic resampling
void resample_trajectories(TrajectoryData& data, int M, int current_M, int target_M, bool is_first_resample) {
    // Calculate normalized weights
    // Implements log-sum-exp trick (https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/)
    double weight_sum = 0.0;
    double max_weight = *std::max_element(data.weights.begin(), data.weights.begin() + current_M);
    
    for (int mu = 0; mu < current_M; ++mu) {
        weight_sum += data.weights[mu];
        data.weights[mu] = std::exp(data.weights[mu] - max_weight);  // Numerical stability
    }
    
    for (int mu = 0; mu < current_M; ++mu) {
        data.weights[mu] /= weight_sum; // normalize weights
    }
    
    // Systematic resampling
    data.resample_indices.clear();
    double step = 1.0 / target_M;
    double u = g_uniform_dist(g_rng) * step;
    
    int i = 0;
    double cumsum = data.weights[0];
    
    for (int j = 0; j < target_M; ++j) {
        double target = u + j * step;
        while (cumsum < target && i < current_M - 1) {
            cumsum += data.weights[++i];
        }
        data.resample_indices.push_back(i);
    }
    
    // Resize for target number of trajectories if needed
    if (is_first_resample) {
        data.X1_nu.resize(target_M * M);
        data.X2_nu.resize(target_M * M);
        data.X3_nu.resize(target_M * M);
        data.weights.resize(target_M);
    }
    
    // Store current trajectories temporarily
    data.X1_mu.assign(data.X1_nu.begin(), data.X1_nu.begin() + current_M * M);
    data.X2_mu.assign(data.X2_nu.begin(), data.X2_nu.begin() + current_M * M);
    data.X3_mu.assign(data.X3_nu.begin(), data.X3_nu.begin() + current_M * M);
    
    // Copy resampled trajectories back
    for (int j = 0; j < target_M; ++j) {
        int src_idx = data.resample_indices[j];
        
        for (int k = 0; k < M; ++k) {
            getX1(data.X1_nu, M, j, k) = data.X1_mu[src_idx * M + k];
            getX2(data.X2_nu, M, j, k) = data.X2_mu[src_idx * M + k];
            getX3(data.X3_nu, M, j, k) = data.X3_mu[src_idx * M + k];
        }

        data.weights[j] = 0.0;  // Reset weights after resampling
    }

    return;
}

/**
 * @brief Compute transfer entropy using specified equation
 * @param data
 * @param M
 * @param k
 * @param current_M
 * @return transfer entropy
 */
double compute_transfer_entropy(const TrajectoryData& data, int M, int k, int current_M) {
    // Placeholder: Implement equations 8, 9, 12, or 13 from main text
    // This would depend on whether dealing with diffusion or jump processes
    
    double te_sum = 0.0;
    double weight_sum = 0.0;
    
    for (int mu = 0; mu < current_M; ++mu) {
        // Bounds checking
        if (mu >= static_cast<int>(data.weights.size()) || 
            mu * M + k >= static_cast<int>(data.X1_nu.size())) {
            continue;
        }
        
        double w = std::exp(data.weights[mu]);
        // Example computation - replace with actual transfer entropy formula
        double x1_val = getX1_const(data.X1_nu, M, mu, k);
        double x2_val = getX2_const(data.X2_nu, M, mu, k);
        double local_te = 0.5 * std::log(1.0 + x1_val * x2_val);
        te_sum += w * local_te;
        weight_sum += w;
    }
    
    return (weight_sum > 0.0) ? te_sum / weight_sum : 0.0;
}

/**
 * @brief Compute transfer entropy via path-weighted sampling
 * @param params
 * @return transfer entropy
 */
std::pair<double, double> tepws(const TEPWSParams& params) {
    // Initialize trajectory data
    TrajectoryData data;
    initialize_trajectory_data(data, params);
    
    // (lines 4-6) Initialize trajectory metadata
    // timestep variable and resampling indicator
    int k = 0; // used for accessing P(X_{2,[0,N]} | X_{1,[0,N]}, X_{3,[0,N]})
    int kappa = params.M2;

    // arrays for cumulative transfer entropy
    std::fill(data.Tnu_a.begin(), data.Tnu_a.end(), 0.0);
    std::fill(data.Tnu_b.begin(), data.Tnu_b.end(), 0.0);
    
    /* 
    Main loop (line 7: repeat)
    */
    int current_M = params.M1;
    bool using_M1 = true;
    
    while (k < params.N) {
        if (k == 0) {
            // STEP 1 (lines 8-10): Generate M1 joint trajectories of
            // (X_{1,[0,N]}^{nu}, X_{2,[0,N]}^{nu}, X_{3,[0,N]}^{nu})
            std::vector<double> x1_init, x2_init, x3_init;
            generate_initial_conditions(x1_init, x2_init, x3_init, params.M1);
            
            // Resize trajectory arrays for M1 trajectories
            data.X1_nu.resize(params.M1 * params.I);
            data.X2_nu.resize(params.M1 * params.I);
            data.X3_nu.resize(params.M1 * params.I);
            
            // Generate M2 samples of initial conditions X_2^{mu}(0)
            // from steady-state trajectory; weights in log-scale w^{mu} = 0
            for (int mu = 0; mu < params.M1; ++mu) {
                data.weights[mu] = 0.0;
                getX1(data.X1_nu, params.I, mu, 0) = x1_init[mu];
                getX2(data.X2_nu, params.I, mu, 0) = x2_init[mu];
                getX3(data.X3_nu, params.I, mu, 0) = x3_init[mu];
            }
        }
        
        // (lines 13-21) Check for first resampling 
        // (lines 27-36) else, check for second resampling
        if (using_M1 && k >= params.M1 / 2) {
            resample_trajectories(data, params.I, params.M1, params.M2, true);
            current_M = params.M2;
            using_M1 = false;

        } else if (!using_M1 && k >= params.M2 / 2) {
            resample_trajectories(data, params.I, params.M2, params.M2, false);
        }
        
        // STEPS 2, 3: Propagate dynamics and update weights
        for (int mu = 0; mu < current_M; ++mu) {
            propagate_reference_dynamics(data, params.I, mu, k, k + 1);
            data.weights[mu] += compute_log_probability_ratio(data, params.I, mu, k);
        }
        
        // Compute transfer entropy (lines 20, 35)
        if (using_M1) {
            data.Tnu_a[k] = compute_transfer_entropy(data, params.I, k, current_M);  // or equation 12
        } else {
            data.Tnu_b[k] = compute_transfer_entropy(data, params.I, k, current_M);  // or equation 13
        }
        
        k++;
    }
    
    // Final computation (lines 38-41)
    int count = 0;
    double T_final = 0.0;
    
    // Average Tnu_a and Tnu_b arrays
    for (int i = 0; i <= params.N; ++i) {
        if (data.Tnu_a[i] != 0.0) {
            T_final += data.Tnu_a[i];
            count++;
        }
        if (data.Tnu_b[i] != 0.0) {
            T_final += data.Tnu_b[i];
            count++;
        }
    }
    
    if (count > 0) {
        T_final /= count;
    }
    
    // Normalize by M1 as indicated in line 41
    T_final /= params.M1;
    
    return std::make_pair(T_final, 0.0);  // Return transfer entropy estimate
}

// Utility function to print algorithm statistics
void print_algorithm_stats(const TEPWSParams& params) {
    std::cout << "TE-PWS Algorithm Parameters:" << std::endl;
    std::cout << "\tProcess (continuous if True, else False): " << params.process << std::endl;
    std::cout << "\tI (timesteps): " << params.I << std::endl;
    std::cout << "\tN (total timesteps): " << params.N << std::endl;
    std::cout << "\tM1 (initial trajectories): " << params.M1 << std::endl;
    std::cout << "\tM2 (resampled trajectories): " << params.M2 << std::endl;
}

// Example usage and test
int main() {
    // Initialize random number generator
    initialize_rng();
    
    // Algorithm parameters - using smaller values for debugging
    TEPWSParams params;
    params.dt = 1.0;        // Timestep
    params.N = 300;        // Number of timesteps  
    params.I = params.N / params.dt;
    params.M1 = 50;         // Initial number of trajectories
    params.M2 = 100;        // Resampled number of trajectories
    params.process = false; // If true, assumes a continuous process, otherwise, a discrete process
    
    // Validate parameters
    if (params.dt > params.N) {
        std::cerr << "Error: timestep (" << params.dt << ") should not exceed number of timesteps (" << params.N << ")" << std::endl;
        params.dt = static_cast<double>(params.N) / 100;
        std::cout << "Adjusted timestep dt to " << params.dt << std::endl;
    }
    
    print_algorithm_stats(params);
    
    // Compute transfer entropy
    std::cout << "\nComputing transfer entropy...\n";
    auto start = std::chrono::high_resolution_clock::now();
    
    auto result = tepws(params);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Transfer Entropy T_{X1->X2}: " << result.first << "\n";
    std::cout << "Computation time: " << duration.count() << " ms\n";
    
    return 0;
}
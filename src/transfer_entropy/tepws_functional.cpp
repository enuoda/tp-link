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

// Algorithm parameters structure
struct TEPWSParams {
    int I;        // Number of timesteps
    int N;        // Total number of timesteps
    int M1, M2;   // Number of trajectories for Monte-Carlo averages
};

// Trajectory data structure
struct TrajectoryData {
    std::vector<double> X1_traj, X2_traj, X3_traj;
    std::vector<double> weights;
    std::vector<double> T1_c, T2_c;  // Cumulative transfer entropy arrays
    
    // Temporary storage for resampling
    std::vector<double> temp_X1, temp_X2, temp_X3;
    std::vector<int> resample_indices;
};

// Initialize random number generator
void initialize_rng(unsigned int seed = 0) {
    if (seed == 0) {
        seed = std::chrono::steady_clock::now().time_since_epoch().count();
    }
    g_rng.seed(seed);
}

// Fast access to trajectory data using flat indexing
inline double& getX1(std::vector<double>& X1_traj, int M, int mu, int k) { 
    return X1_traj[mu * M + k]; 
}

inline double& getX2(std::vector<double>& X2_traj, int M, int mu, int k) { 
    return X2_traj[mu * M + k]; 
}

inline double& getX3(std::vector<double>& X3_traj, int M, int mu, int k) { 
    return X3_traj[mu * M + k]; 
}

inline const double& getX1_const(const std::vector<double>& X1_traj, int M, int mu, int k) { 
    return X1_traj[mu * M + k]; 
}

inline const double& getX2_const(const std::vector<double>& X2_traj, int M, int mu, int k) { 
    return X2_traj[mu * M + k]; 
}

inline const double& getX3_const(const std::vector<double>& X3_traj, int M, int mu, int k) { 
    return X3_traj[mu * M + k]; 
}

// Initialize trajectory data structure
void initialize_trajectory_data(TrajectoryData& data, const TEPWSParams& params) {
    // Pre-allocate all vectors for efficiency
    data.X1_traj.reserve(params.M1 * params.I);
    data.X2_traj.reserve(params.M1 * params.I);
    data.X3_traj.reserve(params.M1 * params.I);
    data.weights.resize(params.M1, 0.0);
    
    data.T1_c.resize(params.N + 1, 0.0);
    data.T2_c.resize(params.N + 1, 0.0);
    
    // Temporary storage
    data.temp_X1.reserve(params.M2 * params.I);
    data.temp_X2.reserve(params.M2 * params.I);
    data.temp_X3.reserve(params.M2 * params.I);
    data.resample_indices.reserve(std::max(params.M1, params.M2));
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
}

// Propagate reference dynamics (placeholder for actual dynamics)
void propagate_reference_dynamics(TrajectoryData& data, int M, int mu, int k_start, int k_end) {
    // Placeholder: Implement actual stochastic dynamics here
    // This should propagate X1_{k,k+1}, X2_{k,k+1}, X3_{k,k+1}
    // using the reference dynamics P0(X1_{k,k+1}, X2_{k,k+1} | X1_{0,k}, X2_{0,k})
    
    for (int k = k_start; k < k_end; ++k) {
        // Simple Ornstein-Uhlenbeck process as example
        double dt = 0.01;
        double noise_strength = 0.1;
        
        getX1(data.X1_traj, M, mu, k + 1) = getX1(data.X1_traj, M, mu, k) - 
            0.1 * getX1(data.X1_traj, M, mu, k) * dt + 
            noise_strength * g_normal_dist(g_rng) * std::sqrt(dt);
            
        getX2(data.X2_traj, M, mu, k + 1) = getX2(data.X2_traj, M, mu, k) - 
            0.1 * getX2(data.X2_traj, M, mu, k) * dt + 
            0.05 * getX1(data.X1_traj, M, mu, k) * dt +  // coupling from X1
            noise_strength * g_normal_dist(g_rng) * std::sqrt(dt);
            
        getX3(data.X3_traj, M, mu, k + 1) = getX3(data.X3_traj, M, mu, k) - 
            0.1 * getX3(data.X3_traj, M, mu, k) * dt + 
            noise_strength * g_normal_dist(g_rng) * std::sqrt(dt);
    }
}

// Compute logarithmic probability ratios for weight updates
double compute_log_probability_ratio(const TrajectoryData& data, int M, int mu, int k) {
    // Placeholder: Implement actual probability ratio computation
    // This should compute ln P(X1_{k,k+1}, X2_{k,k+1} | X1_{0,k}, X2_{0,k}, X3_{0,k}) - 
    //                     ln P(X1_{k,k+1}, X2_{k,k+1} | X1_{0,k}, X2_{0,k})
    
    // Example: Simple coupling strength difference
    double coupling_effect = 0.01 * getX1_const(data.X1_traj, M, mu, k) * getX2_const(data.X2_traj, M, mu, k + 1);
    return coupling_effect;
}

// Efficient resampling using systematic resampling
void resample_trajectories(TrajectoryData& data, int M, int current_M, int target_M, bool is_first_resample) {
    // Calculate normalized weights
    double max_weight = *std::max_element(data.weights.begin(), data.weights.begin() + current_M);
    double weight_sum = 0.0;
    
    for (int mu = 0; mu < current_M; ++mu) {
        data.weights[mu] = std::exp(data.weights[mu] - max_weight);  // Numerical stability
        weight_sum += data.weights[mu];
    }
    
    // Normalize weights
    for (int mu = 0; mu < current_M; ++mu) {
        data.weights[mu] /= weight_sum;
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
        data.X1_traj.resize(target_M * M);
        data.X2_traj.resize(target_M * M);
        data.X3_traj.resize(target_M * M);
        data.weights.resize(target_M);
    }
    
    // Store current trajectories temporarily
    data.temp_X1.assign(data.X1_traj.begin(), data.X1_traj.begin() + current_M * M);
    data.temp_X2.assign(data.X2_traj.begin(), data.X2_traj.begin() + current_M * M);
    data.temp_X3.assign(data.X3_traj.begin(), data.X3_traj.begin() + current_M * M);
    
    // Copy resampled trajectories back
    for (int j = 0; j < target_M; ++j) {
        int src_idx = data.resample_indices[j];
        for (int k = 0; k < M; ++k) {
            getX1(data.X1_traj, M, j, k) = data.temp_X1[src_idx * M + k];
            getX2(data.X2_traj, M, j, k) = data.temp_X2[src_idx * M + k];
            getX3(data.X3_traj, M, j, k) = data.temp_X3[src_idx * M + k];
        }
        data.weights[j] = 0.0;  // Reset weights after resampling
    }
}

// Compute transfer entropy using specified equation
double compute_transfer_entropy(const TrajectoryData& data, int M, int k, int current_M) {
    // Placeholder: Implement equations 8, 9, 12, or 13 from main text
    // This would depend on whether dealing with diffusion or jump processes
    
    double te_sum = 0.0;
    double weight_sum = 0.0;
    
    for (int mu = 0; mu < current_M; ++mu) {
        // Bounds checking
        if (mu >= static_cast<int>(data.weights.size()) || 
            mu * M + k >= static_cast<int>(data.X1_traj.size())) {
            continue;
        }
        
        double w = std::exp(data.weights[mu]);
        // Example computation - replace with actual transfer entropy formula
        double x1_val = getX1_const(data.X1_traj, M, mu, k);
        double x2_val = getX2_const(data.X2_traj, M, mu, k);
        double local_te = 0.5 * std::log(1.0 + x1_val * x2_val);
        te_sum += w * local_te;
        weight_sum += w;
    }
    
    return (weight_sum > 0.0) ? te_sum / weight_sum : 0.0;
}

// Main algorithm implementation
std::pair<double, double> compute_transfer_entropy_pws(const TEPWSParams& params) {
    // Initialize trajectory data
    TrajectoryData data;
    initialize_trajectory_data(data, params);
    
    // Initialize (lines 4-6)
    // int nu = 0;
    std::fill(data.T1_c.begin(), data.T1_c.end(), 0.0);
    std::fill(data.T2_c.begin(), data.T2_c.end(), 0.0);
    
    // Main loop (line 7: repeat)
    int k = 0;
    int current_M = params.M1;
    bool using_M1 = true;
    
    while (k < params.N) {
        if (k == 0) {
            // STEP 1: Generate M1 trajectories (lines 8-10)
            std::vector<double> x1_init, x2_init, x3_init;
            generate_initial_conditions(x1_init, x2_init, x3_init, params.M1);
            
            // Resize trajectory arrays for M1 trajectories
            data.X1_traj.resize(params.M1 * params.I);
            data.X2_traj.resize(params.M1 * params.I);
            data.X3_traj.resize(params.M1 * params.I);
            
            for (int mu = 0; mu < params.M1; ++mu) {
                getX1(data.X1_traj, params.I, mu, 0) = x1_init[mu];
                getX2(data.X2_traj, params.I, mu, 0) = x2_init[mu];
                getX3(data.X3_traj, params.I, mu, 0) = x3_init[mu];
                data.weights[mu] = 0.0;
            }
        }
        
        // Check for first resampling (lines 13-21)
        if (using_M1 && k >= params.M1 / 2) {
            resample_trajectories(data, params.I, params.M1, params.M2, true);
            current_M = params.M2;
            using_M1 = false;
        }
        // Check for second resampling (lines 27-36)
        else if (!using_M1 && k >= params.M2 / 2) {
            resample_trajectories(data, params.I, params.M2, params.M2, false);
        }
        
        // STEPS 2, 3: Propagate dynamics and update weights
        for (int mu = 0; mu < current_M; ++mu) {
            propagate_reference_dynamics(data, params.I, mu, k, k + 1);
            data.weights[mu] += compute_log_probability_ratio(data, params.I, mu, k);
        }
        
        // Compute transfer entropy (lines 20, 35)
        if (using_M1) {
            data.T1_c[k] = compute_transfer_entropy(data, params.I, k, current_M);  // or equation 12
        } else {
            data.T2_c[k] = compute_transfer_entropy(data, params.I, k, current_M);  // or equation 13
        }
        
        k++;
    }
    
    // Final computation (lines 38-41)
    double T_final = 0.0;
    int count = 0;
    
    // Average T1_c and T2_c arrays
    for (int i = 0; i <= params.N; ++i) {
        if (data.T1_c[i] != 0.0) {
            T_final += data.T1_c[i];
            count++;
        }
        if (data.T2_c[i] != 0.0) {
            T_final += data.T2_c[i];
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
    params.I = 4500;       // Number of timesteps per trajectory
    params.N = 3000;       // Total number of timesteps  
    params.M1 = 50;       // Initial number of trajectories
    params.M2 = 100;       // Resampled number of trajectories
    
    // Validate parameters
    if (params.N > params.I) {
        std::cerr << "Error: N (" << params.N << ") should not exceed I (" << params.I << ")" << std::endl;
        params.N = params.I - 1;
        std::cout << "Adjusted N to " << params.N << std::endl;
    }
    
    print_algorithm_stats(params);
    
    // Compute transfer entropy
    std::cout << "\nComputing transfer entropy...\n";
    auto start = std::chrono::high_resolution_clock::now();
    
    auto result = compute_transfer_entropy_pws(params);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Transfer Entropy T_{X1->X2}: " << result.first << "\n";
    std::cout << "Computation time: " << duration.count() << " ms\n";
    
    return 0;
}
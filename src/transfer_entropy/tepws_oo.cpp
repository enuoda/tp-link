/*
09/2025

Transfer entropy calculation vis-a-vis object-oriented programming

Compilation:
------------
- Optimized:
    g++ -std=c++11 -O3 -Wall -march=native -flto -DNDEBUG -ffast-math -funroll-loops -o tepws_oo.x tepws_oo.cpp
- Debug:
    g++ -std=c++11 -g -O0 -Wall -Wextra -fsanitize=address -o tepws_oo_debug.x tepws_oo.cpp

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
#include <memory>
#include <chrono>

class TEPathWeightSampling {
private:
    // Algorithm parameters
    int M;        // Number of timesteps
    int N;        // Total number of timesteps
    int M1, M2;   // Number of trajectories for Monte-Carlo averages
    
    // Random number generation
    std::mt19937 rng;
    std::uniform_real_distribution<double> uniform_dist;
    std::normal_distribution<double> normal_dist;
    
    // Trajectory storage - using flat arrays for cache efficiency
    std::vector<double> X1_traj, X2_traj, X3_traj;  // Current trajectories
    std::vector<double> weights;                      // Path weights w^(μ)
    
    // Transfer entropy arrays
    std::vector<double> T1_c, T2_c;  // Cumulative transfer entropy arrays
    
    // Temporary storage for resampling
    std::vector<double> temp_X1, temp_X2, temp_X3, temp_weights;
    std::vector<int> resample_indices;
    
public:
    TEPathWeightSampling(int timesteps, int total_timesteps, int traj1, int traj2, unsigned int seed = 0)
        : M(timesteps), N(total_timesteps), M1(traj1), M2(traj2),
          rng(seed == 0 ? std::chrono::steady_clock::now().time_since_epoch().count() : seed),
          uniform_dist(0.0, 1.0), normal_dist(0.0, 1.0) {
        
        // Pre-allocate all vectors for efficiency
        X1_traj.reserve(M1 * M);
        X2_traj.reserve(M1 * M);
        X3_traj.reserve(M1 * M);
        weights.resize(M1, 0.0);
        
        T1_c.resize(N + 1, 0.0);
        T2_c.resize(N + 1, 0.0);
        
        // Temporary storage
        temp_X1.reserve(M2 * M);
        temp_X2.reserve(M2 * M);
        temp_X3.reserve(M2 * M);
        temp_weights.resize(M2, 0.0);
        resample_indices.reserve(std::max(M1, M2));
    }
    
    // Fast access to trajectory data using flat indexing
    inline double& getX1(int mu, int k) { return X1_traj[mu * M + k]; }
    inline double& getX2(int mu, int k) { return X2_traj[mu * M + k]; }
    inline double& getX3(int mu, int k) { return X3_traj[mu * M + k]; }
    
    inline const double& getX1(int mu, int k) const { return X1_traj[mu * M + k]; }
    inline const double& getX2(int mu, int k) const { return X2_traj[mu * M + k]; }
    inline const double& getX3(int mu, int k) const { return X3_traj[mu * M + k]; }
    
    // Generate initial conditions from steady-state trajectory
    void generateInitialConditions(std::vector<double>& x1_init, std::vector<double>& x2_init, 
                                 std::vector<double>& x3_init, int num_samples) {
        // Placeholder: In practice, this would sample from the steady-state distribution
        // For demonstration, using Gaussian initial conditions
        x1_init.resize(num_samples);
        x2_init.resize(num_samples);
        x3_init.resize(num_samples);
        
        for (int i = 0; i < num_samples; ++i) {
            x1_init[i] = normal_dist(rng);
            x2_init[i] = normal_dist(rng);
            x3_init[i] = normal_dist(rng);
        }
    }
    
    // Propagate reference dynamics (placeholder for actual dynamics)
    void propagateReferenceDynamics(int mu, int k_start, int k_end) {
        // Placeholder: Implement actual stochastic dynamics here
        // This should propagate X1_{k,k+1}, X2_{k,k+1}, X3_{k,k+1}
        // using the reference dynamics P0(X1_{k,k+1}, X2_{k,k+1} | X1_{0,k}, X2_{0,k})
        
        for (int k = k_start; k < k_end; ++k) {
            // Simple Ornstein-Uhlenbeck process as example
            double dt = 0.01;
            double noise_strength = 0.1;
            
            getX1(mu, k + 1) = getX1(mu, k) - 0.1 * getX1(mu, k) * dt + 
                              noise_strength * normal_dist(rng) * std::sqrt(dt);
            getX2(mu, k + 1) = getX2(mu, k) - 0.1 * getX2(mu, k) * dt + 
                              0.05 * getX1(mu, k) * dt +  // coupling from X1
                              noise_strength * normal_dist(rng) * std::sqrt(dt);
            getX3(mu, k + 1) = getX3(mu, k) - 0.1 * getX3(mu, k) * dt + 
                              noise_strength * normal_dist(rng) * std::sqrt(dt);
        }
    }
    
    // Compute logarithmic probability ratios for weight updates
    double computeLogProbabilityRatio(int mu, int k) {
        // Placeholder: Implement actual probability ratio computation
        // This should compute ln P(X1_{k,k+1}, X2_{k,k+1} | X1_{0,k}, X2_{0,k}, X3_{0,k}) - 
        //                     ln P(X1_{k,k+1}, X2_{k,k+1} | X1_{0,k}, X2_{0,k})
        
        // Example: Simple coupling strength difference
        double coupling_effect = 0.01 * getX1(mu, k) * getX2(mu, k + 1);
        return coupling_effect;
    }
    
    // Efficient resampling using systematic resampling
    void resampleTrajectories(int current_M, int target_M, bool is_first_resample) {
        // Calculate normalized weights
        double max_weight = *std::max_element(weights.begin(), weights.begin() + current_M);
        double weight_sum = 0.0;
        
        for (int mu = 0; mu < current_M; ++mu) {
            weights[mu] = std::exp(weights[mu] - max_weight);  // Numerical stability
            weight_sum += weights[mu];
        }
        
        // Normalize weights
        for (int mu = 0; mu < current_M; ++mu) {
            weights[mu] /= weight_sum;
        }
        
        // Systematic resampling
        resample_indices.clear();
        double step = 1.0 / target_M;
        double u = uniform_dist(rng) * step;
        
        int i = 0;
        double cumsum = weights[0];
        
        for (int j = 0; j < target_M; ++j) {
            double target = u + j * step;
            while (cumsum < target && i < current_M - 1) {
                cumsum += weights[++i];
            }
            resample_indices.push_back(i);
        }
        
        // Copy resampled trajectories
        if (is_first_resample) {
            // Resize for M2 trajectories
            X1_traj.resize(M2 * M);
            X2_traj.resize(M2 * M);
            X3_traj.resize(M2 * M);
            weights.resize(M2);
        }
        
        // Store current trajectories temporarily
        temp_X1.assign(X1_traj.begin(), X1_traj.begin() + current_M * M);
        temp_X2.assign(X2_traj.begin(), X2_traj.begin() + current_M * M);
        temp_X3.assign(X3_traj.begin(), X3_traj.begin() + current_M * M);
        
        // Copy resampled trajectories back
        for (int j = 0; j < target_M; ++j) {
            int src_idx = resample_indices[j];
            for (int k = 0; k < M; ++k) {
                getX1(j, k) = temp_X1[src_idx * M + k];
                getX2(j, k) = temp_X2[src_idx * M + k];
                getX3(j, k) = temp_X3[src_idx * M + k];
            }
            weights[j] = 0.0;  // Reset weights after resampling
        }
    }
    
    // Compute transfer entropy using specified equation
    double computeTransferEntropy(int equation_type, int k, int current_M) {
        // Placeholder: Implement equations 8, 9, 12, or 13 from main text
        // This would depend on whether dealing with diffusion or jump processes
        
        double te_sum = 0.0;
        double weight_sum = 0.0;
        
        for (int mu = 0; mu < current_M; ++mu) {
            double w = std::exp(weights[mu]);
            // Example computation - replace with actual transfer entropy formula
            double local_te = 0.5 * std::log(1.0 + getX1(mu, k) * getX2(mu, k));
            te_sum += w * local_te;
            weight_sum += w;
        }
        
        return (weight_sum > 0.0) ? te_sum / weight_sum : 0.0;
    }
    
    // Main algorithm implementation
    std::pair<double, double> computeTransferEntropy() {
        // Initialize (lines 4-6)
        int nu = 0;
        std::fill(T1_c.begin(), T1_c.end(), 0.0);
        std::fill(T2_c.begin(), T2_c.end(), 0.0);
        
        // Main loop (line 7: repeat)
        int k = 0;
        int current_M = M1;
        bool using_M1 = true;
        
        while (k < N) {
            if (k == 0) {
                // Generate M1 trajectories (lines 8-10)
                std::vector<double> x1_init, x2_init, x3_init;
                generateInitialConditions(x1_init, x2_init, x3_init, M1);
                
                for (int mu = 0; mu < M1; ++mu) {
                    getX1(mu, 0) = x1_init[mu];
                    getX2(mu, 0) = x2_init[mu];
                    getX3(mu, 0) = x3_init[mu];
                    weights[mu] = 0.0;
                }
            }
            
            // Check for first resampling (lines 13-21)
            if (using_M1 && k >= M1 / 2) {
                resampleTrajectories(M1, M2, true);
                current_M = M2;
                using_M1 = false;
            }
            // Check for second resampling (lines 27-36)
            else if (!using_M1 && k >= M2 / 2) {
                resampleTrajectories(M2, M2, false);
            }
            
            // Propagate dynamics and update weights
            for (int mu = 0; mu < current_M; ++mu) {
                propagateReferenceDynamics(mu, k, k + 1);
                weights[mu] += computeLogProbabilityRatio(mu, k);
            }
            
            // Compute transfer entropy (lines 20, 35)
            if (using_M1) {
                T1_c[k] = computeTransferEntropy(9, k, current_M);  // or equation 12
            } else {
                T2_c[k] = computeTransferEntropy(8, k, current_M);  // or equation 13
            }
            
            k++;
        }
        
        // Final computation (lines 38-41)
        double T_final = 0.0;
        int count = 0;
        
        // Average T1_c and T2_c arrays
        for (int i = 0; i <= N; ++i) {
            if (T1_c[i] != 0.0) {
                T_final += T1_c[i];
                count++;
            }
            if (T2_c[i] != 0.0) {
                T_final += T2_c[i];
                count++;
            }
        }
        
        if (count > 0) {
            T_final /= count;
        }
        
        // Normalize by M1 as indicated in line 41
        T_final /= M1;
        
        return std::make_pair(T_final, 0.0);  // Return transfer entropy estimate
    }
    
    // Utility function to get algorithm statistics
    void printStats() const {
        std::cout << "TE-PWS Algorithm Parameters:\n";
        std::cout << "M (timesteps): " << M << "\n";
        std::cout << "N (total timesteps): " << N << "\n";
        std::cout << "M1 (initial trajectories): " << M1 << "\n";
        std::cout << "M2 (resampled trajectories): " << M2 << "\n";
    }
};

// Example usage and test
int main() {
    // Algorithm parameters
    int M = 1000;      // Number of timesteps per trajectory
    int N = 2000;      // Total number of timesteps
    int M1 = 100;      // Initial number of trajectories
    int M2 = 200;      // Resampled number of trajectories
    
    // Create TE-PWS instance
    TEPathWeightSampling te_pws(M, N, M1, M2);
    
    // Print algorithm parameters
    te_pws.printStats();
    
    // Compute transfer entropy
    std::cout << "\nComputing transfer entropy...\n";
    auto start = std::chrono::high_resolution_clock::now();
    
    auto result = te_pws.computeTransferEntropy();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Transfer Entropy T_{X1->X2}: " << result.first << "\n";
    std::cout << "Computation time: " << duration.count() << " ms\n";
    
    return 0;
}

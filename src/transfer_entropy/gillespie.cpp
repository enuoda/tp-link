#include <vector>
#include <random>
#include <cmath>
#include <iostream>

class GillespieSimulator {
private:
    std::mt19937 rng;
    std::uniform_real_distribution<double> uniform_dist;
    
    // System parameters
    int num_states;
    std::vector<std::vector<double>> transition_rates; // Q_a(X) matrix
    
public:
    GillespieSimulator(int n_states, unsigned seed = 42) 
        : num_states(n_states), rng(seed), uniform_dist(0.0, 1.0) {
        
        // Initialize transition rate matrix with arbitrary values
        transition_rates.resize(num_states, std::vector<double>(num_states, 0.0));
        
        // Example: 3-state system with arbitrary transition rates
        if (num_states == 3) {
            transition_rates[0][0] = 1.0;  // State 0 -> State 0
            transition_rates[0][1] = 1.5;  // State 0 -> State 1
            transition_rates[0][2] = 2.0;  // State 0 -> State 2
            transition_rates[0][1] = 1.5;  // State 1 -> State 0
            transition_rates[1][1] = 1.0;  // State 1 -> State 1
            transition_rates[1][2] = 0.5;  // State 1 -> State 2
            transition_rates[2][0] = 2.5;  // State 2 -> State 0
            transition_rates[2][1] = 1.0;  // State 2 -> State 1
            transition_rates[2][2] = 2.0;  // State 2 -> State 2
        }
    }
    
    // Calculate escape propensity λ(t') from current state
    double calculate_escape_propensity(int current_state) {
        double lambda = 0.0;
        for (int j = 0; j < num_states; ++j) {
            if (j != current_state) {
                lambda += transition_rates[current_state][j];
            }
        }
        return lambda;
    }
    
    // Sample next jump time using exponential distribution
    double sample_jump_time(double lambda) {
        if (lambda <= 0.0) return std::numeric_limits<double>::infinity();
        
        double u = uniform_dist(rng);
        return -std::log(u) / lambda;
    }
    
    // Select which transition occurs based on propensities
    int select_transition(int current_state, double lambda) {
        double u = uniform_dist(rng) * lambda;
        double cumulative = 0.0;
        
        for (int j = 0; j < num_states; ++j) {
            if (j != current_state) {
                cumulative += transition_rates[current_state][j];
                if (u <= cumulative) {
                    return j;
                }
            }
        }
        return current_state; // Should not reach here if lambda > 0
    }
    
    // Main Gillespie algorithm simulation
    struct TrajectoryPoint {
        double time;
        int state;
    };
    
    std::vector<TrajectoryPoint> simulate(int initial_state, double max_time, int max_steps = 10000) {
        std::vector<TrajectoryPoint> trajectory;
        
        double current_time = 0.0;
        int current_state = initial_state;
        
        trajectory.push_back({current_time, current_state});
        
        for (int step = 0; step < max_steps && current_time < max_time; ++step) {
            // Calculate escape propensity from current state
            double lambda = calculate_escape_propensity(current_state);
            
            if (lambda <= 0.0) {
                // No transitions possible, system is trapped
                break;
            }
            
            // Sample next jump time
            double tau = sample_jump_time(lambda);
            current_time += tau;
            
            if (current_time >= max_time) {
                break;
            }
            
            // Select which transition occurs
            int new_state = select_transition(current_state, lambda);
            current_state = new_state;
            
            // Record the jump
            trajectory.push_back({current_time, current_state});
        }
        
        return trajectory;
    }
    
    // Calculate trajectory probability (log probability for numerical stability)
    double calculate_log_probability(const std::vector<TrajectoryPoint>& trajectory) {
        if (trajectory.size() < 2) return 0.0;
        
        double log_prob = 0.0;
        
        for (size_t i = 0; i < trajectory.size() - 1; ++i) {
            int state_from = trajectory[i].state;
            int state_to = trajectory[i + 1].state;
            double dt = trajectory[i + 1].time - trajectory[i].time;
            
            // Get transition rate
            double rate = transition_rates[state_from][state_to];
            
            // Get escape propensity
            double lambda = calculate_escape_propensity(state_from);
            
            // Add log probability contribution: ln(rate) - lambda * dt
            log_prob += std::log(rate) - lambda * dt;
        }
        
        return log_prob;
    }
    
    // Print trajectory
    void print_trajectory(const std::vector<TrajectoryPoint>& trajectory) {
        std::cout << "Time\tState\n";
        for (const auto& point : trajectory) {
            std::cout << point.time << "\t" << point.state << "\n";
        }
    }
    
    // Set custom transition rates
    void set_transition_rate(int from_state, int to_state, double rate) {
        if (from_state >= 0 && from_state < num_states && 
            to_state >= 0 && to_state < num_states && 
            from_state != to_state) {
            transition_rates[from_state][to_state] = rate;
        }
    }
};

// Example usage function
void run_gillespie_example() {
    // Create a 3-state system
    GillespieSimulator sim(3);
    
    // Run simulation starting from state 0 for time 10.0
    auto trajectory = sim.simulate(0, 10.0);
    
    // Print results
    std::cout << "Gillespie Algorithm Simulation Results:\n";
    std::cout << "======================================\n";
    sim.print_trajectory(trajectory);
    
    // Calculate and print trajectory probability
    double log_prob = sim.calculate_log_probability(trajectory);
    std::cout << "\nLog probability of trajectory: " << log_prob << "\n";
    std::cout << "Probability of trajectory: " << std::exp(log_prob) << "\n";
    std::cout << "\nNumber of jumps: " << trajectory.size() - 1 << "\n";
}

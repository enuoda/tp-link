#ifndef GILLESPIE_SIMULATOR_H
#define GILLESPIE_SIMULATOR_H

#include <vector>
#include <random>

/**
 * @brief Implementation of the Gillespie algorithm for simulating jump processes
 * 
 * This class implements the exact Gillespie algorithm for simulating stochastic
 * jump processes between discrete states. The algorithm samples jump times from
 * exponential distributions and selects transitions based on their propensities.
 */
class GillespieSimulator {
public:
    /**
     * @brief Structure to represent a point in the trajectory
     */
    struct TrajectoryPoint {
        double time;    ///< Time of the state
        int state;      ///< State index
    };

private:
    std::mt19937 rng;                                    ///< Random number generator
    std::uniform_real_distribution<double> uniform_dist; ///< Uniform distribution [0,1)
    int num_states;                                      ///< Number of discrete states
    std::vector<std::vector<double>> transition_rates;   ///< Transition rate matrix Q_a(X)

public:
    /**
     * @brief Constructor for GillespieSimulator
     * @param n_states Number of discrete states in the system
     * @param seed Random seed for reproducibility (default: 42)
     */
    GillespieSimulator(int n_states, unsigned seed = 42);

    /**
     * @brief Calculate the escape propensity λ(t') from the current state
     * @param current_state Current state index
     * @return Total rate of leaving the current state
     */
    double calculate_escape_propensity(int current_state);

    /**
     * @brief Sample the next jump time from exponential distribution
     * @param lambda Escape propensity (rate parameter)
     * @return Time until next jump
     */
    double sample_jump_time(double lambda);

    /**
     * @brief Select which transition occurs based on propensities
     * @param current_state Current state index
     * @param lambda Total escape propensity
     * @return Index of the next state
     */
    int select_transition(int current_state, double lambda);

    /**
     * @brief Run the main Gillespie algorithm simulation
     * @param initial_state Starting state index
     * @param max_time Maximum simulation time
     * @param max_steps Maximum number of jumps (default: 10000)
     * @return Vector of trajectory points (time, state) pairs
     */
    std::vector<TrajectoryPoint> simulate(int initial_state, double max_time, int max_steps = 10000);

    /**
     * @brief Calculate the log probability of a given trajectory
     * @param trajectory Vector of trajectory points
     * @return Log probability of the trajectory
     */
    double calculate_log_probability(const std::vector<TrajectoryPoint>& trajectory);

    /**
     * @brief Print trajectory to console
     * @param trajectory Vector of trajectory points to print
     */
    void print_trajectory(const std::vector<TrajectoryPoint>& trajectory);

    /**
     * @brief Set a specific transition rate in the rate matrix
     * @param from_state Source state index
     * @param to_state Target state index  
     * @param rate Transition rate value
     */
    void set_transition_rate(int from_state, int to_state, double rate);
};

/**
 * @brief Example function demonstrating usage of the Gillespie algorithm
 */
void run_gillespie_example();

#endif // GILLESPIE_SIMULATOR_H
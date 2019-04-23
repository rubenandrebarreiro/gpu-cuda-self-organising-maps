/**
 *
 * GPU's Self-Organising Maps (S.O.M.)
 *
 * High Performance Computing
 * Work Assignment/Practical Lab's Project #1
 *
 * Description/Steps of operations performed by the GPU CUDA's Kernels:
 * - Implementation of Self-Organising Maps (S.O.M.), using the GPU's parallelism and optimisations
 *   with the support of the CUDA (Compute Unified Device Architecture) API's extensions;
 * - This implementation was made in C++ (C Plus Plus);
 * - This project aims to compare the performances between,
 *   a version with CPU-based operations made by a sequential way,
 *   against a version with GPU-based operations and kernels (supported by CUDA),
 *   taking advantage of GPU's parallelism and optimisations;
 *
 * Authors:
 * - Hervé Miguel Paulino - herve.paulino@fct.unl.pt
 *
 * Modified by:
 * - Ruben Andre Barreiro - r.barreiro@campus.fct.unl.pt
 *
 */

#include "../include/cpu_self_organising_maps_sequential.hpp"
#include <cmath>
#include <cstdlib>
#include <functional>

using namespace std;


namespace gpu_self_organising_maps {

    template<typename T = float>
    // The class of CPU's SOM Sequential
    class cpu_som_sequential : public cpu_self_organising_maps_sequential<T> {

        // Using the class CPU's Self-Organising Maps Sequential as base
        using Base = cpu_self_organising_maps_sequential<T>;

        // The type of the observations
        using observation_type  = typename Base::observation_type;

        // The value's type of all components of an observation
        using value_type  = typename Base::value_type;

        // The size of Distances and Neighborhood matrix
        const unsigned size;

        // The Distance Function which will be bind to be used during the process
        const std::function<float(observation_type&, int)> distance_function;


        // Constructor of CPU's SOM Sequential
        public:
            cpu_som_sequential(const unsigned ncols, const unsigned nrows, string &&input_file, string &&output_file, string&& distance_func) :
                            Base(ncols, nrows, std::move(input_file), std::move(output_file)),
                            size(nrows * ncols),
                            distance_function (distance_func == "euclidean" ?
                                   bind(&cpu_som_sequential::euclidean, this, placeholders::_1, placeholders::_2) :
                                   bind(&cpu_som_sequential::cosine, this, placeholders::_1, placeholders::_2))
            { }

        private:

            /**
             * Returns the index of a 2D matrix to be used in a vector/array.
             *
             * @param row the row in the 2D matrix
             * @param col the column in the 2D matrix
             *
             * @return the index of a 2D matrix to be used in a vector/array
             */
            inline unsigned index_of(unsigned row, unsigned col) const {
                return (row * this->number_cols) + col;
            }

           /**
            * Returns the value of the Euclidean 2D Distance Function, between a given observation and
            * an observation's value in the map.
            *
            * @param obs an observation
            * @param index an index of an observation's value in the map
            *
            * @return the value of the Euclidean 2D Distance Function, between a given observation and
            *         an observation's value in the map
            */
            float euclidean(observation_type &obs, int index) const {
                float result = 0;

                for (unsigned i = 0; i < this->number_features; i++) {
                    auto val = (obs[i] - this->map[index+i]);
                    result += val * val;
                }

                return sqrt(result);
            }

           /**
            * Returns the value of the Cosine Distance Function, between a given observation and
            * an observation's value in the map.
            *
            * @param obs an observation
            * @param index an index of an observation's value in the map
            *
            * @return the value of the Cosine Distance Function, between a given observation and
            *         an observation's value in the map
            */
            float cosine(observation_type &obs, int index) const {
                float num = 0;
                float dem1 = 0;
                float dem2 = 0;

                for (unsigned i = 0; i < this->number_features; i++) {
                    num += obs[i] * this->map[index+i];

                    dem1 += (obs[i] * obs[i]);
                    dem2 += (this->map[index+i] * this->map[index+i]);
                }

                return 1 - ( num / (sqrt(dem1) * sqrt(dem2)));
            }

           /**
            * Returns the index of the BMU (Best Matching Unit),
            * which is the closest vector/array (with the minimum distance) to the
            * current observation that's being analysed.
            *
            * @param distances the distances' map
            *
            * @return the index of the BMU (Best Matching Unit),
            *         which is the closest vector/array (with the minimum distance) to the
            *         current observation that's being analysed
            */
            unsigned arg_min(vector<float> &distances) const {
                unsigned result = 0;
                auto value = distances[0];

                for (unsigned i = 0; i < this->number_rows; i++) {
                    for (unsigned j = 0; j < this->number_cols; j++) {
                        const auto index = index_of(i, j);

                        if (distances[index] < value) {
                            value = distances[index];
                            result = index;
                        }
                    }
                }

                return result;
            }

           /**
            * Returns the Neighborhood's matrix for a given BMU (Best Matching Unit)
            * and for the both, current iteration and current point.
            *
            * @param bmu a given BMU (Best Matching Unit)
            * @param current_point the current point
            * @param num_points the total number of points
            *
            * @return the Neighborhood's matrix for a given BMU (Best Matching Unit)
            *         and for the both, current iteration and current point
            */
            float neighborhood_function(unsigned bmu, unsigned current_point, unsigned num_points) {
                const float theta = (this->max_distance / 2.0f) - ((this->max_distance / 2.0f) * (this->iteration /
                                                                                              static_cast<float>(num_points)));
                const auto sqrDist = (bmu - current_point) * (bmu - current_point);

                return exp( -(sqrDist / (theta * theta)));
            }

           /**
            * Updates the map for a given observation and BMU (Best Matching Unit),
            * with the new data based in the values of the matrices, of both, distances and neighborhood.
            *
            * @param obs an observation value to be analysed
            * @param bmu a given BMU (Best Matching Unit)
            * @param num_points the total number of points
            */
            void update_map(observation_type &obs, unsigned bmu, unsigned num_points) {
                float learning_rate = 1 / static_cast<float>(this->iteration);

                #ifdef DEBUG
                    hpcProjectLog("learning_rate " << learning_rate)
                #endif

                observation_type neighborhood(this->number_cols * this->number_rows);
                for (unsigned i = 0; i < this->number_rows; i++)
                    for (unsigned j = 0; j < this->number_cols; j++) {
                        const auto index = index_of(i, j);

                        neighborhood[index] = neighborhood_function(bmu, index, num_points);
                    }

                #ifdef DEBUG
                    hpcProjectLog("neighborhood " << neighborhood)
                #endif

                for (unsigned i = 0; i < this->number_rows; i++)
                    for (unsigned j = 0; j < this->number_cols; j+=this->number_features) {
                        const auto index = index_of(i, j);

                        for (unsigned f = 0; f < this->number_features; f++) {
                            if (neighborhood[index] > 0.01)
                                this->map[index + f] = this->map[index + f] +
                                                      (learning_rate * neighborhood[index] * (obs[f] - this->map[index + f]));
                        }
                    }
            }

           /**
            * Processes all the information about an observation,
            * accordingly to the SOM (Self-Organising Map) Algorithm
            *
            * @param obs a given observation
            */
            void process_observation(observation_type &obs) {
                this->iteration++;

                #ifdef DEBUG
                    hpcProjectLog("obs " << obs);
                    hpcProjectLog("map " << this->map);
                #endif
                observation_type distances(size);

                for(unsigned i = 0; i < this->number_rows; i++)
                    for(unsigned j = 0; j < this->number_cols; j++) {
                        const auto index = index_of(i, j);

                        distances[index] = distance_function(obs, index);
                    }

                auto bmu = arg_min(distances);

                #ifdef DEBUG
                    hpcProjectLog("distances " << distances);
                    hpcProjectLog("bmu " << bmu);
                #endif

                update_map(obs, bmu, this->dr.get_number_observations());
            }
    };
}

/**
 * The main method to read the arguments from the console and initialises a SOM (Self-Organising Map) Algorithm.
 *
 * @param argc the total number of the arguments read
 * @param argv the values of the arguments read
 *
 * @return 0 or 1
 */
int main(int argc, char* argv[]) {

    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " number_rows number_columns datafile outputfile [distance]\n";

        return 1;
    }

    try {
        if (argc == 5) {
            gpu_self_organising_maps::cpu_som_sequential<float> s(stoi(argv[1]), stoi(argv[2]), string(argv[3]), string(argv[4]), "euclidean");

            s.run();
        }
        else {
            gpu_self_organising_maps::cpu_som_sequential<float> s(stoi(argv[1]), stoi(argv[2]), string(argv[3]), string(argv[4]), string(argv[5]));

            s.run();
        }

    }
    catch (std::runtime_error& e) {
        std::cerr << e.what() << "\n";

        return 1;
    }

    return 0;
}
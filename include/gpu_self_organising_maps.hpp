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

#ifndef GPU_SELF_ORGANISING_MAPS_HPP
#define GPU_SELF_ORGANISING_MAPS_HPP

#include <cmath>
#include <iostream>

#include "hpc_project_1.hpp"
#include "data_reader.hpp"
#include "timer.hpp"

using namespace std;


namespace gpu_self_organising_maps {

    template <typename T = float>
    // The class of GPU's Self-Organising Maps:
    class gpu_self_organising_maps {

        public:
            // The type of the observations
            using observation_type = typename data_reader<T>::observation_type;

            // The value's type of all components of an observation
            using value_type = typename data_reader<T>::value_type;

        protected:

            // The data reader to read the inputs (observations)
            data_reader<T> dr;

            // The number of columns of the map
            const unsigned number_cols;

            // The number of rows of the map
            const unsigned number_rows;

            // The number of features of each observation (position/index) of the map
            const unsigned number_features;

            // The map of observations
            vector<T> map;

            // The maximum distance that can occur in the map of observations
            const float max_distance;

            // The current iteration
            unsigned iteration;

            // The name of the output's file
            const string output_file;

        public:

            // The Constructor of GPU's Self-Organising Maps:
            gpu_self_organising_maps(const unsigned num_cols, const unsigned num_rows,
                                     string&& input_file, string&& output_file,
                                     string&& distance_fun, unsigned seed = 0) :

                dr(input_file),
                number_cols(num_cols),
                number_rows(num_rows),
                number_features(dr.get_number_features()),
                map (number_cols * number_rows * number_features),
                max_distance(sqrt( (num_rows * num_rows) + (num_cols * num_cols) )),
                iteration(0),
                output_file (output_file) {

                    const auto size = ( number_cols * number_rows * number_features );

                    srand(seed);

                    for (unsigned i = 0; i < size; i++)
                        map[i] = static_cast<float> (random()) / static_cast<float> (RAND_MAX);

                }

           /**
            * The runnable method.
            */
            void run() {

                timer<> t;
                t.start();

                while (true) {
                    observation_type obs;
                    dr >> obs;

                    if (dr.eof())
                        break;

                    iteration++;
                    process_observation(obs);
                }

                t.stop();

                t.print_stats(cout);
                cout << " milliseconds\n ";

                write_output();
            }

        protected:

           /**
            * Processes all the information about an observation,
            * accordingly to the SOM (Self-Organising Map) Algorithm
            *
            * @param obs a given observation
            */
            virtual void process_observation(observation_type& obs) {
                hpcProjectLog(obs);
            }

           /**
            * Write the output to a specified file.
            */
            void write_output() {
                ofstream os (output_file);

                for (unsigned i = 0; i < number_rows * number_cols * number_features; i++)
                    os << map[i] << "\n";
            }
    };
}

#endif // GPU_SELF_ORGANISING_MAPS_HPP
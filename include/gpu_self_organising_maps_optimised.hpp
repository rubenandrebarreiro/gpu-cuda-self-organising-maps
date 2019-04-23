#ifndef GPU_SELF_ORGANISING_MAPS_OPTIMISED_HPP
#define GPU_SELF_ORGANISING_MAPS_OPTIMISED_HPP

#include <cmath>
#include <iostream>

#include "hpc_project_1.hpp"
#include "data_reader.hpp"
#include "timer.hpp"

using namespace std;


namespace gpu_self_organising_maps {

    template <typename T = float>
    // The class of GPU's Self-Organising Maps (Optimised):
    class gpu_self_organising_maps_optimised {

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

            // The number of observations processed until the moment
            unsigned num_observations_processed;

            // The name of the output's file
            const string output_file;

        public:

            // The Constructor of GPU's Self-Organising Maps (Optimised):
            gpu_self_organising_maps_optimised(const unsigned num_cols, const unsigned num_rows,
                                               string&& input_file, string&& output_file,
                                               string&& distance_fun, unsigned seed = 0) :

                dr(input_file),
                number_cols(num_cols),
                number_rows(num_rows),
                number_features(dr.get_number_features()),
                map (number_cols * number_rows * number_features),
                max_distance( sqrt( (num_rows * num_rows) + (num_cols * num_cols) ) ),
                num_observations_processed(0),
                iteration(0),
                output_file (output_file) {

                    const auto size = number_cols * number_rows * number_features;

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

                    // The number of observations remaining to be processed,
                    // until the algorithm finish its execution
                    int num_observations_remaining = ( dr.get_number_observations() - num_observations_processed );

                    // If it's possible process four or more observations in parallel
                    if(num_observations_remaining >= 4) {

                        vector<observation_type> observations(4);

                        // Read four consecutive observations, to be processed in parallel
                        dr >> observations[0];
                        dr >> observations[1];
                        dr >> observations[2];
                        dr >> observations[3];

                        if (dr.eof())
                            break;

                        // The iteration counter increments one unit,
                        // since to the first observation, can be processed
                        iteration++;
                        process_observation(observations);

                        // More four observations was processed
                        num_observations_processed += 4;

                        // The iteration counter increments more three units,
                        // since other three observations was processed, in parallel,
                        // together with the first
                        iteration += 3;
                    }

                    // If it's only possible process three observations in parallel
                    else if(num_observations_remaining == 3) {

                        vector<observation_type> observations(3);

                        // Read three consecutive observations, to be processed in parallel
                        dr >> observations[0];
                        dr >> observations[1];
                        dr >> observations[2];

                        if (dr.eof())
                            break;

                        // The iteration counter increments one unit,
                        // since to the first observation, can be processed
                        iteration++;
                        process_observation(observations);

                        // More three observations was processed
                        num_observations_processed += 3;

                        // The iteration counter increments more two units,
                        // since other three observations was processed, in parallel,
                        // together with the first
                        iteration += 2;
                    }

                    // If it's only possible process two observations in parallel
                    else if(num_observations_remaining == 2) {

                        vector<observation_type> observations(2);

                        // Read two consecutive observations, to be processed in parallel
                        dr >> observations[0];
                        dr >> observations[1];

                        if (dr.eof())
                            break;

                        // The iteration counter increments one unit,
                        // since to the first observation, can be processed
                        iteration++;
                        process_observation(observations);

                        // More two observations was processed
                        num_observations_processed += 2;

                        // The iteration counter increments more one unit,
                        // since other two observations was processed, in parallel,
                        // together with the first
                        iteration++;
                    }

                    // If it's only possible process one observation in parallel
                    else {
                        vector<observation_type> observations(1);

                        // Read one consecutive observation, to be processed in parallel
                        dr >> observations[0];

                        if (dr.eof())
                            break;

                        // The iteration counter increments one unit,
                        // since to the first observation, can be processed
                        iteration++;
                        process_observation(observations);

                        // More one observations was processed
                        num_observations_processed++;
                    }
                }

                t.stop();

                t.print_stats(cout);
                cout << " milliseconds\n ";

                write_output();
            }

        protected:

           /**
            * Processes all the information about a vector/array of observations,
            * accordingly to the SOM (Self-Organising Map) Algorithm
            *
            * @param obs a given vector/array of observations
            */
            virtual void process_observation(vector<observation_type>& obs) {
                for(int i = 0; i < obs.size(); i++) {
                    hpcProjectLog(obs[i]);
                }
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
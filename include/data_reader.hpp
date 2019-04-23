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

#ifndef GPU_SELF_ORGANISING_MAPS_DATA_READER_HPP
#define GPU_SELF_ORGANISING_MAPS_DATA_READER_HPP

#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>

using namespace std;

namespace gpu_self_organising_maps {


    template <typename T>
    class data_reader {

        ifstream is;

        unsigned number_observations;

        unsigned number_features;

        public:

            using value_type = T;

            using observation_type = vector<T>;

            explicit data_reader(string& file_name) :
                    number_observations (0),
                    number_features (0){

                is.exceptions ( std::ifstream::failbit );
                is.open(file_name);

                // First read to obtain characteristics
                while (!eof()) {

                    string line;

                    try {
                        line = next_line();

                        if (eof()) // Due to comment lines
                            break;
                    }
                    catch (std::ifstream::failure& e) {
                        break;
                    }

                    istringstream iss(line);
                    string token;
                    unsigned n = 0;

                    while (getline(iss, token, ','))
                        n++;

                    if (number_features == 0)
                        number_features = n;
                    else if (number_features != n)
                        throw std::runtime_error("Error parsing file!!!");

                    number_observations++;
                }

                // Resetting for the iterative reading
                is.clear();
                is.seekg(0);
            }


            ~data_reader() {
                is.close();
            }

            unsigned get_number_observations() const {
                return number_observations;
            }

            unsigned get_number_features() const {
                return number_features;
            }

            bool eof() const {
                return is.eof();
            }

            data_reader& operator>>(observation_type& obs) {

                if (!eof()) {
                    try {
                        istringstream iss(next_line());

                        if (eof()) // Due to comment lines
                            return *this;

                        string token;

                        while (getline(iss, token, ','))
                            obs.push_back(stod(token));
                    }
                    catch (std::ifstream::failure& e) {   }
                }

                return *this;
            }

        private:
            string next_line() {

                string line;
                do {
                    if (is.eof())
                        return "";

                    getline(is, line);
                }
                while (line.empty() || line[0] == '%' || line[0] == '#');

                return line;
            }
    };
}

#endif // GPU_SELF_ORGANISING_MAPS_DATA_READER_HPP
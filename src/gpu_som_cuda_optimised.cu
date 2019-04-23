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
 * - Ruben Andre Barreiro - r.barreiro@campus.fct.unl.pt
 *
 */


#include "../include/gpu_self_organising_maps_CUDA_optimised.hpp"

#include <exception>

using namespace std;


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
            gpu_self_organising_maps::gpu_self_organising_maps_optimised<float> s (stoi(argv[1]), stoi(argv[2]), string(argv[3]), string(argv[4]), "euclidean");

            s.run();
        }
        else {
            gpu_self_organising_maps::gpu_self_organising_maps_optimised<float> s (stoi(argv[1]), stoi(argv[2]), string(argv[3]), string(argv[4]), string(argv[5]));

            s.run();
        }
    }
    catch (runtime_error& e) {
        cerr << "Error processing file " << argv[3] << ": " << e.what() << "\n";

        return 1;
    }

    return 0;
}
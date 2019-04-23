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

#ifndef GPU_SELF_ORGANISING_MAPS_CUDA_HPP
#define GPU_SELF_ORGANISING_MAPS_CUDA_HPP
#endif

#include "gpu_self_organising_maps.hpp"
#include "hpc_project_1.hpp"

using namespace std;

namespace gpu_self_organising_maps {

    /**
     * Returns the matrix of distances on the memory of the GPU,
     * accordingly to the Euclidean 2D Distance Function.
     *
     * A GPU's kernel method to calculate the Euclidean Distance Function,
     * between a map's observation and a specific given observation.
     *
     * @tparam T the template of the observation's type
     *
     * @param map the general map of observations
     * @param obs the current observation
     * @param number_features the number of features of each observation
     * @param map_size the size of the general map
     * @param distances_map the matrix of distances where will be put the distances
     *
     * @return the matrix of distances on the memory of the GPU,
     *         accordingly to the Euclidean Distance Function
     */
    template <typename T>

    __global__ void distance_euclidean(const T* map, const T* obs, const int number_features, const int map_size, float* distances_map) {

        const int index = (blockIdx.x * blockDim.x) + threadIdx.x;

        if(index < map_size) {
            const int index_in_map = index * number_features;

            float total_absolute_two_power = 0.0f;

            for(int f = 0; f < number_features; f++) {
                total_absolute_two_power += pow( ( map[ (index_in_map + f) ] - obs[f] ) , 2.0f);
            }

            distances_map[index] = sqrt(total_absolute_two_power);
        }
    }


    /**
     * Returns the matrix of distances on the memory of the GPU,
     * accordingly to the Cosine Distance Function.
     *
     * A GPU's kernel method to calculate the Cosine Distance Function,
     * between a map's observation and a specific given observation.
     *
     * @tparam T the template of the typename T (the type of all components of each observation)
     *
     * @param map the general map
     * @param obs the current observation
     * @param number_features the number of features of each observation
     * @param map_size the size of the general map
     * @param distances_map the matrix of distances where will be put the distances
     *
     * @return the matrix of distances on the memory of the GPU,
     *         accordingly to the Cosine Distance Function
     */
    template <typename T>

    __global__ void distance_cosine(const T* map, const T* obs, const int number_features, const int map_size, float* distances_map) {

        const int index = (blockIdx.x * blockDim.x) + threadIdx.x;

        if(index < map_size) {
            const int index_in_map = index * number_features;

            float map_two_power_sum = 0.0f;
            float obs_two_power_sum = 0.0f;

            float vectors_cross_product_sum = 0.0f;

            for(int f = 0; f < number_features; f++) {
                map_two_power_sum += pow(map[index_in_map + f], 2.0f);
                obs_two_power_sum += pow(obs[f], 2.0f);

                vectors_cross_product_sum += (map[index_in_map + f] * obs[f]);
            }

            float sqrt_map_two_power_sum = sqrt(map_two_power_sum);
            float sqrt_obs_two_power_sum = sqrt(obs_two_power_sum);

            distances_map[index] = ( 1.0f - ( vectors_cross_product_sum /
                                            ( sqrt_map_two_power_sum * sqrt_obs_two_power_sum ) ) );
        }
    }


    /**
     * Returns the matrix of distances on the memory of the GPU,
     * accordingly to the Manhattan Distance Function.
     *
     * A GPU's kernel method to calculate the Manhattan Distance Function,
     * between a map's observation and a specific given observation.
     *
     * @tparam T the template of the typename T (the type of all components of each observation)
     *
     * @param map the general map of observations
     * @param obs the current observation
     * @param number_features the number of features of each observation
     * @param map_size the size of the general map
     * @param distances_map the matrix of distances where will be put the distances
     *
     * @return the matrix of distances on the memory of the GPU,
     *         accordingly to the Manhattan Distance Function
     */
    template <typename T>

    __global__ void distance_manhattan(const T* map, const T* obs, const int number_features, const int map_size, float* distances_map) {
        const int index = (blockIdx.x * blockDim.x) + threadIdx.x;

        if(index < map_size) {
            const int index_in_map = index * number_features;

            float total_absolute = 0.0f;

            for(int f = 0; f < number_features; f++) {
                total_absolute += abs( (map[index_in_map + f] - obs[f]) );
            }

            distances_map[index] = total_absolute;
        }
    }


    /**
     * Returns the matrix of distances on the memory of the GPU,
     * accordingly to the Minkowski Distance Function.
     *
     * A GPU's kernel method to calculate the Minkowski Distance Function,
     * between a map's observation and a specific given observation.
     *
     * @tparam T the template of the typename T (the type of all components of each observation)
     *
     * @param map the general map
     * @param obs the current observation
     * @param number_features the number of features of each observation
     * @param map_size the size of the general map
     * @param distances_map the matrix of distances where will be put the distances
     *
     * @return the matrix of distances on the memory of the GPU,
     *         accordingly to the Minkowski Distance Function
     */
    template <typename T>

    __global__ void distance_minkowski(const T* map, const T* obs, const int number_features, const int map_size, float* distances_map, float p = 2.0f) {
        const int index = (blockIdx.x * blockDim.x) + threadIdx.x;

        if(index < map_size) {
            const int index_in_map = index * number_features;

            float total_absolute_two_power = 0.0f;

            for(int f = 0; f < number_features; f++) {
                total_absolute_two_power += pow( abs( (map[index_in_map + f] - obs[f]) ), p);
            }

            distances_map[index] = pow( total_absolute_two_power, ( 1.0f / p ) );
        }
    }


    /**
     * Returns the matrix of distances on the memory of the GPU,
     * accordingly to the Chebyshev Distance Function.
     *
     * A GPU's kernel method to calculate the Chebyshev Distance Function,
     * between a map's observation and a specific given observation.
     *
     * @tparam T the template of the typename T (the type of all components of each observation)
     *
     * @param map the general map of observations
     * @param obs the current observation
     * @param number_features the number of features of each observation
     * @param map_size the size of the general map
     * @param distances_map the matrix of distances where will be put the distances
     *
     * @return the matrix of distances on the memory of the GPU,
     *         accordingly to the Chebyshev Distance Function
     */
    template <typename T>

    __global__ void distance_chebyshev(const T* map, const T* obs, const int number_features, const int map_size, float* distances_map) {

        const int index = (blockIdx.x * blockDim.x) + threadIdx.x;

        if(index < map_size) {
            const int index_in_map = index * number_features;

            float max_chebyshev = 0.0f;

            for(int f = 0; f < number_features; f++) {
                max_chebyshev = ( max_chebyshev > ( map[index_in_map + f] - obs[f] ) ) ?
                                  max_chebyshev : ( map[index_in_map + f] - obs[f] );
            }

            distances_map[index] = max_chebyshev;
        }
    }


    /**
     * Returns the arrays/vectors for the values and indexes of
     * the BMUs (Best Matching Units) for each block.
     *
     * A GPU's kernel method to calculate the both, values and indexes of
     * the BMUs (Best Matching Units) for each block.
     *
     * @tparam blockSize the template for the block's size
     *
     * @param distances_map the matrix of distances
     * @param distances_map_size the size of the matrix of distances
     * @param bmu_values the array/vector where the values of
     *        the BMU (Best Matching Unit) for each block will be put
     * @param bmu_indexes the array/vector of the indexes of
     *        the BMU (Best Matching Unit) for each block will be put
     *
     * @return the arrays/vectors for the values and indexes of
     *         the BMUs (Best Matching Units) for each block
     */
    template <unsigned int blockSize>

    __global__ void arg_min(const float* distances_map, int distances_map_size, float* bmu_values, int* bmu_indexes) {

        extern __shared__ float sdata_values[];
        extern __shared__ int sdata_indexes[];

        unsigned int tid = threadIdx.x;
        unsigned int i = (blockSize * 2) * blockIdx.x + tid;

        unsigned int gridSize = (blockSize * 2) * gridDim.x;

        sdata_values[tid] = 0.0f;
        sdata_indexes[tid] = -1;

        while (i < distances_map_size) {
            sdata_indexes[tid] = (distances_map[i] < distances_map[i + blockSize]) ? i : (i + blockSize);
            sdata_values[tid] = (distances_map[i] < distances_map[i + blockSize]) ? distances_map[i] : distances_map[(i + blockSize)];

            i += gridSize;
        }

        __syncthreads();

        if (blockSize >= 512) {
            if (tid < 256) {
                sdata_indexes[tid] = (sdata_values[tid] < sdata_values[tid + 256]) ? tid : (tid + 256);
                sdata_values[tid] = (sdata_values[tid] < sdata_values[tid + 256]) ? sdata_values[tid] : sdata_values[tid + 256];
            }

            __syncthreads();
        }

        if (blockSize >= 256) {
            if (tid < 128) {
                sdata_indexes[tid] = (sdata_values[tid] < sdata_values[tid + 128]) ? tid : (tid + 128);
                sdata_values[tid] = (sdata_values[tid] < sdata_values[tid + 128]) ? sdata_values[tid] : sdata_values[tid + 128];
            }

            __syncthreads();
        }

        if (blockSize >= 128) {
            if (tid < 64) {
                sdata_indexes[tid] = (sdata_values[tid] < sdata_values[tid + 64]) ? tid : (tid + 64);
                sdata_values[tid] = (sdata_values[tid] < sdata_values[tid + 64]) ? sdata_values[tid] : sdata_values[tid + 64];
            }

            __syncthreads();
        }

        if (tid < 32) {

            if (blockSize >= 64) {
                sdata_indexes[tid] = (sdata_values[tid] < sdata_values[tid + 32]) ? tid : (tid + 32);
                sdata_values[tid] = (sdata_values[tid] < sdata_values[tid + 32]) ? sdata_values[tid] : sdata_values[tid + 32];
            }

            if (blockSize >= 32) {
                sdata_indexes[tid] = (sdata_values[tid] < sdata_values[tid + 16]) ? tid : (tid + 16);
                sdata_values[tid] = (sdata_values[tid] < sdata_values[tid + 16]) ? sdata_values[tid] : sdata_values[tid + 16];
            }

            if (blockSize >= 16) {
                sdata_indexes[tid] = (sdata_values[tid] < sdata_values[tid + 8]) ? tid : (tid + 8);
                sdata_values[tid] = (sdata_values[tid] < sdata_values[tid + 8]) ? sdata_values[tid] : sdata_values[tid + 8];
            }

            if (blockSize >= 8) {
                sdata_indexes[tid] = (sdata_values[tid] < sdata_values[tid + 4]) ? tid : (tid + 4);
                sdata_values[tid] = (sdata_values[tid] < sdata_values[tid + 4]) ? sdata_values[tid] : sdata_values[tid + 4];
            }

            if (blockSize >= 4) {
                sdata_indexes[tid] = (sdata_values[tid] < sdata_values[tid + 2]) ? tid : (tid + 2);
                sdata_values[tid] = (sdata_values[tid] < sdata_values[tid + 2]) ? sdata_values[tid] : sdata_values[tid + 2];
            }

            if (blockSize >= 2) {
                sdata_indexes[tid] = (sdata_values[tid] < sdata_values[tid + 1]) ? tid : (tid + 1);
                sdata_values[tid] = (sdata_values[tid] < sdata_values[tid + 1]) ? sdata_values[tid] : sdata_values[tid + 1];
            }
        }

        if (tid == 0) {
            bmu_indexes[blockIdx.x] = sdata_indexes[0];
            bmu_values[blockIdx.x] = sdata_values[0];
        }
    }


    /**
     * Returns the neighborhood's matrix for a given BMU (Best Matching Unit)
     * and for the both, current iteration and current point.
     *
     * A GPU's kernel method to calculate the Neighborhood's matrix,
     * for a given BMU (Best Matching Unit) and for the both, current iteration and current point.
     *
     * @tparam T the template of the typename T (the type of all components of each observation)
     *
     * @param iteration the current iteration of the algorithm/process
     * @param number_features the number of components of each observation
     * @param bmu the previously calculated BMU (Best Matching Unit)
     * @param num_inputs the number of inputs (observations)
     * @param distances_map_size the size of the matrix of distances
     * @param max_distance the maximum distance that can occur in the map
     * @param neighborhood the neighborhood's matrix where will be put the neighborhood's values
     *
     * @return the neighborhood's matrix the memory of the GPU,
     *         accordingly to the previously described procedure
     */
    template <typename T>

    __global__ void neighborhood_function(const int iteration, const int number_features, const int bmu, const int num_inputs,
                                          const int distances_map_size, const float max_distance, T* neighborhood) {

        const unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;

        if(index < distances_map_size) {

            float theta = ( ( max_distance / 2.0f) - ( ( max_distance / 2.0f ) * ( iteration / num_inputs ) ) );

            float sqrDist = pow( abs( bmu - index ), 2.0f );

            float n = exp( -1.0f * ( sqrDist / pow(theta, 2.0f) ) );

            neighborhood[index] = ( n > 0.01f ) ? n : 0.0f;
        }
    }


    /**
     * Returns the updated general map of observations, on the memory of the GPU,
     * for a given observation and neighborhood's matrix, as also, the current iteration,
     * related to the learning rate formula.
     *
     * A GPU's kernel method to update the general map of observations,
     * for a given observation and neighborhood's matrix, as also, the current iteration,
     * related to the learning rate formula.
     *
     * @tparam T the template of the typename T (the type of all components of each observation)
     *
     * @param iteration the current iteration of the algorithm/process
     * @param number_features the number of components of each observation
     * @param obs the current observation, that it's being analysed
     * @param map_size the size of the general map of observations
     * @param neighborhood the neighborhood's matrix
     * @param map the general map of observations, which will be updated
     *
     * @return the updated general map of observations, on the memory of the GPU,
     *         accordingly to the previously described procedure
     */
    template <typename T>

    __global__ void update_map(const int iteration, const int number_features, const T* obs, const int map_size, const T* neighborhood, T* map) {
        const int index = (blockIdx.x * blockDim.x) + threadIdx.x;

        float learn_rate = ( 1 / static_cast<float>(iteration) );

        if(index < map_size) {
            const int index_in_map = index * number_features;

            for(int f = 0; f < number_features; f++) {
                map[index_in_map + f] += ( ( learn_rate * neighborhood[index] * ( obs[f] - map[index_in_map + f] ) ) ) ;
            }
        }
    }


    template <typename T = float>

    // Class of GPU's Self Organising Maps on CUDA
    class gpu_self_organising_maps_CUDA: public gpu_self_organising_maps<T> {

        // Using the class GPU's Self-Organising Maps as base
        using Base = gpu_self_organising_maps<T>;

        // The observation's type
        using observation_type = typename Base::observation_type ;

        // The size of an observation
        const unsigned observation_size;

        // The size of the blocks in the vectors/arrays of CUDA in the Device's memory (GPU)
        static constexpr auto block_size = 512;

        // The Distance Function which will be bind to be used during the process
        const std::function<void(T*, T*, int, int, float*)>  distance_function;


        // The general map (matrix or cube) of observations to be kept in the Device's memory (GPU)
        T* map_gpu;

        // The vector/array of observations to be kept in the Device's memory (GPU)
        T* obs_gpu;

        // The matrix of distances to be kept in the Device's memory (GPU)
        T* distances_gpu;

        // The vector/array of the candidate values to
        // the BMU (Best Matching Unit) to be kept in the Device's memory (GPU)
        float* bmu_values_gpu;

        // The vector/array of the indexes of the candidate values to
        // the BMU (Best Matching Unit) to be kept in the Device's memory (GPU)
        int* bmu_indexes_gpu;

        // The matrix of neighborhood to be kept in the Device's memory (GPU)
        T* neighborhood_gpu;


        public:
            gpu_self_organising_maps_CUDA(const unsigned ncols, const unsigned nrows, string&& input_file, string&& output_file,
                                          string&& distance_fun, unsigned seed = 0) :
                Base(ncols, nrows, std::move(input_file), std::move(output_file), seed),
                observation_size (this->number_features  * sizeof(T)),
                distance_function ( distance_fun == "euclidean" ?

                                    // The Euclidean Distance Function
                                    bind(&distance_euclidean, this, placeholders::_1, placeholders::_2) :

                                    // NOTE:
                                    // - If you want to test with other Distance Function than
                                    //   Cosine Distance Function, comment the following line,
                                    //   and uncomment one of the following next lines after that one,
                                    //   related to the Distance Function that you want to test

                                    // The Cosine Distance Function
                                    bind(&distance_cosine, this, placeholders::_1, placeholders::_2))

                                    // The Manhattan Distance Function
                                    //bind(&distance_manhattan, this, placeholders::_1, placeholders::_2))

                                    // The Minkowski Distance Function
                                    //bind(&distance_minkowski, this, placeholders::_1, placeholders::_2))

                                    // The Chebyshev Distance Function
                                    //bind(&distance_chebyshev, this, placeholders::_1, placeholders::_2))

            {
                // The size of the general map of observations in bytes
                const unsigned map_size_bytes = nrows * ncols * observation_size;

                // The size of the distance's matrix in bytes
                const unsigned distances_map_size_bytes = nrows * ncols * sizeof(float);

                // Allocate memory on the GPU for the general map of the observations
                cudaMalloc(&map_gpu, map_size_bytes);

                // Allocate memory on the GPU for the current observation,
                // which is being analysed
                cudaMalloc(&obs_gpu, observation_size);

                // Allocate memory on the GPU for the distances' matrix
                cudaMalloc(&distances_gpu, distances_map_size_bytes);

                // Allocate memory on the GPU for the neighborhood's matrix
                cudaMalloc(&neighborhood_gpu, ncols * nrows * sizeof(float));

                // Copy the data of the general map of the observations
                // from the CPU's memory (Host) to the GPU's memory (Device)
                cudaMemcpy(map_gpu, this->map.data(), map_size_bytes, cudaMemcpyHostToDevice);
            }


        protected:

            virtual void process_observation(observation_type& obs) {

                // Copy the data of the current observation, which is being analysed,
                // from the CPU's memory (Host) to the GPU's memory (Device)
                cudaMemcpy(obs_gpu, obs.data(), observation_size, cudaMemcpyHostToDevice);

                // The total size of the general map of observations
                // by the product of number_rows * number_cols * number_features
                const auto map_size = (this->number_rows * this->number_cols * this->number_features);

                // The total size of the distances' or neighborhood's matrix
                // by the product of the number_rows * number_cols
                const int distances_map_size = (this->number_rows * this->number_cols);

                // The number of blocks that will accessed by the threads
                const unsigned number_blocks = (this->number_rows * this->number_cols + block_size - 1) / block_size;

                // The GPU's kernel to calculate the distances' matrix
                distance_function<<<number_blocks, block_size>>>(map_gpu, obs_gpu, this->number_features, distances_map_size, distances_gpu);


                #ifdef DEBUG
                    vector<float> distances (distances_size);
                    cudaMemcpy(distances.data(), distances_gpu, distances_size * sizeof(float), cudaMemcpyDeviceToHost);

                    hpcProjectLog("distance to " << obs << ": " << distances << "\n");
                #endif

                // Allocate memory on the GPU for the values and indexes of
                // the BMU (Best Matching Unit) for each block
                cudaMalloc(&bmu_values_gpu, number_blocks * sizeof(float));
                cudaMalloc(&bmu_indexes_gpu, number_blocks * sizeof(int));

                // The GPU's kernel to calculate the BMU (Best Matching Unit) for each block
                arg_min<<<number_blocks, block_size>>>(distances_gpu, distances_map_size,
                                                       bmu_values_gpu, bmu_indexes_gpu);

                // The vectors/arrays to keep the values and indexes of
                // the BMU (Best Matching Unit) for each block, on the memory of the CPU
                vector<float> bmu_values (number_blocks);
                vector<int> bmu_indexes (number_blocks);

                // Copy the data of the values and indexes of the BMU (Best Matching Unit) for each block,
                // from the CPU's memory (Host) to the GPU's memory (Device)
                cudaMemcpy(bmu_values.data(), bmu_values_gpu,
                           ( number_blocks * sizeof(float) ), cudaMemcpyDeviceToHost);
                cudaMemcpy(bmu_indexes.data(), bmu_indexes_gpu,
                           ( number_blocks * sizeof(int) ), cudaMemcpyDeviceToHost);

                // The minimum global value of the BMU (Best Matching Unit) of all reviewed blocks,
                // and the respectively index
                float min_value = std::numeric_limits<float>::max();
                int min_value_index = -1;

                // Loop to find the minimum global value of the BMU (Best Matching Unit) of all reviewed blocks,
                // and the respectively index
                for(int i = 0; i < block_size; i++) {
                    hpcProjectLog(bmu_values[i]);

                    min_value = std::min(min_value, bmu_values[i]);

                    if(min_value == bmu_values[i]) {
                        min_value_index = bmu_indexes[i];
                    }
                }

                // Delete the vectors/arrays, previously defined, to keep the values and indexes of
                // the BMU (Best Matching Unit) for each block, on the memory of the CPU
                delete[] bmu_values.data();
                delete[] bmu_indexes.data();

                // The final index of the minimum global value of
                // the BMU (Best Matching Unit) of all reviewed blocks
                vector<int> bmu_final (1);
                bmu_final[0] = min_value_index;

                #ifdef DEBUG
                    hpcProjectLog("bmu to " << obs << " is at: " << bmu_final << "\n");
                #endif

                // Free the previously allocated memory on the GPU for the
                // distances' matrix, values and indexes of
                // the BMU (Best Matching Unit) for each block
                cudaFree(&distances_gpu);
                cudaFree(&bmu_values_gpu);
                cudaFree(&bmu_indexes_gpu);

                // The vector/array of the index of the minimum global value of
                // the BMU (Best Matching Unit) of all reviewed blocks
                // to be kept in the Device's memory (GPU)
                int* bmu_final_gpu;

                // Allocate memory on the GPU for the vector/array of
                // the index of the minimum global value of
                // the BMU (Best Matching Unit) of all reviewed blocks
                cudaMalloc(&bmu_final_gpu, sizeof(int));

                // Copy the data of the index of the minimum global value of
                // the BMU (Best Matching Unit) of all reviewed blocks,
                // from the CPU's memory (Host) to the GPU's memory (Device)
                cudaMemcpy(bmu_final_gpu, bmu_final.data(), sizeof(int), cudaMemcpyHostToDevice);

                // The GPU's kernel to calculate the neighborhood's matrix
                neighborhood_function<<<number_blocks, block_size>>>(this->iteration, this->number_features. bmu_final_gpu,
                                                                     this->dr.get_number_observations(), distances_map_size,
                                                                     this->max_distance, neighborhood_gpu);

                // Free the previously allocated memory on the GPU for the vector/array of
                // the index of the minimum global value of
                // the BMU (Best Matching Unit) of all reviewed blocks
                cudaFree(&bmu_final_gpu);

                #ifdef DEBUG
                    vector<float> neighborhood (this->number_rows * this->number_cols);
                    cudaMemcpy(neighborhood.data(), neighborhood_gpu, ncols * nrows * sizeof(float), cudaMemcpyDeviceToHost);
                    hpcProjectLog("bmu to " << obs << " is at: " << bmu << "\n");
                #endif

                // The GPU's kernel to update the general map of observations
                update_map<<<number_blocks, block_size>>>(this->iteration, this->number_features, obs_gpu, map_size, neighborhood_gpu, map_gpu);

                #ifdef DEBUG
                    const unsigned map_size_bytes = this->number_rows * this->number_cols * observation_size;
                    cudaMemcpy(this->map.data(), map_gpu, map_size_bytes, cudaMemcpyDeviceToHost);
                    hpcProjectLog("the updated map given the obs [" << obs << "] and the bmu [" << bmu << "] is " << map "\n");
                #endif

                // Free the previously allocated memory on the GPU for
                // the general map of observations, the current observation which is being analysed,
                // and for the neighborhood's matrix
                cudaFree(&map_gpu);
                cudaFree(&obs_gpu);
                cudaFree(&neighborhood_gpu);
            }
    };
}
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

#ifndef GPU_SELF_ORGANISING_MAPS_CUDA_OPTIMISED_HPP
#define GPU_SELF_ORGANISING_MAPS_CUDA_OPTIMISED_HPP
#endif

#include "gpu_self_organising_maps_optimised.hpp"
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
     * @tparam T the template of the typename T (the type of all components of each observation)
     * @tparam blockSize the template of the block's size
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
    template<typename T, unsigned int blockSize>

    __global__ void distance_euclidean(const T *map, const T *obs, const int number_features, const int map_size,
                                       float *distances_map) {

        extern __shared__ float sdata[];

        unsigned int tid = threadIdx.x;
        const int index = (blockIdx.x * blockDim.x) + tid;

        sdata[tid] = 0.0f;

        if (index < map_size) {
            const int index_in_map = index * number_features;

            if (tid < number_features) {
                sdata[tid] = pow((map[(index_in_map + tid)] - obs[tid]), 2.0f);

                __syncthreads();

                if (blockSize >= 512) {
                    if (tid < 256) {
                        sdata[tid] += sdata[tid + 256];
                    }

                    __syncthreads();
                }

                if (blockSize >= 256) {
                    if (tid < 128) {
                        sdata[tid] += sdata[tid + 128];
                    }

                    __syncthreads();
                }

                if (blockSize >= 128) {
                    if (tid < 64) {
                        sdata[tid] += sdata[tid + 64];
                    }

                    __syncthreads();
                }

                if (tid < 32) {
                    if (blockSize >= 64) {
                        sdata[tid] += sdata[tid + 32];
                    }

                    if (blockSize >= 32) {
                        sdata[tid] += sdata[tid + 16];
                    }

                    if (blockSize >= 16) {
                        sdata[tid] += sdata[tid + 8];
                    }

                    if (blockSize >= 8) {
                        sdata[tid] += sdata[tid + 4];
                    }

                    if (blockSize >= 4) {
                        sdata[tid] += sdata[tid + 2];
                    }

                    if (blockSize >= 2) {
                        sdata[tid] += sdata[tid + 1];
                    }
                }
            }

            distances_map[index] = sqrt(sdata[0]);
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
     * @tparam blockSize the template of the block's size
     *
     * @param map the general map of observations
     * @param obs the current observation
     * @param number_features the number of features of each observation
     * @param map_size the size of the general map
     * @param distances_map the matrix of distances where will be put the distances
     *
     * @return the matrix of distances on the memory of the GPU,
     *         accordingly to the Cosine Distance Function
     */
    template<typename T, unsigned int blockSize>

    __global__ void
    distance_cosine(const T *map, const T *obs, const int number_features, const int map_size,
                    float *distances_map) {

        extern __shared__ float sdata_map_two_power[];
        extern __shared__ float sdata_obs_two_power[];
        extern __shared__ float sdata_vectors_cross_product[];

        unsigned int tid = threadIdx.x;
        const int index = (blockIdx.x * blockDim.x) + tid;

        sdata_map_two_power[tid] = 0.0f;
        sdata_obs_two_power[tid] = 0.0f;
        sdata_vectors_cross_product[tid] = 0.0f;

        if (index < map_size) {
            const int index_in_map = index * number_features;

            if (tid < number_features) {
                sdata_map_two_power[tid] = pow(map[index_in_map + tid], 2.0f);
                sdata_obs_two_power[tid] = pow(obs[tid], 2.0f);

                sdata_vectors_cross_product[tid] = (map[index_in_map + tid] * obs[tid]);

                __syncthreads();

                if (blockSize >= 512) {
                    if (tid < 256) {
                        sdata_map_two_power[tid] += sdata_map_two_power[tid + 256];
                        sdata_obs_two_power[tid] += sdata_obs_two_power[tid + 256];

                        sdata_vectors_cross_product[tid] += sdata_vectors_cross_product[tid + 256];
                    }

                    __syncthreads();
                }

                if (blockSize >= 256) {
                    if (tid < 128) {
                        sdata_map_two_power[tid] += sdata_map_two_power[tid + 128];
                        sdata_obs_two_power[tid] += sdata_obs_two_power[tid + 128];

                        sdata_vectors_cross_product[tid] += sdata_vectors_cross_product[tid + 128];
                    }

                    __syncthreads();
                }

                if (blockSize >= 128) {
                    if (tid < 64) {
                        sdata_map_two_power[tid] += sdata_map_two_power[tid + 64];
                        sdata_obs_two_power[tid] += sdata_obs_two_power[tid + 64];

                        sdata_vectors_cross_product[tid] += sdata_vectors_cross_product[tid + 64];
                    }

                    __syncthreads();
                }

                if (tid < 32) {
                    if (blockSize >= 64) {
                        sdata_map_two_power[tid] += sdata_map_two_power[tid + 32];
                        sdata_obs_two_power[tid] += sdata_obs_two_power[tid + 32];

                        sdata_vectors_cross_product[tid] += sdata_vectors_cross_product[tid + 32];
                    }

                    if (blockSize >= 32) {
                        sdata_map_two_power[tid] += sdata_map_two_power[tid + 16];
                        sdata_obs_two_power[tid] += sdata_obs_two_power[tid + 16];

                        sdata_vectors_cross_product[tid] += sdata_vectors_cross_product[tid + 16];
                    }

                    if (blockSize >= 16) {
                        sdata_map_two_power[tid] += sdata_map_two_power[tid + 8];
                        sdata_obs_two_power[tid] += sdata_obs_two_power[tid + 8];

                        sdata_vectors_cross_product[tid] += sdata_vectors_cross_product[tid + 8];
                    }

                    if (blockSize >= 8) {
                        sdata_map_two_power[tid] += sdata_map_two_power[tid + 4];
                        sdata_obs_two_power[tid] += sdata_obs_two_power[tid + 4];

                        sdata_vectors_cross_product[tid] += sdata_vectors_cross_product[tid + 4];
                    }

                    if (blockSize >= 4) {
                        sdata_map_two_power[tid] += sdata_map_two_power[tid + 2];
                        sdata_obs_two_power[tid] += sdata_obs_two_power[tid + 2];

                        sdata_vectors_cross_product[tid] += sdata_vectors_cross_product[tid + 2];
                    }

                    if (blockSize >= 2) {
                        sdata_map_two_power[tid] += sdata_map_two_power[tid + 1];
                        sdata_obs_two_power[tid] += sdata_obs_two_power[tid + 1];

                        sdata_vectors_cross_product[tid] += sdata_vectors_cross_product[tid + 1];
                    }
                }
            }

            float sqrt_map_two_power_sum = sqrt(sdata_map_two_power[0]);
            float sqrt_obs_two_power_sum = sqrt(sdata_obs_two_power[0]);

            float vectors_cross_product_sum = sdata_vectors_cross_product[0];

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
     * @tparam blockSize the template of the block's size
     *
     * @param map the general map
     * @param obs the current observation
     * @param number_features the number of features of each observation
     * @param map_size the size of the general map
     * @param distances_map the matrix of distances where will be put the distances
     *
     * @return the matrix of distances on the memory of the GPU,
     *         accordingly to the Manhattan Distance Function
     */
    template<typename T, unsigned int blockSize>

    __global__ void distance_manhattan(const T *map, const T *obs, const int number_features, const int map_size,
                                       float *distances_map) {

        extern __shared__ float sdata[];

        unsigned int tid = threadIdx.x;
        const int index = (blockIdx.x * blockDim.x) + tid;

        sdata[tid] = 0.0f;

        if (index < map_size) {
            const int index_in_map = index * number_features;

            if (tid < number_features) {
                sdata[tid] = abs( (map[(index_in_map + tid)] - obs[tid]) );

                __syncthreads();

                if (blockSize >= 512) {
                    if (tid < 256) {
                        sdata[tid] += sdata[tid + 256];
                    }

                    __syncthreads();
                }

                if (blockSize >= 256) {
                    if (tid < 128) {
                        sdata[tid] += sdata[tid + 128];
                    }

                    __syncthreads();
                }

                if (blockSize >= 128) {
                    if (tid < 64) {
                        sdata[tid] += sdata[tid + 64];
                    }

                    __syncthreads();
                }

                if (tid < 32) {
                    if (blockSize >= 64) {
                        sdata[tid] += sdata[tid + 32];
                    }

                    if (blockSize >= 32) {
                        sdata[tid] += sdata[tid + 16];
                    }

                    if (blockSize >= 16) {
                        sdata[tid] += sdata[tid + 8];
                    }

                    if (blockSize >= 8) {
                        sdata[tid] += sdata[tid + 4];
                    }

                    if (blockSize >= 4) {
                        sdata[tid] += sdata[tid + 2];
                    }

                    if (blockSize >= 2) {
                        sdata[tid] += sdata[tid + 1];
                    }
                }
            }

            distances_map[index] = sdata[0];
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
     * @tparam blockSize the template of the block's size
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
    template<typename T, unsigned int blockSize>

    __global__ void
    distance_minkowski(const T *map, const T *obs, const int number_features, const int map_size, float *distances_map,
                       float p = 2.0f) {

        extern __shared__ float sdata[];

        unsigned int tid = threadIdx.x;
        const int index = (blockIdx.x * blockDim.x) + tid;

        sdata[tid] = 0.0f;

        if (index < map_size) {
            const int index_in_map = index * number_features;

            if (tid < number_features) {
                sdata[tid] = pow( abs( (map[index_in_map + tid] - obs[tid]) ), p);

                __syncthreads();

                if (blockSize >= 512) {
                    if (tid < 256) {
                        sdata[tid] += sdata[tid + 256];
                    }

                    __syncthreads();
                }

                if (blockSize >= 256) {
                    if (tid < 128) {
                        sdata[tid] += sdata[tid + 128];
                    }

                    __syncthreads();
                }

                if (blockSize >= 128) {
                    if (tid < 64) {
                        sdata[tid] += sdata[tid + 64];
                    }

                    __syncthreads();
                }

                if (tid < 32) {
                    if (blockSize >= 64) {
                        sdata[tid] += sdata[tid + 32];
                    }

                    if (blockSize >= 32) {
                        sdata[tid] += sdata[tid + 16];
                    }

                    if (blockSize >= 16) {
                        sdata[tid] += sdata[tid + 8];
                    }

                    if (blockSize >= 8) {
                        sdata[tid] += sdata[tid + 4];
                    }

                    if (blockSize >= 4) {
                        sdata[tid] += sdata[tid + 2];
                    }

                    if (blockSize >= 2) {
                        sdata[tid] += sdata[tid + 1];
                    }
                }
            }

            distances_map[index] = pow( sdata[0], ( 1.0f / p ) );
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
     * @tparam blockSize the template of the block's size
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
    template<typename T, unsigned int blockSize>

    __global__ void distance_chebyshev(const T *map, const T *obs, const int number_features, const int map_size,
                                       float *distances_map) {

        extern __shared__ float sdata[];

        unsigned int tid = threadIdx.x;
        const int index = (blockIdx.x * blockDim.x) + tid;

        sdata[tid] = 0.0f;

        if (index < map_size) {
            const int index_in_map = index * number_features;

            if (tid < number_features) {
                sdata[tid] = ( sdata[tid] > ( map[index_in_map + tid] - obs[tid] ) ) ?
                               sdata[tid] : ( map[index_in_map + tid] - obs[tid] );

                __syncthreads();

                if (blockSize >= 512) {
                    if (tid < 256) {
                        sdata[tid] = ( sdata[tid] > sdata[tid + 256] ) ?
                                       sdata[tid] : sdata[tid + 256];
                    }

                    __syncthreads();
                }

                if (blockSize >= 256) {
                    if (tid < 128) {
                        sdata[tid] = ( sdata[tid] > sdata[tid + 128] ) ?
                                       sdata[tid] : sdata[tid + 128];
                    }

                    __syncthreads();
                }

                if (blockSize >= 128) {
                    if (tid < 64) {
                        sdata[tid] = ( sdata[tid] > sdata[tid + 64] ) ?
                                       sdata[tid] : sdata[tid + 64];
                    }

                    __syncthreads();
                }

                if (tid < 32) {
                    if (blockSize >= 64) {
                        sdata[tid] = ( sdata[tid] > sdata[tid + 32] ) ?
                                       sdata[tid] : sdata[tid + 32];
                    }

                    if (blockSize >= 32) {
                        sdata[tid] = ( sdata[tid] > sdata[tid + 16] ) ?
                                       sdata[tid] : sdata[tid + 16];
                    }

                    if (blockSize >= 16) {
                        sdata[tid] = ( sdata[tid] > sdata[tid + 8] ) ?
                                       sdata[tid] : sdata[tid + 8];
                    }

                    if (blockSize >= 8) {
                        sdata[tid] = ( sdata[tid] > sdata[tid + 4] ) ?
                                       sdata[tid] : sdata[tid + 4];
                    }

                    if (blockSize >= 4) {
                        sdata[tid] = ( sdata[tid] > sdata[tid + 2] ) ?
                                       sdata[tid] : sdata[tid + 2];
                    }

                    if (blockSize >= 2) {
                        sdata[tid] = ( sdata[tid] > sdata[tid + 1] ) ?
                                       sdata[tid] : sdata[tid + 1];
                    }
                }
            }

            distances_map[index] = sdata[0];
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
    template<unsigned int blockSize>

    __global__ void arg_min(const float *distances_map, int distances_map_size, float *bmu_values, int *bmu_indexes) {

        extern __shared__ float sdata_values[];
        extern __shared__ int sdata_indexes[];

        unsigned int tid = threadIdx.x;
        unsigned int i = (blockSize * 2) * blockIdx.x + tid;

        unsigned int gridSize = (blockSize * 2) * gridDim.x;

        sdata_values[tid] = 0.0f;
        sdata_indexes[tid] = -1;

        while (i < distances_map_size) {
            sdata_indexes[tid] = (distances_map[i] < distances_map[i + blockSize]) ? i : (i + blockSize);
            sdata_values[tid] = (distances_map[i] < distances_map[i + blockSize]) ? distances_map[i] : distances_map[(
                    i + blockSize)];

            i += gridSize;
        }

        __syncthreads();

        if (blockSize >= 512) {
            if (tid < 256) {
                sdata_indexes[tid] = (sdata_values[tid] < sdata_values[tid + 256]) ? tid : (tid + 256);
                sdata_values[tid] = (sdata_values[tid] < sdata_values[tid + 256]) ? sdata_values[tid] : sdata_values[
                        tid + 256];
            }

            __syncthreads();
        }

        if (blockSize >= 256) {
            if (tid < 128) {
                sdata_indexes[tid] = (sdata_values[tid] < sdata_values[tid + 128]) ? tid : (tid + 128);
                sdata_values[tid] = (sdata_values[tid] < sdata_values[tid + 128]) ? sdata_values[tid] : sdata_values[
                        tid + 128];
            }

            __syncthreads();
        }

        if (blockSize >= 128) {
            if (tid < 64) {
                sdata_indexes[tid] = (sdata_values[tid] < sdata_values[tid + 64]) ? tid : (tid + 64);
                sdata_values[tid] = (sdata_values[tid] < sdata_values[tid + 64]) ? sdata_values[tid] : sdata_values[
                        tid + 64];
            }

            __syncthreads();
        }

        if (tid < 32) {

            if (blockSize >= 64) {
                sdata_indexes[tid] = (sdata_values[tid] < sdata_values[tid + 32]) ? tid : (tid + 32);
                sdata_values[tid] = (sdata_values[tid] < sdata_values[tid + 32]) ? sdata_values[tid] : sdata_values[
                        tid + 32];
            }

            if (blockSize >= 32) {
                sdata_indexes[tid] = (sdata_values[tid] < sdata_values[tid + 16]) ? tid : (tid + 16);
                sdata_values[tid] = (sdata_values[tid] < sdata_values[tid + 16]) ? sdata_values[tid] : sdata_values[
                        tid + 16];
            }

            if (blockSize >= 16) {
                sdata_indexes[tid] = (sdata_values[tid] < sdata_values[tid + 8]) ? tid : (tid + 8);
                sdata_values[tid] = (sdata_values[tid] < sdata_values[tid + 8]) ? sdata_values[tid] : sdata_values[tid +
                                                                                                                   8];
            }

            if (blockSize >= 8) {
                sdata_indexes[tid] = (sdata_values[tid] < sdata_values[tid + 4]) ? tid : (tid + 4);
                sdata_values[tid] = (sdata_values[tid] < sdata_values[tid + 4]) ? sdata_values[tid] : sdata_values[tid +
                                                                                                                   4];
            }

            if (blockSize >= 4) {
                sdata_indexes[tid] = (sdata_values[tid] < sdata_values[tid + 2]) ? tid : (tid + 2);
                sdata_values[tid] = (sdata_values[tid] < sdata_values[tid + 2]) ? sdata_values[tid] : sdata_values[tid +
                                                                                                                   2];
            }

            if (blockSize >= 2) {
                sdata_indexes[tid] = (sdata_values[tid] < sdata_values[tid + 1]) ? tid : (tid + 1);
                sdata_values[tid] = (sdata_values[tid] < sdata_values[tid + 1]) ? sdata_values[tid] : sdata_values[tid +
                                                                                                                   1];
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
    template<typename T>

    __global__ void
    neighborhood_function(const int iteration, const int number_features, const int bmu, const int num_inputs,
                          const int distances_map_size, const float max_distance, T *neighborhood) {

        const unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;

        if (index < distances_map_size) {

            float theta = ((max_distance / 2.0f) - ((max_distance / 2.0f) * (iteration / num_inputs)));

            float sqrDist = pow(abs(bmu - index), 2.0f);

            float n = exp(-1.0f * (sqrDist / pow(theta, 2.0f)));

            neighborhood[index] = (n > 0.01f) ? n : 0.0f;
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
    template<typename T>

    __global__ void
    update_map(const int iteration, const int number_features, const T *obs, const int map_size, const T *neighborhood,
               T *map) {
        const int index = (blockIdx.x * blockDim.x) + threadIdx.x;

        float learn_rate = (1 / static_cast<float>(iteration));

        if (index < map_size) {
            const int index_in_map = index * number_features;

            for (int f = 0; f < number_features; f++) {
                map[index_in_map + f] += ((learn_rate * neighborhood[index] * (obs[f] - map[index_in_map + f])));
            }
        }
    }


    template<typename T = float>

    // Class of GPU's Self Organising Maps on CUDA (Optimised)
    class gpu_self_organising_maps_CUDA_optimised : public gpu_self_organising_maps_optimised<T> {

        // Using the class GPU's Self-Organising Maps (Optimised) as base
        using Base = gpu_self_organising_maps_optimised<T>;

        // The observation's type
        using observation_type = typename Base::observation_type;

        // The size of an observation
        const unsigned observation_size;

        // The size of the blocks in the vectors/arrays of CUDA in the Device's memory (GPU)
        static constexpr auto block_size = 512;

        // The Distance Function which will be bind to be used during the process
        const std::function<void(T *, T *, int, int, float *)> distance_function;


        // The general map (matrix or cube) of observations to be kept in the Device's memory (GPU)
        T *map_gpu;

        // The vector/array of observations to be kept in the Device's memory (GPU)
        T *obs_gpu_1, *obs_gpu_2, *obs_gpu_3, *obs_gpu_4;

        // The vector/array of observations to be kept in the Device's memory (GPU) as Host
        T *obs_host_1, *obs_host_2, *obs_host_3, *obs_host_4;

        // The matrix of distances to be kept in the Device's memory (GPU)
        T *distances_gpu_1, *distances_gpu_2, *distances_gpu_3, *distances_gpu_4;

        // The matrix of distances to be kept in the Device's memory (GPU) as Host
        T *distances_host_1, *distances_host_2, *distances_host_3, *distances_host_4;

        // The vector/array of the candidate values to
        // the BMU (Best Matching Unit) to be kept in the Device's memory (GPU)
        float *bmu_values_gpu_1, *bmu_values_gpu_2, *bmu_values_gpu_3, *bmu_values_gpu_4;

        // The vector/array of the candidate values to
        // the BMU (Best Matching Unit) to be kept in the Device's memory (GPU) as Host
        float *bmu_values_host_1, *bmu_values_host_2, *bmu_values_host_3, *bmu_values_host_4;

        // The vector/array of the indexes of the candidate values to
        // the BMU (Best Matching Unit) to be kept in the Device's memory (GPU)
        int *bmu_indexes_gpu_1, *bmu_indexes_gpu_2, *bmu_indexes_gpu_3, *bmu_indexes_gpu_4;

        // The vector/array of the indexes of the candidate values to
        // the BMU (Best Matching Unit) to be kept in the Device's memory (GPU) as Host
        int *bmu_indexes_host_1, *bmu_indexes_host_2, *bmu_indexes_host_3, *bmu_indexes_host_4;

        // The matrix of neighborhood to be kept in the Device's memory (GPU)
        T *neighborhood_gpu_1, *neighborhood_gpu_2, *neighborhood_gpu_3, *neighborhood_gpu_4;

        // The matrix of neighborhood to be kept in the Device's memory (GPU) as Host
        T *neighborhood_host_1, *neighborhood_host_2, *neighborhood_host_3, *neighborhood_host_4;

        // The total size of the general map of observations
        // by the product of number_rows * number_cols * number_features
        const int map_size;

        // The total size of the distances' or neighborhood's matrix
        // by the product of the number_rows * number_cols
        const int distances_map_size;

        // The number of blocks that will accessed by the threads
        const unsigned number_blocks;

        // The size of the general map of observations in bytes
        const unsigned map_size_bytes;

        // The size of the distance's matrix in bytes
        const unsigned distances_map_size_bytes;



        public:
            gpu_self_organising_maps_CUDA_optimised(const unsigned ncols, const unsigned nrows, string &&input_file,
                                                string &&output_file,
                                                string &&distance_fun, unsigned seed = 0) :
                Base(ncols, nrows, std::move(input_file), std::move(output_file), seed),
                observation_size(this->number_features * sizeof(T)),
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

                    // The total size of the general map of observations
                    // by the product of number_rows * number_cols * number_features
                    map_size = (this->number_rows * this->number_cols * this->number_features);

                    // The total size of the distances' or neighborhood's matrix
                    // by the product of the number_rows * number_cols
                    distances_map_size = (this->number_rows * this->number_cols);

                    // The number of blocks that will accessed by the threads
                    number_blocks = (this->number_rows * this->number_cols + block_size - 1) / block_size;

                    // The size of the general map of observations in bytes
                    map_size_bytes = nrows * ncols * observation_size;

                    // The size of the distance's matrix in bytes
                    distances_map_size_bytes = nrows * ncols * sizeof(float);

                    // Allocate memory on the GPU for the general map of the observations
                    cudaMalloc(&map_gpu, map_size_bytes);

                    // Copy the data of the general map of the observations
                    // from the CPU's memory (Host) to the GPU's memory (Device)
                    cudaMemcpy(map_gpu, this->map.data(), map_size_bytes, cudaMemcpyHostToDevice);
                }


        protected:

            virtual void process_observation(vector<observation_type> &obs) {

                // If it's possible process four observations in parallel
                if (obs->size() == 4) {

                    // Four CUDA's Streams for parallelism
                    cudaStream_t s1, s2, s3, s4;

                    // Create four CUDA's Streams for parallelism
                    cudaStreamCreate(&s1);
                    cudaStreamCreate(&s2);
                    cudaStreamCreate(&s3);
                    cudaStreamCreate(&s4);

                    // Allocate memory on the GPU for the current four observations,
                    // which are being analysed
                    cudaMalloc(&obs_gpu_1, observation_size);
                    cudaMalloc(&obs_gpu_2, observation_size);
                    cudaMalloc(&obs_gpu_3, observation_size);
                    cudaMalloc(&obs_gpu_4, observation_size);

                    // Allocate memory on the GPU for the current four observations,
                    // which are being analysed as Host
                    cudaMallocHost(&obs_host_1, observation_size);
                    cudaMallocHost(&obs_host_2, observation_size);
                    cudaMallocHost(&obs_host_3, observation_size);
                    cudaMallocHost(&obs_host_4, observation_size);

                    // Copy the data of the current four observations, which are being analysed,
                    // from the CPU's memory (Host) to the GPU's memory (Device)
                    cudaMemcpyAsync(obs_host_1, obs[0].data(), observation_size, cudaMemcpyHostToDevice, s1);
                    cudaMemcpyAsync(obs_host_2, obs[1].data(), observation_size, cudaMemcpyHostToDevice, s2);
                    cudaMemcpyAsync(obs_host_3, obs[2].data(), observation_size, cudaMemcpyHostToDevice, s3);
                    cudaMemcpyAsync(obs_host_4, obs[3].data(), observation_size, cudaMemcpyHostToDevice, s4);

                    // Copy the data of the current four observations, which are being analysed,
                    // from the GPU's memory (Device) as Host to the GPU's memory (Device)
                    cudaMemcpyAsync(obs_gpu_1, obs_host_1, observation_size, cudaMemcpyHostToDevice, s1);
                    cudaMemcpyAsync(obs_gpu_2, obs_host_2, observation_size, cudaMemcpyHostToDevice, s2);
                    cudaMemcpyAsync(obs_gpu_3, obs_host_3, observation_size, cudaMemcpyHostToDevice, s3);
                    cudaMemcpyAsync(obs_gpu_4, obs_host_4, observation_size, cudaMemcpyHostToDevice, s4);

                    // Allocate memory on the GPU for the current four distances' matrices,
                    // which are being processed
                    cudaMalloc(&distances_gpu_1, distances_map_size_bytes);
                    cudaMalloc(&distances_gpu_2, distances_map_size_bytes);
                    cudaMalloc(&distances_gpu_3, distances_map_size_bytes);
                    cudaMalloc(&distances_gpu_4, distances_map_size_bytes);

                    // Launch four GPU's kernels to calculate the distances' matrices, in parallel
                    distance_function << < number_blocks, block_size, 0, s1 >> >
                            (map_gpu, obs_gpu_1, this->number_features, distances_map_size, distances_gpu_1);
                    distance_function << < number_blocks, block_size, 0, s2 >> >
                            (map_gpu, obs_gpu_2, this->number_features, distances_map_size, distances_gpu_2);
                    distance_function << < number_blocks, block_size, 0, s3 >> >
                            (map_gpu, obs_gpu_3, this->number_features, distances_map_size, distances_gpu_3);
                    distance_function << < number_blocks, block_size, 0, s4>> >
                            (map_gpu, obs_gpu_4, this->number_features, distances_map_size, distances_gpu_4);

                    #ifdef DEBUG
                        vector<float> distances_1 (distances_size);
                        cudaMemcpyAsync(distances_1.data(), distances_gpu_1, distances_map_size, cudaMemcpyDeviceToHost, s1);

                        hpcProjectLog("distance to " << obs[0] << ": " << distances_1 << "\n");


                        vector<float> distances_2 (distances_size);
                        cudaMemcpyAsync(distances_2.data(), distances_gpu_2, distances_map_size, cudaMemcpyDeviceToHost, s2);

                        hpcProjectLog("distance to " << obs[1] << ": " << distances_2 << "\n");


                        vector<float> distances_3 (distances_size);
                        cudaMemcpyAsync(distances_3.data(), distances_gpu_3, distances_map_size, cudaMemcpyDeviceToHost, s3);

                        hpcProjectLog("distance to " << obs[2] << ": " << distances_3 << "\n");


                        vector<float> distances_4 (distances_size);
                        cudaMemcpyAsync(distances_4.data(), distances_gpu_4, distances_map_size, cudaMemcpyDeviceToHost, s4);

                        hpcProjectLog("distance to " << obs[3] << ": " << distances_4 << "\n");
                    #endif

                    // Allocate memory on the GPU for the values of
                    // the four BMUs (Best Matching Units) for each block
                    cudaMalloc(&bmu_values_gpu_1, number_blocks * sizeof(float));
                    cudaMalloc(&bmu_values_gpu_2, number_blocks * sizeof(float));
                    cudaMalloc(&bmu_values_gpu_3, number_blocks * sizeof(float));
                    cudaMalloc(&bmu_values_gpu_4, number_blocks * sizeof(float));

                    // Allocate memory on the GPU for the values of
                    // the four BMUs (Best Matching Units) for each block as Host
                    cudaMallocHost(&bmu_values_host_1, number_blocks * sizeof(float));
                    cudaMallocHost(&bmu_values_host_2, number_blocks * sizeof(float));
                    cudaMallocHost(&bmu_values_host_3, number_blocks * sizeof(float));
                    cudaMallocHost(&bmu_values_host_4, number_blocks * sizeof(float));

                    // Allocate memory on the GPU for the indexes of
                    // the four BMUs (Best Matching Units) for each block
                    cudaMalloc(&bmu_indexes_gpu_1, number_blocks * sizeof(int));
                    cudaMalloc(&bmu_indexes_gpu_2, number_blocks * sizeof(int));
                    cudaMalloc(&bmu_indexes_gpu_3, number_blocks * sizeof(int));
                    cudaMalloc(&bmu_indexes_gpu_4, number_blocks * sizeof(int));

                    // Allocate memory on the GPU for the indexes of
                    // the four BMUs (Best Matching Units) for each block as Host
                    cudaMallocHost(&bmu_indexes_host_1, number_blocks * sizeof(int));
                    cudaMallocHost(&bmu_indexes_host_2, number_blocks * sizeof(int));
                    cudaMallocHost(&bmu_indexes_host_3, number_blocks * sizeof(int));
                    cudaMallocHost(&bmu_indexes_host_4, number_blocks * sizeof(int));

                    // Launch four GPU's kernels to calculate the BMUs (Best Matching Units) for each block, in parallel
                    arg_min << < number_blocks, block_size, 0, s1 >> >
                            (distances_gpu_1, distances_map_size, bmu_values_gpu_1, bmu_indexes_gpu_1);
                    arg_min << < number_blocks, block_size, 0, s2 >> >
                            (distances_gpu_2, distances_map_size, bmu_values_gpu_2, bmu_indexes_gpu_2);
                    arg_min << < number_blocks, block_size, 0, s3 >> >
                            (distances_gpu_3, distances_map_size, bmu_values_gpu_3, bmu_indexes_gpu_3);
                    arg_min << < number_blocks, block_size, 0, s4 >> >
                            (distances_gpu_4, distances_map_size, bmu_values_gpu_4, bmu_indexes_gpu_4);

                    // The vectors/arrays to keep the values of
                    // the four BMUs (Best Matching Units) for each block, on the memory of the CPU
                    vector<float> bmu_values_1(number_blocks);
                    vector<float> bmu_values_2(number_blocks);
                    vector<float> bmu_values_3(number_blocks);
                    vector<float> bmu_values_4(number_blocks);

                    // The vectors/arrays to keep the indexes of the values of
                    // the four BMUs (Best Matching Units) for each block, on the memory of the CPU
                    vector<int> bmu_indexes_1(number_blocks);
                    vector<int> bmu_indexes_2(number_blocks);
                    vector<int> bmu_indexes_3(number_blocks);
                    vector<int> bmu_indexes_4(number_blocks);

                    // Copy the data of the four values of the BMUs (Best Matching Units) for each block,
                    // from the GPU's memory (Device) to the GPU's memory (Device) as Host
                    cudaMemcpyAsync(bmu_values_host_1, bmu_values_gpu_1, (number_blocks * sizeof(float)),
                              cudaMemcpyDeviceToHost, s1);
                    cudaMemcpyAsync(bmu_values_host_2, bmu_values_gpu_2, (number_blocks * sizeof(float)),
                              cudaMemcpyDeviceToHost, s2);
                    cudaMemcpyAsync(bmu_values_host_3, bmu_values_gpu_3, (number_blocks * sizeof(float)),
                              cudaMemcpyDeviceToHost, s3);
                    cudaMemcpyAsync(bmu_values_host_4, bmu_values_gpu_4, (number_blocks * sizeof(float)),
                              cudaMemcpyDeviceToHost, s4);

                    // Copy the data of the four values of the BMUs (Best Matching Units) for each block,
                    // from the GPU's memory (Device) to the CPU's memory (Host)
                    cudaMemcpyAsync(bmu_values_1.data(), bmu_values_host_1, (number_blocks * sizeof(float)),
                              cudaMemcpyDeviceToHost, s1);
                    cudaMemcpyAsync(bmu_values_2.data(), bmu_values_host_2, (number_blocks * sizeof(float)),
                              cudaMemcpyDeviceToHost, s2);
                    cudaMemcpyAsync(bmu_values_3.data(), bmu_values_host_3, (number_blocks * sizeof(float)),
                              cudaMemcpyDeviceToHost, s3);
                    cudaMemcpyAsync(bmu_values_4.data(), bmu_values_host_4, (number_blocks * sizeof(float)),
                              cudaMemcpyDeviceToHost, s4);

                    // Copy the data of the four indexes of the BMUs (Best Matching Units) for each block,
                    // from the GPU's memory (Device) to the GPU's memory (Device) as Host
                    cudaMemcpyAsync(bmu_indexes_host_1, bmu_indexes_gpu_1, (number_blocks * sizeof(int)),
                              cudaMemcpyDeviceToHost, s1);
                    cudaMemcpyAsync(bmu_indexes_host_2, bmu_indexes_gpu_2, (number_blocks * sizeof(int)),
                              cudaMemcpyDeviceToHost, s2);
                    cudaMemcpyAsync(bmu_indexes_host_3, bmu_indexes_gpu_3, (number_blocks * sizeof(int)),
                              cudaMemcpyDeviceToHost, s3);
                    cudaMemcpyAsync(bmu_indexes_host_4, bmu_indexes_gpu_4, (number_blocks * sizeof(int)),
                              cudaMemcpyDeviceToHost, s4);

                    // Copy the data of the four indexes of the BMUs (Best Matching Units) for each block,
                    // from the GPU's memory (Device) to the CPU's memory (Host)
                    cudaMemcpyAsync(bmu_indexes_1.data(), bmu_indexes_host_1, (number_blocks * sizeof(int)),
                              cudaMemcpyDeviceToHost, s1);
                    cudaMemcpyAsync(bmu_indexes_2.data(), bmu_indexes_host_2, (number_blocks * sizeof(int)),
                              cudaMemcpyDeviceToHost, s2);
                    cudaMemcpyAsync(bmu_indexes_3.data(), bmu_indexes_host_3, (number_blocks * sizeof(int)),
                              cudaMemcpyDeviceToHost, s3);
                    cudaMemcpyAsync(bmu_indexes_4.data(), bmu_indexes_host_4, (number_blocks * sizeof(int)),
                              cudaMemcpyDeviceToHost, s4);

                    // The minimum four global values of
                    // the BMUs (Best Matching Units) of all reviewed blocks
                    vector<float> min_values(4);
                    min_values[0] = std::numeric_limits<float>::max();
                    min_values[1] = std::numeric_limits<float>::max();
                    min_values[2] = std::numeric_limits<float>::max();
                    min_values[3] = std::numeric_limits<float>::max();

                    // The indexes of the minimum four global values of
                    // the BMUs (Best Matching Units) of all reviewed blocks
                    vector<int> min_values_indexes(4);
                    min_values_indexes[0] = -1;
                    min_values_indexes[1] = -1;
                    min_values_indexes[2] = -1;
                    min_values_indexes[3] = -1;

                    // Loop to find the minimum four global values of
                    // the BMUs (Best Matching Units) of all reviewed blocks,
                    // and the respectively indexes
                    for (int i = 0; i < block_size; i++) {
                        hpcProjectLog(bmu_values_1[i]);
                        hpcProjectLog(bmu_values_2[i]);
                        hpcProjectLog(bmu_values_3[i]);
                        hpcProjectLog(bmu_values_4[i]);

                        min_values[0] = std::min(min_values[0], bmu_values_1[i]);
                        min_values[1] = std::min(min_values[1], bmu_values_2[i]);
                        min_values[2] = std::min(min_values[2], bmu_values_3[i]);
                        min_values[3] = std::min(min_values[3], bmu_values_4[i]);

                        if (min_values[0] == bmu_values_1[i]) {
                            min_values_indexes[0] = bmu_indexes_1[i];
                        }

                        if (min_values[1] == bmu_values_2[i]) {
                            min_values_indexes[1] = bmu_indexes_2[i];
                        }

                        if (min_values[2] == bmu_values_3[i]) {
                            min_values_indexes[2] = bmu_indexes_3[i];
                        }

                        if (min_values[3] == bmu_values_4[i]) {
                            min_values_indexes[3] = bmu_indexes_4[i];
                        }
                    }

                    // Delete the four vectors/arrays, previously defined, to keep the values of
                    // the BMUs (Best Matching Units) for each block, on the memory of the CPU
                    delete[] bmu_values_1.data();
                    delete[] bmu_values_2.data();
                    delete[] bmu_values_3.data();
                    delete[] bmu_values_4.data();

                    // Delete the four vectors/arrays, previously defined,
                    // to keep the indexes of the values of
                    // the BMUs (Best Matching Units) for each block, on the memory of the CPU
                    delete[] bmu_indexes_1.data();
                    delete[] bmu_indexes_2.data();
                    delete[] bmu_indexes_3.data();
                    delete[] bmu_indexes_4.data();

                    // The final four indexes of the minimum global values of
                    // the BMUs (Best Matching Units) of all reviewed blocks
                    vector<int> bmu_final_1(1);
                    vector<int> bmu_final_2(1);
                    vector<int> bmu_final_3(1);
                    vector<int> bmu_final_4(1);
                    bmu_final_1[0] = min_values_indexes[0];
                    bmu_final_2[0] = min_values_indexes[1];
                    bmu_final_3[0] = min_values_indexes[2];
                    bmu_final_4[0] = min_values_indexes[3];

                    #ifdef DEBUG
                        hpcProjectLog("bmu to " << obs[0] << " is at: " << bmu_final_1[0] << "\n");
                        hpcProjectLog("bmu to " << obs[1] << " is at: " << bmu_final_2[0] << "\n");
                        hpcProjectLog("bmu to " << obs[2] << " is at: " << bmu_final_3[0] << "\n");
                        hpcProjectLog("bmu to " << obs[3] << " is at: " << bmu_final_4[0] << "\n");
                    #endif

                    // Free the previously four allocated memories on the GPU for the
                    // distances' matrices
                    cudaFree(&distances_gpu_1);
                    cudaFree(&distances_gpu_2);
                    cudaFree(&distances_gpu_3);
                    cudaFree(&distances_gpu_4);

                    // Free the previously four allocated memories on the GPU for the
                    // distances' matrices as Host
                    cudaFree(&distances_host_1);
                    cudaFree(&distances_host_2);
                    cudaFree(&distances_host_3);
                    cudaFree(&distances_host_4);

                    // Free the previously four allocated memories on the GPU for the
                    // values of the BMUs (Best Matching Units) for each block
                    cudaFree(&bmu_values_gpu_1);
                    cudaFree(&bmu_values_gpu_2);
                    cudaFree(&bmu_values_gpu_3);
                    cudaFree(&bmu_values_gpu_4);

                    // Free the previously four allocated memories on the GPU for the
                    // values of the BMUs (Best Matching Units) for each block as Host
                    cudaFree(&bmu_values_host_1);
                    cudaFree(&bmu_values_host_2);
                    cudaFree(&bmu_values_host_3);
                    cudaFree(&bmu_values_host_4);

                    // Free the previously four allocated memories on the GPU for the
                    // indexes of the values of the BMUs (Best Matching Units) for each block
                    cudaFree(&bmu_indexes_gpu_1);
                    cudaFree(&bmu_indexes_gpu_2);
                    cudaFree(&bmu_indexes_gpu_3);
                    cudaFree(&bmu_indexes_gpu_4);

                    // Free the previously four allocated memories on the GPU for the
                    // indexes of the values of the BMUs (Best Matching Units) for each block as Host
                    cudaFree(&bmu_indexes_host_1);
                    cudaFree(&bmu_indexes_host_2);
                    cudaFree(&bmu_indexes_host_3);
                    cudaFree(&bmu_indexes_host_4);

                    // The four vectors/arrays of the indexes of the minimum global values of
                    // the BMUs (Best Matching Units) of all reviewed blocks
                    // to be kept in the Device's memory (GPU)
                    int *bmu_final_gpu_1;
                    int *bmu_final_gpu_2;
                    int *bmu_final_gpu_3;
                    int *bmu_final_gpu_4;

                    // The four vectors/arrays of the indexes of the minimum global values of
                    // the BMUs (Best Matching Units) of all reviewed blocks
                    // to be kept in the Device's memory (GPU) as Host
                    int *bmu_final_host_1;
                    int *bmu_final_host_2;
                    int *bmu_final_host_3;
                    int *bmu_final_host_4;

                    // Allocate four memories on the GPU for the vectors/arrays of
                    // the indexes of the minimum global values of
                    // the BMUs (Best Matching Units) of all reviewed blocks
                    cudaMalloc(&bmu_final_gpu_1, sizeof(int));
                    cudaMalloc(&bmu_final_gpu_2, sizeof(int));
                    cudaMalloc(&bmu_final_gpu_3, sizeof(int));
                    cudaMalloc(&bmu_final_gpu_4, sizeof(int));

                    // Allocate four memories on the GPU for the vectors/arrays of
                    // the indexes of the minimum global values of
                    // the BMUs (Best Matching Units) of all reviewed blocks as Host
                    cudaMallocHost(&bmu_final_host_1, sizeof(int));
                    cudaMallocHost(&bmu_final_host_2, sizeof(int));
                    cudaMallocHost(&bmu_final_host_3, sizeof(int));
                    cudaMallocHost(&bmu_final_host_4, sizeof(int));

                    // Copy the data of the indexes of the four minimum global values of
                    // the BMUs (Best Matching Units) of all reviewed blocks,
                    // from the CPU's memory (Host) to the GPU's memory (Device) as Host
                    cudaMemcpyAsync(bmu_final_host_1, bmu_final_1.data(), sizeof(int), cudaMemcpyHostToDevice);
                    cudaMemcpyAsync(bmu_final_host_2, bmu_final_2.data(), sizeof(int), cudaMemcpyHostToDevice);
                    cudaMemcpyAsync(bmu_final_host_3, bmu_final_3.data(), sizeof(int), cudaMemcpyHostToDevice);
                    cudaMemcpyAsync(bmu_final_host_4, bmu_final_4.data(), sizeof(int), cudaMemcpyHostToDevice);

                    // Copy the data of the indexes of the four minimum global values of
                    // the BMUs (Best Matching Units) of all reviewed blocks,
                    // from the GPU's memory (Device) as Host to the GPU's memory (Device)
                    cudaMemcpyAsync(bmu_final_gpu_1, bmu_final_host_1, sizeof(int), cudaMemcpyHostToDevice, s1);
                    cudaMemcpyAsync(bmu_final_gpu_2, bmu_final_host_2, sizeof(int), cudaMemcpyHostToDevice, s2);
                    cudaMemcpyAsync(bmu_final_gpu_3, bmu_final_host_3, sizeof(int), cudaMemcpyHostToDevice, s3);
                    cudaMemcpyAsync(bmu_final_gpu_4, bmu_final_host_4, sizeof(int), cudaMemcpyHostToDevice, s4);

                    // Allocate four memories on the GPU for the neighborhood's matrices
                    cudaMalloc(&neighborhood_gpu_1, distances_map_size_bytes);
                    cudaMalloc(&neighborhood_gpu_2, distances_map_size_bytes);
                    cudaMalloc(&neighborhood_gpu_3, distances_map_size_bytes);
                    cudaMalloc(&neighborhood_gpu_4, distances_map_size_bytes);

                    // Launch four GPU's kernels to calculate the neighborhood's matrices, in parallel
                    neighborhood_function << < number_blocks, block_size, 0, s1 >> >
                            (this->iteration, this->number_features, bmu_final_gpu_1,
                             this->dr.get_number_observations(), distances_map_size,
                             this->max_distance, neighborhood_gpu_1);
                    neighborhood_function << < number_blocks, block_size, 0, s2 >> >
                            ((this->iteration + 1), this->number_features, bmu_final_gpu_2,
                              this->dr.get_number_observations(), distances_map_size,
                              this->max_distance, neighborhood_gpu_2);
                    neighborhood_function << < number_blocks, block_size, 0, s3 >> >
                            ((this->iteration + 2), this->number_features, bmu_final_gpu_3,
                              this->dr.get_number_observations(), distances_map_size,
                              this->max_distance, neighborhood_gpu_3);
                    neighborhood_function << < number_blocks, block_size, 0, s4 >> >
                            ((this->iteration + 3), this->number_features, bmu_final_gpu_4,
                              this->dr.get_number_observations(), distances_map_size,
                              this->max_distance, neighborhood_gpu_4);

                    // Free the previously allocated memory on the GPU for the vectors/arrays of
                    // the four indexes of the minimum global values of
                    // the BMUs (Best Matching Units) of all reviewed blocks
                    cudaFree(&bmu_final_gpu_1);
                    cudaFree(&bmu_final_gpu_2);
                    cudaFree(&bmu_final_gpu_3);
                    cudaFree(&bmu_final_gpu_4);

                    // Free the previously allocated memory on the GPU for the vectors/arrays of
                    // the four indexes of the minimum global values of
                    // the BMUs (Best Matching Units) of all reviewed blocks as Host
                    cudaFree(&bmu_final_host_1);
                    cudaFree(&bmu_final_host_2);
                    cudaFree(&bmu_final_host_3);
                    cudaFree(&bmu_final_host_4);

                    #ifdef DEBUG
                        vector<float> neighborhood_1 (this->number_rows * this->number_cols);
                        cudaMemcpyAsync(neighborhood_host_1, neighborhood_gpu_1, ncols * nrows * sizeof(float), cudaMemcpyDeviceToHost, s1);
                        cudaMemcpyAsync(neighborhood_1.data(), neighborhood_host_1, ncols * nrows * sizeof(float), cudaMemcpyDeviceToHost, s1);

                        hpcProjectLog("neighborhood to " << bmu_final_1[0] << " is: " << neighborhood_1.data() << "\n");


                        vector<float> neighborhood_2 (this->number_rows * this->number_cols);
                        cudaMemcpyAsync(neighborhood_host_2, neighborhood_gpu_2, ncols * nrows * sizeof(float), cudaMemcpyDeviceToHost, s2);
                        cudaMemcpyAsync(neighborhood_2.data(), neighborhood_host_2, ncols * nrows * sizeof(float), cudaMemcpyDeviceToHost, s2);

                        hpcProjectLog("neighborhood to " << bmu_final_2[0] << " is: " << neighborhood_2.data() << "\n");


                        vector<float> neighborhood_3 (this->number_rows * this->number_cols);
                        cudaMemcpyAsync(neighborhood_host_3, neighborhood_gpu_3, ncols * nrows * sizeof(float), cudaMemcpyDeviceToHost, s3);
                        cudaMemcpyAsync(neighborhood_3.data(), neighborhood_host_3, ncols * nrows * sizeof(float), cudaMemcpyDeviceToHost, s3);

                        hpcProjectLog("neighborhood to " << bmu_final_3[0] << " is: " << neighborhood_3.data() << "\n");


                        vector<float> neighborhood_4 (this->number_rows * this->number_cols);
                        cudaMemcpyAsync(neighborhood_host_4, neighborhood_gpu_4, ncols * nrows * sizeof(float), cudaMemcpyDeviceToHost, s4);
                        cudaMemcpyAsync(neighborhood_4.data(), neighborhood_host_4, ncols * nrows * sizeof(float), cudaMemcpyDeviceToHost, s4);

                        hpcProjectLog("neighborhood to " << bmu_final_4[0] << " is: " << neighborhood_4.data() << "\n");
                    #endif

                    // Launch four GPU's kernels to update the general maps of observations, in parallel
                    update_map << < number_blocks, block_size, 0, s1 >> >
                            (this->iteration, this->number_features, obs_gpu_1, map_size, neighborhood_gpu_1, map_gpu);
                    update_map << < number_blocks, block_size, 0, s2 >> >
                            ((this->iteration + 1), this->number_features, obs_gpu_2, map_size, neighborhood_gpu_2, map_gpu);
                    update_map << < number_blocks, block_size, 0, s3 >> >
                            ((this->iteration + 2), this->number_features, obs_gpu_3, map_size, neighborhood_gpu_3, map_gpu);
                    update_map << < number_blocks, block_size, 0, s4 >> >
                            ((this->iteration + 3), this->number_features, obs_gpu_4, map_size, neighborhood_gpu_4, map_gpu);


                    #ifdef DEBUG
                        const unsigned map_size_bytes = this->number_rows * this->number_cols * observation_size;
                        cudaMemcpy(this->map.data(), map_gpu, map_size_bytes, cudaMemcpyDeviceToHost);
                        hpcProjectLog("the updated map given the obs [" << obs << "] and the bmu [" << bmu << "] is " << map "\n");
                    #endif

                    // Free the previously allocated memory on the GPU for
                    // the general map of observations
                    cudaFree(&map_gpu);

                    // Free the previously four allocated memories on the GPU for
                    // the current observations which are being analysed
                    cudaFree(&obs_gpu_1);
                    cudaFree(&obs_gpu_2);
                    cudaFree(&obs_gpu_3);
                    cudaFree(&obs_gpu_4);

                    // Free the previously four allocated memories on the GPU for
                    // the current observations which are being analysed as Host
                    cudaFree(&obs_host_1);
                    cudaFree(&obs_host_2);
                    cudaFree(&obs_host_3);
                    cudaFree(&obs_host_4);

                    // Free the previously four allocated memories on the GPU for
                    // the neighborhood's matrices as Host
                    cudaFree(&neighborhood_gpu_1);
                    cudaFree(&neighborhood_gpu_2);
                    cudaFree(&neighborhood_gpu_3);
                    cudaFree(&neighborhood_gpu_4);

                    // Free the previously four allocated memories on the GPU for
                    // the neighborhood's matrices as Host
                    cudaFree(&neighborhood_host_1);
                    cudaFree(&neighborhood_host_2);
                    cudaFree(&neighborhood_host_3);
                    cudaFree(&neighborhood_host_4);
                }

                // If it's possible process three observations in parallel
                else if (obs->size() == 3) {

                    // Three CUDA's Streams for parallelism
                    cudaStream_t s1, s2, s3;

                    // Create three CUDA's Streams for parallelism
                    cudaStreamCreate(&s1);
                    cudaStreamCreate(&s2);
                    cudaStreamCreate(&s3);

                    // Allocate memory on the GPU for the current three observations,
                    // which are being analysed
                    cudaMalloc(&obs_gpu_1, observation_size);
                    cudaMalloc(&obs_gpu_2, observation_size);
                    cudaMalloc(&obs_gpu_3, observation_size);

                    // Allocate memory on the GPU for the current three observations,
                    // which are being analysed as Host
                    cudaMallocHost(&obs_host_1, observation_size);
                    cudaMallocHost(&obs_host_2, observation_size);
                    cudaMallocHost(&obs_host_3, observation_size);

                    // Copy the data of the current three observations, which are being analysed,
                    // from the CPU's memory (Host) to the GPU's memory (Device)
                    cudaMemcpyAsync(obs_host_1, obs[0].data(), observation_size, cudaMemcpyHostToDevice, s1);
                    cudaMemcpyAsync(obs_host_2, obs[1].data(), observation_size, cudaMemcpyHostToDevice, s2);
                    cudaMemcpyAsync(obs_host_3, obs[2].data(), observation_size, cudaMemcpyHostToDevice, s3);

                    // Copy the data of the current three observations, which are being analysed,
                    // from the GPU's memory (Device) as Host to the GPU's memory (Device)
                    cudaMemcpyAsync(obs_gpu_1, obs_host_1, observation_size, cudaMemcpyHostToDevice, s1);
                    cudaMemcpyAsync(obs_gpu_2, obs_host_2, observation_size, cudaMemcpyHostToDevice, s2);
                    cudaMemcpyAsync(obs_gpu_3, obs_host_3, observation_size, cudaMemcpyHostToDevice, s3);

                    // Allocate memory on the GPU for the current three distances' matrices,
                    // which are being processed
                    cudaMalloc(&distances_gpu_1, distances_map_size_bytes);
                    cudaMalloc(&distances_gpu_2, distances_map_size_bytes);
                    cudaMalloc(&distances_gpu_3, distances_map_size_bytes);

                    // Launch three GPU's kernels to calculate the distances' matrices, in parallel
                    distance_function << < number_blocks, block_size, 0, s1 >> >
                            (map_gpu, obs_gpu_1, this->number_features, distances_map_size, distances_gpu_1);
                    distance_function << < number_blocks, block_size, 0, s2 >> >
                            (map_gpu, obs_gpu_2, this->number_features, distances_map_size, distances_gpu_2);
                    distance_function << < number_blocks, block_size, 0, s3 >> >
                            (map_gpu, obs_gpu_3, this->number_features, distances_map_size, distances_gpu_3);

                    #ifdef DEBUG
                        vector<float> distances_1 (distances_size);
                        cudaMemcpyAsync(distances_1.data(), distances_gpu_1, distances_map_size, cudaMemcpyDeviceToHost, s1);

                        hpcProjectLog("distance to " << obs[0] << ": " << distances_1 << "\n");


                        vector<float> distances_2 (distances_size);
                        cudaMemcpyAsync(distances_2.data(), distances_gpu_2, distances_map_size, cudaMemcpyDeviceToHost, s2);

                        hpcProjectLog("distance to " << obs[1] << ": " << distances_2 << "\n");


                        vector<float> distances_3 (distances_size);
                        cudaMemcpyAsync(distances_3.data(), distances_gpu_3, distances_map_size, cudaMemcpyDeviceToHost, s3);

                        hpcProjectLog("distance to " << obs[2] << ": " << distances_3 << "\n");
                    #endif

                    // Allocate memory on the GPU for the values of
                    // the three BMUs (Best Matching Units) for each block
                    cudaMalloc(&bmu_values_gpu_1, number_blocks * sizeof(float));
                    cudaMalloc(&bmu_values_gpu_2, number_blocks * sizeof(float));
                    cudaMalloc(&bmu_values_gpu_3, number_blocks * sizeof(float));

                    // Allocate memory on the GPU for the values of
                    // the three BMUs (Best Matching Units) for each block as Host
                    cudaMallocHost(&bmu_values_host_1, number_blocks * sizeof(float));
                    cudaMallocHost(&bmu_values_host_2, number_blocks * sizeof(float));
                    cudaMallocHost(&bmu_values_host_3, number_blocks * sizeof(float));

                    // Allocate memory on the GPU for the indexes of
                    // the three BMUs (Best Matching Units) for each block
                    cudaMalloc(&bmu_indexes_gpu_1, number_blocks * sizeof(int));
                    cudaMalloc(&bmu_indexes_gpu_2, number_blocks * sizeof(int));
                    cudaMalloc(&bmu_indexes_gpu_3, number_blocks * sizeof(int));

                    // Allocate memory on the GPU for the indexes of
                    // the three BMUs (Best Matching Units) for each block as Host
                    cudaMallocHost(&bmu_indexes_host_1, number_blocks * sizeof(int));
                    cudaMallocHost(&bmu_indexes_host_2, number_blocks * sizeof(int));
                    cudaMallocHost(&bmu_indexes_host_3, number_blocks * sizeof(int));

                    // Launch three GPU's kernels to calculate the BMUs (Best Matching Units) for each block, in parallel
                    arg_min << < number_blocks, block_size, 0, s1 >> >
                            (distances_gpu_1, distances_map_size, bmu_values_gpu_1, bmu_indexes_gpu_1);
                    arg_min << < number_blocks, block_size, 0, s2 >> >
                            (distances_gpu_2, distances_map_size, bmu_values_gpu_2, bmu_indexes_gpu_2);
                    arg_min << < number_blocks, block_size, 0, s3 >> >
                            (distances_gpu_3, distances_map_size, bmu_values_gpu_3, bmu_indexes_gpu_3);

                    // The vectors/arrays to keep the values of
                    // the three BMUs (Best Matching Units) for each block, on the memory of the CPU
                    vector<float> bmu_values_1(number_blocks);
                    vector<float> bmu_values_2(number_blocks);
                    vector<float> bmu_values_3(number_blocks);

                    // The vectors/arrays to keep the indexes of the values of
                    // the three BMUs (Best Matching Units) for each block, on the memory of the CPU
                    vector<int> bmu_indexes_1(number_blocks);
                    vector<int> bmu_indexes_2(number_blocks);
                    vector<int> bmu_indexes_3(number_blocks);

                    // Copy the data of the three values of the BMUs (Best Matching Units) for each block,
                    // from the GPU's memory (Device) to the GPU's memory (Device) as Host
                    cudaMemcpyAsync(bmu_values_host_1, bmu_values_gpu_1, (number_blocks * sizeof(float)),
                              cudaMemcpyDeviceToHost, s1);
                    cudaMemcpyAsync(bmu_values_host_2, bmu_values_gpu_2, (number_blocks * sizeof(float)),
                              cudaMemcpyDeviceToHost, s2);
                    cudaMemcpyAsync(bmu_values_host_3, bmu_values_gpu_3, (number_blocks * sizeof(float)),
                              cudaMemcpyDeviceToHost, s3);

                    // Copy the data of the three values of the BMUs (Best Matching Units) for each block,
                    // from the GPU's memory (Device) to the CPU's memory (Host)
                    cudaMemcpyAsync(bmu_values_1.data(), bmu_values_host_1, (number_blocks * sizeof(float)),
                              cudaMemcpyDeviceToHost, s1);
                    cudaMemcpyAsync(bmu_values_2.data(), bmu_values_host_2, (number_blocks * sizeof(float)),
                              cudaMemcpyDeviceToHost, s2);
                    cudaMemcpyAsync(bmu_values_3.data(), bmu_values_host_3, (number_blocks * sizeof(float)),
                              cudaMemcpyDeviceToHost, s3);

                    // Copy the data of the three indexes of the BMUs (Best Matching Units) for each block,
                    // from the GPU's memory (Device) to the GPU's memory (Device) as Host
                    cudaMemcpyAsync(bmu_indexes_host_1, bmu_indexes_gpu_1, (number_blocks * sizeof(int)),
                              cudaMemcpyDeviceToHost, s1);
                    cudaMemcpyAsync(bmu_indexes_host_2, bmu_indexes_gpu_2, (number_blocks * sizeof(int)),
                              cudaMemcpyDeviceToHost, s2);
                    cudaMemcpyAsync(bmu_indexes_host_3, bmu_indexes_gpu_3, (number_blocks * sizeof(int)),
                              cudaMemcpyDeviceToHost, s3);

                    // Copy the data of the three indexes of the BMUs (Best Matching Units) for each block,
                    // from the GPU's memory (Device) to the CPU's memory (Host)
                    cudaMemcpyAsync(bmu_indexes_1.data(), bmu_indexes_host_1, (number_blocks * sizeof(int)),
                              cudaMemcpyDeviceToHost, s1);
                    cudaMemcpyAsync(bmu_indexes_2.data(), bmu_indexes_host_2, (number_blocks * sizeof(int)),
                              cudaMemcpyDeviceToHost, s2);
                    cudaMemcpyAsync(bmu_indexes_3.data(), bmu_indexes_host_3, (number_blocks * sizeof(int)),
                              cudaMemcpyDeviceToHost, s3);

                    // The minimum three global values of
                    // the BMUs (Best Matching Units) of all reviewed blocks
                    vector<float> min_values(3);
                    min_values[0] = std::numeric_limits<float>::max();
                    min_values[1] = std::numeric_limits<float>::max();
                    min_values[2] = std::numeric_limits<float>::max();

                    // The indexes of the minimum three global values of
                    // the BMUs (Best Matching Units) of all reviewed blocks
                    vector<int> min_values_indexes(3);
                    min_values_indexes[0] = -1;
                    min_values_indexes[1] = -1;
                    min_values_indexes[2] = -1;

                    // Loop to find the minimum three global values of
                    // the BMUs (Best Matching Units) of all reviewed blocks,
                    // and the respectively indexes
                    for (int i = 0; i < block_size; i++) {
                        hpcProjectLog(bmu_values_1[i]);
                        hpcProjectLog(bmu_values_2[i]);
                        hpcProjectLog(bmu_values_3[i]);

                        min_values[0] = std::min(min_values[0], bmu_values_1[i]);
                        min_values[1] = std::min(min_values[1], bmu_values_2[i]);
                        min_values[2] = std::min(min_values[2], bmu_values_3[i]);

                        if (min_values[0] == bmu_values_1[i]) {
                            min_values_indexes[0] = bmu_indexes_1[i];
                        }

                        if (min_values[1] == bmu_values_2[i]) {
                            min_values_indexes[1] = bmu_indexes_2[i];
                        }

                        if (min_values[2] == bmu_values_3[i]) {
                            min_values_indexes[2] = bmu_indexes_3[i];
                        }
                    }

                    // Delete the three vectors/arrays, previously defined, to keep the values of
                    // the BMUs (Best Matching Units) for each block, on the memory of the CPU
                    delete[] bmu_values_1.data();
                    delete[] bmu_values_2.data();
                    delete[] bmu_values_3.data();

                    // Delete the three vectors/arrays, previously defined,
                    // to keep the indexes of the values of
                    // the BMUs (Best Matching Units) for each block, on the memory of the CPU
                    delete[] bmu_indexes_1.data();
                    delete[] bmu_indexes_2.data();
                    delete[] bmu_indexes_3.data();

                    // The final three indexes of the minimum global values of
                    // the BMUs (Best Matching Units) of all reviewed blocks
                    vector<int> bmu_final_1(1);
                    vector<int> bmu_final_2(1);
                    vector<int> bmu_final_3(1);
                    bmu_final_1[0] = min_values_indexes[0];
                    bmu_final_2[0] = min_values_indexes[1];
                    bmu_final_3[0] = min_values_indexes[2];

                    #ifdef DEBUG
                        hpcProjectLog("bmu to " << obs[0] << " is at: " << bmu_final_1[0] << "\n");
                        hpcProjectLog("bmu to " << obs[1] << " is at: " << bmu_final_2[0] << "\n");
                        hpcProjectLog("bmu to " << obs[2] << " is at: " << bmu_final_3[0] << "\n");
                    #endif

                    // Free the previously three allocated memories on the GPU for the
                    // distances' matrices
                    cudaFree(&distances_gpu_1);
                    cudaFree(&distances_gpu_2);
                    cudaFree(&distances_gpu_3);

                    // Free the previously three allocated memories on the GPU for the
                    // distances' matrices as Host
                    cudaFree(&distances_host_1);
                    cudaFree(&distances_host_2);
                    cudaFree(&distances_host_3);

                    // Free the previously three allocated memories on the GPU for the
                    // values of the BMUs (Best Matching Units) for each block
                    cudaFree(&bmu_values_gpu_1);
                    cudaFree(&bmu_values_gpu_2);
                    cudaFree(&bmu_values_gpu_3);

                    // Free the previously three allocated memories on the GPU for the
                    // values of the BMUs (Best Matching Units) for each block as Host
                    cudaFree(&bmu_values_host_1);
                    cudaFree(&bmu_values_host_2);
                    cudaFree(&bmu_values_host_3);

                    // Free the previously three allocated memories on the GPU for the
                    // indexes of the values of the BMUs (Best Matching Units) for each block
                    cudaFree(&bmu_indexes_gpu_1);
                    cudaFree(&bmu_indexes_gpu_2);
                    cudaFree(&bmu_indexes_gpu_3);

                    // Free the previously three allocated memories on the GPU for the
                    // indexes of the values of the BMUs (Best Matching Units) for each block as Host
                    cudaFree(&bmu_indexes_host_1);
                    cudaFree(&bmu_indexes_host_2);
                    cudaFree(&bmu_indexes_host_3);

                    // The three vectors/arrays of the indexes of the minimum global values of
                    // the BMUs (Best Matching Units) of all reviewed blocks
                    // to be kept in the Device's memory (GPU)
                    int *bmu_final_gpu_1;
                    int *bmu_final_gpu_2;
                    int *bmu_final_gpu_3;

                    // The three vectors/arrays of the indexes of the minimum global values of
                    // the BMUs (Best Matching Units) of all reviewed blocks
                    // to be kept in the Device's memory (GPU) as Host
                    int *bmu_final_host_1;
                    int *bmu_final_host_2;
                    int *bmu_final_host_3;

                    // Allocate three memories on the GPU for the vectors/arrays of
                    // the indexes of the minimum global values of
                    // the BMUs (Best Matching Units) of all reviewed blocks
                    cudaMalloc(&bmu_final_gpu_1, sizeof(int));
                    cudaMalloc(&bmu_final_gpu_2, sizeof(int));
                    cudaMalloc(&bmu_final_gpu_3, sizeof(int));

                    // Allocate three memories on the GPU for the vectors/arrays of
                    // the indexes of the minimum global values of
                    // the BMUs (Best Matching Units) of all reviewed blocks as Host
                    cudaMallocHost(&bmu_final_host_1, sizeof(int));
                    cudaMallocHost(&bmu_final_host_2, sizeof(int));
                    cudaMallocHost(&bmu_final_host_3, sizeof(int));

                    // Copy the data of the indexes of the three minimum global values of
                    // the BMUs (Best Matching Units) of all reviewed blocks,
                    // from the CPU's memory (Host) to the GPU's memory (Device) as Host
                    cudaMemcpyAsync(bmu_final_host_1, bmu_final_1.data(), sizeof(int), cudaMemcpyHostToDevice, s1);
                    cudaMemcpyAsync(bmu_final_host_2, bmu_final_2.data(), sizeof(int), cudaMemcpyHostToDevice, s2);
                    cudaMemcpyAsync(bmu_final_host_3, bmu_final_3.data(), sizeof(int), cudaMemcpyHostToDevice, s3);

                    // Copy the data of the indexes of the three minimum global values of
                    // the BMUs (Best Matching Units) of all reviewed blocks,
                    // from the GPU's memory (Device) as Host to the GPU's memory (Device)
                    cudaMemcpyAsync(bmu_final_gpu_1, bmu_final_host_1, sizeof(int), cudaMemcpyHostToDevice, s1);
                    cudaMemcpyAsync(bmu_final_gpu_2, bmu_final_host_2, sizeof(int), cudaMemcpyHostToDevice, s2);
                    cudaMemcpyAsync(bmu_final_gpu_3, bmu_final_host_3, sizeof(int), cudaMemcpyHostToDevice, s3);

                    // Allocate three memories on the GPU for the neighborhood's matrices
                    cudaMalloc(&neighborhood_gpu_1, distances_map_size_bytes);
                    cudaMalloc(&neighborhood_gpu_2, distances_map_size_bytes);
                    cudaMalloc(&neighborhood_gpu_3, distances_map_size_bytes);

                    // Launch three GPU's kernels to calculate the neighborhood's matrices, in parallel
                    neighborhood_function << < number_blocks, block_size, 0, s1 >> >
                            (this->iteration, this->number_features, bmu_final_gpu_1,
                             this->dr.get_number_observations(), distances_map_size,
                             this->max_distance, neighborhood_gpu_1);
                    neighborhood_function << < number_blocks, block_size, 0, s2 >> >
                            ((this->iteration + 1), this->number_features, bmu_final_gpu_2,
                              this->dr.get_number_observations(), distances_map_size,
                              this->max_distance, neighborhood_gpu_2);
                    neighborhood_function << < number_blocks, block_size, 0, s3 >> >
                            ((this->iteration + 2), this->number_features, bmu_final_gpu_3,
                              this->dr.get_number_observations(), distances_map_size,
                              this->max_distance, neighborhood_gpu_3);

                    // Free the previously allocated memory on the GPU for the vectors/arrays of
                    // the three indexes of the minimum global values of
                    // the BMUs (Best Matching Units) of all reviewed blocks
                    cudaFree(&bmu_final_gpu_1);
                    cudaFree(&bmu_final_gpu_2);
                    cudaFree(&bmu_final_gpu_3);

                    // Free the previously allocated memory on the GPU for the vectors/arrays of
                    // the three indexes of the minimum global values of
                    // the BMUs (Best Matching Units) of all reviewed blocks as Host
                    cudaFree(&bmu_final_host_1);
                    cudaFree(&bmu_final_host_2);
                    cudaFree(&bmu_final_host_3);

                    #ifdef DEBUG
                        vector<float> neighborhood_1 (this->number_rows * this->number_cols);
                        cudaMemcpyAsync(neighborhood_host_1, neighborhood_gpu_1, ncols * nrows * sizeof(float), cudaMemcpyDeviceToHost, s1);
                        cudaMemcpyAsync(neighborhood_1.data(), neighborhood_host_1, ncols * nrows * sizeof(float), cudaMemcpyDeviceToHost, s1);

                        hpcProjectLog("neighborhood to " << bmu_final_1[0] << " is: " << neighborhood_1.data() << "\n");


                        vector<float> neighborhood_2 (this->number_rows * this->number_cols);
                        cudaMemcpyAsync(neighborhood_host_2, neighborhood_gpu_2, ncols * nrows * sizeof(float), cudaMemcpyDeviceToHost, s2);
                        cudaMemcpyAsync(neighborhood_2.data(), neighborhood_host_2, ncols * nrows * sizeof(float), cudaMemcpyDeviceToHost, s2);

                        hpcProjectLog("neighborhood to " << bmu_final_2[0] << " is: " << neighborhood_2.data() << "\n");


                        vector<float> neighborhood_3 (this->number_rows * this->number_cols);
                        cudaMemcpyAsync(neighborhood_host_3, neighborhood_gpu_3, ncols * nrows * sizeof(float), cudaMemcpyDeviceToHost, s3);
                        cudaMemcpyAsync(neighborhood_3.data(), neighborhood_host_3, ncols * nrows * sizeof(float), cudaMemcpyDeviceToHost, s3);

                        hpcProjectLog("neighborhood to " << bmu_final_3[0] << " is: " << neighborhood_3.data() << "\n");

                #endif

                // Launch three GPU's kernels to update the general maps of observations, in parallel
                update_map << < number_blocks, block_size, 0, s1 >> >
                        (this->iteration, this->number_features, obs_gpu_1, map_size, neighborhood_gpu_1, map_gpu);
                update_map << < number_blocks, block_size, 0, s2 >> >
                        ((this->iteration + 1), this->number_features, obs_gpu_2, map_size, neighborhood_gpu_2, map_gpu);
                update_map << < number_blocks, block_size, 0, s3 >> >
                        ((this->iteration + 2), this->number_features, obs_gpu_3, map_size, neighborhood_gpu_3, map_gpu);

                #ifdef DEBUG
                    const unsigned map_size_bytes = this->number_rows * this->number_cols * observation_size;
                    cudaMemcpy(this->map.data(), map_gpu, map_size_bytes, cudaMemcpyDeviceToHost);
                    hpcProjectLog("the updated map given the obs [" << obs << "] and the bmu [" << bmu << "] is " << map "\n");
                #endif

                // Free the previously allocated memory on the GPU for
                // the general map of observations
                cudaFree(&map_gpu);

                // Free the previously three allocated memories on the GPU for
                // the current observations which are being analysed
                cudaFree(&obs_gpu_1);
                cudaFree(&obs_gpu_2);
                cudaFree(&obs_gpu_3);

                // Free the previously three allocated memories on the GPU for
                // the current observations which are being analysed as Host
                cudaFree(&obs_host_1);
                cudaFree(&obs_host_2);
                cudaFree(&obs_host_3);

                // Free the previously three allocated memories on the GPU for
                // the neighborhood's matrices as Host
                cudaFree(&neighborhood_gpu_1);
                cudaFree(&neighborhood_gpu_2);
                cudaFree(&neighborhood_gpu_3);

                // Free the previously three allocated memories on the GPU for
                // the neighborhood's matrices as Host
                cudaFree(&neighborhood_host_1);
                cudaFree(&neighborhood_host_2);
                cudaFree(&neighborhood_host_3);
            }

            // If it's possible process two observations in parallel
            else if (obs->size() == 2) {

                // Two CUDA's Streams for parallelism
                cudaStream_t s1, s2;

                // Create two CUDA's Streams for parallelism
                cudaStreamCreate(&s1);
                cudaStreamCreate(&s2);

                // Allocate memory on the GPU for the current two observations,
                // which are being analysed
                cudaMalloc(&obs_gpu_1, observation_size);
                cudaMalloc(&obs_gpu_2, observation_size);

                // Allocate memory on the GPU for the current two observations,
                // which are being analysed as Host
                cudaMallocHost(&obs_host_1, observation_size);
                cudaMallocHost(&obs_host_2, observation_size);

                // Copy the data of the current two observations, which are being analysed,
                // from the CPU's memory (Host) to the GPU's memory (Device)
                cudaMemcpyAsync(obs_host_1, obs[0].data(), observation_size, cudaMemcpyHostToDevice, s1);
                cudaMemcpyAsync(obs_host_2, obs[1].data(), observation_size, cudaMemcpyHostToDevice, s2);

                // Copy the data of the current two observations, which are being analysed,
                // from the GPU's memory (Device) as Host to the GPU's memory (Device)
                cudaMemcpyAsync(obs_gpu_1, obs_host_1, observation_size, cudaMemcpyHostToDevice, s1);
                cudaMemcpyAsync(obs_gpu_2, obs_host_2, observation_size, cudaMemcpyHostToDevice, s2);

                // Allocate memory on the GPU for the current two distances' matrices,
                // which are being processed
                cudaMalloc(&distances_gpu_1, distances_map_size_bytes);
                cudaMalloc(&distances_gpu_2, distances_map_size_bytes);

                // Launch two GPU's kernels to calculate the distances' matrices, in parallel
                distance_function << < number_blocks, block_size, 0, s1 >> >
                        (map_gpu, obs_gpu_1, this->number_features, distances_map_size, distances_gpu_1);
                distance_function << < number_blocks, block_size, 0, s2 >> >
                        (map_gpu, obs_gpu_2, this->number_features, distances_map_size, distances_gpu_2);

                #ifdef DEBUG
                    vector<float> distances_1 (distances_size);
                    cudaMemcpyAsync(distances_1.data(), distances_gpu_1, distances_map_size, cudaMemcpyDeviceToHost, s1);

                    hpcProjectLog("distance to " << obs[0] << ": " << distances_1 << "\n");


                    vector<float> distances_2 (distances_size);
                    cudaMemcpyAsync(distances_2.data(), distances_gpu_2, distances_map_size, cudaMemcpyDeviceToHost, s2);

                    hpcProjectLog("distance to " << obs[1] << ": " << distances_2 << "\n");
                #endif

                // Allocate memory on the GPU for the values of
                // the two BMUs (Best Matching Units) for each block
                cudaMalloc(&bmu_values_gpu_1, number_blocks * sizeof(float));
                cudaMalloc(&bmu_values_gpu_2, number_blocks * sizeof(float));

                // Allocate memory on the GPU for the values of
                // the two BMUs (Best Matching Units) for each block as Host
                cudaMallocHost(&bmu_values_host_1, number_blocks * sizeof(float));
                cudaMallocHost(&bmu_values_host_2, number_blocks * sizeof(float));

                // Allocate memory on the GPU for the indexes of
                // the two BMUs (Best Matching Units) for each block
                cudaMalloc(&bmu_indexes_gpu_1, number_blocks * sizeof(int));
                cudaMalloc(&bmu_indexes_gpu_2, number_blocks * sizeof(int));

                // Allocate memory on the GPU for the indexes of
                // the two BMUs (Best Matching Units) for each block as Host
                cudaMallocHost(&bmu_indexes_host_1, number_blocks * sizeof(int));
                cudaMallocHost(&bmu_indexes_host_2, number_blocks * sizeof(int));

                // Launch two GPU's kernels to calculate the BMUs (Best Matching Units) for each block, in parallel
                arg_min << < number_blocks, block_size, 0, s1 >> >
                        (distances_gpu_1, distances_map_size, bmu_values_gpu_1, bmu_indexes_gpu_1);
                arg_min << < number_blocks, block_size, 0, s2 >> >
                        (distances_gpu_2, distances_map_size, bmu_values_gpu_2, bmu_indexes_gpu_2);

                // The vectors/arrays to keep the values of
                // the two BMUs (Best Matching Units) for each block, on the memory of the CPU
                vector<float> bmu_values_1(number_blocks);
                vector<float> bmu_values_2(number_blocks);

                // The vectors/arrays to keep the indexes of the values of
                // the two BMUs (Best Matching Units) for each block, on the memory of the CPU
                vector<int> bmu_indexes_1(number_blocks);
                vector<int> bmu_indexes_2(number_blocks);

                // Copy the data of the two values of the BMUs (Best Matching Units) for each block,
                // from the GPU's memory (Device) to the GPU's memory (Device) as Host
                cudaMemcpyAsync(bmu_values_host_1, bmu_values_gpu_1, (number_blocks * sizeof(float)),
                          cudaMemcpyDeviceToHost, s1);
                cudaMemcpyAsync(bmu_values_host_2, bmu_values_gpu_2, (number_blocks * sizeof(float)),
                          cudaMemcpyDeviceToHost, s2);

                // Copy the data of the two values of the BMUs (Best Matching Units) for each block,
                // from the GPU's memory (Device) to the CPU's memory (Host)
                cudaMemcpyAsync(bmu_values_1.data(), bmu_values_host_1, (number_blocks * sizeof(float)),
                          cudaMemcpyDeviceToHost, s1);
                cudaMemcpyAsync(bmu_values_2.data(), bmu_values_host_2, (number_blocks * sizeof(float)),
                          cudaMemcpyDeviceToHost, s2);

                // Copy the data of the two indexes of the BMUs (Best Matching Units) for each block,
                // from the GPU's memory (Device) to the GPU's memory (Device) as Host
                cudaMemcpyAsync(bmu_indexes_host_1, bmu_indexes_gpu_1, (number_blocks * sizeof(int)),
                          cudaMemcpyDeviceToHost, s1);
                cudaMemcpyAsync(bmu_indexes_host_2, bmu_indexes_gpu_2, (number_blocks * sizeof(int)),
                          cudaMemcpyDeviceToHost, s2);

                // Copy the data of the two indexes of the BMUs (Best Matching Units) for each block,
                // from the GPU's memory (Device) to the CPU's memory (Host)
                cudaMemcpyAsync(bmu_indexes_1.data(), bmu_indexes_host_1, (number_blocks * sizeof(int)),
                          cudaMemcpyDeviceToHost, s1);
                cudaMemcpyAsync(bmu_indexes_2.data(), bmu_indexes_host_2, (number_blocks * sizeof(int)),
                          cudaMemcpyDeviceToHost, s2);

                // The minimum two global values of
                // the BMUs (Best Matching Units) of all reviewed blocks
                vector<float> min_values(2);
                min_values[0] = std::numeric_limits<float>::max();
                min_values[1] = std::numeric_limits<float>::max();

                // The indexes of the minimum two global values of
                // the BMUs (Best Matching Units) of all reviewed blocks
                vector<int> min_values_indexes(2);
                min_values_indexes[0] = -1;
                min_values_indexes[1] = -1;

                // Loop to find the minimum two global values of
                // the BMUs (Best Matching Units) of all reviewed blocks,
                // and the respectively indexes
                for (int i = 0; i < block_size; i++) {
                    hpcProjectLog(bmu_values_1[i]);
                    hpcProjectLog(bmu_values_2[i]);

                    min_values[0] = std::min(min_values[0], bmu_values_1[i]);
                    min_values[1] = std::min(min_values[1], bmu_values_2[i]);

                    if (min_values[0] == bmu_values_1[i]) {
                        min_values_indexes[0] = bmu_indexes_1[i];
                    }

                    if (min_values[1] == bmu_values_2[i]) {
                        min_values_indexes[1] = bmu_indexes_2[i];
                    }
                }

                // Delete the two vectors/arrays, previously defined, to keep the values of
                // the BMUs (Best Matching Units) for each block, on the memory of the CPU
                delete[] bmu_values_1.data();
                delete[] bmu_values_2.data();

                // Delete the two vectors/arrays, previously defined,
                // to keep the indexes of the values of
                // the BMUs (Best Matching Units) for each block, on the memory of the CPU
                delete[] bmu_indexes_1.data();
                delete[] bmu_indexes_2.data();

                // The final two indexes of the minimum global values of
                // the BMUs (Best Matching Units) of all reviewed blocks
                vector<int> bmu_final_1(1);
                vector<int> bmu_final_2(1);
                bmu_final_1[0] = min_values_indexes[0];
                bmu_final_2[0] = min_values_indexes[1];

                #ifdef DEBUG
                    hpcProjectLog("bmu to " << obs[0] << " is at: " << bmu_final_1[0] << "\n");
                    hpcProjectLog("bmu to " << obs[1] << " is at: " << bmu_final_2[0] << "\n");
                #endif

                // Free the previously two allocated memories on the GPU for the
                // distances' matrices
                cudaFree(&distances_gpu_1);
                cudaFree(&distances_gpu_2);

                // Free the previously two allocated memories on the GPU for the
                // distances' matrices as Host
                cudaFree(&distances_host_1);
                cudaFree(&distances_host_2);

                // Free the previously two allocated memories on the GPU for the
                // values of the BMUs (Best Matching Units) for each block
                cudaFree(&bmu_values_gpu_1);
                cudaFree(&bmu_values_gpu_2);

                // Free the previously two allocated memories on the GPU for the
                // values of the BMUs (Best Matching Units) for each block as Host
                cudaFree(&bmu_values_host_1);
                cudaFree(&bmu_values_host_2);

                // Free the previously two allocated memories on the GPU for the
                // indexes of the values of the BMUs (Best Matching Units) for each block
                cudaFree(&bmu_indexes_gpu_1);
                cudaFree(&bmu_indexes_gpu_2);

                // Free the previously two allocated memories on the GPU for the
                // indexes of the values of the BMUs (Best Matching Units) for each block as Host
                cudaFree(&bmu_indexes_host_1);
                cudaFree(&bmu_indexes_host_2);

                // The two vectors/arrays of the indexes of the minimum global values of
                // the BMUs (Best Matching Units) of all reviewed blocks
                // to be kept in the Device's memory (GPU)
                int *bmu_final_gpu_1;
                int *bmu_final_gpu_2;

                // The two vectors/arrays of the indexes of the minimum global values of
                // the BMUs (Best Matching Units) of all reviewed blocks
                // to be kept in the Device's memory (GPU) as Host
                int *bmu_final_host_1;
                int *bmu_final_host_2;

                // Allocate two memories on the GPU for the vectors/arrays of
                // the indexes of the minimum global values of
                // the BMUs (Best Matching Units) of all reviewed blocks
                cudaMalloc(&bmu_final_gpu_1, sizeof(int));
                cudaMalloc(&bmu_final_gpu_2, sizeof(int));

                // Allocate two memories on the GPU for the vectors/arrays of
                // the indexes of the minimum global values of
                // the BMUs (Best Matching Units) of all reviewed blocks as Host
                cudaMallocHost(&bmu_final_host_1, sizeof(int));
                cudaMallocHost(&bmu_final_host_2, sizeof(int));

                // Copy the data of the indexes of the two minimum global values of
                // the BMUs (Best Matching Units) of all reviewed blocks,
                // from the CPU's memory (Host) to the GPU's memory (Device) as Host
                cudaMemcpyAsync(bmu_final_host_1, bmu_final_1.data(), sizeof(int), cudaMemcpyHostToDevice, s1);
                cudaMemcpyAsync(bmu_final_host_2, bmu_final_2.data(), sizeof(int), cudaMemcpyHostToDevice, s2);

                // Copy the data of the indexes of the two minimum global values of
                // the BMUs (Best Matching Units) of all reviewed blocks,
                // from the GPU's memory (Device) as Host to the GPU's memory (Device)
                cudaMemcpyAsync(bmu_final_gpu_1, bmu_final_host_1, sizeof(int), cudaMemcpyHostToDevice, s1);
                cudaMemcpyAsync(bmu_final_gpu_2, bmu_final_host_2, sizeof(int), cudaMemcpyHostToDevice, s2);

                // Allocate two memories on the GPU for the neighborhood's matrices
                cudaMalloc(&neighborhood_gpu_1, distances_map_size_bytes);
                cudaMalloc(&neighborhood_gpu_2, distances_map_size_bytes);

                // Launch two GPU's kernels to calculate the neighborhood's matrices, in parallel
                neighborhood_function << < number_blocks, block_size, 0, s1 >> >
                        (this->iteration, this->number_features, bmu_final_gpu_1,
                         this->dr.get_number_observations(), distances_map_size,
                         this->max_distance, neighborhood_gpu_1);
                neighborhood_function << < number_blocks, block_size, 0, s2 >> >
                        ((this->iteration + 1), this->number_features, bmu_final_gpu_2,
                          this->dr.get_number_observations(), distances_map_size,
                          this->max_distance, neighborhood_gpu_2);

                // Free the previously allocated memory on the GPU for the vectors/arrays of
                // the two indexes of the minimum global values of
                // the BMUs (Best Matching Units) of all reviewed blocks
                cudaFree(&bmu_final_gpu_1);
                cudaFree(&bmu_final_gpu_2);

                // Free the previously allocated memory on the GPU for the vectors/arrays of
                // the two indexes of the minimum global values of
                // the BMUs (Best Matching Units) of all reviewed blocks as Host
                cudaFree(&bmu_final_host_1);
                cudaFree(&bmu_final_host_2);

                #ifdef DEBUG
                    vector<float> neighborhood_1 (this->number_rows * this->number_cols);
                    cudaMemcpyAsync(neighborhood_host_1, neighborhood_gpu_1, ncols * nrows * sizeof(float), cudaMemcpyDeviceToHost, s1);
                    cudaMemcpyAsync(neighborhood_1.data(), neighborhood_host_1, ncols * nrows * sizeof(float), cudaMemcpyDeviceToHost, s1);

                    hpcProjectLog("neighborhood to " << bmu_final_1[0] << " is: " << neighborhood_1.data() << "\n");


                    vector<float> neighborhood_2 (this->number_rows * this->number_cols);
                    cudaMemcpyAsync(neighborhood_host_2, neighborhood_gpu_2, ncols * nrows * sizeof(float), cudaMemcpyDeviceToHost, s2);
                    cudaMemcpyAsync(neighborhood_2.data(), neighborhood_host_2, ncols * nrows * sizeof(float), cudaMemcpyDeviceToHost, s2);

                    hpcProjectLog("neighborhood to " << bmu_final_2[0] << " is: " << neighborhood_2.data() << "\n");

                #endif

                // Launch two GPU's kernels to update the general maps of observations, in parallel
                update_map << < number_blocks, block_size, 0, s1 >> >
                        (this->iteration, this->number_features, obs_gpu_1, map_size, neighborhood_gpu_1, map_gpu);
                update_map << < number_blocks, block_size, 0, s2 >> >
                        ((this->iteration + 1), this->number_features, obs_gpu_2, map_size, neighborhood_gpu_2, map_gpu);

                #ifdef DEBUG
                    const unsigned map_size_bytes = this->number_rows * this->number_cols * observation_size;
                    cudaMemcpy(this->map.data(), map_gpu, map_size_bytes, cudaMemcpyDeviceToHost);
                    hpcProjectLog("the updated map given the obs [" << obs << "] and the bmu [" << bmu << "] is " << map "\n");
                #endif

                // Free the previously allocated memory on the GPU for
                // the general map of observations
                cudaFree(&map_gpu);

                // Free the previously two allocated memories on the GPU for
                // the current observations which are being analysed
                cudaFree(&obs_gpu_1);
                cudaFree(&obs_gpu_2);

                // Free the previously two allocated memories on the GPU for
                // the current observations which are being analysed as Host
                cudaFree(&obs_host_1);
                cudaFree(&obs_host_2);

                // Free the previously two allocated memories on the GPU for
                // the neighborhood's matrices
                cudaFree(&neighborhood_gpu_1);
                cudaFree(&neighborhood_gpu_2);

                // Free the previously two allocated memories on the GPU for
                // the neighborhood's matrices as Host
                cudaFree(&neighborhood_host_1);
                cudaFree(&neighborhood_host_2);
            }

            // If it's possible process only one observation
            else {

                // Allocate memory on the GPU for the current observation,
                // which is being analysed
                cudaMalloc(&obs_gpu_1, observation_size);

                // Copy the data of the current observation, which is being analysed,
                // from the CPU's memory (Host) to the GPU's memory (Device)
                cudaMemcpy(obs_gpu_1, obs[0].data(), observation_size, cudaMemcpyHostToDevice);

                // Allocate memory on the GPU for the current distances' matrix,
                // which is being processed
                cudaMalloc(&distances_gpu_1, distances_map_size_bytes);

                // Launch a GPU's kernel to calculate the distances' matrix
                distance_function << < number_blocks, block_size >> >
                         (map_gpu, obs_gpu_1, this->number_features, distances_map_size, distances_gpu_1);

                #ifdef DEBUG
                    vector<float> distances_1 (distances_size);
                    cudaMemcpy(distances_1.data(), distances_gpu_1, distances_map_size, cudaMemcpyDeviceToHost);

                    hpcProjectLog("distance to " << obs[0] << ": " << distances_1 << "\n");
                #endif

                // Allocate memory on the GPU for the value of
                // the BMU (Best Matching Unit) for each block
                cudaMalloc(&bmu_values_gpu_1, number_blocks * sizeof(float));

                // Allocate memory on the GPU for the index of
                // the two BMU (Best Matching Unit) for each block
                cudaMalloc(&bmu_indexes_gpu_1, number_blocks * sizeof(int));

                // Launch a GPU's kernel to calculate the BMU (Best Matching Unit) for each block
                arg_min << < number_blocks, block_size >> >
                        (distances_gpu_1, distances_map_size, bmu_values_gpu_1, bmu_indexes_gpu_1);

                // The vector/array to keep the value of
                // the BMU (Best Matching Unit) for each block, on the memory of the CPU
                vector<float> bmu_values_1(number_blocks);

                // The vector/array to keep the index of the value of
                // the BMU (Best Matching Unit) for each block, on the memory of the CPU
                vector<int> bmu_indexes_1(number_blocks);

                // Copy the data of the value of the BMU (Best Matching Unit) for each block,
                // from the GPU's memory (Device) to the CPU's memory (Host)
                cudaMemcpy(bmu_values_1.data(), bmu_values_host_1, (number_blocks * sizeof(float)),
                           cudaMemcpyDeviceToHost);

                // Copy the data of the indexe of the BMU (Best Matching Unit) for each block,
                // from the GPU's memory (Device) to the CPU's memory (Host)
                cudaMemcpy(bmu_indexes_1.data(), bmu_indexes_host_1, (number_blocks * sizeof(int)),
                           cudaMemcpyDeviceToHost);

                // The minimum global value of
                // the BMU (Best Matching Unit) of all reviewed blocks
                vector<float> min_values(1);
                min_values[0] = std::numeric_limits<float>::max();

                // The indexes of the minimum global value of
                // the BMU (Best Matching Unit) of all reviewed blocks
                vector<int> min_values_indexes(1);
                min_values_indexes[0] = -1;

                // Loop to find the minimum global value of
                // the BMU (Best Matching Unit) of all reviewed blocks,
                // and the respectively index
                for (int i = 0; i < block_size; i++) {
                    hpcProjectLog(bmu_values_1[i]);

                    min_values[0] = std::min(min_values[0], bmu_values_1[i]);

                    if (min_values[0] == bmu_values_1[i]) {
                        min_values_indexes[0] = bmu_indexes_1[i];
                    }
                }

                // Delete the vector/array, previously defined, to keep the value of
                // the BMU (Best Matching Unit) for each block, on the memory of the CPU
                delete[] bmu_values_1.data();

                // Delete the vector/array, previously defined,
                // to keep the index of the value of
                // the BMU (Best Matching Unit) for each block, on the memory of the CPU
                delete[] bmu_indexes_1.data();

                // The final index of the minimum global value of
                // the BMU (Best Matching Unit) of all reviewed blocks
                vector<int> bmu_final_1(1);
                bmu_final_1[0] = min_values_indexes[0];

                #ifdef DEBUG
                    hpcProjectLog("bmu to " << obs[0] << " is at: " << bmu_final_1[0] << "\n");
                #endif

                // Free the previously allocated memory on the GPU for the
                // distances' matrix
                cudaFree(&distances_gpu_1);

                // Free the previously allocated memory on the GPU for the
                // value of the BMU (Best Matching Unit) for each block
                cudaFree(&bmu_values_gpu_1);

                // Free the previously allocated memory on the GPU for the
                // index of the value of the BMU (Best Matching Unit) for each block
                cudaFree(&bmu_indexes_gpu_1);

                // The vector/array of the index of the minimum global value of
                // the BMU (Best Matching Unit) of all reviewed blocks
                // to be kept in the Device's memory (GPU)
                int *bmu_final_gpu_1;

                // Allocate memory on the GPU for the vector/array of
                // the index of the minimum global value of
                // the BMU (Best Matching Unit) of all reviewed blocks
                cudaMalloc(&bmu_final_gpu_1, sizeof(int));

                // Copy the data of the indexes of the two minimum global values of
                // the BMUs (Best Matching Units) of all reviewed blocks,
                // from the CPU's memory (Host) to the GPU's memory (Device)
                cudaMemcpy(bmu_final_gpu_1, bmu_final_1.data(), sizeof(int), cudaMemcpyHostToDevice);

                // Allocate memory on the GPU for the neighborhood's matrix
                cudaMalloc(&neighborhood_gpu_1, distances_map_size_bytes);

                // Launch a GPU's kernel to calculate the neighborhood's matrix
                neighborhood_function << < number_blocks, block_size >> > (this->iteration, this->number_features, bmu_final_gpu_1,
                        this->dr.get_number_observations(), distances_map_size,
                        this->max_distance, neighborhood_gpu_1);

                // Free the previously allocated memory on the GPU for the vector/array of
                // the index of the minimum global value of
                // the BMU (Best Matching Unit) of all reviewed blocks
                cudaFree(&bmu_final_gpu_1);

                #ifdef DEBUG
                    vector<float> neighborhood_1 (this->number_rows * this->number_cols);
                    cudaMemcpy(neighborhood_1.data(), neighborhood_gpu_1, ncols * nrows * sizeof(float), cudaMemcpyDeviceToHost);

                    hpcProjectLog("neighborhood to " << bmu_final_1[0] << " is: " << neighborhood_1.data() << "\n");
                #endif

                // Launch a GPU's kernel to update the general map of observations
                update_map << < number_blocks, block_size >> >
                         (this->iteration, this->number_features, obs_gpu_1, map_size, neighborhood_gpu_1, map_gpu);

                #ifdef DEBUG
                    const unsigned map_size_bytes = this->number_rows * this->number_cols * observation_size;
                    cudaMemcpy(this->map.data(), map_gpu, map_size_bytes, cudaMemcpyDeviceToHost);
                    hpcProjectLog("the updated map given the obs [" << obs << "] and the bmu [" << bmu << "] is " << map "\n");
                #endif

                // Free the previously allocated memory on the GPU for
                // the general map of observations
                cudaFree(&map_gpu);

                // Free the previously allocated memory on the GPU for
                // the current observation which is being analysed
                cudaFree(&obs_gpu_1);

                // Free the previously allocated memory on the GPU for
                // the neighborhood's matrix
                cudaFree(&neighborhood_gpu_1);
            }
        }
    };
}
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

#ifndef GPU_SELF_ORGANISING_MAPS_HPC_PROJECT_1_HPP
#define GPU_SELF_ORGANISING_MAPS_HPC_PROJECT_1_HPP

#include <array>
#include <vector>
#include <algorithm>
#include <iostream>
#include <iterator>

// #include <gtest/gtest.h>

using namespace std;

constexpr auto THREADS_PER_BLOCK = 512;

// Print a log, given a message
#define hpcProjectLog(message) { std::cout << "[INFO] " <<  ( message ) << std::endl; }


template <typename T, size_t Size>
ostream& operator<<(ostream& out, const array<T, Size>& a) {

    out << "[ ";
    copy(a.begin(),
         a.end(),
         ostream_iterator<T>(out, " "));
    out << "]";

    return out;
}

template <typename T>
ostream& operator<<(ostream& out, const vector<T>& a) {

    out << "[ ";
    copy(a.begin(),
         a.end(),
         ostream_iterator<T>(out, " "));
    out << "]";

    return out;

}


/**
 * Assert that the contents of two vectors are the same
 */
template<typename Container>
inline void expect_container_eq(Container &a, Container &b) {

    EXPECT_EQ(a.size(), b.size());
    auto aptr = a.data();
    auto bptr = b.data();
    for (std::size_t i = 0; i < a.size(); i++)
        EXPECT_EQ(aptr[i], bptr[i]);
}

#endif // GPU_SELF_ORGANISING_MAPS_HPC_PROJECT_1_HPP
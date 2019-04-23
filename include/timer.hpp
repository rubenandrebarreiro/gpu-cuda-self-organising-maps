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

#ifndef GPU_SELF_ORGANISING_MAPS_TIMER_HPP
#define GPU_SELF_ORGANISING_MAPS_TIMER_HPP

#include <chrono>
#include <cmath>
#include <map>
#include <memory>

#define s_main "main"

namespace gpu_self_organising_maps {

    template <class Duration = std::chrono::milliseconds>

    // Class of Timer
    class timer {

        using Clock = std::chrono::high_resolution_clock;
        using TimePoint = std::chrono::time_point<Clock>;
        using ElapseTime = typename Duration::rep;

        struct stage {

            std::unique_ptr<ElapseTime[]> measurements;

            unsigned current_measurement;

            ElapseTime computed_average;

            TimePoint start;

            //TimePoint end;

            stage(unsigned number_measurements, TimePoint&& start) :
                    measurements (std::unique_ptr<ElapseTime[]>(new ElapseTime[number_measurements])),
                    current_measurement (0),
                    computed_average (0),
                    start (std::forward<TimePoint>(start))
            {}
        };

        public:

            // Constructor:
            explicit timer(const unsigned num_measurements = 1, const float percentage = 0) :
                number_measurements(num_measurements) {

                if (number_measurements > 1) {
                    accounted_measurements = number_measurements - roundf(number_measurements * percentage);

                    account_from = accounted_measurements == number_measurements ?
                                       0 :
                                       roundf((number_measurements - accounted_measurements) / 2.0 - 1.0);

                    account_to = account_from + accounted_measurements - 1;
                }
                else
                    account_from = account_to = 0;
            }


            void start(const std::string& stage_name = s_main) {
                m_stages.emplace(stage_name, stage(number_measurements, Clock::now()));
            }


            ElapseTime stop(const std::string& stage_name = s_main) {
                auto now  = Clock::now();

                timer::stage &s = m_stages.at(stage_name);
                s.measurements[s.current_measurement] = std::chrono::duration_cast<Duration>(now - s.start).count();
                s.computed_average = 0;

                accounted_measurements++;

                return s.measurements[s.current_measurement++];
            }


            ElapseTime reset(const std::string& name = s_main) {
                timer::stage &s = m_stages.at(name);
                s.current_measurement = 0;
            }


            ElapseTime average(const std::string& stage_name = s_main) {
                stage &s = m_stages.at(stage_name);

                if (accounted_measurements <= 1)
                    return s.measurements[0];

                if (s.computed_average == 0) {
                    qsort(s.measurements.get(), number_measurements, sizeof(TimePoint), compare);

                    for (unsigned int i = account_from; i <= account_to; i++)
                        s.computed_average += s.measurements[i];

                    s.computed_average /= accounted_measurements;
                }

                return s.computed_average;
            }

            // Return the standard deviation
            double std_deviation(const std::string& stage_name = s_main) {
                stage &s = m_stages.at(stage_name);
                average(stage_name);

                if (accounted_measurements <= 1)
                    return 0;

                double variance = 0.0;

                for (unsigned int i = account_from; i <= account_to; i++) {
                    auto aux = s.measurements[i] - s.computed_average;
                    variance += aux * aux;
                }

                variance /= accounted_measurements;

                return std::sqrt(variance);
            }

            // Prints the statistics collected
            void print_stats(std::ostream& out,
                             const std::string& stage_name = s_main,
                             const bool cvs = true) {

                stage &s = m_stages.at(stage_name);

                if (number_measurements <= 1)
                    out << s.measurements[0];

                else if (cvs) {
                    out << accounted_measurements << "/" << number_measurements <<

                    "," <<
                    " & " << s.measurements[account_from] <<
                    " & " << s.measurements[account_to] <<
                    " & " << average(stage_name) <<
                    " & " << std_deviation(stage_name);
                }
                else {
                    out << "statistics (middle " << accounted_measurements << " of " << number_measurements <<
                        " measurements) in " << ":" << std::endl <<
                        "\tAverage: " << average(stage_name) << std::endl <<
                        "\tMaximum: " << s.measurements[account_to] << std::endl <<
                        "\tMinimum: " << s.measurements[account_from] << std::endl <<
                        "\tStandard deviation: " << std_deviation(stage_name) << std::endl;
                }
            }

        private:

            std::map<std::string, stage> m_stages;

            unsigned accounted_measurements;
            unsigned account_from;
            unsigned account_to;
            const unsigned number_measurements;


           /**
            * Auxiliary compare function
            */
            static int compare(const void *a, const void *b) {
                if ((*(TimePoint *) a < *(TimePoint *) b))
                    return -1;

                return *(TimePoint *) a > *(TimePoint *) b;
            }
    };
}

#endif /* GPU_SELF_ORGANISING_MAPS_TIMER_HPP */


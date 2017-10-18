#include "lasso_minimizer_objective_fun.h"

#include <iostream>

namespace analyze_tools {
namespace opt {

double theta_norm_squared(const input_array &theta) {
    double result = 0;
    const auto n_timesteps = static_cast<std::size_t>(theta.shape()[0]);
    const auto n_reactions = static_cast<std::size_t>(theta.shape()[1]);
    const auto n_species = static_cast<std::size_t>(theta.shape()[2]);

    for (std::size_t t = 0; t < n_timesteps; ++t) {
        for (std::size_t s = 0; s < n_species; ++s) {
            for (std::size_t r = 0; r < n_reactions; ++r) {
                const auto tval = theta.at(t, r, s);
                result += tval * tval;
            }
        }
    }
    return result;
}

/**
 * RMSD
 */
double score(const input_array &propensities, const input_array &theta, const input_array &dX) {
    double result = 0;
    if (theta.ndim() != 3) {
        throw std::invalid_argument("invalid dims");
    }
    const auto n_timesteps = static_cast<std::size_t>(theta.shape()[0]);
    const auto n_reactions = static_cast<std::size_t>(theta.shape()[1]);
    const auto n_species = static_cast<std::size_t>(theta.shape()[2]);
    for (std::size_t t = 0; t < n_timesteps; ++t) {
        for (std::size_t s = 0; s < n_species; ++s) {
            auto x = dX.at(t, s);
            for (std::size_t r = 0; r < n_reactions; ++r) {
                x -= propensities.at(r) * theta.at(t, r, s);
            }
            result += x * x;
        }
    }
    // std::cout << "curr result: " << result << ", n_timesteps= " << n_timesteps << std::endl;
    result *= (1. / (n_timesteps));
    return std::sqrt(result);
}

void least_squares_function(input_array &result, const input_array &propensities, const input_array &theta,
                            const input_array &dX) {
    const auto n_timesteps = static_cast<std::size_t>(theta.shape()[0]);
    const auto n_reactions = static_cast<std::size_t>(theta.shape()[1]);
    const auto n_species = static_cast<std::size_t>(theta.shape()[2]);
    auto data = result.mutable_data();

    if(result.ndim() != 1) {
        throw std::invalid_argument("invalid result dims! (got ndim=" + std::to_string(result.ndim()) + ")");
    }
    if(static_cast<std::size_t>(result.shape()[0]) != n_timesteps * n_species) {
        throw std::invalid_argument("invalid shape of result! (got shape[0]=" + std::to_string(n_reactions) + ")");
    }

    for (std::size_t t = 0; t < n_timesteps; ++t) {
        for (std::size_t s = 0; s < n_species; ++s) {
            auto x = dX.at(t, s);
            for (std::size_t r = 0; r < n_reactions; ++r) {
                x -= propensities.at(r) * theta.at(t, r, s);
            }
            data[s + n_species*t] = x;
        }
    }

    {
        // apply prefactor
        for (std::size_t i = 0; i < n_timesteps * n_species; ++i) {
            data[i] *= 1. / (2. * n_timesteps);
        }
    }
}

input_array elastic_net_objective_function_jac(const input_array &propensities,
                                               const double alpha, const double l1_ratio, const input_array &theta,
                                               const input_array &dX) {
    input_array result;
    const auto n_timesteps = static_cast<std::size_t>(theta.shape()[0]);
    const auto n_reactions = static_cast<std::size_t>(theta.shape()[1]);
    const auto n_species = static_cast<std::size_t>(theta.shape()[2]);

    {
        std::vector<std::size_t> shape;
        shape.push_back(static_cast<std::size_t>(n_reactions));
        result.resize(shape);
    }
    // auto data = result.mutable_data();

    // for each element in the 1xr jacobi matrix...
    for(std::size_t i = 0; i < n_reactions; ++i) {
        // first set it to 0
        result.mutable_at(i) = 0;
        // then calculate the frobenius part
        for (std::size_t t = 0; t < n_timesteps; ++t) {
            for (std::size_t s = 0; s < n_species; ++s) {
                auto x = dX.at(t, s);
                auto theta_t_i_s = theta.at(t, i, s);
                for(std::size_t r = 0; r < n_reactions; ++r) {
                    x -= propensities.at(r) * theta.at(t, r, s);
                }
                result.mutable_at(i) += theta_t_i_s * x;
            }
        }
        result.mutable_at(i) *= -1. / (2.*n_timesteps);

        // now the l1 regularization
        result.mutable_at(i) += alpha * l1_ratio;

        // now the l2 regularization
        result.mutable_at(i) += alpha * (1. - l1_ratio) * propensities.at(i);

    }


    return result;
}

double elastic_net_objective_function(const input_array &propensities, const double alpha, const double l1_ratio,
                                      const input_array &theta, const input_array &dX) {
    double result = 0;
    if (theta.ndim() != 3) {
        throw std::invalid_argument("invalid dims");
    }
    const auto n_timesteps = static_cast<std::size_t>(theta.shape()[0]);
    const auto n_reactions = static_cast<std::size_t>(theta.shape()[1]);
    const auto n_species = static_cast<std::size_t>(theta.shape()[2]);
    for (std::size_t t = 0; t < n_timesteps; ++t) {
        for (std::size_t s = 0; s < n_species; ++s) {
            auto x = dX.at(t, s);
            for (std::size_t r = 0; r < n_reactions; ++r) {
                x -= propensities.at(r) * theta.at(t, r, s);
            }
            result += x * x;
        }
    }
    result *= 1. / (2. * n_timesteps);
    double regulator = 0;
    {
        double regulator_l1 = 0;
        for (std::size_t r = 0; r < n_reactions; ++r) {
            regulator_l1 += std::abs(propensities.at(r));
        }
        regulator_l1 *= alpha * l1_ratio;
        regulator += regulator_l1;
    }
    if (l1_ratio < 1.0 && alpha != 0) {
        double l2_norm_squared = 0;
        for (std::size_t r = 0; r < n_reactions; ++r) {
            l2_norm_squared += propensities.at(r) * propensities.at(r);
        }
        l2_norm_squared *= 0.5 * alpha * (1 - l1_ratio);
        regulator += l2_norm_squared;
    }
    return result + regulator;
}

double lasso_cost_fun(const input_array &propensities, const double alpha,
                      const input_array &theta, const input_array &dX) {
    return elastic_net_objective_function(propensities, alpha, 1.0, theta, dX);
}

}
}
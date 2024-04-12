//
// Created by Frank on 1/24/2024.
//

#ifndef DISTRIBUTEDSBP_SPENCE_HPP
#define DISTRIBUTEDSBP_SPENCE_HPP

double polevl(double x, double coef[], int N);

/// Computes the integral
///
///                     x
///                     -
///                    | | log t
/// spence(x)  =  -    |   ----- dt
///                  | |   t - 1
///                  -
///                  1
///
/// for x >= 0. A rational approximation gives the integral in the interval (0.5, 1.5). Transformation formulas for
/// 1/x and 1-x are employed outside the basic expansion range.
double spence(double x);

#endif //DISTRIBUTEDSBP_SPENCE_HPP

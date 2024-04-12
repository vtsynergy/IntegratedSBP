/***
 * Utility functions.
 */
#ifndef SBP_UTILS_HPP
#define SBP_UTILS_HPP

// #include <filesystem>
#include <fstream>
#include <iostream>
#include <cmath>
#include <execution>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "args.hpp"
#include "blockmodel/sparse/typedefs.hpp"
#include "fs.hpp"

namespace utils {

typedef struct proposal_and_edge_counts_t {
    long proposal;
    long num_out_neighbor_edges;
    long num_in_neighbor_edges;
    long num_neighbor_edges;
} ProposalAndEdgeCounts;

/// Builds the base path for the graph and true assignment .tsv files.
/// Assumes the file is saved in the following directory:
/// <args.directory>/<args.type>/<args.overlap>Overlap_<args.blocksizevar>BlockSizeVar
/// Assumes the graph file is named:
/// <args.type>_<args.overlap>Overlap_<args.blocksizevar>BlockSizeVar_<args.numvertices>_nodes.tsv
/// Assumes the true assignment file is named:
/// <args.type>_<args.overlap>Overlap_<args.blocksizevar>BlockSizeVar_<args.numvertices>_trueBlockmodel.tsv
std::string build_filepath();

/// Argsort based on radix sort in descending order
std::vector<long> argsort(const std::vector<long>& v);

/// Divides all elements in a MapVector<long> by a scalar, and stores the result in `result`
inline void div(const MapVector<long> &lhs, const double &rhs, SparseVector<double> &result) {
    for (const std::pair<const long, long> &pair : lhs) {
        result.idx.push_back(pair.first);
        result.data.push_back((double) pair.second / rhs);
    }
}

/// Extends vector a with vector b, similar to python a.extend(b)
template <typename T> inline void extend(std::vector<T> &a, std::vector<T> &b) {
    a.reserve(a.size() + b.size());
    a.insert(a.end(), b.begin(), b.end());
}

/// Assumes filepath corresponds to the path of a CSV file, and reads it as such.
/// All data stored as strings.
/// Note: does NOT differentiate between header row and data rows, and does NOT do data type conversion.
std::vector<std::vector<std::string>> read_csv(fs::path &filepath);

/// Inserts the given edge into the neighbors list. Assumes the graph is unweighted.
void insert(NeighborList &neighbors, long from, long to);

/// Inserts the given edge into the neighbors list, avoiding duplicates. Assumes the graph is unweighted.
void insert_nodup(NeighborList &neighbors, long from, long to);

/// Inserts the given pair into the map if the element does not already exist. Returns true if the insertion happened,
/// false otherwise.
bool insert(std::unordered_map<long, long> &map, long key, long value);

/// Concatenates two vectors without modifying them.
template <typename T> inline std::vector<T> concatenate(std::vector<T> &a, std::vector<T> &b) {
    std::vector<T> result;
    result.insert(result.end(), a.begin(), a.end());
    result.insert(result.end(), b.begin(), b.end());
    return result;
}

/// Returns a vector filled with a constant value.
template <typename T> inline std::vector<T> constant(long size, T value) {
    std::vector<T> result(size, value);
    return result;
}

/// Radix sort in descending order
void radix_sort(std::vector<long>& v);

/// Radix sort in descending order for pairs
void radix_sort(std::vector<std::pair<long, long>>& v);

template<typename T, typename Y> void radix_sort(std::vector<std::pair<T, Y>> &v) {
    if (v.empty()) {
        return;
    }

    // isolate integer byte by index.
    auto bmask = [](Y x, size_t i) {
        return (x >> i*8) & 0xFF;
    };

    // allocate temporary buffer.
    std::vector<std::pair<T, Y>> sorted(v.size());

    // for each byte in integer (assuming 4-byte int).
    for (size_t i, j = 0; j < sizeof(long); j++) {
        // initialize counter to zero;
        size_t h[256] = {}, start;

        // histogram.
        // count each occurrence of indexed-byte value.
        for (i = 0; i < v.size(); i++)
            h[255 - bmask(v[i].second, j)]++;

        // accumulate.
        // generate positional offsets. adjust starting point
        // if most significant digit.
        start = (j != 7) ? 0 : 128;
        for ( i = 1+start; i < 256+start; i++ )
            h[i % 256] += h[(i-1) % 256];

        // distribute.
        // stable reordering of elements. backward to avoid shifting
        // the counter array.
        for ( i = v.size(); i > 0; i-- ) {
            sorted[--h[255 - bmask(v[i-1].second, j)]] = v[i-1];
        }

        std::swap(v, sorted);
    }
}

/// Returns a vector filled with values in the range[start, start+size).
template <typename T> inline std::vector<T> range(long start, long size) {
    // TODO: may be faster using push_backs, instead of initializing and then modifying
    std::vector<T> result(size, 0);
    std::iota(result.begin(), result.end(), start);
    return result;
}

/// Partially sorts the indices of an array in ascending order according to the values of the array.
inline std::vector<long> partial_sort_indices(const std::vector<double> &unsorted, long pivot) {
    // initialize original index locations
    std::vector<long> indices = utils::range<long>(0, unsorted.size());
    // partially sort indices based on comparing values in unsorted
    std::nth_element(indices.data(), indices.data() + pivot, indices.data() + indices.size(),
              [unsorted](size_t i1, size_t i2) { return unsorted[i1] < unsorted[i2]; });
    // std::nth_element(std::execution::par_unseq, indices.data(), indices.data() + pivot, indices.data() + indices.size(),
    //           [unsorted](size_t i1, size_t i2) { return unsorted[i1] < unsorted[i2]; });
    return indices;
}

/// Returns the sum of the elements in a vector.
template <typename T> inline T sum(const std::vector<T> &vector) {
    T result = 0;
    for (const T &value : vector) {
        result += value;
    }
    return result;
}

/// Sorts the indices of an array in descending order according to the values of the array
template <typename T> inline std::vector<long> argsort(const std::vector<T> &unsorted) {
    // initialize original index locations
    std::vector<long> indices = utils::range<long>(0, unsorted.size());
    // sort indexes based on comparing values in unsorted
    std::sort(std::execution::par_unseq, indices.data(), indices.data() + indices.size(),
              [&unsorted](size_t i1, size_t i2) { return unsorted[i1] > unsorted[i2]; });
    return indices;
}

/// A radix sort implementation
//static void radix_sort(std::vector<long>& v) {
//    if (v.empty()) {
//        return;
//    }
//
//    constexpr long num_bits = 8; // number of bits in a byte
//    constexpr long num_buckets = 1 << num_bits; // number of possible byte values
//    constexpr long mask = num_buckets - 1; // mask to extract the least significant byte
//
//    long max_element = *std::max_element(v.begin(), v.end());
//    long num_passes = (sizeof(long) + num_bits - 1) / num_bits; // number of passes needed for all bytes
//    std::vector<long> counts(num_buckets);
//
//    std::vector<long> sorted(v.size());
//    for (long pass = 0; pass < num_passes; pass++) {
//        std::memset(counts.data(), 0, num_buckets * sizeof(long)); // reset counts
//
//        for (long i = 0; i < v.size(); i++) {
//            long byte = (v[i] >> (num_bits * pass)) & mask;
//            counts[byte]++;
//        }
//
//        for (long i = 1; i < num_buckets; i++) {
//            counts[i] += counts[i - 1];
//        }
//
//        for (long i = v.size() - 1; i >= 0; i--) {
//            long byte = (v[i] >> (num_bits * pass)) & mask;
//            sorted[--counts[byte]] = v[i];
//        }
//
//        std::copy(sorted.begin(), sorted.end(), v.begin());
//    }
//}

/// A radix sort-based argsort
//static std::vector<long> argsort(const std::vector<long>& v) {
//    if (v.empty()) {
//        return std::vector<long>();
//    }
//
//    constexpr long num_bits = 8; // number of bits in a byte
//    constexpr long num_buckets = 1 << num_bits; // number of possible byte values
//    constexpr long mask = num_buckets - 1; // mask to extract the least significant byte
//
//    long max_element = *std::max_element(v.begin(), v.end());
//    long num_passes = (sizeof(long) + num_bits - 1) / num_bits; // number of passes needed for all bytes
//    std::vector<long> counts(num_buckets);
//
//    std::vector<long> indices(v.size());
//    std::vector<long> sorted_indices(v.size());
//    for (long i = 0; i < v.size(); i++) {
//        indices[i] = i;
//    }
//
//    for (long pass = 0; pass < num_passes; pass++) {
//        std::memset(counts.data(), 0, num_buckets * sizeof(long)); // reset counts
//
//        for (long i = 0; i < v.size(); i++) {
//            long byte = (v[i] >> (num_bits * pass)) & mask;
//            counts[byte]++;
//        }
//
//        for (long i = 1; i < num_buckets; i++) {
//            counts[i] += counts[i - 1];
//        }
//
//        for (long i = v.size() - 1; i >= 0; i--) {
//            long byte = (v[i] >> (num_bits * pass)) & mask;
//            sorted_indices[--counts[byte]] = indices[i];
//        }
//
//        std::swap(indices, sorted_indices);
//    }
//
//    return indices;
//}

/// Returns the sum of the elements in a vector, where sum and vector types are different.
template <typename T, typename Y> inline T sum(const MapVector<Y> &vector) {
    T result = 0;
    for (const std::pair<long, Y> &pair : vector) {
        result += pair.second;
    }
    return result;
}

/// Creates a SparseVector by only considering the non-zero elements of vector.
template <typename T> inline SparseVector<T> to_sparse(const std::vector<T> &vector) {
    SparseVector<T> result;
    for (long i = 0; i < vector.size(); ++i) {
        T value = vector[i];
        if (value != 0) {
            result.idx.push_back(i);
            result.data.push_back(value);
        } 
    }
    return result;
}

/// Casts the values in vector to type double.
/// Relies on an implicit cast from vector type T to double.
template <typename T> inline std::vector<double> to_double(const std::vector<T> &vector) {
    return std::vector<double>(vector.begin(), vector.end());
}

/// Casts the values in vector to type float.
/// Relies on an implicit cast from vector type T to double.
template <typename T> inline std::vector<float> to_float(const std::vector<T> &vector) {
    return std::vector<float>(vector.begin(), vector.end());
}

/// Casts the values in vector to type long.
/// Relies on an implicit cast from vector type T to long.
template <typename T> inline std::vector<long> to_long(const std::vector<T> &vector) {
    return std::vector<long>(vector.begin(), vector.end());
}

/// Returns the natural log of every value in vector.
/// Relies on an implicit conversion from type T to double.
template <typename T> inline std::vector<T> nat_log(const std::vector<T> &vector) {
    std::vector<T> result;
    for (const T &value : vector) {
        double temp = logf(value);
        assert(!std::isnan(temp));
        assert(!std::isinf(temp));
        result.push_back(temp);
//        result.push_back(logf(value));
    }
    return result;
}

/// Returns the index of the maximum element in vector.
template <typename T> inline long argmax(const std::vector<T> &vector) {
    T max_value = vector[0];
    long max_index = 0;
    for (long i = 1; i < (long) vector.size(); ++i) {
        /// The following link can compute this without branching (could be useful for GPUs)
        /// https://www.geeksforgeeks.org/compute-the-minimum-or-maximum-max-of-two-integers-without-branching/
        if (vector[i] > max_value) {
            max_value = vector[i];
            max_index = i;
        }
    }
    return max_index;
}

/// prints an array
template <typename T> inline void print(const T vector[], size_t vector_size) {
//    size_t vector_bytes = sizeof(vector);
//    if (vector_bytes == 0) {
//        std::cout << "[]" << std::endl;
//        return;
//    }
//    size_t elem_bytes = sizeof(vector[0]);
//    size_t vector_size = vector_bytes / elem_bytes;
    std::cout << "[" << vector[0] << ", ";
    for (size_t num_printed = 1; num_printed < vector_size - 1; num_printed++) {
        if (num_printed % 25 == 0) {
            std::cout << std::endl << " ";
        }
        std::cout << vector[num_printed] << ", ";
    }
    std::cout << vector[vector_size - 1] << "]" << std::endl;
}

/// prints a sparse vector
template <typename T> inline void print(const MapVector<T> vector) {
    if (vector.empty()) {
        std::cout << "[]" << std::endl;
        return;
    }
    long i = 0;
    for (const std::pair<long, T> &element : vector) {
        if (i == 0) {
            std::cout << "[" << element.first << ": " << element.second << ", ";
        } else {
            std::cout << element.first << ": " << element.second << ", ";
        }
        i += 1;
        if (i % 25 == 0) {
            std::cout << std::endl << " ";
        }
    }
    std::cout << " ]" << std::endl;
}

/// prints a vector
template <typename T> inline void print(const std::vector<T> &vector) {
    if (vector.empty()) {
        std::cout << "[]" << std::endl;
        return;
    }
    std::cout << "[" << vector[0] << ", ";
    for (size_t num_printed = 1; num_printed < vector.size() - 1; num_printed++) {
        if (num_printed % 25 == 0) {
            std::cout << std::endl << " ";
        }
        std::cout << vector[num_printed] << ", ";
    }
    std::cout << vector[vector.size() - 1] << "]" << std::endl;
}

/// prints a vector, cuts vector off at 20 elements
template <typename T> inline void print_short(const std::vector<T> &vector) {
    if (vector.empty()) {
        std::cout << "[]" << std::endl;
        return;
    }
    std::cout << "[" << vector[0] << ", ";
    if (vector.size() <= 20) {
        for (size_t num_printed = 1; num_printed < vector.size() - 1; num_printed++) {
            if (num_printed % 25 == 0) {
                std::cout << std::endl << " ";
            }
            std::cout << vector[num_printed] << ", ";
        }
        std::cout << vector[vector.size() - 1] << "]" << std::endl;
    }else {
        for (size_t num_printed = 1; num_printed < 10; num_printed++) {
            if (num_printed % 25 == 0) {
                std::cout << std::endl << " ";
            }
            std::cout << vector[num_printed] << ", ";
        }
        std::cout << " ... ";
        for (size_t num_printed = vector.size() - 10; num_printed < vector.size() - 1; num_printed++) {
            if (num_printed % 25 == 0) {
                std::cout << std::endl << " ";
            }
            std::cout << vector[num_printed] << ", ";
        }
    }
    std::cout << vector[vector.size() - 1] << "]" << std::endl;
}

}  // namespace utils

/// Allows elementwise multiplication of two std::vector<double> objects.
inline std::vector<double> operator*(const std::vector<double> &lhs, const std::vector<double> &rhs) {
    std::vector<double> result(lhs.size());
    for (size_t i = 0; i < lhs.size(); ++i) {
        result[i] = lhs[i] * rhs[i];
    }
    return result;
}

/// Allows elementwise multiplication of two std::vector<float> objects.
inline std::vector<float> operator*(const std::vector<float> &lhs, const std::vector<float> &rhs) {
    std::vector<float> result(lhs.size());
    for (size_t i = 0; i < lhs.size(); ++i) {
        result[i] = lhs[i] * rhs[i];
    }
    return result;
}

/// Allows elementwise division of two std::vector<double> objects.
inline std::vector<double> operator/(const std::vector<double> &lhs, const std::vector<double> &rhs) {
    std::vector<double> result(lhs.size());
    for (size_t i = 0; i < lhs.size(); ++i) {
        result[i] = lhs[i] / rhs[i];
    }
    return result;
}

/// Allows elementwise division of a std::vector<float> and a scalar.
inline std::vector<double> operator/(const std::vector<double> &lhs, const double &rhs) {
    std::vector<double> result(lhs.size());
    for (size_t i = 0; i < lhs.size(); ++i) {
        result[i] = lhs[i] / rhs;
    }
    return result;
}

/// Allows elementwise division of two std::vector<float> objects.
inline std::vector<float> operator/(const std::vector<float> &lhs, const std::vector<float> &rhs) {
    std::vector<float> result(lhs.size());
    for (size_t i = 0; i < lhs.size(); ++i) {
        result[i] = lhs[i] / rhs[i];
    }
    return result;
}

/// Allows elementwise division of a std::vector<float> and a scalar.
inline std::vector<float> operator/(const std::vector<float> &lhs, const float &rhs) {
    std::vector<float> result(lhs.size());
    for (size_t i = 0; i < lhs.size(); ++i) {
        result[i] = lhs[i] / rhs;
    }
    return result;
}

/// Allows elementwise multiplication of a std::vector<double> and a scalar.
inline std::vector<double> operator*(const std::vector<double> &lhs, const long &rhs) {
    std::vector<double> result(lhs.size());
    for (size_t i = 0; i < lhs.size(); ++i) {
        result[i] = lhs[i] * (double) rhs;
    }
    return result;
}

/// Allows elementwise addition of two std::vector<double> objects.
inline std::vector<double> operator+(const std::vector<double> &lhs, const std::vector<double> &rhs) {
    std::vector<double> result(lhs.size());
    for (size_t i = 0; i < lhs.size(); ++i) {
        result[i] = lhs[i] + rhs[i];
    }
    return result;
}

/// Allows elementwise addition of two std::vector<float> objects.
inline std::vector<float> operator+(const std::vector<float> &lhs, const std::vector<float> &rhs) {
    std::vector<float> result(lhs.size());
    for (size_t i = 0; i < lhs.size(); ++i) {
        result[i] = lhs[i] + rhs[i];
    }
    return result;
}

/// Allows elementwise addition of two std::vector<long> objects.
inline std::vector<long> operator+(const std::vector<long> &lhs, const std::vector<long> &rhs) {
    std::vector<long> result(lhs.size());
    for (size_t i = 0; i < lhs.size(); ++i) {
        result[i] = lhs[i] + rhs[i];
    }
    return result;
}

#endif // SBP_UTILS_HPP

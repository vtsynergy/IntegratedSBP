#include "utils.hpp"

#include "mpi_data.hpp"

namespace utils {

std::vector<long> argsort(const std::vector<long> &v) {
    if (v.empty()) {
        return {};
    }

    // isolate integer byte by index.
    auto bmask = [](long x, size_t i) {
        return (static_cast<unsigned long>(x) >> i*8) & 0xFF;
    };

    // allocate temporary buffer.
    std::vector<long> indices(v.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::vector<long> new_indices(v.size());
    std::vector<long> v_copy(v);
    std::vector<long> new_v(v.size());

    // for each byte in integer (assuming 4-byte int).
    for (size_t i, j = 0; j < sizeof(long); j++) {
        // initialize counter to zero;
        size_t h[256] = {}, start;

        // histogram.
        // count each occurrence of indexed-byte value.
        for (i = 0; i < v_copy.size(); i++)
            h[255 - bmask(v_copy[i], j)]++;

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
            size_t k = --h[255 - bmask(v_copy[i - 1], j)];
            new_indices[k] = indices[i - 1];
            new_v[k] = v_copy[i - 1];
//            new_indices[--h[255 - bmask(v[i-1], j)]] = indices[i-1];
        }

        std::swap(indices, new_indices);
        std::swap(v_copy, new_v);
    }
    return indices;
}

std::string build_filepath() {
//    std::ostringstream filepath_stream;
//    filepath_stream << args.directory << "/" << args.type << "/" << args.overlap << "Overlap_" << args.blocksizevar;
//    filepath_stream << "BlockSizeVar/" << args.type << "_" << args.overlap << "Overlap_" << args.blocksizevar;
//    filepath_stream << "BlockSizeVar_" << args.numvertices << "_nodes";
//    // TODO: Add capability to process multiple "streaming" graph parts
//    std::string filepath = filepath_stream.str();
    std::string filepath = args.filepath;
    if (!fs::exists(filepath + ".tsv") && !fs::exists(filepath + ".mtx")) {
        std::cerr << "ERROR " << "File doesn't exist: " << filepath + ".tsv/.mtx" << std::endl;
        exit(-1);
    }
    return filepath;
}

std::vector<std::vector<std::string>> read_csv(fs::path &filepath) {
    std::vector<std::vector<std::string>> contents;
    if (!fs::exists(filepath)) {
        if (mpi.rank == 0)
            std::cerr << "ERROR " << "File doesn't exist: " << filepath << std::endl;
        return contents;
    }
    std::string line;
    std::ifstream file(filepath);
    long items = 0;
    while (std::getline(file, line)) {
        std::vector<std::string> row;
        std::stringstream line_stream(line);
        std::string value;
        while (line_stream >> value) {
            row.push_back(value);
            items++;
        }
        contents.push_back(row);
    }
    if (mpi.rank == 0)
        std::cout << "Read in " << contents.size() << " lines and " << items << " values." << std::endl;
    return contents;
}

void insert(NeighborList &neighbors, long from, long to) {
    if (from >= (long) neighbors.size()) {
        std::vector<std::vector<long>> padding(from - neighbors.size() + 1, std::vector<long>());
        neighbors.insert(neighbors.end(), padding.begin(), padding.end());
    }
    neighbors[from].push_back(to);
}

void insert_nodup(NeighborList &neighbors, long from, long to) {
    if (from >= (long) neighbors.size()) {
        std::vector<std::vector<long>> padding(from - neighbors.size() + 1, std::vector<long>());
        neighbors.insert(neighbors.end(), padding.begin(), padding.end());
    }
    for (const long &neighbor: neighbors[from])
        if (neighbor == to) return;
    neighbors[from].push_back(to);
}

bool insert(std::unordered_map<long, long> &map, long key, long value) {
    std::pair<std::unordered_map<long, long>::iterator, bool> result = map.insert(std::make_pair(key, value));
    return result.second;
}

void radix_sort(std::vector<long> &v) {
    if (v.empty()) {
        return;
    }

    // isolate integer byte by index.
    auto bmask = [](long x, size_t i) {
        return (static_cast<unsigned long>(x) >> i*8) & 0xFF;
    };

    // allocate temporary buffer.
    std::vector<long> sorted(v.size());

    // for each byte in integer (assuming 4-byte int).
    for (size_t i, j = 0; j < sizeof(long); j++) {
        // initialize counter to zero;
        size_t h[256] = {}, start;

        // histogram.
        // count each occurrence of indexed-byte value.
        for (i = 0; i < v.size(); i++)
            h[255 - bmask(v[i], j)]++;

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
            sorted[--h[255 - bmask(v[i-1], j)]] = v[i-1];
        }

        std::swap(v, sorted);
    }
}

void radix_sort(std::vector<std::pair<long, long>> &v) {
    if (v.empty()) {
        return;
    }

    // isolate integer byte by index.
    auto bmask = [](long x, size_t i) {
        return (static_cast<unsigned long>(x) >> i*8) & 0xFF;
    };

    // allocate temporary buffer.
    std::vector<std::pair<long, long>> sorted(v.size());

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

//void radix_sort(std::vector<std::pair<long, long>> &v) {
//    if (v.empty()) {
//        return;
//    }
//
//    constexpr long num_bits = 8; // number of bits in a byte
//    constexpr long num_buckets = 1 << num_bits; // number of possible byte values
//    constexpr long mask = num_buckets - 1; // mask to extract the least significant byte
//
//    long max_element = (*std::max_element(v.begin(), v.end(),
//                                         [](const auto &p1, const auto &p2) { return p1.second > p2.second; })).second;
//    long num_passes = (sizeof(long) + num_bits - 1) / num_bits; // number of passes needed for all bytes
//    std::vector<long> counts(num_buckets);
//
//    std::vector<std::pair<long, long>> sorted_v(v.size());
//
//    for (long pass = 0; pass < num_passes; pass++) {
//        std::fill(counts.begin(), counts.end(), 0); // reset counts
//
//        for (const auto &elem: v) {
//            long byte = (max_element - (elem.second >> (num_bits * pass))) &
//                       mask; // changed to max_element - ... to sort in descending order
//            counts[byte]++;
//        }
//
//        for (long i = num_buckets - 2; i >= 0; i--) { // changed to process buckets in reverse order
//            counts[i] += counts[i + 1];
//        }
//
//        for (long i = v.size() - 1; i >= 0; i--) {
//            long byte = (max_element - (v[i].second >> (num_bits * pass))) &
//                       mask; // changed to max_element - ... to sort in descending order
//            sorted_v[--counts[byte]] = v[i];
//        }
//
//        std::swap(v, sorted_v);
//    }
//}

}  // namespace utils

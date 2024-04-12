#include "evaluate.hpp"

#include "entropy.hpp"

namespace evaluate {

double calculate_f1_score(long num_vertices, Hungarian::Matrix &contingency_table) {
    // The number of vertex pairs = |V| choose 2 = |V|! / (2! * (|V| - 2)!) = (|V| * |V|-1) / 2
    double num_pairs = (num_vertices * (num_vertices - 1.0)) / 2.0;
    const long nrows = contingency_table.size();
    const long ncols = contingency_table[0].size();
    std::vector<double> rowsums(nrows, 0);
    std::vector<double> colsums(ncols, 0);
    double cell_pairs = 0;  // num_agreement_same
    double cells_squared = 0;  // sum_table_squared
    double correctly_classified = 0.0;
    for (long row = 0; row < nrows; ++row) {
        for (long col = 0; col < ncols; ++col) {
            long value = contingency_table[row][col];
            rowsums[row] += value;
            colsums[col] += value;
            cell_pairs += (value * (value - 1));
            cells_squared += (value * value);
            if (row == col)
                correctly_classified += value;
        }
    }
    cell_pairs /= 2.0;
    double row_pairs = 0;  // num_same_in_b1
    double col_pairs = 0;  // num_same_in_b2
    double rowsums_squared = 0;  // sum_rowsum_squared
    double colsums_squared = 0;  // sum_colsum_squared
    for (double sum: rowsums) {
        row_pairs += (sum * (sum - 1));
        rowsums_squared += (sum * sum);
    }
    for (double sum: colsums) {
        col_pairs += (sum * (sum - 1));
        colsums_squared += (sum * sum);
    }
    row_pairs /= 2.0;
    col_pairs /= 2.0;
    double num_agreement_diff = (num_vertices * num_vertices) + cells_squared;
    num_agreement_diff -= (rowsums_squared + colsums_squared);
    num_agreement_diff /= 2.0;
    double num_agreement = cell_pairs + num_agreement_diff;
    double rand_index = num_agreement / num_pairs;
    double recall = cell_pairs / row_pairs;
    double precision = cell_pairs / col_pairs;
    double f1_score = (2.0 * precision * recall) / (precision + recall);

    std::cout << "Accuracy: " << correctly_classified / double(num_vertices) << std::endl;
    std::cout << "Rand index: " << rand_index << std::endl;
    // TODO: precision & recall could be flipped. Figure out when that is so...
    std::cout << "Recall: " << recall << std::endl;
    std::cout << "Precision: " << precision << std::endl;
    std::cout << "F1 Score: " << f1_score << std::endl;
    return f1_score;
}

double calculate_nmi(long num_vertices, Hungarian::Matrix &contingency_table) {
    std::vector<std::vector<double>> joint_probability;
    double sum = num_vertices;
    size_t nrows = contingency_table.size();
    size_t ncols = contingency_table[0].size();

    std::vector<double> marginal_prob_b1(nrows, 0.0);
    std::vector<double> marginal_prob_b2(ncols, 0.0);
    for (size_t i = 0; i < nrows; ++i) {
        joint_probability.emplace_back(std::vector<double>());
        for (size_t j = 0; j < ncols; ++j) {
            double value = contingency_table[i][j] / sum;
            joint_probability[i].push_back(value);
            marginal_prob_b1[i] += value;
            marginal_prob_b2[j] += value;
        }
    }
    double H_b1 = -1.0 * utils::sum<double>(marginal_prob_b1 * utils::nat_log(marginal_prob_b1));
    double H_b2 = -1.0 * utils::sum<double>(marginal_prob_b2 * utils::nat_log(marginal_prob_b2));
//    std::vector<std::vector<double>> marginal_product;
    double mi = 0.0;
    for (size_t i = 0; i < nrows; ++i) {
//        std::vector<double> row;
        for (size_t j = 0; j < ncols; ++j) {
            double marginal_product = marginal_prob_b1[i] * marginal_prob_b2[j];
            double joint_prob = joint_probability[i][j];
            if (joint_prob == 0.0) continue;
            mi += joint_prob * log(joint_prob / marginal_product);
//            row.push_back(marginal_prob_b1[i] * marginal_prob_b2[j]);
        }
//        marginal_product.push_back(row);
    }
    double nmi = mi / sqrt(H_b1 * H_b2);
    std::cout << "MI = " << mi << " H_b1 = " << H_b1 << " H_b2 = " << H_b2 << " NMI = " << nmi << std::endl;
    return nmi;
}

Eval evaluate_blockmodel(const Graph &graph, Blockmodel &blockmodel) {
    Hungarian::Matrix contingency_table = hungarian(graph, blockmodel);
    double f1_score = calculate_f1_score(graph.num_vertices(), contingency_table);
    double nmi = calculate_nmi(graph.num_vertices(), contingency_table);
    std::vector<long> true_assignment(graph.assignment());
    long true_num_blocks = 1 + *std::max_element(true_assignment.begin(), true_assignment.end());
    Blockmodel true_blockmodel(true_num_blocks, graph, 0.5, true_assignment);
    double true_entropy = entropy::mdl(true_blockmodel, graph);  // .num_vertices(), graph.num_edges());
    std::cout << "true entropy = " << true_entropy << std::endl;
    return Eval { f1_score, nmi, true_entropy };
}

Hungarian::Matrix hungarian(const Graph &graph, Blockmodel &blockmodel) {
    // Create contingency table
    long num_true_communities = 0;
    std::unordered_map<long, long> translator;  // TODO: add some kind of if statement for whether to use this or not
    translator[-1] = -1;
    for (long community: graph.assignment()) {
        if (community > -1) {
            if (translator.insert(std::unordered_map<long, long>::value_type(community, num_true_communities)).second) {
                num_true_communities++;
            };
        }
    }
    std::vector<long> true_assignment(graph.assignment());
    for (size_t i = 0; i < true_assignment.size(); ++i) {
        true_assignment[i] = translator[true_assignment[i]];
    }
    std::cout << "Blockmodel correctness evaluation" << std::endl;
    std::cout << "Number of vertices: " << graph.num_vertices() << std::endl;
    std::cout << "Number of communities in true assignment: " << num_true_communities << std::endl;
    std::cout << "Number of communities in alg. assignment: " << blockmodel.getNum_blocks() << std::endl;
    std::vector<long> rows, cols;
    long nrows, ncols;
    if (num_true_communities < blockmodel.getNum_blocks()) {
        rows = true_assignment;
        cols = blockmodel.block_assignment();
        nrows = num_true_communities;
        ncols = blockmodel.getNum_blocks();
    } else {
        rows = blockmodel.block_assignment();
        cols = true_assignment;
        nrows = blockmodel.getNum_blocks();
        ncols = num_true_communities;
    }
    std::vector<std::vector<int>> contingency_table(nrows, std::vector<int>(ncols, 0));
    for (long i = 0; i < graph.num_vertices(); ++i) {
        long row_block = rows[i];
        long col_block = cols[i];
        if (true_assignment[i] > -1) {
            contingency_table[row_block][col_block]++;
        }
    }

    Hungarian::Result result = Hungarian::Solve(contingency_table, Hungarian::MODE_MAXIMIZE_UTIL);
    if (!result.success) {
        std::cout << "Failed to find solution :(" << std::endl;
        exit(-1);
    }

    std::vector<long> assignment(result.assignment.size(), 0);
    for (size_t row = 0; row < result.assignment.size(); ++row) {
        for (size_t col = 0; col < result.assignment[0].size(); ++col) {
            if (result.assignment[row][col] == 1) {
                assignment[row] = (long) col;
            }
        }
    }

    std::vector<std::vector<int>> new_contingency_table(nrows, std::vector<int>(ncols, 0));
    for (long original_column = 0; original_column < ncols; ++original_column) {
        long new_column = assignment[original_column];
        for (long row = 0; row < nrows; ++row) {
            new_contingency_table[row][original_column] = contingency_table[row][new_column];
        }
    }

    // Make sure rows represent algorithm communities, columns represent true communities
    if (num_true_communities < blockmodel.getNum_blocks()) {
        Hungarian::Matrix transpose_contingency_table(ncols, std::vector<int>(nrows, 0));
        for (long row = 0; row < nrows; ++row) {
            for (long col = 0; col < ncols; ++col) {
                transpose_contingency_table[col][row] = new_contingency_table[row][col];
            }
        }
        new_contingency_table = transpose_contingency_table;
        long temp = ncols;
        ncols = nrows;
        nrows = temp;
    }

    std::cout << "Contingency Table" << std::endl;
    for (long i = 0; i < nrows; ++i) {
        if (nrows > 50) {
            if (i == 25) std::cout << "...\n";
            if (i > 25 && i < (nrows - 25)) continue;
        }
        for (long j = 0; j < ncols; ++j) {
            std::cout << new_contingency_table[i][j] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    return new_contingency_table;
}

} // namespace evaluate

/***
 * Stores a Graph.
 */
#ifndef SBP_EVALUATE_HPP
#define SBP_EVALUATE_HPP

#include <set>

#include "hungarian.hpp"

#include "graph.hpp"
#include "blockmodel/blockmodel.hpp"

namespace evaluate {

struct Eval {
    double f1_score;
    double nmi;
    double true_mdl;
};

double calculate_f1_score(long num_vertices, Hungarian::Matrix &contingency_table);

double calculate_nmi(long num_vertices, Hungarian::Matrix &contingency_table);

Eval evaluate_blockmodel(const Graph &graph, Blockmodel &blockmodel);

Hungarian::Matrix hungarian(const Graph &graph, Blockmodel &blockmodel);

}

#endif // SBP_EVALUATE_HPP

#ifndef PTM_BDA_CLASSIFIER_H
#define PTM_BDA_CLASSIFIER_H

#include <vector>

#include "ptm_neighbor_finder.h"
#include "ptm_analysis_types.h"

double bda_median_ptm_bcc_lattice(
    const std::vector<PTMAtomResult>& ptm,
    double fallback
);

BDAResult run_ptm_based_bda(
    const SystemData& sys,
    const CellListNeighborFinder& finder,
    const std::vector<PTMAtomResult>& ptm,
    const BDAConfig& cfg
);

void print_bda_debug_summary(const BDAResult& bda);

#endif
#ifndef PTM_RUNNER_H
#define PTM_RUNNER_H

#include <vector>

#include "ptm_neighbor_finder.h"
#include "ptm_analysis_types.h"

struct PTMRunConfig {
    int flags;
    bool output_conventional_orientation;

    double rmsd_cutoff_sc;
    double rmsd_cutoff_fcc;
    double rmsd_cutoff_hcp;
    double rmsd_cutoff_bcc;
    double rmsd_cutoff_ico;
    double rmsd_cutoff_dhex;
    double rmsd_cutoff_dcub;
    double rmsd_cutoff_graphene;

    double omega_rmsd_strict;
    double omega_rmsd_loose;

    bool debug_ptm_index;
    int debug_first_n_atoms;

    PTMRunConfig();
};

void run_ptm_for_system(
    SystemData& sys,
    const PTMRunConfig& cfg,
    std::vector<PTMAtomResult>& ptm_results
);

int filter_ptm_type_by_rmsd(int raw_type, double rmsd, const PTMRunConfig& cfg);

int omega_site_from_ptm_type(int type);

int omega_confidence_from_rmsd(int type, double rmsd, const PTMRunConfig& cfg);

#endif
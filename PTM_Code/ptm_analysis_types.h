#ifndef PTM_ANALYSIS_TYPES_H
#define PTM_ANALYSIS_TYPES_H

#include <vector>

enum BDADefect {
    BDA_BLK = 0,
    BDA_SRF = 1,
    BDA_VCN = 2,
    BDA_DSL = 3,
    BDA_TWN = 4,
    BDA_PLF = 5,
    BDA_ELS = 6
};

struct BDAConfig {
    char chunk_axis = 'x';
    double chunk_width = 5.0;
    double lattice_parameter = 3.20;

    double auto_search_factor = 1.80;
    int chunk_sample_stride = 1;
    int chunk_min_samples = 30;
    bool chunk_use_ptm_bcc_atoms = true;
    bool chunk_fill_empty_from_nearest = true;
    int chunk_smooth_half_window = 0;

    double max_unidentified_fraction = 0.005;
    bool keep_unidentified = false;
};

struct BDAResult {
    std::vector<int> defect;
    std::vector<int> coord;
    std::vector<int> chunk_id;
    std::vector<double> csp;
    std::vector<double> local_a;
    std::vector<double> local_cutoff;

    int count[7];

    BDAResult() {
        for (int i = 0; i < 7; i++) count[i] = 0;
    }
};

struct PTMAtomResult {
    int raw_ptm;
    int ptm_type;
    int vis_type;

    int omega_flag;
    int omega_site;
    int omega_confidence;

    double rmsd;
    double scale;
    double f_res[3];
    double f_res_scalar;

    double q[4];

    int bda_defect;
    int bda_coord;
    int bda_chunk_id;
    double bda_csp;
    double bda_local_a;
    double bda_local_cutoff;

    double interatomic_distance;
    double lattice_constant;
};

struct BDANeighbor {
    int idx;
    double d;
    double d2;
    double dr[3];
};
#endif
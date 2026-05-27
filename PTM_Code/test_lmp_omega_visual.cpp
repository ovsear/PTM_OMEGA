#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstddef>

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <map>
#include <chrono>

#include <limits>
#include <numeric>

#include "ptm_functions.h"
#include "ptm_constants.h"

#include "ptm_neighbor_finder.h"

#include "ptm_analysis_types.h"
#include "ptm_bda_classifier.h"

#include "ptm_runner.h"
#include "ptm_io_lammps.h"

#ifdef _OPENMP
#include <omp.h>
#endif

// ============================================================
// User parameters
// ============================================================

static const int PTM_FLAGS =
    PTM_CHECK_SC |
    PTM_CHECK_FCC |
    PTM_CHECK_HCP |
    PTM_CHECK_BCC |
//    PTM_CHECK_ICO |
//    PTM_CHECK_DCUB |
//    PTM_CHECK_DHEX |
//    PTM_CHECK_GRAPHENE |
    PTM_CHECK_OMEGA_A |
    PTM_CHECK_OMEGA_B;

static const double RMSD_CUTOFF_FCC = 0.10;
static const double RMSD_CUTOFF_HCP = 0.10;
static const double RMSD_CUTOFF_BCC = 0.10;
static const double RMSD_CUTOFF_ICO = 0.08;
static const double RMSD_CUTOFF_SC  = 0.06;

static const double RMSD_CUTOFF_DCUB     = 0.10;
static const double RMSD_CUTOFF_DHEX     = 0.10;
static const double RMSD_CUTOFF_GRAPHENE = 0.08;

static const double OMEGA_RMSD_STRICT = 0.06;
static const double OMEGA_RMSD_LOOSE  = 0.10;

// Print first N atoms in terminal for sanity check.
static const int NUM_PRINT_ATOMS = 10;

static const double USER_BCC_A0_GUESS = 3.20;
static const double PTM_TOPK_SEARCH_FACTOR = 2.0;

static const bool DEBUG_NEIGHBOR_CACHE = false;
static const bool DEBUG_PTM_INDEX = false;

// ============================================================
// Data structures
// ============================================================
static double elapsed_sec(const std::chrono::high_resolution_clock::time_point& a,
                          const std::chrono::high_resolution_clock::time_point& b)
{
    return std::chrono::duration<double>(b - a).count();
}
// ============================================================
// PTM-based BDA module
// ============================================================

static int count_insufficient_ptm_neighbors(const SystemData& sys, int required_points)
{
    int bad = 0;
    int min_count = required_points;
    int max_count = 0;

    for (size_t i = 0; i < sys.neigh_count.size(); i++) {
        int c = sys.neigh_count[i];
        if (c < required_points) bad++;
        if (i == 0 || c < min_count) min_count = c;
        if (c > max_count) max_count = c;
    }

    std::printf(
        "PTM neighbor cache check: required=%d bad=%d min_count=%d max_count=%d\n",
        required_points, bad, min_count, max_count
    );

    return bad;
}

// ============================================================
// Main per-file PTM analysis
// ============================================================

static void run_file(const char* filename)
{
    auto t0 = std::chrono::high_resolution_clock::now();

    SystemData sys;
    if (!read_lammps_atomic_auto(filename, sys, -1)) {
        std::fprintf(stderr, "ERROR: failed to read %s\n", filename);
        return;
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    std::printf("\n===== %s =====\n", filename);
    std::printf("atoms = %zu\n", sys.atoms.size());

    double a_search_guess = USER_BCC_A0_GUESS;

    // For BCC, the 19th neighbor lies around sqrt(2)*a.
    // 2.0*a is conservative for distorted/shocked BCC.
    double neighbor_max_cutoff = PTM_TOPK_SEARCH_FACTOR * a_search_guess;

    CellListNeighborFinder finder;

    int bad_neighbors = 0;
    for (int attempt = 0; attempt < 5; attempt++) {
        finder.build(sys, neighbor_max_cutoff);
        build_neighbour_cache(sys, finder, PTM_MAX_INPUT_POINTS);

        int bad_unique = 0;

        for (int i = 0; i < (int)sys.atoms.size(); i++) {
            std::vector<size_t> ids;

            const NeighborEntry* nb =
                &sys.neigh_cache[(size_t)i * (size_t)sys.neigh_stride];

            for (int k = 0; k < sys.neigh_count[i]; k++) {
                ids.push_back(nb[k].idx);
            }

            std::sort(ids.begin(), ids.end());

            for (size_t k = 1; k < ids.size(); k++) {
                if (ids[k] == ids[k - 1]) {
                    bad_unique++;
                    break;
                }
            }
        }

        std::printf("PTM neighbor cache unique check: bad_unique=%d\n", bad_unique);

        if (DEBUG_NEIGHBOR_CACHE)
        {
            int i = 0;
            const NeighborEntry* nb =
                &sys.neigh_cache[(size_t)i * (size_t)sys.neigh_stride];

            std::printf("DEBUG cache atom 0: count=%d stride=%d\n",
                        sys.neigh_count[i], sys.neigh_stride);

            for (int k = 0; k < sys.neigh_count[i] && k < 19; k++) {
                std::printf(
                    "  cache k=%2d idx=%zu type=%d d=%.8g dr=(%.8g %.8g %.8g)\n",
                    k,
                    nb[k].idx,
                    (int)nb[k].number,
                    std::sqrt(nb[k].d2),
                    nb[k].dr[0],
                    nb[k].dr[1],
                    nb[k].dr[2]
                );
            }
        }

        std::printf(
            "Neighbor finder attempt %d: a_search_guess=%.8g max_cutoff=%.8g cells=%d x %d x %d\n",
            attempt,
            a_search_guess,
            neighbor_max_cutoff,
            finder.nx(),
            finder.ny(),
            finder.nz()
        );

        bad_neighbors = count_insufficient_ptm_neighbors(sys, PTM_MAX_INPUT_POINTS);

        if (bad_neighbors == 0) {
            break;
        }

        neighbor_max_cutoff *= 1.25;
    }

    if (bad_neighbors > 0) {
        std::printf(
            "WARNING: %d atoms still have fewer than PTM_MAX_INPUT_POINTS neighbors after adaptive cutoff.\n",
            bad_neighbors
        );
    }

    auto t2 = std::chrono::high_resolution_clock::now();

    // Multi-threaded with OpenMP.

    std::vector<PTMAtomResult> ptm_results;

    if (DEBUG_PTM_INDEX)
    {
        int ordering[PTM_MAX_INPUT_POINTS];
        size_t nbr_indices[PTM_MAX_INPUT_POINTS];
        int32_t numbers[PTM_MAX_INPUT_POINTS];
        double nbr_pos[PTM_MAX_INPUT_POINTS][3];

        int nret = get_neighbours_lmp(
            &sys,
            0,
            0,
            PTM_MAX_INPUT_POINTS,
            ordering,
            nbr_indices,
            numbers,
            nbr_pos
        );

        std::printf("DEBUG callback atom 0: nret=%d\n", nret);

        for (int k = 0; k < nret; k++) {
            double d2 =
                nbr_pos[k][0] * nbr_pos[k][0] +
                nbr_pos[k][1] * nbr_pos[k][1] +
                nbr_pos[k][2] * nbr_pos[k][2];

            std::printf(
                "  k=%2d ordering=%2d idx=%zu type=%d dr=(%.8g %.8g %.8g) d=%.8g\n",
                k,
                ordering[k],
                nbr_indices[k],
                (int)numbers[k],
                nbr_pos[k][0],
                nbr_pos[k][1],
                nbr_pos[k][2],
                std::sqrt(d2)
            );
        }
    }

    PTMRunConfig ptm_cfg;
    ptm_cfg.flags = PTM_FLAGS;
    ptm_cfg.output_conventional_orientation = true;

    ptm_cfg.rmsd_cutoff_sc = 0.10;
    ptm_cfg.rmsd_cutoff_fcc = 0.10;
    ptm_cfg.rmsd_cutoff_hcp = 0.10;
    ptm_cfg.rmsd_cutoff_bcc = 0.10;
    ptm_cfg.rmsd_cutoff_ico = 0.10;

    ptm_cfg.omega_rmsd_strict = 0.10;
    ptm_cfg.omega_rmsd_loose = 0.13;

    ptm_cfg.debug_ptm_index = false;
    ptm_cfg.debug_first_n_atoms = 10;

    run_ptm_for_system(sys, ptm_cfg, ptm_results);

    auto t3 = std::chrono::high_resolution_clock::now();

    // ------------------------------------------------------------
    // PTM-based BDA after PTM classification
    // ------------------------------------------------------------
    BDAConfig bda_cfg;
    
    double ptm_global_a = bda_median_ptm_bcc_lattice(ptm_results, USER_BCC_A0_GUESS);

    if (!std::isfinite(ptm_global_a) || ptm_global_a < 2.0 || ptm_global_a > 5.0) {
        std::printf(
            "WARNING: invalid ptm_global_a=%.8g, fallback to USER_BCC_A0_GUESS=%.8g\n",
            ptm_global_a,
            USER_BCC_A0_GUESS
        );
        ptm_global_a = USER_BCC_A0_GUESS;
    }

    bda_cfg.lattice_parameter = ptm_global_a;

    bda_cfg.chunk_smooth_half_window = 0;
    bda_cfg.chunk_axis = 'x';
    bda_cfg.chunk_width = 10.0;
    bda_cfg.auto_search_factor = 1.80;
    bda_cfg.chunk_min_samples = 30;
    bda_cfg.chunk_use_ptm_bcc_atoms = true;

    BDAResult bda = run_ptm_based_bda(sys, finder, ptm_results, bda_cfg);

    for (size_t i = 0; i < ptm_results.size(); i++) {
        ptm_results[i].bda_defect = bda.defect[i];
        ptm_results[i].bda_coord = bda.coord[i];
        ptm_results[i].bda_chunk_id = bda.chunk_id[i];
        ptm_results[i].bda_csp = bda.csp[i];
        ptm_results[i].bda_local_a = bda.local_a[i];
        ptm_results[i].bda_local_cutoff = bda.local_cutoff[i];
    }

    print_bda_debug_summary(bda);

    std::map<int, int> counts;
    std::map<int, double> rmsd_sum;
    std::map<int, double> rmsd_max;
    std::map<int, int> raw_counts;

    for (size_t i = 0; i < sys.atoms.size(); i++) {
        int raw_type = ptm_results[i].raw_ptm;
        int type = ptm_results[i].ptm_type;
        double rmsd = ptm_results[i].rmsd;

        raw_counts[raw_type]++;

        counts[type]++;
        rmsd_sum[type] += rmsd;

        if (counts[type] == 1 || rmsd > rmsd_max[type]) {
            rmsd_max[type] = rmsd;
        }

        if ((int)i < NUM_PRINT_ATOMS) {
            std::printf(
                "atom %4zu id=%4d atom_type=%d -> raw=%-8s filtered=%-8s "
                "conf=%d rmsd=%.8g scale=%.8g F_res=%.8g\n",
                i,
                sys.atoms[i].id,
                sys.atoms[i].type,
                type_name(raw_type),
                type_name(type),
                ptm_results[i].omega_confidence,
                ptm_results[i].rmsd,
                ptm_results[i].scale,
                ptm_results[i].f_res_scalar
            );
        }
    }

    // End of Multi-threaded section.

    std::printf("\nRaw summary before omega cutoff:\n");
    for (std::map<int, int>::const_iterator it = raw_counts.begin(); it != raw_counts.end(); ++it) {
        std::printf(
            "%-10s type=%2d count=%5d\n",
            type_name(it->first),
            it->first,
            it->second
        );
    }

    std::printf("\nFiltered summary:\n");
    for (std::map<int, int>::const_iterator it = counts.begin(); it != counts.end(); ++it) {
        int t = it->first;
        int c = it->second;

        double mean_rmsd = 0.0;
        if (c > 0) {
            mean_rmsd = rmsd_sum[t] / c;
        }

        std::printf(
            "%-10s type=%2d count=%5d mean_rmsd=%.8g max_rmsd=%.8g\n",
            type_name(t),
            t,
            c,
            mean_rmsd,
            rmsd_max[t]
        );
    }

    auto t4 = std::chrono::high_resolution_clock::now();

    std::string outdump = make_output_dump_name(filename);
    write_ptm_dump(outdump, sys, ptm_results);

    auto t5 = std::chrono::high_resolution_clock::now();

    std::fprintf(stderr,
    "Timing: read=%.6f s, build_neigh=%.6f s, ptm=%.6f s, bda=%.6f s, write=%.6f s, total=%.6f s\n",
    elapsed_sec(t0, t1),
    elapsed_sec(t1, t2),
    elapsed_sec(t2, t3),
    elapsed_sec(t3, t4),
    elapsed_sec(t4, t5),
    elapsed_sec(t0, t5)
);
}

// ============================================================
// Program entry
// ============================================================

int main(int argc, char** argv)
{
    int ret = ptm_initialize_global();

    std::printf("ptm_initialize_global ret = %d\n", ret);

    if (ret != 0) {
        return ret;
    }

    if (argc < 2) {
        std::printf("\nUsage:\n");
        std::printf("  ./test_lmp_omega_visual file1.lmp [file2.lmp ...]\n\n");

        std::printf("Example:\n");
        std::printf("  ./test_lmp_omega_visual ./testModel/C32_omega_ideal.lmp ./testModel/C32_omega_zhang.lmp\n\n");

        return 0;
    }

    for (int i = 1; i < argc; i++) {
        run_file(argv[i]);
    }

    return 0;
}

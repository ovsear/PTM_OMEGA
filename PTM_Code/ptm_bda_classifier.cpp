#include "ptm_bda_classifier.h"

#include <cstdio>
#include <cmath>
#include <algorithm>
#include <limits>
#include <numeric>
#include <chrono>

#include "ptm_constants.h"

static double elapsed_sec(const std::chrono::high_resolution_clock::time_point& a,
                          const std::chrono::high_resolution_clock::time_point& b)
{
    return std::chrono::duration<double>(b - a).count();
}

static double bda_axis_coord(const SystemData& sys, int i, char axis)
{
    if (axis == 'x' || axis == 'X') return sys.atoms[i].x[0];
    if (axis == 'y' || axis == 'Y') return sys.atoms[i].x[1];
    return sys.atoms[i].x[2];
}

static double bda_median(std::vector<double> vals, double fallback)
{
    vals.erase(
        std::remove_if(vals.begin(), vals.end(),
            [](double v) { return !std::isfinite(v); }),
        vals.end()
    );

    if (vals.empty()) return fallback;

    std::sort(vals.begin(), vals.end());
    size_t n = vals.size();

    if (n % 2 == 1) return vals[n / 2];
    return 0.5 * (vals[n / 2 - 1] + vals[n / 2]);
}

static void bda_fill_nan_by_nearest(std::vector<double>& values, double fallback)
{
    const int n = (int)values.size();
    std::vector<int> valid;

    for (int i = 0; i < n; i++) {
        if (std::isfinite(values[i])) valid.push_back(i);
    }

    if (valid.empty()) {
        for (int i = 0; i < n; i++) values[i] = fallback;
        return;
    }

    for (int i = 0; i < n; i++) {
        if (std::isfinite(values[i])) continue;

        int best = valid[0];
        int best_dist = std::abs(valid[0] - i);

        for (size_t k = 1; k < valid.size(); k++) {
            int dd = std::abs(valid[k] - i);
            if (dd < best_dist) {
                best = valid[k];
                best_dist = dd;
            }
        }

        values[i] = values[best];
    }
}

static void bda_median_smooth(std::vector<double>& values, int half_window)
{
    if (half_window <= 0) return;

    const int n = (int)values.size();
    std::vector<double> out = values;

    for (int i = 0; i < n; i++) {
        std::vector<double> vv;
        int lo = std::max(0, i - half_window);
        int hi = std::min(n, i + half_window + 1);

        for (int j = lo; j < hi; j++) {
            if (std::isfinite(values[j])) vv.push_back(values[j]);
        }

        if (!vv.empty()) out[i] = bda_median(vv, values[i]);
    }

    values.swap(out);
}

static bool bda_is_ptm_bcc_for_cutoff(const PTMAtomResult& r)
{
    return r.ptm_type == PTM_MATCH_BCC || r.raw_ptm == PTM_MATCH_BCC;
}

static double bda_ptm_lattice_guess_one(const PTMAtomResult& r)
{
    if (r.ptm_type != PTM_MATCH_BCC && r.raw_ptm != PTM_MATCH_BCC) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    if (std::isfinite(r.lattice_constant) && r.lattice_constant > 0.0) {
        return r.lattice_constant;
    }

    if (std::isfinite(r.interatomic_distance) && r.interatomic_distance > 0.0) {
        return 2.0 / std::sqrt(3.0) * r.interatomic_distance;
    }

    return std::numeric_limits<double>::quiet_NaN();
}

// ============================================================
// PTM phase handling for BDA
// ============================================================
//
// BDA is intended to classify BCC defects, not transformed FCC/HCP/omega phases.
// Therefore, clear phase products are excluded from BDA defect rules.
// They remain visible through raw_ptm / ptm_type in the output.

static int bda_effective_ptm_type(const PTMAtomResult& r)
{
    /*
    // Prefer filtered PTM type.
    if (r.ptm_type != PTM_MATCH_NONE) {
        return r.ptm_type;
    }

    // If filtered type is NONE but raw type is a clear phase,
    // keep raw phase identity for BDA exclusion.
    return r.raw_ptm;
    */
    return r.ptm_type;
}

static bool bda_is_excluded_phase_for_bda(const PTMAtomResult& r)
{
    int t = bda_effective_ptm_type(r);

    return t == PTM_MATCH_FCC ||
           t == PTM_MATCH_HCP;
}

// ============================================================
// PTM phase handling for BDA
// ============================================================


double bda_median_ptm_bcc_lattice(
    const std::vector<PTMAtomResult>& ptm,
    double fallback
)
{
    std::vector<double> vals;
    vals.reserve(ptm.size());

    for (size_t i = 0; i < ptm.size(); i++) {
        double a = bda_ptm_lattice_guess_one(ptm[i]);
        if (std::isfinite(a) && a > 0.0) {
            vals.push_back(a);
        }
    }

    return bda_median(vals, fallback);
}

static void bda_determine_chunk_cutoffs(
    const SystemData& sys,
    const std::vector<PTMAtomResult>& ptm,
    const BDAConfig& cfg,
    std::vector<int>& chunk_ids,
    std::vector<double>& local_a,
    std::vector<double>& local_cutoff
)
{
    const int n = (int)sys.atoms.size();

    chunk_ids.assign(n, 0);
    local_a.assign(n, cfg.lattice_parameter);
    local_cutoff.assign(n, ((std::sqrt(2.0) + 1.0) / 2.0) * cfg.lattice_parameter);

    if (n <= 0) return;

    double min_coord = bda_axis_coord(sys, 0, cfg.chunk_axis);
    for (int i = 1; i < n; i++) {
        min_coord = std::min(min_coord, bda_axis_coord(sys, i, cfg.chunk_axis));
    }

    const double width = cfg.chunk_width;
    if (width <= 0.0) return;

    int max_chunk = 0;
    for (int i = 0; i < n; i++) {
        int cid = (int)std::floor((bda_axis_coord(sys, i, cfg.chunk_axis) - min_coord) / width);
        if (cid < 0) cid = 0;
        chunk_ids[i] = cid;
        max_chunk = std::max(max_chunk, cid);
    }

    const int n_chunks = max_chunk + 1;

    std::vector<double> chunk_ptm_a(n_chunks, std::numeric_limits<double>::quiet_NaN());

    for (int c = 0; c < n_chunks; c++) {
        std::vector<double> vals;

        for (int i = 0; i < n; i++) {
            if (chunk_ids[i] != c) continue;

            double a = bda_ptm_lattice_guess_one(ptm[i]);
            if (std::isfinite(a) && a > 0.0) {
                vals.push_back(a);
            }
        }

        chunk_ptm_a[c] = bda_median(vals, std::numeric_limits<double>::quiet_NaN());
    }

    bda_fill_nan_by_nearest(chunk_ptm_a, cfg.lattice_parameter);

    std::vector<double> r14(n, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> r15(n, std::numeric_limits<double>::quiet_NaN());

    int stride = std::max(1, cfg.chunk_sample_stride);

    #pragma omp parallel for schedule(dynamic, 256)
    for (int i = 0; i < n; i += stride) {
        if (sys.neigh_stride <= 0 || sys.neigh_cache.empty()) continue;
        if (sys.neigh_count[i] < 16) continue;

        const NeighborEntry* nb =
            &sys.neigh_cache[(size_t)i * (size_t)sys.neigh_stride];

        // nb[0] = center
        // nb[14] = 14th nearest neighbor
        // nb[15] = 15th nearest neighbor
        double rr14 = std::sqrt(nb[14].d2);
        double rr15 = std::sqrt(nb[15].d2);

        if (rr15 > rr14 && rr14 > 0.0) {
            r14[i] = rr14;
            r15[i] = rr15;
        }
    }

    std::vector<int> global_candidates;
    global_candidates.reserve(n);

    for (int i = 0; i < n; i++) {
        if (std::isfinite(r14[i]) && std::isfinite(r15[i]) && r15[i] > r14[i]) {
            global_candidates.push_back(i);
        }
    }

    std::vector<int> global_bcc_candidates;
    if (cfg.chunk_use_ptm_bcc_atoms) {
        for (size_t k = 0; k < global_candidates.size(); k++) {
            int i = global_candidates[k];
            if (bda_is_ptm_bcc_for_cutoff(ptm[i])) {
                global_bcc_candidates.push_back(i);
            }
        }

        if ((int)global_bcc_candidates.size() >= std::max(10, cfg.chunk_min_samples)) {
            global_candidates.swap(global_bcc_candidates);
        }
    }

    std::vector<double> global_r14_values;
    std::vector<double> global_r15_values;

    for (size_t k = 0; k < global_candidates.size(); k++) {
        int i = global_candidates[k];
        global_r14_values.push_back(r14[i]);
        global_r15_values.push_back(r15[i]);
    }

    double global_r14 = bda_median(global_r14_values, cfg.lattice_parameter);
    double global_r15 = bda_median(
        global_r15_values,
        2.0 * ((std::sqrt(2.0) + 1.0) / 2.0) * cfg.lattice_parameter - global_r14
    );

    if (!(global_r15 > global_r14)) {
        global_r14 = cfg.lattice_parameter;
        global_r15 = std::sqrt(2.0) * cfg.lattice_parameter;
    }

    double global_cutoff = 0.5 * (global_r14 + global_r15);

    std::vector<double> chunk_a(n_chunks, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> chunk_r15(n_chunks, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> chunk_cutoff(n_chunks, std::numeric_limits<double>::quiet_NaN());

    for (int c = 0; c < n_chunks; c++) {
        std::vector<int> idx;
        std::vector<int> idx_bcc;

        for (int i = 0; i < n; i++) {
            if (chunk_ids[i] != c) continue;
            if (!std::isfinite(r14[i]) || !std::isfinite(r15[i]) || !(r15[i] > r14[i])) continue;

            idx.push_back(i);
            if (bda_is_ptm_bcc_for_cutoff(ptm[i])) idx_bcc.push_back(i);
        }

        if (cfg.chunk_use_ptm_bcc_atoms && (int)idx_bcc.size() >= cfg.chunk_min_samples) {
            idx.swap(idx_bcc);
        }

        if ((int)idx.size() >= cfg.chunk_min_samples) {
            std::vector<double> va, v15;
            for (size_t k = 0; k < idx.size(); k++) {
                va.push_back(r14[idx[k]]);
                v15.push_back(r15[idx[k]]);
            }

            double med_r14 = bda_median(va, global_r14);
            double med_r15 = bda_median(v15, global_r15);

            if (med_r15 > med_r14 && med_r14 > 0.0) {
                chunk_a[c] = med_r14;
                chunk_r15[c] = med_r15;
                chunk_cutoff[c] = 0.5 * (med_r14 + med_r15);
            }
        }
    }

    if (cfg.chunk_fill_empty_from_nearest) {
    bda_fill_nan_by_nearest(chunk_a, std::numeric_limits<double>::quiet_NaN());
    bda_fill_nan_by_nearest(chunk_r15, std::numeric_limits<double>::quiet_NaN());
    bda_fill_nan_by_nearest(chunk_cutoff, std::numeric_limits<double>::quiet_NaN());
    }

    for (int c = 0; c < n_chunks; c++) {
        if (!std::isfinite(chunk_a[c])) {
            chunk_a[c] = chunk_ptm_a[c];
        }

        if (!std::isfinite(chunk_a[c])) {
            chunk_a[c] = global_r14;
        }

        if (!std::isfinite(chunk_a[c]) || chunk_a[c] <= 0.0) {
            chunk_a[c] = cfg.lattice_parameter;
        }

        if (!std::isfinite(chunk_r15[c]) || chunk_r15[c] <= chunk_a[c]) {
            chunk_r15[c] = std::sqrt(2.0) * chunk_a[c];
        }

        if (!std::isfinite(chunk_cutoff[c]) || chunk_cutoff[c] <= chunk_a[c]) {
            chunk_cutoff[c] = 0.5 * (chunk_a[c] + chunk_r15[c]);
        }
    }

    bda_median_smooth(chunk_a, cfg.chunk_smooth_half_window);
    bda_median_smooth(chunk_r15, cfg.chunk_smooth_half_window);

    for (int c = 0; c < n_chunks; c++) {
        chunk_cutoff[c] = 0.5 * (chunk_a[c] + chunk_r15[c]);
    }

    for (int i = 0; i < n; i++) {
        int c = chunk_ids[i];
        local_a[i] = chunk_a[c];
        local_cutoff[i] = chunk_cutoff[c];
    }
}

static void bda_build_neighbor_lists(
    const CellListNeighborFinder& finder,
    const std::vector<double>& local_cutoff,
    std::vector<std::vector<int> >& neighbor_lists
)
{
    const int n = (int)local_cutoff.size();
    neighbor_lists.assign(n, std::vector<int>());

    #pragma omp parallel for schedule(dynamic, 256)
    for (int i = 0; i < n; i++) {
        std::vector<NeighborQueryResult> q;

        // BDA coordination only needs neighbor indices.
        // No distance sorting is needed here.
        finder.query_cutoff(i, local_cutoff[i], q, false);

        std::vector<int> lst;
        lst.reserve(q.size());

        for (size_t k = 0; k < q.size(); k++) {
            lst.push_back(q[k].idx);
        }

        neighbor_lists[i].swap(lst);
    }
}

static void bda_compute_coordination(
    const std::vector<std::vector<int> >& neighbor_lists,
    std::vector<int>& coord
)
{
    const int n = (int)neighbor_lists.size();
    coord.assign(n, 0);

    for (int i = 0; i < n; i++) {
        coord[i] = (int)neighbor_lists[i].size();
    }
}

static double bda_pair_sum_recursive(
    const std::vector<BDANeighbor>& nbs,
    bool used[8],
    int used_count
)
{
    if (used_count >= 8) return 0.0;

    int first = -1;
    for (int i = 0; i < 8; i++) {
        if (!used[i]) {
            first = i;
            break;
        }
    }

    if (first < 0) return 0.0;

    used[first] = true;
    double best = std::numeric_limits<double>::max();

    for (int j = first + 1; j < 8; j++) {
        if (used[j]) continue;

        used[j] = true;

        double sx = nbs[first].dr[0] + nbs[j].dr[0];
        double sy = nbs[first].dr[1] + nbs[j].dr[1];
        double sz = nbs[first].dr[2] + nbs[j].dr[2];

        double pair_val = sx * sx + sy * sy + sz * sz;
        double rest = bda_pair_sum_recursive(nbs, used, used_count + 2);

        best = std::min(best, pair_val + rest);

        used[j] = false;
    }

    used[first] = false;
    return best;
}

static double bda_compute_csp8_one_from_cache(const SystemData& sys, int i)
{
    if (sys.neigh_stride <= 0 || sys.neigh_cache.empty()) return 0.0;
    if (sys.neigh_count[i] < 9) return 0.0;

    const NeighborEntry* nb =
        &sys.neigh_cache[(size_t)i * (size_t)sys.neigh_stride];

    std::vector<BDANeighbor> first8;
    first8.reserve(8);

    for (int k = 1; k <= 8; k++) {
        BDANeighbor x;
        x.idx = (int)nb[k].idx;
        x.d2 = nb[k].d2;
        x.d = std::sqrt(nb[k].d2);
        x.dr[0] = nb[k].dr[0];
        x.dr[1] = nb[k].dr[1];
        x.dr[2] = nb[k].dr[2];
        first8.push_back(x);
    }

    bool used[8] = {false, false, false, false, false, false, false, false};
    return bda_pair_sum_recursive(first8, used, 0);
}

static void bda_compute_csp8_all(
    const SystemData& sys,
    std::vector<double>& csp
)
{
    const int n = (int)sys.atoms.size();
    csp.assign(n, 0.0);

    #pragma omp parallel for schedule(dynamic, 256)
    for (int i = 0; i < n; i++) {
        csp[i] = bda_compute_csp8_one_from_cache(sys, i);
    }
}

static int bda_struct_from_ptm(const PTMAtomResult& r)
{
    // For BDA, raw BCC and filtered BCC should both be treated as BCC-like.
    // This preserves distorted BCC atoms that may be rejected by the RMSD filter.
    if (r.ptm_type == PTM_MATCH_BCC || r.raw_ptm == PTM_MATCH_BCC) {
        return PTM_MATCH_BCC;
    }

    // If filtered type is NONE but raw type is FCC/HCP/OMEGA,
    // keep the raw phase identity so transformed phase atoms can be excluded.
    if (r.ptm_type == PTM_MATCH_NONE) {
        return r.raw_ptm;
    }

    return r.ptm_type;
}

static bool bda_nonperfect_ptm(
    int i,
    const std::vector<int>& atom_struct,
    const std::vector<int>& coord,
    const std::vector<double>& csp,
    const std::vector<PTMAtomResult>& ptm,
    const BDAConfig& cfg
)
{
    // if (atom_struct[i] != PTM_MATCH_BCC) return true;
    // if (coord[i] != 14) return true;
    // if (csp[i] > cfg.csp_candidate_threshold) return true;
    // if (ptm[i].rmsd > cfg.bcc_rmsd_candidate_threshold) return true;
    // return false;
    return atom_struct[i] != PTM_MATCH_BCC || coord[i] != 14;
}

static std::vector<int> bda_get_nonperfect_neighbors(
    int i,
    const std::vector<std::vector<int> >& neighbor_lists,
    const std::vector<char>& nonperfect
)
{
    std::vector<int> out;
    const std::vector<int>& lst = neighbor_lists[i];

    for (size_t k = 0; k < lst.size(); k++) {
        int nb = lst[k];
        if (nonperfect[nb]) out.push_back(nb);
    }

    return out;
}

static int bda_common_neighbor_defect(
    int i,
    const std::vector<std::vector<int> >& neighbor_lists,
    const std::vector<char>& nonperfect,
    const std::vector<int>& atom_defect
)
{
    int defect_count[7] = {0, 0, 0, 0, 0, 0, 0};
    std::vector<int> neighbors = bda_get_nonperfect_neighbors(i, neighbor_lists, nonperfect);

    for (size_t k = 0; k < neighbors.size(); k++) {
        int nb = neighbors[k];
        for (int d = 1; d <= 5; d++) {
            if (atom_defect[nb] == d) defect_count[d]++;
        }
    }

    int max_count = 0;
    int best_defect = BDA_ELS;
    int n_best = 0;

    for (int d = 1; d <= 5; d++) {
        if (defect_count[d] > max_count && defect_count[d] >= 3) {
            max_count = defect_count[d];
            best_defect = d;
            n_best = 1;
        }
        else if (defect_count[d] == max_count && defect_count[d] >= 3) {
            n_best++;
        }
    }

    if (n_best == 1) return best_defect;
    return BDA_ELS;
}

static std::vector<int> bda_classify(
    const std::vector<int>& atom_struct,
    const std::vector<int>& coord,
    const std::vector<double>& csp,
    const std::vector<std::vector<int> >& neighbor_lists,
    const std::vector<PTMAtomResult>& ptm,
    const BDAConfig& cfg
)
{
    const int n = (int)coord.size();

    std::vector<int> atom_defect(n, -1);
    std::vector<char> nonperfect(n, 0);
    std::vector<char> excluded_phase(n, 0);

    for (int i = 0; i < n; i++) {
        if (bda_is_excluded_phase_for_bda(ptm[i])) {
            excluded_phase[i] = 1;

            // Do not classify transformed FCC/HCP/omega atoms as BCC defects.
            // They remain identifiable through raw_ptm / ptm_type.
            atom_defect[i] = BDA_ELS;
            nonperfect[i] = 0;
            continue;
        }

        nonperfect[i] = bda_nonperfect_ptm(i, atom_struct, coord, csp, ptm, cfg) ? 1 : 0;
    }

    auto get_neighbors = [&](int i) -> std::vector<int> {
        return bda_get_nonperfect_neighbors(i, neighbor_lists, nonperfect);
    };

    auto is_neighbor2surface = [&](int i) -> bool {
        if (atom_struct[i] != PTM_MATCH_BCC && coord[i] < 14) {
            int count = 0;
            std::vector<int> neighbors = get_neighbors(i);

            for (size_t kk = 0; kk < neighbors.size(); kk++) {
                int nb = neighbors[kk];
                if (atom_struct[nb] != PTM_MATCH_BCC &&
                    (atom_defect[nb] == BDA_SRF || coord[nb] <= 11)) {
                    count++;
                    atom_defect[nb] = BDA_SRF;
                }
            }

            if (count >= 4) {
                atom_defect[i] = BDA_SRF;
                return true;
            }
        }

        return false;
    };

    auto is_surface = [&](int i) -> bool {
        if (atom_struct[i] != PTM_MATCH_BCC && (coord[i] <= 11 || is_neighbor2surface(i))) {
            atom_defect[i] = BDA_SRF;
            return true;
        }
        return false;
    };

    auto is_dislo = [&](int i) -> bool {
        int c = coord[i];

        if (c >= 12 && c != 14) {
            int nr_14 = 0;
            int nr_non14 = 0;
            std::vector<int> neighbors = get_neighbors(i);

            for (size_t kk = 0; kk < neighbors.size(); kk++) {
                int nb = neighbors[kk];
                if (coord[nb] != 14) nr_non14++;
                if (coord[nb] == 14) nr_14++;
            }

            if (nr_non14 > nr_14) {
                atom_defect[i] = BDA_DSL;
                return true;
            }

            return false;
        }
        else if (c == 14) {
            int nr_14 = 0;
            int nr_non14 = 0;
            std::vector<int> neighbors = get_neighbors(i);
            int nr_nonperfect = (int)neighbors.size();
            int nr_perfect = c - nr_nonperfect;

            for (size_t kk = 0; kk < neighbors.size(); kk++) {
                int nb = neighbors[kk];
                if (coord[nb] >= 12 && coord[nb] != 14) nr_non14++;
                if (coord[nb] == 14) nr_14++;
            }

            if (nr_non14 >= 4 && nr_14 <= 6 && nr_perfect <= 4) {
                atom_defect[i] = BDA_DSL;
                return true;
            }

            return false;
        }

        return false;
    };

    auto is_vac = [&](int i) -> bool {
        int c = coord[i];
        double s = csp[i];

        if (c == 13 && s < 1.0) {
            int nr_12_4 = 0;
            int nr_13 = 0;
            std::vector<int> neighbors = get_neighbors(i);
            int nr_perfect = c - (int)neighbors.size();

            for (size_t kk = 0; kk < neighbors.size(); kk++) {
                int nb = neighbors[kk];
                if (atom_struct[nb] != PTM_MATCH_BCC && coord[nb] == 13 && csp[nb] > 4.0) nr_13++;
                if (atom_struct[nb] != PTM_MATCH_BCC && coord[nb] == 12 && csp[nb] > 4.0) nr_12_4++;
            }

            if ((nr_13 == 4 && nr_perfect == 9) ||
                (nr_13 == 2 && nr_12_4 == 2 && nr_perfect == 9)) {
                atom_defect[i] = BDA_VCN;
                return true;
            }

            return false;
        }

        if (c == 13 && s > 4.0) {
            int nr_12_1 = 0, nr_12_4 = 0;
            int nr_13_1 = 0, nr_13_4 = 0, nr_13 = 0;
            std::vector<int> neighbors = get_neighbors(i);
            int nr_perfect = c - (int)neighbors.size();

            for (size_t kk = 0; kk < neighbors.size(); kk++) {
                int nb = neighbors[kk];

                if (atom_struct[nb] != PTM_MATCH_BCC && coord[nb] == 12 && csp[nb] < 1.0) nr_12_1++;
                if (atom_struct[nb] != PTM_MATCH_BCC && coord[nb] == 12 && csp[nb] > 4.0) nr_12_4++;
                if (atom_struct[nb] != PTM_MATCH_BCC && coord[nb] == 13 && csp[nb] < 1.0) nr_13_1++;
                if (atom_struct[nb] != PTM_MATCH_BCC && coord[nb] == 13 && csp[nb] > 4.0) nr_13_4++;
                if (atom_struct[nb] != PTM_MATCH_BCC && coord[nb] == 13) nr_13++;
            }

            if ((nr_13_1 == 3 && nr_13_4 == 3 && nr_perfect == 7) ||
                (nr_12_1 == 2 && nr_12_4 == 2 && nr_13_1 == 1 && nr_13_4 == 1 && nr_perfect == 7) ||
                (nr_13 == 6 && nr_perfect == 7) ||
                (nr_13_4 == 4 && nr_perfect > 7)) {
                atom_defect[i] = BDA_VCN;
                return true;
            }

            return false;
        }

        if (c == 12 && s > 4.0) {
            int nr_12_1 = 0, nr_12_4 = 0;
            int nr_13_1 = 0, nr_13_4 = 0;
            std::vector<int> neighbors = get_neighbors(i);
            int nr_perfect = c - (int)neighbors.size();

            for (size_t kk = 0; kk < neighbors.size(); kk++) {
                int nb = neighbors[kk];

                if (atom_struct[nb] != PTM_MATCH_BCC && coord[nb] == 12 && csp[nb] < 1.0) nr_12_1++;
                if (atom_struct[nb] != PTM_MATCH_BCC && coord[nb] == 12 && csp[nb] > 4.0) nr_12_4++;
                if (atom_struct[nb] != PTM_MATCH_BCC && coord[nb] == 13 && csp[nb] < 1.0) nr_13_1++;
                if (atom_struct[nb] != PTM_MATCH_BCC && coord[nb] == 13 && csp[nb] > 4.0) nr_13_4++;
            }

            if (nr_12_1 == 2 && nr_12_4 == 1 &&
                nr_13_1 == 2 && nr_13_4 == 4 && nr_perfect == 3) {
                atom_defect[i] = BDA_VCN;
                return true;
            }

            return false;
        }

        if (c == 12 && s < 1.0) {
            int nr_12_4 = 0;
            int nr_13_4 = 0;
            std::vector<int> neighbors = get_neighbors(i);
            int nr_perfect = c - (int)neighbors.size();

            for (size_t kk = 0; kk < neighbors.size(); kk++) {
                int nb = neighbors[kk];

                if (atom_struct[nb] != PTM_MATCH_BCC && coord[nb] == 12 && csp[nb] > 4.0) nr_12_4++;
                if (atom_struct[nb] != PTM_MATCH_BCC && coord[nb] == 13 && csp[nb] > 4.0) nr_13_4++;
            }

            if (nr_12_4 == 2 && nr_13_4 == 4 && nr_perfect == 6) {
                atom_defect[i] = BDA_VCN;
                return true;
            }

            return false;
        }

        return false;
    };

    auto is_twin = [&](int i) -> bool {
        int c = coord[i];
        double s = csp[i];

        if (c == 13 && s > 4.5) {
            int nr_13 = 0;
            int nr_14 = 0;
            std::vector<int> neighbors = get_neighbors(i);
            int nr_perfect = c - (int)neighbors.size();

            for (size_t kk = 0; kk < neighbors.size(); kk++) {
                int nb = neighbors[kk];
                if (atom_struct[nb] != PTM_MATCH_BCC && coord[nb] == 13) nr_13++;
                if (atom_struct[nb] != PTM_MATCH_BCC && coord[nb] == 14) nr_14++;
            }

            if (nr_13 == 5 && nr_14 == 2 && nr_perfect == 6) {
                atom_defect[i] = BDA_TWN;
                return true;
            }
        }

        if (c == 14 && s > 8.0) {
            std::vector<int> neighbors = get_neighbors(i);
            int nr_perfect = c - (int)neighbors.size();

            if (nr_perfect <= 8) {
                atom_defect[i] = BDA_TWN;
                return true;
            }
        }

        if (c == 14) {
            int nr_13 = 0;
            int nr_14 = 0;
            std::vector<int> neighbors = get_neighbors(i);
            int nr_perfect = c - (int)neighbors.size();

            for (size_t kk = 0; kk < neighbors.size(); kk++) {
                int nb = neighbors[kk];
                if (atom_struct[nb] != PTM_MATCH_BCC && coord[nb] == 13) nr_13++;
                if (atom_struct[nb] != PTM_MATCH_BCC && coord[nb] == 14) nr_14++;
            }

            if (nr_perfect >= 6 && nr_perfect <= 9 &&
                (nr_14 >= 4 || (nr_13 == 4 && nr_14 == 2))) {
                atom_defect[i] = BDA_TWN;
                return true;
            }

            return false;
        }

        if (c == 13 && s < 1.0) {
            int nr_14 = 0;
            std::vector<int> neighbors = get_neighbors(i);

            for (size_t kk = 0; kk < neighbors.size(); kk++) {
                int nb = neighbors[kk];
                if (atom_struct[nb] != PTM_MATCH_BCC &&
                    coord[nb] == 14 && csp[nb] > 8.0) {
                    nr_14++;
                }
            }

            if (nr_14 == 4) {
                atom_defect[i] = BDA_TWN;
                return true;
            }

            return false;
        }

        return false;
    };

    auto is_planarfault = [&](int i) -> bool {
        int c = coord[i];
        int st = atom_struct[i];

        if (st != PTM_MATCH_BCC && c == 12) {
            int nr_12 = 0;
            int nr_13 = 0;
            std::vector<int> neighbors = get_neighbors(i);
            int nr_perfect = c - (int)neighbors.size();

            for (size_t kk = 0; kk < neighbors.size(); kk++) {
                int nb = neighbors[kk];
                if (coord[nb] == 12) nr_12++;
                if (coord[nb] == 13) nr_13++;
            }

            if (nr_12 >= 9 && nr_perfect == 0) {
                atom_defect[i] = BDA_PLF;
                return true;
            }
            else if ((3 <= nr_12 && nr_12 <= 6) &&
                     (7 <= nr_13 && nr_13 <= 9) &&
                     nr_perfect == 0) {
                atom_defect[i] = BDA_PLF;
                return true;
            }
            else if (nr_12 >= 6 && nr_13 >= 3 && nr_perfect == 0) {
                atom_defect[i] = BDA_PLF;
                return true;
            }

            return false;
        }

        if (c == 13) {
            int nr_12 = 0;
            int nr_13 = 0;
            std::vector<int> neighbors = get_neighbors(i);
            int nr_perfect = c - (int)neighbors.size();

            for (size_t kk = 0; kk < neighbors.size(); kk++) {
                int nb = neighbors[kk];
                if (coord[nb] == 12) nr_12++;
                if (coord[nb] == 13) nr_13++;
            }

            if (nr_12 + nr_13 == 9 && nr_13 >= 7) {
                atom_defect[i] = BDA_PLF;
                return true;
            }
            else if (nr_13 == 6 && nr_12 == 3 && nr_perfect == 4) {
                atom_defect[i] = BDA_PLF;
                return true;
            }
            else if (nr_13 == 6 && nr_12 <= 1 && nr_perfect >= 6) {
                atom_defect[i] = BDA_PLF;
                return true;
            }
            else if (nr_13 >= 7 && nr_12 <= 4 && nr_perfect <= 3) {
                atom_defect[i] = BDA_PLF;
                return true;
            }

            return false;
        }

        return false;
    };

    std::vector<int> identified;
    std::vector<int> unidentified;
    int defect_atoms = 0;

    for (int i = 0; i < n; i++) {
        if (atom_defect[i] != -1) continue;

        if (nonperfect[i]) {
            defect_atoms++;

            if (is_surface(i)) {
                // already assigned
            }
            else if (is_vac(i)) {
                identified.push_back(i);
            }
            else if (is_twin(i)) {
                identified.push_back(i);
            }
            else if (is_planarfault(i)) {
                identified.push_back(i);
            }
            else if (is_dislo(i)) {
                identified.push_back(i);
            }
            else {
                atom_defect[i] = BDA_ELS;
                unidentified.push_back(i);
            }
        }
        else {
            atom_defect[i] = BDA_BLK;
        }
    }

    for (size_t k = 0; k < identified.size(); k++) {
        int i = identified[k];
        int cd = bda_common_neighbor_defect(i, neighbor_lists, nonperfect, atom_defect);
        if (atom_defect[i] != cd) {
            unidentified.push_back(i);
        }
    }

    for (size_t k = 0; k < unidentified.size(); k++) {
        atom_defect[unidentified[k]] = BDA_ELS;
    }

    if (defect_atoms > 0 && !cfg.keep_unidentified) {
        int previous_len = -1;

        while (((double)unidentified.size() / (double)defect_atoms > cfg.max_unidentified_fraction) &&
               ((int)unidentified.size() != previous_len)) {
            previous_len = (int)unidentified.size();

            std::vector<int> old_list = unidentified;
            unidentified.clear();

            for (size_t k = 0; k < old_list.size(); k++) {
                int i = old_list[k];
                int cd = bda_common_neighbor_defect(i, neighbor_lists, nonperfect, atom_defect);

                if (atom_defect[i] != cd) {
                    atom_defect[i] = cd;
                }
                else {
                    atom_defect[i] = BDA_ELS;
                    unidentified.push_back(i);
                }
            }
        }
    }

    for (int i = 0; i < n; i++) {
        if (atom_defect[i] < 0) atom_defect[i] = BDA_ELS;
    }

    return atom_defect;
}

BDAResult run_ptm_based_bda(
    const SystemData& sys,
    const CellListNeighborFinder& finder,
    const std::vector<PTMAtomResult>& ptm,
    const BDAConfig& cfg
)
{
    const int n = (int)sys.atoms.size();

    BDAResult out;
    out.defect.assign(n, BDA_ELS);
    out.coord.assign(n, 0);
    out.chunk_id.assign(n, 0);
    out.csp.assign(n, 0.0);
    out.local_a.assign(n, cfg.lattice_parameter);
    out.local_cutoff.assign(n, ((std::sqrt(2.0) + 1.0) / 2.0) * cfg.lattice_parameter);

    if (n <= 0) return out;

    auto tb0 = std::chrono::high_resolution_clock::now();

    bda_determine_chunk_cutoffs(sys, ptm, cfg, out.chunk_id, out.local_a, out.local_cutoff);

    auto tb1 = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<int> > neighbor_lists;
    bda_build_neighbor_lists(finder, out.local_cutoff, neighbor_lists);

    auto tb2 = std::chrono::high_resolution_clock::now();

    bda_compute_coordination(neighbor_lists, out.coord);

    auto tb3 = std::chrono::high_resolution_clock::now();

    bda_compute_csp8_all(sys, out.csp);

    auto tb4 = std::chrono::high_resolution_clock::now();

    std::vector<int> atom_struct(n, PTM_MATCH_NONE);
    for (int i = 0; i < n; i++) {
        atom_struct[i] = bda_struct_from_ptm(ptm[i]);
    }

    out.defect = bda_classify(atom_struct, out.coord, out.csp, neighbor_lists, ptm, cfg);

    auto tb5 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n; i++) {
        int d = out.defect[i];
        if (0 <= d && d <= 6) out.count[d]++;
    }

    std::fprintf(stderr,
        "BDA timing: chunk_cutoff=%.6f s, build_bda_neighbors=%.6f s, coord=%.6f s, csp=%.6f s, classify=%.6f s\n",
        elapsed_sec(tb0, tb1),
        elapsed_sec(tb1, tb2),
        elapsed_sec(tb2, tb3),
        elapsed_sec(tb3, tb4),
        elapsed_sec(tb4, tb5)
    );

    return out;
}

void print_bda_debug_summary(const BDAResult& bda)
{
    int n = (int)bda.defect.size();
    if (n <= 0) return;

    auto minmaxmean_double = [&](const std::vector<double>& v,
                                 double& vmin,
                                 double& vmax,
                                 double& vmean) {
        vmin = std::numeric_limits<double>::max();
        vmax = -std::numeric_limits<double>::max();
        vmean = 0.0;
        int cnt = 0;

        for (size_t i = 0; i < v.size(); i++) {
            if (!std::isfinite(v[i])) continue;
            vmin = std::min(vmin, v[i]);
            vmax = std::max(vmax, v[i]);
            vmean += v[i];
            cnt++;
        }

        if (cnt > 0) vmean /= (double)cnt;
        else {
            vmin = 0.0;
            vmax = 0.0;
            vmean = 0.0;
        }
    };

    double amin, amax, amean;
    double cmin, cmax, cmean;
    double smin, smax, smean;

    minmaxmean_double(bda.local_a, amin, amax, amean);
    minmaxmean_double(bda.local_cutoff, cmin, cmax, cmean);
    minmaxmean_double(bda.csp, smin, smax, smean);

    int coord_hist[41] = {0};
    int coord_gt40 = 0;
    for (size_t i = 0; i < bda.coord.size(); i++) {
        int c = bda.coord[i];
        if (0 <= c && c <= 40) coord_hist[c]++;
        else if (c > 40) coord_gt40++;
    }

    std::printf("BDA debug local_a: min=%g max=%g mean=%g\n", amin, amax, amean);
    std::printf("BDA debug cutoff : min=%g max=%g mean=%g\n", cmin, cmax, cmean);
    std::printf("BDA debug csp    : min=%g max=%g mean=%g\n", smin, smax, smean);

    std::printf("BDA coord histogram:");
    for (int c = 0; c <= 20; c++) {
        if (coord_hist[c] > 0) {
            std::printf(" %d:%d", c, coord_hist[c]);
        }
    }
    if (coord_gt40 > 0) std::printf(" >40:%d", coord_gt40);
    std::printf("\n");

    std::printf(
        "BDA counts: bulk=%d surface=%d vacancy=%d dislocation=%d twin=%d planar_fault=%d else=%d\n",
        bda.count[BDA_BLK],
        bda.count[BDA_SRF],
        bda.count[BDA_VCN],
        bda.count[BDA_DSL],
        bda.count[BDA_TWN],
        bda.count[BDA_PLF],
        bda.count[BDA_ELS]
    );
}

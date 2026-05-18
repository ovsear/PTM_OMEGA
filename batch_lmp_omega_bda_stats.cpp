#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <chrono>
#include <algorithm>

#include "ptm_functions.h"
#include "ptm_constants.h"
#include "ptm_initialize_data.h"

#include "ptm_neighbor_finder.h"
#include "ptm_analysis_types.h"
#include "ptm_runner.h"
#include "ptm_bda_classifier.h"
#include "ptm_io_lammps.h"

// ============================================================
// User defaults
// ============================================================

static const double USER_BCC_A0_GUESS = 3.20;
static const double PTM_TOPK_SEARCH_FACTOR = 2.0;

static const int PTM_FLAGS_FULL =
    PTM_CHECK_SC |
    PTM_CHECK_FCC |
    PTM_CHECK_HCP |
    PTM_CHECK_BCC |
    PTM_CHECK_ICO |
    PTM_CHECK_DCUB |
    PTM_CHECK_DHEX |
    PTM_CHECK_GRAPHENE |
    PTM_CHECK_OMEGA_A |
    PTM_CHECK_OMEGA_B;

static const int PTM_FLAGS_METALLIC_OMEGA =
    PTM_CHECK_FCC |
    PTM_CHECK_HCP |
    PTM_CHECK_BCC |
    PTM_CHECK_OMEGA_A |
    PTM_CHECK_OMEGA_B;

struct BatchOptions {
    std::vector<std::string> input_files;
    std::string list_file;
    std::string output_csv;

    bool write_visual = false;
    bool use_full_flags = false;
    bool verbose = false;
};

struct FrameSummary {
    std::string file;
    int status = 0;

    int natoms = 0;

    int count_none = 0;
    int count_fcc = 0;
    int count_hcp = 0;
    int count_bcc = 0;
    int count_ico = 0;
    int count_sc = 0;
    int count_dcub = 0;
    int count_dhex = 0;
    int count_graphene = 0;
    int count_omega_a = 0;
    int count_omega_b = 0;
    int count_unknown = 0;

    int omega_total = 0;
    int omega_strict = 0;
    int omega_loose = 0;

    int bda_bulk = 0;
    int bda_surface = 0;
    int bda_vacancy = 0;
    int bda_dislocation = 0;
    int bda_twin = 0;
    int bda_planar_fault = 0;
    int bda_else = 0;

    int omega_on_bda_twin = 0;
    int omega_on_bda_planar_fault = 0;
    int omega_on_bda_twin_or_planar_fault = 0;

    double omega_on_bda_twin_ratio = 0.0;
    double omega_on_bda_planar_fault_ratio = 0.0;
    double omega_on_bda_twin_or_planar_fault_ratio = 0.0;

    double mean_omega_rmsd = 0.0;
    double mean_omega_fres = 0.0;

    double read_time = 0.0;
    double build_neigh_time = 0.0;
    double ptm_time = 0.0;
    double bda_time = 0.0;
    double write_time = 0.0;
    double total_time = 0.0;
};

static double elapsed_sec(
    const std::chrono::high_resolution_clock::time_point& a,
    const std::chrono::high_resolution_clock::time_point& b
)
{
    return std::chrono::duration<double>(b - a).count();
}

static bool is_omega(int type)
{
    return type == PTM_MATCH_OMEGA_A || type == PTM_MATCH_OMEGA_B;
}

static void print_usage(const char* prog)
{
    std::fprintf(
        stderr,
        "Usage:\n"
        "  %s file1.lmp file2.lmp ... --out summary.csv\n"
        "  %s --list filelist.txt --out summary.csv\n\n"
        "Options:\n"
        "  --out FILE       Output CSV file. Default: summary.csv\n"
        "  --list FILE      Text file containing one input path per line.\n"
        "  --visual         Also write per atom OVITO dump for each frame.\n"
        "  --metallic       Use FCC/HCP/BCC/OMEGA_A/OMEGA_B flags only. Default.\n"
        "  --full           Use full PTM flags. \n"
        "  --verbose        Print per file progress.\n",
        prog, prog
    );
}

static bool parse_args(int argc, char** argv, BatchOptions& opt)
{
    opt.output_csv = "summary.csv";

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];

        if (a == "--out") {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "ERROR: --out requires a filename\n");
                return false;
            }
            opt.output_csv = argv[++i];
        }
        else if (a == "--list") {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "ERROR: --list requires a filename\n");
                return false;
            }
            opt.list_file = argv[++i];
        }
        else if (a == "--visual") {
            opt.write_visual = true;
        }
        else if (a == "--metallic") {
            opt.use_full_flags = false;
        }
        else if (a == "--full") {
            opt.use_full_flags = true;
        }
        else if (a == "--verbose") {
            opt.verbose = true;
        }
        else if (a == "--help" || a == "-h") {
            return false;
        }
        else {
            opt.input_files.push_back(a);
        }
    }

    if (!opt.list_file.empty()) {
        std::ifstream fin(opt.list_file.c_str());
        if (!fin) {
            std::fprintf(stderr, "ERROR: cannot open list file %s\n", opt.list_file.c_str());
            return false;
        }

        std::string line;
        while (std::getline(fin, line)) {
            if (line.empty()) continue;
            if (line[0] == '#') continue;
            opt.input_files.push_back(line);
        }
    }

    if (opt.input_files.empty()) {
        std::fprintf(stderr, "ERROR: no input files\n");
        return false;
    }

    return true;
}

static void setup_ptm_config(PTMRunConfig& cfg, bool use_full_flags)
{
    cfg.flags = use_full_flags ? PTM_FLAGS_FULL : PTM_FLAGS_METALLIC_OMEGA;
    cfg.output_conventional_orientation = true;

    cfg.rmsd_cutoff_sc = 0.10;
    cfg.rmsd_cutoff_fcc = 0.10;
    cfg.rmsd_cutoff_hcp = 0.10;
    cfg.rmsd_cutoff_bcc = 0.10;
    cfg.rmsd_cutoff_ico = 0.10;

    cfg.omega_rmsd_strict = 0.10;
    cfg.omega_rmsd_loose = 0.15;

    cfg.debug_ptm_index = false;
    cfg.debug_first_n_atoms = 0;
}

static void setup_bda_config(BDAConfig& cfg, double lattice_guess)
{
    cfg.lattice_parameter = lattice_guess;
    cfg.chunk_axis = 'x';
    cfg.chunk_width = 10.0;

    cfg.auto_search_factor = 1.80;
    cfg.chunk_sample_stride = 1;
    cfg.chunk_min_samples = 30;
    cfg.chunk_use_ptm_bcc_atoms = true;
    cfg.chunk_fill_empty_from_nearest = true;
    cfg.chunk_smooth_half_window = 0;

    cfg.max_unidentified_fraction = 0.005;
    cfg.keep_unidentified = false;
}

static bool build_finder_and_ptm_cache(
    SystemData& sys,
    CellListNeighborFinder& finder,
    double a_guess,
    bool verbose
)
{
    double search_guess = a_guess;
    if (search_guess <= 0.0 || !std::isfinite(search_guess)) {
        search_guess = USER_BCC_A0_GUESS;
    }

    double max_cutoff = search_guess * PTM_TOPK_SEARCH_FACTOR;

    for (int attempt = 0; attempt < 5; attempt++) {
        finder.build(sys, max_cutoff);
        build_neighbour_cache(sys, finder, PTM_MAX_INPUT_POINTS);

        int bad_count = 0;
        int bad_unique = 0;
        int min_count = PTM_MAX_INPUT_POINTS;
        int max_count = 0;

        for (size_t i = 0; i < sys.atoms.size(); i++) {
            int c = sys.neigh_count[i];
            min_count = std::min(min_count, c);
            max_count = std::max(max_count, c);

            if (c < PTM_MAX_INPUT_POINTS) {
                bad_count++;
            }

            const NeighborEntry* nb =
                &sys.neigh_cache[i * (size_t)sys.neigh_stride];

            std::vector<size_t> ids;
            ids.reserve((size_t)c);

            for (int k = 0; k < c; k++) {
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

        if (verbose) {
            std::printf(
                "Neighbor finder attempt %d: max_cutoff=%g cells=%d x %d x %d "
                "bad=%d bad_unique=%d min_count=%d max_count=%d\n",
                attempt,
                max_cutoff,
                finder.nx(),
                finder.ny(),
                finder.nz(),
                bad_count,
                bad_unique,
                min_count,
                max_count
            );
        }

        if (bad_count == 0 && bad_unique == 0) {
            return true;
        }

        max_cutoff *= 1.5;
    }

    return false;
}

static void count_ptm_type(FrameSummary& s, int type)
{
    switch (type) {
        case PTM_MATCH_NONE: s.count_none++; break;
        case PTM_MATCH_FCC: s.count_fcc++; break;
        case PTM_MATCH_HCP: s.count_hcp++; break;
        case PTM_MATCH_BCC: s.count_bcc++; break;
        case PTM_MATCH_ICO: s.count_ico++; break;
        case PTM_MATCH_SC: s.count_sc++; break;
        case PTM_MATCH_DCUB: s.count_dcub++; break;
        case PTM_MATCH_DHEX: s.count_dhex++; break;
        case PTM_MATCH_GRAPHENE: s.count_graphene++; break;
        case PTM_MATCH_OMEGA_A: s.count_omega_a++; break;
        case PTM_MATCH_OMEGA_B: s.count_omega_b++; break;
        default: s.count_unknown++; break;
    }
}

static void fill_summary_counts(
    FrameSummary& s,
    const std::vector<PTMAtomResult>& ptm,
    const BDAResult& bda
)
{
    double omega_rmsd_sum = 0.0;
    double omega_fres_sum = 0.0;

    for (size_t i = 0; i < ptm.size(); i++) {
        const PTMAtomResult& r = ptm[i];

        count_ptm_type(s, r.ptm_type);

        bool omega = is_omega(r.ptm_type);
        if (omega) {
            s.omega_total++;

            if (r.omega_confidence == 2) {
                s.omega_strict++;
            }
            else if (r.omega_confidence == 1) {
                s.omega_loose++;
            }

            omega_rmsd_sum += r.rmsd;
            omega_fres_sum += r.f_res_scalar;

            if (i < bda.defect.size()) {
                if (bda.defect[i] == BDA_TWN) {
                    s.omega_on_bda_twin++;
                }

                if (bda.defect[i] == BDA_PLF) {
                    s.omega_on_bda_planar_fault++;
                }

                if (bda.defect[i] == BDA_TWN || bda.defect[i] == BDA_PLF) {
                    s.omega_on_bda_twin_or_planar_fault++;
                }
            }
        }
    }

    s.bda_bulk = bda.count[BDA_BLK];
    s.bda_surface = bda.count[BDA_SRF];
    s.bda_vacancy = bda.count[BDA_VCN];
    s.bda_dislocation = bda.count[BDA_DSL];
    s.bda_twin = bda.count[BDA_TWN];
    s.bda_planar_fault = bda.count[BDA_PLF];
    s.bda_else = bda.count[BDA_ELS];

    if (s.omega_total > 0) {
        s.mean_omega_rmsd = omega_rmsd_sum / (double)s.omega_total;
        s.mean_omega_fres = omega_fres_sum / (double)s.omega_total;

        s.omega_on_bda_twin_ratio =
            (double)s.omega_on_bda_twin / (double)s.bda_twin;

        s.omega_on_bda_planar_fault_ratio =
            (double)s.omega_on_bda_planar_fault / (double)s.bda_planar_fault;

        s.omega_on_bda_twin_or_planar_fault_ratio =
            (double)s.omega_on_bda_twin_or_planar_fault / ((double)s.bda_twin + (double)s.bda_planar_fault);
    }
 
}

static FrameSummary process_one_file(
    const std::string& file,
    const BatchOptions& opt,
    const PTMRunConfig& ptm_cfg
)
{
    FrameSummary s;
    s.file = file;
    s.status = 0;

    auto t0 = std::chrono::high_resolution_clock::now();

    SystemData sys;

    auto t_read0 = std::chrono::high_resolution_clock::now();
    if (!read_lammps_atomic_auto(file.c_str(), sys, -1)) {
        s.status = 1;
        auto t_end = std::chrono::high_resolution_clock::now();
        s.total_time = elapsed_sec(t0, t_end);
        return s;
    }
    auto t_read1 = std::chrono::high_resolution_clock::now();

    s.natoms = (int)sys.atoms.size();

    double a_density = estimate_bcc_lattice_from_density(sys, USER_BCC_A0_GUESS);

    CellListNeighborFinder finder;

    auto t_neigh0 = std::chrono::high_resolution_clock::now();
    bool neigh_ok = build_finder_and_ptm_cache(sys, finder, a_density, opt.verbose);
    auto t_neigh1 = std::chrono::high_resolution_clock::now();

    if (!neigh_ok) {
        s.status = 2;
        auto t_end = std::chrono::high_resolution_clock::now();
        s.read_time = elapsed_sec(t_read0, t_read1);
        s.build_neigh_time = elapsed_sec(t_neigh0, t_neigh1);
        s.total_time = elapsed_sec(t0, t_end);
        return s;
    }

    std::vector<PTMAtomResult> ptm_results;

    auto t_ptm0 = std::chrono::high_resolution_clock::now();
    run_ptm_for_system(sys, ptm_cfg, ptm_results);
    auto t_ptm1 = std::chrono::high_resolution_clock::now();

    BDAConfig bda_cfg;
    double ptm_global_a = bda_median_ptm_bcc_lattice(ptm_results, USER_BCC_A0_GUESS);
    setup_bda_config(bda_cfg, ptm_global_a);

    auto t_bda0 = std::chrono::high_resolution_clock::now();
    BDAResult bda = run_ptm_based_bda(sys, finder, ptm_results, bda_cfg);
    auto t_bda1 = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < ptm_results.size(); i++) {
        ptm_results[i].bda_defect = bda.defect[i];
        ptm_results[i].bda_coord = bda.coord[i];
        ptm_results[i].bda_chunk_id = bda.chunk_id[i];
        ptm_results[i].bda_csp = bda.csp[i];
        ptm_results[i].bda_local_a = bda.local_a[i];
        ptm_results[i].bda_local_cutoff = bda.local_cutoff[i];
    }

    auto t_write0 = std::chrono::high_resolution_clock::now();
    if (opt.write_visual) {
        std::string outdump = make_output_dump_name(file);
        write_ptm_dump(outdump, sys, ptm_results);
    }
    auto t_write1 = std::chrono::high_resolution_clock::now();

    fill_summary_counts(s, ptm_results, bda);

    auto t_end = std::chrono::high_resolution_clock::now();

    s.read_time = elapsed_sec(t_read0, t_read1);
    s.build_neigh_time = elapsed_sec(t_neigh0, t_neigh1);
    s.ptm_time = elapsed_sec(t_ptm0, t_ptm1);
    s.bda_time = elapsed_sec(t_bda0, t_bda1);
    s.write_time = elapsed_sec(t_write0, t_write1);
    s.total_time = elapsed_sec(t0, t_end);

    return s;
}

static void write_csv_header(std::ofstream& fout)
{
    fout
        << "file,status,natoms,"
        << "none,fcc,hcp,bcc,ico,sc,dcub,dhex,graphene,omega_a,omega_b,unknown,"
        << "omega_total,omega_strict,omega_loose,"
        << "bda_bulk,bda_surface,bda_vacancy,bda_dislocation,bda_twin,bda_planar_fault,bda_else,"
        << "omega_on_bda_twin,omega_on_bda_planar_fault,omega_on_bda_twin_or_planar_fault,"
        << "omega_on_bda_twin_ratio,omega_on_bda_planar_fault_ratio,omega_on_bda_twin_or_planar_fault_ratio,"
        << "mean_omega_rmsd,mean_omega_fres,"
        << "read_time,build_neigh_time,ptm_time,bda_time,write_time,total_time\n";
}

static void write_csv_row(std::ofstream& fout, const FrameSummary& s)
{
    fout
        << s.file << ","
        << s.status << ","
        << s.natoms << ","

        << s.count_none << ","
        << s.count_fcc << ","
        << s.count_hcp << ","
        << s.count_bcc << ","
        << s.count_ico << ","
        << s.count_sc << ","
        << s.count_dcub << ","
        << s.count_dhex << ","
        << s.count_graphene << ","
        << s.count_omega_a << ","
        << s.count_omega_b << ","
        << s.count_unknown << ","

        << s.omega_total << ","
        << s.omega_strict << ","
        << s.omega_loose << ","

        << s.bda_bulk << ","
        << s.bda_surface << ","
        << s.bda_vacancy << ","
        << s.bda_dislocation << ","
        << s.bda_twin << ","
        << s.bda_planar_fault << ","
        << s.bda_else << ","

        << s.omega_on_bda_twin << ","
        << s.omega_on_bda_planar_fault << ","
        << s.omega_on_bda_twin_or_planar_fault << ","

        << s.omega_on_bda_twin_ratio << ","
        << s.omega_on_bda_planar_fault_ratio << ","
        << s.omega_on_bda_twin_or_planar_fault_ratio << ","

        << s.mean_omega_rmsd << ","
        << s.mean_omega_fres << ","

        << s.read_time << ","
        << s.build_neigh_time << ","
        << s.ptm_time << ","
        << s.bda_time << ","
        << s.write_time << ","
        << s.total_time << "\n";
}

int main(int argc, char** argv)
{
    BatchOptions opt;
    if (!parse_args(argc, argv, opt)) {
        print_usage(argv[0]);
        return 1;
    }

    int ret = ptm_initialize_global();
    std::printf("ptm_initialize_global ret = %d\n", ret);

    PTMRunConfig ptm_cfg;
    setup_ptm_config(ptm_cfg, opt.use_full_flags);

    std::ofstream fout(opt.output_csv.c_str());
    if (!fout) {
        std::fprintf(stderr, "ERROR: cannot write output CSV %s\n", opt.output_csv.c_str());
        return 1;
    }

    write_csv_header(fout);

    for (size_t i = 0; i < opt.input_files.size(); i++) {
        const std::string& file = opt.input_files[i];

        if (opt.verbose) {
            std::printf("[%zu/%zu] %s\n", i + 1, opt.input_files.size(), file.c_str());
        }

        FrameSummary s = process_one_file(file, opt, ptm_cfg);
        write_csv_row(fout, s);

        std::printf(
            "%s status=%d natoms=%d BCC=%d OMEGA_A=%d OMEGA_B=%d "
            "BDA_TWN=%d BDA_PLF=%d total=%.4f s\n",
            file.c_str(),
            s.status,
            s.natoms,
            s.count_bcc,
            s.count_omega_a,
            s.count_omega_b,
            s.bda_twin,
            s.bda_planar_fault,
            s.total_time
        );
    }

    std::printf("wrote summary CSV: %s\n", opt.output_csv.c_str());

    return 0;
}
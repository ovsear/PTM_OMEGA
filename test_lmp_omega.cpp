// test_lmp_omega.cpp
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

#include "ptm_functions.h"
#include "ptm_constants.h"
#include "ptm_initialize_data.h"

struct Atom {
    int id;
    int type;
    double x[3];
};

struct SystemData {
    std::vector<Atom> atoms;

    double xlo, xhi;
    double ylo, yhi;
    double zlo, zhi;
    double xy, xz, yz;

    double H[3][3];
    double invH[3][3];
    double origin[3];
};

static void invert_3x3(const double A[3][3], double invA[3][3])
{
    double det =
        A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
        A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
        A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);

    if (std::fabs(det) < 1e-14) {
        std::fprintf(stderr, "ERROR: singular box matrix\n");
        std::exit(1);
    }

    double idet = 1.0 / det;

    invA[0][0] =  (A[1][1] * A[2][2] - A[1][2] * A[2][1]) * idet;
    invA[0][1] = -(A[0][1] * A[2][2] - A[0][2] * A[2][1]) * idet;
    invA[0][2] =  (A[0][1] * A[1][2] - A[0][2] * A[1][1]) * idet;

    invA[1][0] = -(A[1][0] * A[2][2] - A[1][2] * A[2][0]) * idet;
    invA[1][1] =  (A[0][0] * A[2][2] - A[0][2] * A[2][0]) * idet;
    invA[1][2] = -(A[0][0] * A[1][2] - A[0][2] * A[1][0]) * idet;

    invA[2][0] =  (A[1][0] * A[2][1] - A[1][1] * A[2][0]) * idet;
    invA[2][1] = -(A[0][0] * A[2][1] - A[0][1] * A[2][0]) * idet;
    invA[2][2] =  (A[0][0] * A[1][1] - A[0][1] * A[1][0]) * idet;
}

static void matvec(const double A[3][3], const double v[3], double out[3])
{
    for (int i = 0; i < 3; i++)
        out[i] = A[i][0] * v[0] + A[i][1] * v[1] + A[i][2] * v[2];
}

static bool read_lammps_data(const char* filename, SystemData& sys)
{
    std::ifstream fin(filename);
    if (!fin) {
        std::fprintf(stderr, "ERROR: cannot open %s\n", filename);
        return false;
    }

    sys.xy = 0.0;
    sys.xz = 0.0;
    sys.yz = 0.0;
    sys.xlo = sys.ylo = sys.zlo = 0.0;
    sys.xhi = sys.yhi = sys.zhi = 0.0;

    std::string line;
    int natoms_header = -1;
    bool in_atoms = false;
    int blank_after_atoms = 0;

    while (std::getline(fin, line)) {
        if (line.empty()) {
            if (in_atoms)
                blank_after_atoms++;
            continue;
        }

        std::stringstream ss(line);
        std::vector<std::string> tok;
        std::string t;
        while (ss >> t)
            tok.push_back(t);

        if (tok.size() >= 2 && tok[1] == "atoms") {
            natoms_header = std::atoi(tok[0].c_str());
            continue;
        }

        if (tok.size() >= 4 && tok[2] == "xlo" && tok[3] == "xhi") {
            sys.xlo = std::atof(tok[0].c_str());
            sys.xhi = std::atof(tok[1].c_str());
            continue;
        }

        if (tok.size() >= 4 && tok[2] == "ylo" && tok[3] == "yhi") {
            sys.ylo = std::atof(tok[0].c_str());
            sys.yhi = std::atof(tok[1].c_str());
            continue;
        }

        if (tok.size() >= 4 && tok[2] == "zlo" && tok[3] == "zhi") {
            sys.zlo = std::atof(tok[0].c_str());
            sys.zhi = std::atof(tok[1].c_str());
            continue;
        }

        if (tok.size() >= 6 && tok[3] == "xy" && tok[4] == "xz" && tok[5] == "yz") {
            sys.xy = std::atof(tok[0].c_str());
            sys.xz = std::atof(tok[1].c_str());
            sys.yz = std::atof(tok[2].c_str());
            continue;
        }

        if (tok.size() >= 1 && tok[0] == "Atoms") {
            in_atoms = true;
            blank_after_atoms = 0;
            continue;
        }

        if (in_atoms) {
            if (tok.size() < 5)
                continue;

            Atom a;
            a.id = std::atoi(tok[0].c_str());
            a.type = std::atoi(tok[1].c_str());
            a.x[0] = std::atof(tok[2].c_str());
            a.x[1] = std::atof(tok[3].c_str());
            a.x[2] = std::atof(tok[4].c_str());
            sys.atoms.push_back(a);

            if (natoms_header > 0 && (int)sys.atoms.size() >= natoms_header)
                break;
        }
    }

    sys.origin[0] = sys.xlo;
    sys.origin[1] = sys.ylo;
    sys.origin[2] = sys.zlo;

    sys.H[0][0] = sys.xhi - sys.xlo;
    sys.H[0][1] = sys.xy;
    sys.H[0][2] = sys.xz;

    sys.H[1][0] = 0.0;
    sys.H[1][1] = sys.yhi - sys.ylo;
    sys.H[1][2] = sys.yz;

    sys.H[2][0] = 0.0;
    sys.H[2][1] = 0.0;
    sys.H[2][2] = sys.zhi - sys.zlo;

    invert_3x3(sys.H, sys.invH);

    if (natoms_header > 0 && (int)sys.atoms.size() != natoms_header) {
        std::fprintf(stderr, "ERROR: expected %d atoms, read %zu atoms\n",
                     natoms_header, sys.atoms.size());
        return false;
    }

    return true;
}

static void minimum_image_delta(const SystemData& sys, int i, int j, double dr[3])
{
    double ri[3] = {
        sys.atoms[i].x[0] - sys.origin[0],
        sys.atoms[i].x[1] - sys.origin[1],
        sys.atoms[i].x[2] - sys.origin[2]
    };

    double rj[3] = {
        sys.atoms[j].x[0] - sys.origin[0],
        sys.atoms[j].x[1] - sys.origin[1],
        sys.atoms[j].x[2] - sys.origin[2]
    };

    double fi[3], fj[3], df[3];
    matvec(sys.invH, ri, fi);
    matvec(sys.invH, rj, fj);

    for (int k = 0; k < 3; k++) {
        df[k] = fj[k] - fi[k];
        df[k] -= std::round(df[k]);
    }

    matvec(sys.H, df, dr);
}

static int get_neighbours_lmp(
    void* vdata,
    size_t /*unused*/,
    size_t atom_index,
    int num,
    int* ordering,
    size_t* nbr_indices,
    int32_t* numbers,
    double (*nbr_pos)[3]
)
{
    SystemData* sys = static_cast<SystemData*>(vdata);
    int n = (int)sys->atoms.size();
    int i = (int)atom_index;

    struct Candidate {
        double d2;
        int idx;
        double dr[3];
    };

    std::vector<Candidate> cand;
    cand.reserve(n);

    // PTM expects point 0 to be the central atom.
    ordering[0] = 0;
    nbr_indices[0] = (size_t)i;
    numbers[0] = sys->atoms[i].type;
    nbr_pos[0][0] = 0.0;
    nbr_pos[0][1] = 0.0;
    nbr_pos[0][2] = 0.0;

    for (int j = 0; j < n; j++) {
        if (j == i)
            continue;

        double dr[3];
        minimum_image_delta(*sys, i, j, dr);

        double d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];

        Candidate c;
        c.d2 = d2;
        c.idx = j;
        c.dr[0] = dr[0];
        c.dr[1] = dr[1];
        c.dr[2] = dr[2];

        cand.push_back(c);
    }

    std::sort(cand.begin(), cand.end(), [](const Candidate& a, const Candidate& b) {
        if (a.d2 != b.d2)
            return a.d2 < b.d2;
        return a.idx < b.idx;
    });

    int nout = std::min(num, (int)cand.size() + 1);

    for (int k = 1; k < nout; k++) {
        const Candidate& c = cand[k - 1];

        ordering[k] = k;
        nbr_indices[k] = (size_t)c.idx;
        numbers[k] = sys->atoms[c.idx].type;

        nbr_pos[k][0] = c.dr[0];
        nbr_pos[k][1] = c.dr[1];
        nbr_pos[k][2] = c.dr[2];
    }

    return nout;
}

static const char* type_name(int type)
{
    switch (type) {
        case PTM_MATCH_NONE: return "NONE";
        case PTM_MATCH_FCC: return "FCC";
        case PTM_MATCH_HCP: return "HCP";
        case PTM_MATCH_BCC: return "BCC";
        case PTM_MATCH_ICO: return "ICO";
        case PTM_MATCH_SC: return "SC";
        case PTM_MATCH_DCUB: return "DCUB";
        case PTM_MATCH_DHEX: return "DHEX";
        case PTM_MATCH_GRAPHENE: return "GRAPHENE";
#ifdef PTM_MATCH_OMEGA_A
        case PTM_MATCH_OMEGA_A: return "OMEGA_A";
#endif
#ifdef PTM_MATCH_OMEGA_B
        case PTM_MATCH_OMEGA_B: return "OMEGA_B";
#endif
        default: return "UNKNOWN";
    }
}

static void run_file(const char* filename)
{
    SystemData sys;
    if (!read_lammps_data(filename, sys))
        return;

    std::printf("\n===== %s =====\n", filename);
    std::printf("atoms = %zu\n", sys.atoms.size());

    std::map<int, int> counts;
    std::map<int, double> rmsd_sum;
    std::map<int, double> rmsd_max;

    ptm_local_handle_t local = ptm_initialize_local();

    int flags =
        PTM_CHECK_SC |
        PTM_CHECK_FCC |
        PTM_CHECK_HCP |
        PTM_CHECK_BCC |
        PTM_CHECK_ICO |
        PTM_CHECK_OMEGA_A |
        PTM_CHECK_OMEGA_B ;

    for (size_t i = 0; i < sys.atoms.size(); i++) {
        
        /*
        std::printf("calling ptm_index: i=%zu id=%d lmp_type=%d\n",
                i, sys.atoms[i].id, sys.atoms[i].type);
        */
        std::fflush(stdout);

        int32_t type = PTM_MATCH_NONE;
        int32_t alloy_type = 0;
        double scale = 0.0;
        double rmsd = 0.0;
        double q[4] = {0, 0, 0, 0};
        double F[9] = {0};
        double F_res = 0.0;
        double U[9] = {0};
        double P[9] = {0};
        double interatomic_distance = 0.0;
        double lattice_constant = 0.0;
        int best_template_index = 0;
        const double (*best_template)[3] = NULL;
        int8_t output_indices[PTM_MAX_INPUT_POINTS];

        int ret = ptm_index(
            local,
            i,
            get_neighbours_lmp,
            &sys,
            flags,
            false,
            &type,
            &alloy_type,
            &scale,
            &rmsd,
            q,
            F,
            &F_res,
            U,
            P,
            &interatomic_distance,
            &lattice_constant,
            &best_template_index,
            &best_template,
            output_indices
        );

        if (ret != 0) {
            std::printf("atom %zu: ptm_index ret = %d\n", i, ret);
            continue;
        }

        int raw_type = type;
        double raw_rmsd = rmsd;

        if ((type == PTM_MATCH_OMEGA_A || type == PTM_MATCH_OMEGA_B) && rmsd > 0.10) {
            type = PTM_MATCH_NONE;
        }

        counts[type]++;
        rmsd_sum[type] += rmsd;
        if (counts[type] == 1 || rmsd > rmsd_max[type])
            rmsd_max[type] = rmsd;

        if (i < 10) {
            std::printf(
                "atom %4zu id=%4d lmp_type=%d -> raw=%-8s filtered=%-8s rmsd=%.8g scale=%.8g F_res=%.8g\n",
                i,
                sys.atoms[i].id,
                sys.atoms[i].type,
                type_name(raw_type),
                type_name(type),
                raw_rmsd,
                scale,
                F_res
            );
        }
    }

    ptm_uninitialize_local(local);

    std::printf("\nSummary:\n");
    for (std::map<int, int>::const_iterator it = counts.begin(); it != counts.end(); ++it) {
        int type = it->first;
        int count = it->second;
        double mean_rmsd = rmsd_sum[type] / count;

        std::printf(
            "%-10s type=%2d count=%5d mean_rmsd=%.8g max_rmsd=%.8g\n",
            type_name(type),
            type,
            count,
            mean_rmsd,
            rmsd_max[type]
        );
    }
}

int main()
{
    int ret = ptm_initialize_global();
    std::printf("ptm_initialize_global ret = %d\n", ret);

    if (ret != 0)
        return ret;

    run_file("./testModel/C32_omega_ideal.lmp");
    run_file("./testModel/C32_omega_zhang.lmp");
    run_file("./testModel/BCC.lmp");
    run_file("./testModel/HCP.lmp");
    run_file("./testModel/FCC.lmp");
    run_file("./testModel/BCC_strained.lmp");
    run_file("./testModel/anisotropic_scale.lmp");
    run_file("./testModel/combined_strain_noise.lmp");
    run_file("./testModel/iso_scale_0p98.lmp");
    run_file("./testModel/random_uniform_0p01a.lmp");
    run_file("./testModel/shear_xy_0p05.lmp");
    run_file("./testModel/thermal_gaussian_0p01a.lmp");
    run_file("./testModel/thermal_gaussian_0p03a.lmp");

    return 0;
}
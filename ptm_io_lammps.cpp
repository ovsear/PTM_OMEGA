#include "ptm_io_lammps.h"

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

#include "ptm_constants.h"

const char* type_name(int type)
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
        case PTM_MATCH_OMEGA_A: return "OMEGA_A";
        case PTM_MATCH_OMEGA_B: return "OMEGA_B";
        default: return "UNKNOWN";
    }
}


bool read_lammps_data_atomic(const char* filename, SystemData& sys)
{
    std::ifstream fin(filename);
    if (!fin) {
        std::fprintf(stderr, "ERROR: cannot open %s\n", filename);
        return false;
    }

    sys.atoms.clear();

    sys.xy = 0.0;
    sys.xz = 0.0;
    sys.yz = 0.0;

    sys.xlo = 0.0;
    sys.xhi = 0.0;
    sys.ylo = 0.0;
    sys.yhi = 0.0;
    sys.zlo = 0.0;
    sys.zhi = 0.0;

    sys.origin[0] = 0.0;
    sys.origin[1] = 0.0;
    sys.origin[2] = 0.0;

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            sys.H[i][j] = 0.0;
            sys.invH[i][j] = 0.0;
        }
    }

    std::string line;
    int natoms_header = -1;
    bool in_atoms = false;

    bool has_restricted_x = false;
    bool has_restricted_y = false;
    bool has_restricted_z = false;

    bool has_avec = false;
    bool has_bvec = false;
    bool has_cvec = false;
    bool has_origin = false;

    double avec[3] = {0.0, 0.0, 0.0};
    double bvec[3] = {0.0, 0.0, 0.0};
    double cvec[3] = {0.0, 0.0, 0.0};
    double origin[3] = {0.0, 0.0, 0.0};

    while (std::getline(fin, line)) {
        if (line.empty()) {
            continue;
        }

        std::stringstream ss(line);
        std::vector<std::string> tok;
        std::string t;

        while (ss >> t) {
            tok.push_back(t);
        }

        if (tok.empty()) {
            continue;
        }

        if (tok.size() >= 2 && tok[1] == "atoms") {
            natoms_header = std::atoi(tok[0].c_str());
            continue;
        }

        // ------------------------------------------------------------
        // Standard restricted triclinic / orthogonal LAMMPS data
        // ------------------------------------------------------------
        if (tok.size() >= 4 && tok[2] == "xlo" && tok[3] == "xhi") {
            sys.xlo = std::atof(tok[0].c_str());
            sys.xhi = std::atof(tok[1].c_str());
            has_restricted_x = true;
            continue;
        }

        if (tok.size() >= 4 && tok[2] == "ylo" && tok[3] == "yhi") {
            sys.ylo = std::atof(tok[0].c_str());
            sys.yhi = std::atof(tok[1].c_str());
            has_restricted_y = true;
            continue;
        }

        if (tok.size() >= 4 && tok[2] == "zlo" && tok[3] == "zhi") {
            sys.zlo = std::atof(tok[0].c_str());
            sys.zhi = std::atof(tok[1].c_str());
            has_restricted_z = true;
            continue;
        }

        if (tok.size() >= 6 && tok[3] == "xy" && tok[4] == "xz" && tok[5] == "yz") {
            sys.xy = std::atof(tok[0].c_str());
            sys.xz = std::atof(tok[1].c_str());
            sys.yz = std::atof(tok[2].c_str());
            continue;
        }

        // ------------------------------------------------------------
        // General triclinic LAMMPS data written by OVITO or rotation tests:
        //     ax ay az avec
        //     bx by bz bvec
        //     cx cy cz cvec
        //     ox oy oz abc origin
        // ------------------------------------------------------------
        if (tok.size() >= 4 && tok[3] == "avec") {
            avec[0] = std::atof(tok[0].c_str());
            avec[1] = std::atof(tok[1].c_str());
            avec[2] = std::atof(tok[2].c_str());
            has_avec = true;
            continue;
        }

        if (tok.size() >= 4 && tok[3] == "bvec") {
            bvec[0] = std::atof(tok[0].c_str());
            bvec[1] = std::atof(tok[1].c_str());
            bvec[2] = std::atof(tok[2].c_str());
            has_bvec = true;
            continue;
        }

        if (tok.size() >= 4 && tok[3] == "cvec") {
            cvec[0] = std::atof(tok[0].c_str());
            cvec[1] = std::atof(tok[1].c_str());
            cvec[2] = std::atof(tok[2].c_str());
            has_cvec = true;
            continue;
        }

        if (tok.size() >= 5 && tok[3] == "abc" && tok[4] == "origin") {
            origin[0] = std::atof(tok[0].c_str());
            origin[1] = std::atof(tok[1].c_str());
            origin[2] = std::atof(tok[2].c_str());
            has_origin = true;
            continue;
        }

        if (tok.size() >= 1 && tok[0] == "Atoms") {
            in_atoms = true;
            continue;
        }

        if (in_atoms) {
            if (tok.size() < 5) {
                continue;
            }

            Atom a;
            a.id = std::atoi(tok[0].c_str());
            a.type = std::atoi(tok[1].c_str());
            a.x[0] = std::atof(tok[2].c_str());
            a.x[1] = std::atof(tok[3].c_str());
            a.x[2] = std::atof(tok[4].c_str());

            sys.atoms.push_back(a);

            if (natoms_header > 0 && (int)sys.atoms.size() >= natoms_header) {
                break;
            }
        }
    }

    if (natoms_header > 0 && (int)sys.atoms.size() != natoms_header) {
        std::fprintf(
            stderr,
            "ERROR: expected %d atoms, read %zu atoms from %s\n",
            natoms_header,
            sys.atoms.size(),
            filename
        );
        return false;
    }

    bool has_general_box = has_avec && has_bvec && has_cvec;

    if (has_general_box) {
        // General triclinic:
        // r = origin + H * s
        // H columns are avec, bvec, cvec.
        sys.origin[0] = has_origin ? origin[0] : 0.0;
        sys.origin[1] = has_origin ? origin[1] : 0.0;
        sys.origin[2] = has_origin ? origin[2] : 0.0;

        sys.H[0][0] = avec[0];
        sys.H[1][0] = avec[1];
        sys.H[2][0] = avec[2];

        sys.H[0][1] = bvec[0];
        sys.H[1][1] = bvec[1];
        sys.H[2][1] = bvec[2];

        sys.H[0][2] = cvec[0];
        sys.H[1][2] = cvec[1];
        sys.H[2][2] = cvec[2];

        // These fields are only used by the old restricted dump writer.
        // For general boxes, set a simple bounding range as fallback.
        double xmin = sys.atoms.empty() ? 0.0 : sys.atoms[0].x[0];
        double xmax = xmin;
        double ymin = sys.atoms.empty() ? 0.0 : sys.atoms[0].x[1];
        double ymax = ymin;
        double zmin = sys.atoms.empty() ? 0.0 : sys.atoms[0].x[2];
        double zmax = zmin;

        for (size_t i = 0; i < sys.atoms.size(); i++) {
            xmin = std::min(xmin, sys.atoms[i].x[0]);
            xmax = std::max(xmax, sys.atoms[i].x[0]);
            ymin = std::min(ymin, sys.atoms[i].x[1]);
            ymax = std::max(ymax, sys.atoms[i].x[1]);
            zmin = std::min(zmin, sys.atoms[i].x[2]);
            zmax = std::max(zmax, sys.atoms[i].x[2]);
        }

        double pad = 1.0e-6;
        sys.xlo = xmin - pad;
        sys.xhi = xmax + pad;
        sys.ylo = ymin - pad;
        sys.yhi = ymax + pad;
        sys.zlo = zmin - pad;
        sys.zhi = zmax + pad;

        sys.xy = 0.0;
        sys.xz = 0.0;
        sys.yz = 0.0;
    }
    else if (has_restricted_x && has_restricted_y && has_restricted_z) {
        // Restricted triclinic / orthogonal:
        // r = origin + H * s
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
    }
    else {
        std::fprintf(
            stderr,
            "ERROR: cannot find valid box in %s. "
            "Expected xlo/xhi style or avec/bvec/cvec style.\n",
            filename
        );
        return false;
    }

    invert_3x3(sys.H, sys.invH);

    return true;
}

static void reset_system_box_and_atoms(SystemData& sys)
{
    sys.atoms.clear();

    sys.xlo = 0.0;
    sys.xhi = 0.0;
    sys.ylo = 0.0;
    sys.yhi = 0.0;
    sys.zlo = 0.0;
    sys.zhi = 0.0;

    sys.xy = 0.0;
    sys.xz = 0.0;
    sys.yz = 0.0;

    sys.origin[0] = 0.0;
    sys.origin[1] = 0.0;
    sys.origin[2] = 0.0;

    sys.neigh_stride = 0;
    sys.neigh_count.clear();
    sys.neigh_cache.clear();

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            sys.H[i][j] = 0.0;
            sys.invH[i][j] = 0.0;
        }
    }
}

static int find_column(const std::vector<std::string>& cols, const char* name)
{
    for (size_t i = 0; i < cols.size(); i++) {
        if (cols[i] == name) return (int)i;
    }
    return -1;
}

static bool parse_dump_box(
    std::ifstream& fin,
    const std::string& box_header,
    SystemData& sys
)
{
    std::string line;
    std::vector<std::string> lines;

    for (int i = 0; i < 3; i++) {
        if (!std::getline(fin, line)) {
            std::fprintf(stderr, "ERROR: unexpected EOF while reading BOX BOUNDS\n");
            return false;
        }
        lines.push_back(line);
    }

    bool triclinic = false;
    if (box_header.find("xy") != std::string::npos &&
        box_header.find("xz") != std::string::npos &&
        box_header.find("yz") != std::string::npos) {
        triclinic = true;
    }

    double xlo_bound = 0.0, xhi_bound = 0.0;
    double ylo_bound = 0.0, yhi_bound = 0.0;
    double zlo_bound = 0.0, zhi_bound = 0.0;
    double xy = 0.0, xz = 0.0, yz = 0.0;

    if (!triclinic) {
        {
            std::stringstream ss(lines[0]);
            ss >> sys.xlo >> sys.xhi;
        }
        {
            std::stringstream ss(lines[1]);
            ss >> sys.ylo >> sys.yhi;
        }
        {
            std::stringstream ss(lines[2]);
            ss >> sys.zlo >> sys.zhi;
        }

        sys.xy = 0.0;
        sys.xz = 0.0;
        sys.yz = 0.0;
    }
    else {
        {
            std::stringstream ss(lines[0]);
            ss >> xlo_bound >> xhi_bound >> xy;
        }
        {
            std::stringstream ss(lines[1]);
            ss >> ylo_bound >> yhi_bound >> xz;
        }
        {
            std::stringstream ss(lines[2]);
            ss >> zlo_bound >> zhi_bound >> yz;
        }

        double xlo_shift = std::min(0.0, std::min(xy, std::min(xz, xy + xz)));
        double xhi_shift = std::max(0.0, std::max(xy, std::max(xz, xy + xz)));

        double ylo_shift = std::min(0.0, yz);
        double yhi_shift = std::max(0.0, yz);

        sys.xlo = xlo_bound - xlo_shift;
        sys.xhi = xhi_bound - xhi_shift;

        sys.ylo = ylo_bound - ylo_shift;
        sys.yhi = yhi_bound - yhi_shift;

        sys.zlo = zlo_bound;
        sys.zhi = zhi_bound;

        sys.xy = xy;
        sys.xz = xz;
        sys.yz = yz;
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

    return true;
}

static void frac_to_cart_lammps_box(
    const SystemData& sys,
    double xs,
    double ys,
    double zs,
    double x[3]
)
{
    double f[3] = {xs, ys, zs};
    double r[3];
    matvec(sys.H, f, r);

    x[0] = sys.origin[0] + r[0];
    x[1] = sys.origin[1] + r[1];
    x[2] = sys.origin[2] + r[2];
}

bool read_lammps_dump_atomic(
    const char* filename,
    SystemData& sys,
    int frame_index
)
{
    std::ifstream fin(filename);
    if (!fin) {
        std::fprintf(stderr, "ERROR: cannot open %s\n", filename);
        return false;
    }

    SystemData last_sys;
    bool found_frame = false;

    std::string line;
    int current_frame = -1;

    while (std::getline(fin, line)) {
        if (line.find("ITEM: TIMESTEP") != 0) {
            continue;
        }

        current_frame++;

        std::string timestep_line;
        if (!std::getline(fin, timestep_line)) {
            break;
        }

        std::string number_header;
        if (!std::getline(fin, number_header)) {
            break;
        }

        if (number_header.find("ITEM: NUMBER OF ATOMS") != 0) {
            std::fprintf(stderr, "ERROR: expected ITEM: NUMBER OF ATOMS in %s\n", filename);
            return false;
        }

        std::string natoms_line;
        if (!std::getline(fin, natoms_line)) {
            break;
        }

        int natoms = std::atoi(natoms_line.c_str());
        if (natoms <= 0) {
            std::fprintf(stderr, "ERROR: invalid atom count %d in %s\n", natoms, filename);
            return false;
        }

        std::string box_header;
        if (!std::getline(fin, box_header)) {
            break;
        }

        if (box_header.find("ITEM: BOX BOUNDS") != 0) {
            std::fprintf(stderr, "ERROR: expected ITEM: BOX BOUNDS in %s\n", filename);
            return false;
        }

        SystemData frame_sys;
        reset_system_box_and_atoms(frame_sys);

        if (!parse_dump_box(fin, box_header, frame_sys)) {
            return false;
        }

        std::string atoms_header;
        if (!std::getline(fin, atoms_header)) {
            break;
        }

        if (atoms_header.find("ITEM: ATOMS") != 0) {
            std::fprintf(stderr, "ERROR: expected ITEM: ATOMS in %s\n", filename);
            return false;
        }

        std::stringstream hs(atoms_header);
        std::string tmp;
        std::vector<std::string> cols;

        hs >> tmp; // ITEM:
        hs >> tmp; // ATOMS
        while (hs >> tmp) {
            cols.push_back(tmp);
        }

        int col_id = find_column(cols, "id");
        int col_type = find_column(cols, "type");

        int col_x = find_column(cols, "x");
        int col_y = find_column(cols, "y");
        int col_z = find_column(cols, "z");

        int col_xu = find_column(cols, "xu");
        int col_yu = find_column(cols, "yu");
        int col_zu = find_column(cols, "zu");

        int col_xs = find_column(cols, "xs");
        int col_ys = find_column(cols, "ys");
        int col_zs = find_column(cols, "zs");

        int col_xsu = find_column(cols, "xsu");
        int col_ysu = find_column(cols, "ysu");
        int col_zsu = find_column(cols, "zsu");

        bool has_cart =
            (col_x >= 0 && col_y >= 0 && col_z >= 0);

        bool has_unwrapped =
            (col_xu >= 0 && col_yu >= 0 && col_zu >= 0);

        bool has_scaled =
            (col_xs >= 0 && col_ys >= 0 && col_zs >= 0);

        bool has_scaled_unwrapped =
            (col_xsu >= 0 && col_ysu >= 0 && col_zsu >= 0);

        if (col_type < 0) {
            std::fprintf(stderr, "ERROR: dump must contain atom type column: %s\n", filename);
            return false;
        }

        if (!has_cart && !has_unwrapped && !has_scaled && !has_scaled_unwrapped) {
            std::fprintf(
                stderr,
                "ERROR: dump must contain x/y/z, xu/yu/zu, xs/ys/zs, or xsu/ysu/zsu columns: %s\n",
                filename
            );
            return false;
        }

        frame_sys.atoms.clear();
        frame_sys.atoms.reserve((size_t)natoms);

        for (int i = 0; i < natoms; i++) {
            if (!std::getline(fin, line)) {
                std::fprintf(stderr, "ERROR: unexpected EOF in atom section of %s\n", filename);
                return false;
            }

            std::stringstream ss(line);
            std::vector<std::string> tok;
            std::string t;

            while (ss >> t) {
                tok.push_back(t);
            }

            if ((int)tok.size() < (int)cols.size()) {
                std::fprintf(stderr, "ERROR: malformed atom line in %s\n", filename);
                return false;
            }

            Atom a;
            a.id = (col_id >= 0) ? std::atoi(tok[(size_t)col_id].c_str()) : (i + 1);
            a.type = std::atoi(tok[(size_t)col_type].c_str());

            if (has_cart) {
                a.x[0] = std::atof(tok[(size_t)col_x].c_str());
                a.x[1] = std::atof(tok[(size_t)col_y].c_str());
                a.x[2] = std::atof(tok[(size_t)col_z].c_str());
            }
            else if (has_unwrapped) {
                a.x[0] = std::atof(tok[(size_t)col_xu].c_str());
                a.x[1] = std::atof(tok[(size_t)col_yu].c_str());
                a.x[2] = std::atof(tok[(size_t)col_zu].c_str());
            }
            else if (has_scaled) {
                double xs = std::atof(tok[(size_t)col_xs].c_str());
                double ys = std::atof(tok[(size_t)col_ys].c_str());
                double zs = std::atof(tok[(size_t)col_zs].c_str());

                frac_to_cart_lammps_box(frame_sys, xs, ys, zs, a.x);
            }
            else {
                double xs = std::atof(tok[(size_t)col_xsu].c_str());
                double ys = std::atof(tok[(size_t)col_ysu].c_str());
                double zs = std::atof(tok[(size_t)col_zsu].c_str());

                frac_to_cart_lammps_box(frame_sys, xs, ys, zs, a.x);
            }

            frame_sys.atoms.push_back(a);
        }

        if (frame_index >= 0) {
            if (current_frame == frame_index) {
                sys = frame_sys;
                return true;
            }
        }
        else {
            // Default: keep the last frame.
            last_sys = frame_sys;
            found_frame = true;
        }
    }

    if (frame_index >= 0) {
        std::fprintf(
            stderr,
            "ERROR: frame_index=%d not found in dump %s\n",
            frame_index,
            filename
        );
        return false;
    }

    if (!found_frame) {
        std::fprintf(stderr, "ERROR: no dump frame found in %s\n", filename);
        return false;
    }

    sys = last_sys;
    return true;
}

static bool file_looks_like_lammps_dump(const char* filename)
{
    std::ifstream fin(filename);
    if (!fin) {
        return false;
    }

    std::string line;
    for (int i = 0; i < 20 && std::getline(fin, line); i++) {
        if (line.find("ITEM: TIMESTEP") == 0) {
            return true;
        }
    }

    return false;
}

bool read_lammps_atomic_auto(
    const char* filename,
    SystemData& sys,
    int frame_index
)
{
    if (file_looks_like_lammps_dump(filename)) {
        return read_lammps_dump_atomic(filename, sys, frame_index);
    }

    return read_lammps_data_atomic(filename, sys);
}

std::string make_output_dump_name(const std::string& input_name)
{
    std::string out = input_name;

    size_t slash = out.find_last_of("/\\");
    std::string dir = "";
    std::string base = out;

    if (slash != std::string::npos) {
        dir = out.substr(0, slash + 1);
        base = out.substr(slash + 1);
    }

    size_t dot = base.find_last_of('.');
    if (dot != std::string::npos) {
        base = base.substr(0, dot);
    }

    return dir + base + "_ptm.dump";
}

void write_ptm_dump(
    const std::string& outname,
    const SystemData& sys,
    const std::vector<PTMAtomResult>& results
)
{
    std::ofstream fout(outname.c_str());

    if (!fout) {
        std::printf("ERROR: cannot write %s\n", outname.c_str());
        return;
    }

    // LAMMPS triclinic dump uses bounded box values.
    double xlo_bound = sys.xlo + std::min(std::min(0.0, sys.xy), std::min(sys.xz, sys.xy + sys.xz));
    double xhi_bound = sys.xhi + std::max(std::max(0.0, sys.xy), std::max(sys.xz, sys.xy + sys.xz));

    double ylo_bound = sys.ylo + std::min(0.0, sys.yz);
    double yhi_bound = sys.yhi + std::max(0.0, sys.yz);

    double zlo_bound = sys.zlo;
    double zhi_bound = sys.zhi;

    fout << "ITEM: TIMESTEP\n";
    fout << "0\n";

    fout << "ITEM: NUMBER OF ATOMS\n";
    fout << sys.atoms.size() << "\n";

    fout << "ITEM: BOX BOUNDS xy xz yz pp pp pp\n";
    fout << xlo_bound << " " << xhi_bound << " " << sys.xy << "\n";
    fout << ylo_bound << " " << yhi_bound << " " << sys.xz << "\n";
    fout << zlo_bound << " " << zhi_bound << " " << sys.yz << "\n";

    fout << "ITEM: ATOMS "
         << "id type atom_type x y z "
         << "raw_ptm ptm_type omega_flag omega_site omega_confidence "
         << "rmsd scale f_res "
         << "q0 q1 q2 q3 "
         << "bda_defect bda_coord bda_csp bda_chunk_id bda_local_a bda_local_cutoff\n";

    for (size_t i = 0; i < sys.atoms.size(); i++) {
        const Atom& a = sys.atoms[i];
        const PTMAtomResult& r = results[i];

        fout << a.id << " "
             << r.vis_type << " "
             << a.type << " "
             << a.x[0] << " "
             << a.x[1] << " "
             << a.x[2] << " "
             << r.raw_ptm << " "
             << r.ptm_type << " "
             << r.omega_flag << " "
             << r.omega_site << " "
             << r.omega_confidence << " "
             << r.rmsd << " "
             << r.scale << " "
             << r.f_res_scalar << " "
             << r.q[0] << " "
             << r.q[1] << " "
             << r.q[2] << " "
             << r.q[3] << " "
             << r.bda_defect << " "
             << r.bda_coord << " "
             << r.bda_csp << " "
             << r.bda_chunk_id << " "
             << r.bda_local_a << " "
             << r.bda_local_cutoff << "\n";
    }

    fout.close();

    std::printf("wrote visual dump: %s\n", outname.c_str());
}

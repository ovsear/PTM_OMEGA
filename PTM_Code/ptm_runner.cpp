#include "ptm_runner.h"

#include <cstdio>
#include <cmath>
#include <cstring>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "ptm_constants.h"
#include "ptm_initialize_data.h"

extern "C" int ptm_index(
    ptm_local_handle_t local_handle,
    size_t atom_index,
    int (*get_neighbours)(
        void* vdata,
        size_t _unused_lammps_variable,
        size_t atom_index,
        int num,
        int* ordering,
        size_t* nbr_indices,
        int32_t* numbers,
        double (*nbr_pos)[3]
    ),
    void* nbrlist,
    int32_t flags,
    bool output_conventional_orientation,
    int32_t* p_type,
    int32_t* p_alloy_type,
    double* p_scale,
    double* p_rmsd,
    double* q,
    double* F,
    double* F_res,
    double* U,
    double* P,
    double* p_interatomic_distance,
    double* p_lattice_constant,
    int* p_best_template_index,
    const double (**p_best_template)[3],
    int8_t* output_indices
);

PTMRunConfig::PTMRunConfig()
{
    flags = 0;
    output_conventional_orientation = true;

    rmsd_cutoff_sc = 0.10;
    rmsd_cutoff_fcc = 0.10;
    rmsd_cutoff_hcp = 0.10;
    rmsd_cutoff_bcc = 0.10;
    rmsd_cutoff_ico = 0.10;

    rmsd_cutoff_dhex = 0.10;
    rmsd_cutoff_dcub = 0.10;
    rmsd_cutoff_graphene = 0.10;

    omega_rmsd_strict = 0.06;
    omega_rmsd_loose = 0.10;

    debug_ptm_index = false;
    debug_first_n_atoms = 10;
}

int omega_site_from_ptm_type(int type)
{
    if (type == PTM_MATCH_OMEGA_A) return 1;
    if (type == PTM_MATCH_OMEGA_B) return 2;
    return 0;
}

int omega_confidence_from_rmsd(int type, double rmsd, const PTMRunConfig& cfg)
{
    if (type != PTM_MATCH_OMEGA_A && type != PTM_MATCH_OMEGA_B) {
        return 0;
    }

    if (rmsd <= cfg.omega_rmsd_strict) {
        return 2;
    }

    if (rmsd <= cfg.omega_rmsd_loose) {
        return 1;
    }

    return 0;
}

int filter_ptm_type_by_rmsd(int raw_type, double rmsd, const PTMRunConfig& cfg)
{
    if (raw_type == PTM_MATCH_NONE) {
        return PTM_MATCH_NONE;
    }

    if (raw_type == PTM_MATCH_SC && rmsd > cfg.rmsd_cutoff_sc) {
        return PTM_MATCH_NONE;
    }

    if (raw_type == PTM_MATCH_FCC && rmsd > cfg.rmsd_cutoff_fcc) {
        return PTM_MATCH_NONE;
    }

    if (raw_type == PTM_MATCH_HCP && rmsd > cfg.rmsd_cutoff_hcp) {
        return PTM_MATCH_NONE;
    }

    if (raw_type == PTM_MATCH_BCC && rmsd > cfg.rmsd_cutoff_bcc) {
        return PTM_MATCH_NONE;
    }

    if (raw_type == PTM_MATCH_ICO && rmsd > cfg.rmsd_cutoff_ico) {
        return PTM_MATCH_NONE;
    }

    if (raw_type == PTM_MATCH_DHEX && rmsd > cfg.rmsd_cutoff_dhex) {
        return PTM_MATCH_NONE;
    }

    if (raw_type == PTM_MATCH_DCUB && rmsd > cfg.rmsd_cutoff_dcub) {
        return PTM_MATCH_NONE;
    }

    if (raw_type == PTM_MATCH_GRAPHENE && rmsd > cfg.rmsd_cutoff_graphene) {
        return PTM_MATCH_NONE;
    }

    if ((raw_type == PTM_MATCH_OMEGA_A || raw_type == PTM_MATCH_OMEGA_B) &&
        rmsd > cfg.omega_rmsd_loose) {
        return PTM_MATCH_NONE;
    }

    return raw_type;
}

static void init_ptm_atom_result(PTMAtomResult& r)
{
    r.raw_ptm = PTM_MATCH_NONE;
    r.ptm_type = PTM_MATCH_NONE;
    r.vis_type = PTM_MATCH_NONE;

    r.omega_flag = 0;
    r.omega_site = 0;
    r.omega_confidence = 0;

    r.rmsd = 0.0;
    r.scale = 0.0;

    r.f_res[0] = 0.0;
    r.f_res[1] = 0.0;
    r.f_res[2] = 0.0;
    r.f_res_scalar = 0.0;

    r.q[0] = 1.0;
    r.q[1] = 0.0;
    r.q[2] = 0.0;
    r.q[3] = 0.0;

    r.bda_defect = BDA_ELS;
    r.bda_coord = 0;
    r.bda_chunk_id = -1;
    r.bda_csp = 0.0;
    r.bda_local_a = 0.0;
    r.bda_local_cutoff = 0.0;

    r.interatomic_distance = 0.0;
    r.lattice_constant = 0.0;
}

void run_ptm_for_system(
    SystemData& sys,
    const PTMRunConfig& cfg,
    std::vector<PTMAtomResult>& ptm_results
)
{
    const int n = (int)sys.atoms.size();

    ptm_results.assign((size_t)n, PTMAtomResult());

    #pragma omp parallel
    {
        ptm_local_handle_t local = ptm_initialize_local();

        #pragma omp for schedule(dynamic, 256)
        for (int i = 0; i < n; i++) {
            PTMAtomResult rr;
            init_ptm_atom_result(rr);

            int32_t type = PTM_MATCH_NONE;
            int32_t alloy_type = 0;

            double scale = 0.0;
            double rmsd = 0.0;
            double q[4] = {1.0, 0.0, 0.0, 0.0};
            double F[9] = {0.0};
            double F_res[3] = {0.0, 0.0, 0.0};
            double U[9] = {0.0};
            double P[9] = {0.0};

            double interatomic_distance = 0.0;
            double lattice_constant = 0.0;

            int best_template_index = -1;
            const double (*best_template)[3] = NULL;
            int8_t output_indices[PTM_MAX_INPUT_POINTS];

            std::memset(output_indices, 0, sizeof(output_indices));

            int ret = ptm_index(
                local,
                (size_t)i,
                get_neighbours_lmp,
                &sys,
                cfg.flags,
                cfg.output_conventional_orientation,
                &type,
                &alloy_type,
                &scale,
                &rmsd,
                q,
                F,
                F_res,
                U,
                P,
                &interatomic_distance,
                &lattice_constant,
                &best_template_index,
                &best_template,
                output_indices
            );

            if (cfg.debug_ptm_index && i < cfg.debug_first_n_atoms) {
                #pragma omp critical
                {
                    std::printf(
                        "DEBUG ptm_index: i=%d id=%d ret=%d type=%d rmsd=%.9g scale=%.9g\n",
                        i,
                        sys.atoms[i].id,
                        ret,
                        (int)type,
                        rmsd,
                        scale
                    );
                }
            }

            if (ret != 0) {
                type = PTM_MATCH_NONE;
                rmsd = 0.0;
                scale = 0.0;
                F_res[0] = 0.0;
                F_res[1] = 0.0;
                F_res[2] = 0.0;
                interatomic_distance = 0.0;
                lattice_constant = 0.0;
            }

            rr.raw_ptm = (int)type;
            rr.ptm_type = filter_ptm_type_by_rmsd((int)type, rmsd, cfg);
            rr.vis_type = rr.ptm_type;

            rr.rmsd = rmsd;
            rr.scale = scale;

            rr.q[0] = q[0];
            rr.q[1] = q[1];
            rr.q[2] = q[2];
            rr.q[3] = q[3];

            rr.f_res[0] = F_res[0];
            rr.f_res[1] = F_res[1];
            rr.f_res[2] = F_res[2];
            rr.f_res_scalar = std::sqrt(F_res[0] + F_res[1] + F_res[2]);

            rr.interatomic_distance = interatomic_distance;
            rr.lattice_constant = lattice_constant;

            rr.omega_site = omega_site_from_ptm_type(rr.ptm_type);
            rr.omega_confidence = omega_confidence_from_rmsd(rr.ptm_type, rr.rmsd, cfg);
            rr.omega_flag = rr.omega_site > 0 ? 1 : 0;

            ptm_results[(size_t)i] = rr;
        }

        ptm_uninitialize_local(local);
    }
}
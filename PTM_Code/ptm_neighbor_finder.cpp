#include "ptm_neighbor_finder.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

void invert_3x3(const double A[3][3], double invA[3][3])
{
    double det =
        A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
        A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
        A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);

    if (std::fabs(det) < 1.0e-14) {
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

void matvec(const double A[3][3], const double v[3], double out[3])
{
    for (int i = 0; i < 3; i++) {
        out[i] = A[i][0] * v[0] + A[i][1] * v[1] + A[i][2] * v[2];
    }
}

void minimum_image_delta(const SystemData& sys, int i, int j, double dr[3])
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

    double fi[3];
    double fj[3];
    double df[3];

    matvec(sys.invH, ri, fi);
    matvec(sys.invH, rj, fj);

    for (int k = 0; k < 3; k++) {
        df[k] = fj[k] - fi[k];
        df[k] -= std::floor(df[k] + 0.5);
    }

    matvec(sys.H, df, dr);
}

static bool neighbor_query_less(
    const NeighborQueryResult& a,
    const NeighborQueryResult& b
)
{
    if (std::fabs(a.d2 - b.d2) > 1.0e-14) {
        return a.d2 < b.d2;
    }
    return a.idx < b.idx;
}

CellListNeighborFinder::CellListNeighborFinder()
    : sys_(NULL),
      max_cutoff_(0.0),
      nx_(1),
      ny_(1),
      nz_(1)
{
}

double CellListNeighborFinder::wrap01(double x)
{
    x -= std::floor(x);
    if (x >= 1.0) x -= 1.0;
    if (x < 0.0) x += 1.0;
    return x;
}

void CellListNeighborFinder::atom_fractional(int i, double s[3]) const
{
    const SystemData& sys = *sys_;

    double r[3] = {
        sys.atoms[i].x[0] - sys.origin[0],
        sys.atoms[i].x[1] - sys.origin[1],
        sys.atoms[i].x[2] - sys.origin[2]
    };

    matvec(sys.invH, r, s);

    s[0] = wrap01(s[0]);
    s[1] = wrap01(s[1]);
    s[2] = wrap01(s[2]);
}

void CellListNeighborFinder::atom_to_cell(
    int i,
    int& ix,
    int& iy,
    int& iz
) const
{
    double s[3];
    atom_fractional(i, s);

    ix = (int)std::floor(s[0] * nx_);
    iy = (int)std::floor(s[1] * ny_);
    iz = (int)std::floor(s[2] * nz_);

    if (ix < 0) ix = 0;
    if (iy < 0) iy = 0;
    if (iz < 0) iz = 0;

    if (ix >= nx_) ix = nx_ - 1;
    if (iy >= ny_) iy = ny_ - 1;
    if (iz >= nz_) iz = nz_ - 1;
}

int CellListNeighborFinder::cell_index(int ix, int iy, int iz) const
{
    return (iz * ny_ + iy) * nx_ + ix;
}

void CellListNeighborFinder::make_periodic_range(
    int c,
    int h,
    int n,
    std::vector<int>& out
)
{
    out.clear();

    if (n <= 1 || 2 * h + 1 >= n) {
        for (int i = 0; i < n; i++) {
            out.push_back(i);
        }
        return;
    }

    for (int d = -h; d <= h; d++) {
        int x = c + d;
        while (x < 0) x += n;
        while (x >= n) x -= n;
        out.push_back(x);
    }

    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
}

void CellListNeighborFinder::halo_for_cutoff(
    double cutoff,
    int& hx,
    int& hy,
    int& hz
) const
{
    const SystemData& sys = *sys_;

    double r0 = std::sqrt(
        sys.invH[0][0] * sys.invH[0][0] +
        sys.invH[0][1] * sys.invH[0][1] +
        sys.invH[0][2] * sys.invH[0][2]
    );

    double r1 = std::sqrt(
        sys.invH[1][0] * sys.invH[1][0] +
        sys.invH[1][1] * sys.invH[1][1] +
        sys.invH[1][2] * sys.invH[1][2]
    );

    double r2 = std::sqrt(
        sys.invH[2][0] * sys.invH[2][0] +
        sys.invH[2][1] * sys.invH[2][1] +
        sys.invH[2][2] * sys.invH[2][2]
    );

    hx = (int)std::ceil(cutoff * r0 * nx_);
    hy = (int)std::ceil(cutoff * r1 * ny_);
    hz = (int)std::ceil(cutoff * r2 * nz_);

    hx = std::max(1, std::min(hx, nx_));
    hy = std::max(1, std::min(hy, ny_));
    hz = std::max(1, std::min(hz, nz_));
}

void CellListNeighborFinder::build(const SystemData& sys, double max_cutoff)
{
    sys_ = &sys;
    max_cutoff_ = std::max(max_cutoff, 1.0e-12);

    const int n = (int)sys.atoms.size();

    double ax = std::sqrt(
        sys.H[0][0] * sys.H[0][0] +
        sys.H[1][0] * sys.H[1][0] +
        sys.H[2][0] * sys.H[2][0]
    );

    double ay = std::sqrt(
        sys.H[0][1] * sys.H[0][1] +
        sys.H[1][1] * sys.H[1][1] +
        sys.H[2][1] * sys.H[2][1]
    );

    double az = std::sqrt(
        sys.H[0][2] * sys.H[0][2] +
        sys.H[1][2] * sys.H[1][2] +
        sys.H[2][2] * sys.H[2][2]
    );

    nx_ = std::max(1, (int)std::floor(ax / max_cutoff_));
    ny_ = std::max(1, (int)std::floor(ay / max_cutoff_));
    nz_ = std::max(1, (int)std::floor(az / max_cutoff_));

    nx_ = std::min(nx_, std::max(1, n));
    ny_ = std::min(ny_, std::max(1, n));
    nz_ = std::min(nz_, std::max(1, n));

    const int n_cells = nx_ * ny_ * nz_;

    head_.assign(n_cells, -1);
    next_.assign(n, -1);
    atom_cell_.assign(n, 0);

    for (int i = 0; i < n; i++) {
        int ix, iy, iz;
        atom_to_cell(i, ix, iy, iz);

        int cid = cell_index(ix, iy, iz);
        atom_cell_[i] = cid;

        next_[i] = head_[cid];
        head_[cid] = i;
    }
}

void CellListNeighborFinder::query_cutoff(
    int center,
    double cutoff,
    std::vector<NeighborQueryResult>& out,
    bool sort_output
) const
{
    out.clear();

    if (sys_ == NULL) {
        return;
    }

    const SystemData& sys = *sys_;
    const int n = (int)sys.atoms.size();

    if (center < 0 || center >= n || cutoff <= 0.0) {
        return;
    }

    const double cutoff2 = cutoff * cutoff;

    int icx, icy, icz;
    atom_to_cell(center, icx, icy, icz);

    int hx, hy, hz;
    halo_for_cutoff(cutoff, hx, hy, hz);

    std::vector<int> xs, ys, zs;
    make_periodic_range(icx, hx, nx_, xs);
    make_periodic_range(icy, hy, ny_, ys);
    make_periodic_range(icz, hz, nz_, zs);

    std::vector<unsigned char> seen((size_t)n, 0);
    seen[(size_t)center] = 1;

    for (size_t ax = 0; ax < xs.size(); ax++) {
        for (size_t ay = 0; ay < ys.size(); ay++) {
            for (size_t az = 0; az < zs.size(); az++) {
                int cid = cell_index(xs[ax], ys[ay], zs[az]);

                for (int j = head_[cid]; j != -1; j = next_[j]) {
                    
                    if (j < 0 || j >= n) {
                        continue;
                    }

                    if (seen[(size_t)j]) {
                        continue;
                    }
                    seen[(size_t)j] = 1;

                    if (j == center) {
                        continue;
                    }

                    double dr[3];
                    minimum_image_delta(sys, center, j, dr);

                    double d2 =
                        dr[0] * dr[0] +
                        dr[1] * dr[1] +
                        dr[2] * dr[2];

                    if (d2 <= cutoff2 && d2 > 1.0e-20) {
                        NeighborQueryResult q;
                        q.idx = j;
                        q.number = sys.atoms[j].type;
                        q.d2 = d2;
                        q.d = std::sqrt(d2);
                        q.dr[0] = dr[0];
                        q.dr[1] = dr[1];
                        q.dr[2] = dr[2];
                        q.pbc_shift[0] = 0;
                        q.pbc_shift[1] = 0;
                        q.pbc_shift[2] = 0;
                        out.push_back(q);
                    }
                }
            }
        }
    }

    if (sort_output) {
        std::sort(out.begin(), out.end(), neighbor_query_less);
    }
}

void CellListNeighborFinder::query_topk(
    int center,
    int k,
    std::vector<NeighborQueryResult>& out
) const
{
    out.clear();

    if (sys_ == NULL || k <= 0) {
        return;
    }

    const SystemData& sys = *sys_;
    const int n = (int)sys.atoms.size();

    if (center < 0 || center >= n) {
        return;
    }

    NeighborQueryResult self;
    self.idx = center;
    self.number = sys.atoms[center].type;
    self.d = 0.0;
    self.d2 = 0.0;
    self.dr[0] = 0.0;
    self.dr[1] = 0.0;
    self.dr[2] = 0.0;
    self.pbc_shift[0] = 0;
    self.pbc_shift[1] = 0;
    self.pbc_shift[2] = 0;

    out.push_back(self);

    const int want = k - 1;
    if (want <= 0) {
        return;
    }

    std::vector<NeighborQueryResult> best;
    best.reserve(want);

    auto insert_best = [&](const NeighborQueryResult& q) {
        if ((int)best.size() < want) {
            best.push_back(q);
            std::sort(best.begin(), best.end(), neighbor_query_less);
            return;
        }

        if (!neighbor_query_less(q, best.back())) {
            return;
        }

        best.back() = q;
        std::sort(best.begin(), best.end(), neighbor_query_less);
    };

    int icx, icy, icz;
    atom_to_cell(center, icx, icy, icz);

    int hx, hy, hz;
    halo_for_cutoff(max_cutoff_, hx, hy, hz);

    std::vector<int> xs, ys, zs;
    make_periodic_range(icx, hx, nx_, xs);
    make_periodic_range(icy, hy, ny_, ys);
    make_periodic_range(icz, hz, nz_, zs);

    std::vector<unsigned char> seen((size_t)n, 0);
    seen[(size_t)center] = 1;

    const double cutoff2 = max_cutoff_ * max_cutoff_;

    for (size_t ax = 0; ax < xs.size(); ax++) {
        for (size_t ay = 0; ay < ys.size(); ay++) {
            for (size_t az = 0; az < zs.size(); az++) {
                int cid = cell_index(xs[ax], ys[ay], zs[az]);

                for (int j = head_[cid]; j != -1; j = next_[j]) {
                    if (j < 0 || j >= n) {
                        continue;
                    }

                    if (seen[(size_t)j]) {
                        continue;
                    }
                    seen[(size_t)j] = 1;

                    double dr[3];
                    minimum_image_delta(sys, center, j, dr);

                    double d2 =
                        dr[0] * dr[0] +
                        dr[1] * dr[1] +
                        dr[2] * dr[2];

                    if (d2 <= 1.0e-20 || d2 > cutoff2) {
                        continue;
                    }

                    NeighborQueryResult q;
                    q.idx = j;
                    q.number = sys.atoms[j].type;
                    q.d2 = d2;
                    q.d = std::sqrt(d2);
                    q.dr[0] = dr[0];
                    q.dr[1] = dr[1];
                    q.dr[2] = dr[2];
                    q.pbc_shift[0] = 0;
                    q.pbc_shift[1] = 0;
                    q.pbc_shift[2] = 0;

                    insert_best(q);
                }
            }
        }
    }

    for (size_t i = 0; i < best.size(); i++) {
        out.push_back(best[i]);
    }
}

double CellListNeighborFinder::max_cutoff() const
{
    return max_cutoff_;
}

int CellListNeighborFinder::nx() const
{
    return nx_;
}

int CellListNeighborFinder::ny() const
{
    return ny_;
}

int CellListNeighborFinder::nz() const
{
    return nz_;
}

double estimate_bcc_lattice_from_density(
    const SystemData& sys,
    double fallback
)
{
    const int n = (int)sys.atoms.size();
    if (n <= 0) {
        return fallback;
    }

    double det =
        sys.H[0][0] * (sys.H[1][1] * sys.H[2][2] - sys.H[1][2] * sys.H[2][1]) -
        sys.H[0][1] * (sys.H[1][0] * sys.H[2][2] - sys.H[1][2] * sys.H[2][0]) +
        sys.H[0][2] * (sys.H[1][0] * sys.H[2][1] - sys.H[1][1] * sys.H[2][0]);

    double volume = std::fabs(det);
    if (!(volume > 0.0)) {
        return fallback;
    }

    // BCC has 2 atoms in the conventional cubic cell.
    // This is only a search-radius estimate.
    double a = std::pow(2.0 * volume / (double)n, 1.0 / 3.0);

    if (!std::isfinite(a) || a <= 0.0) {
        return fallback;
    }

    return a;
}

void build_neighbour_cache(
    SystemData& sys,
    const CellListNeighborFinder& finder,
    int max_points
)
{
    const int n = (int)sys.atoms.size();

    if (max_points <= 0) {
        std::fprintf(stderr, "ERROR: max_points <= 0 in build_neighbour_cache\n");
        std::exit(1);
    }

    sys.neigh_stride = max_points;
    sys.neigh_count.assign(n, 0);
    sys.neigh_cache.assign((size_t)n * (size_t)max_points, NeighborEntry());

    #pragma omp parallel for schedule(dynamic, 256)
    for (int i = 0; i < n; i++) {
        std::vector<NeighborQueryResult> q;
        finder.query_topk(i, max_points, q);

        NeighborEntry* out =
            &sys.neigh_cache[(size_t)i * (size_t)max_points];

        int nout = std::min(max_points, (int)q.size());

        for (int k = 0; k < nout; k++) {
            out[k].d2 = q[k].d2;
            out[k].idx = (size_t)q[k].idx;
            out[k].number = q[k].number;
            out[k].dr[0] = q[k].dr[0];
            out[k].dr[1] = q[k].dr[1];
            out[k].dr[2] = q[k].dr[2];
        }

        sys.neigh_count[i] = nout;
    }
}

int get_neighbours_lmp(
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

    const int n = (int)sys->atoms.size();

    if (num <= 0) {
        return 0;
    }

    for (int k = 0; k < num; k++) {
        ordering[k] = -1;
        nbr_indices[k] = (size_t)-1;
        numbers[k] = 0;
        nbr_pos[k][0] = 0.0;
        nbr_pos[k][1] = 0.0;
        nbr_pos[k][2] = 0.0;
    }

    if (atom_index >= (size_t)n) {
        std::fprintf(
            stderr,
            "ERROR get_neighbours_lmp cached: atom_index=%zu out of range, n=%d\n",
            atom_index,
            n
        );
        return 0;
    }

    if (sys->neigh_stride <= 0 || sys->neigh_cache.empty()) {
        std::fprintf(
            stderr,
            "ERROR get_neighbours_lmp cached: neighbour cache is not built\n"
        );
        return 0;
    }

    const int i = (int)atom_index;
    const int cached_count = sys->neigh_count[i];
    const int nout = std::min(num, cached_count);

    const NeighborEntry* in =
        &sys->neigh_cache[(size_t)i * (size_t)sys->neigh_stride];

    for (int k = 0; k < nout; k++) {
        ordering[k] = k;
        nbr_indices[k] = in[k].idx;
        numbers[k] = in[k].number;

        nbr_pos[k][0] = in[k].dr[0];
        nbr_pos[k][1] = in[k].dr[1];
        nbr_pos[k][2] = in[k].dr[2];
    }

    return nout;
}
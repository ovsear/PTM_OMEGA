#ifndef PTM_NEIGHBOR_FINDER_H
#define PTM_NEIGHBOR_FINDER_H

#include <vector>
#include <cstddef>
#include <cstdint>

struct Atom {
    int id;
    int type;
    double x[3];
};

struct NeighborEntry {
    double d2;
    size_t idx;       // 0-based atom index
    int32_t number;   // atom type
    double dr[3];     // minimum-image vector from center to neighbor
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

    int neigh_stride;
    std::vector<int> neigh_count;
    std::vector<NeighborEntry> neigh_cache;

    SystemData() :
        xlo(0.0), xhi(0.0),
        ylo(0.0), yhi(0.0),
        zlo(0.0), zhi(0.0),
        xy(0.0), xz(0.0), yz(0.0),
        neigh_stride(0)
    {
        origin[0] = origin[1] = origin[2] = 0.0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                H[i][j] = 0.0;
                invH[i][j] = 0.0;
            }
        }
    }
};

struct NeighborQueryResult {
    int idx;
    int32_t number;
    double d;
    double d2;
    double dr[3];
    int pbc_shift[3];
};

void invert_3x3(const double A[3][3], double invA[3][3]);

void matvec(const double A[3][3], const double v[3], double out[3]);

void minimum_image_delta(const SystemData& sys, int i, int j, double dr[3]);

class CellListNeighborFinder {
public:
    CellListNeighborFinder();

    void build(const SystemData& sys, double max_cutoff);

    // Returns all neighbors within cutoff, excluding the center atom.
    void query_cutoff(
        int center,
        double cutoff,
        std::vector<NeighborQueryResult>& out,
        bool sort_output = true
    ) const;

    // Returns K nearest points, including center atom as out[0].
    void query_topk(
        int center,
        int k,
        std::vector<NeighborQueryResult>& out
    ) const;

    double max_cutoff() const;

    int nx() const;
    int ny() const;
    int nz() const;

private:
    const SystemData* sys_;
    double max_cutoff_;

    int nx_, ny_, nz_;
    std::vector<int> head_;
    std::vector<int> next_;
    std::vector<int> atom_cell_;

    static double wrap01(double x);

    void atom_fractional(int i, double s[3]) const;

    void atom_to_cell(int i, int& ix, int& iy, int& iz) const;

    int cell_index(int ix, int iy, int iz) const;

    void halo_for_cutoff(double cutoff, int& hx, int& hy, int& hz) const;

    static void make_periodic_range(
        int c,
        int h,
        int n,
        std::vector<int>& out
    );
};

double estimate_bcc_lattice_from_density(
    const SystemData& sys,
    double fallback
);

void build_neighbour_cache(
    SystemData& sys,
    const CellListNeighborFinder& finder,
    int max_points
);

// PTM callback. This function reads sys.neigh_cache.
int get_neighbours_lmp(
    void* vdata,
    size_t unused,
    size_t atom_index,
    int num,
    int* ordering,
    size_t* nbr_indices,
    int32_t* numbers,
    double (*nbr_pos)[3]
);

#endif
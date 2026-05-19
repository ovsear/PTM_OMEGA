#ifndef PTM_IO_LAMMPS_H
#define PTM_IO_LAMMPS_H

#include <string>
#include <vector>

#include "ptm_neighbor_finder.h"
#include "ptm_analysis_types.h"

bool read_lammps_data_atomic(
    const char* filename,
    SystemData& sys
);

bool read_lammps_dump_atomic(
    const char* filename,
    SystemData& sys,
    int frame_index = -1,
    bool preserve_original_columns = false
);

bool read_lammps_atomic_auto(
    const char* filename,
    SystemData& sys,
    int frame_index = -1,
    bool preserve_original_columns = false
);

void write_ptm_dump(
    const std::string& outname,
    const SystemData& sys,
    const std::vector<PTMAtomResult>& results
);

std::string make_output_dump_name(
    const std::string& input_name
);

const char* type_name(int type);

#endif
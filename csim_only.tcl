# build.tcl (auto-generated)

# Create a project
open_project -reset hls_project
# Add design files
add_files source/gemm.cpp
add_files -tb source/tb.cpp -cflags "-Wno-unknown-pragmas"

# Set the top-level function
set_top gemm

# ########################################################
# Create a solution

open_solution -flow_target vitis -reset "gemm2"

# Device Selection: vck190
set_part {xcvc1902-vsva2197-2MP-e-S}

create_clock -period 300MHz -name default

# Disabling automatic pipelining of loops
config_compile -pipeline_loops 0
config_interface -m_axi_conservative_mode=0
config_compile -enable_auto_rewind
config_rtl -add_register_in_block_condition=false

# Disable bugged (?) srl mapping
config_storage fifo -impl auto -auto_srl_max_bits 0 -auto_srl_max_depth 0
config_compile -unsafe_math_optimizations

# Disabling automatic partition
config_array_partition -throughput_driven off
config_array_partition -complete_threshold 1

csim_design

exit
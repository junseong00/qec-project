# OOC (block-level) synth + place + route for pipelined FFN on Zynq-7020.
# Run:  vivado -mode batch -source synth_ffn_pipe.tcl
# NOTE: single-threaded — Win11 NoDefaultCurrentDirectoryInExePath policy breaks
# the multithread helper child-process spawn (couldn't read unimacro_*.tcl).
set_param general.maxThreads 1
set rtl C:/Users/js030/Downloads/fpga_handoff/rtl
set rpt C:/Users/js030/Downloads/fpga_handoff/reports

read_verilog $rtl/ffn_pipe.v
synth_design -top ffn -part xc7z020clg400-1 -mode out_of_context -include_dirs $rtl
create_clock -name clk -period 10.000 [get_ports clk]
opt_design
place_design
route_design

report_utilization -file $rpt/utilization_ffn_pipe_impl.txt
report_timing_summary -delay_type max -max_paths 5 -file $rpt/timing_ffn_pipe_impl.txt

# concise console summary
set wns [get_property SLACK [get_timing_paths -delay_type max -max_paths 1 -nworst 1]]
puts "FFN_PIPE_WNS $wns"
set luts [llength [get_cells -hier -filter {REF_NAME =~ LUT*}]]
puts "FFN_PIPE_DONE"

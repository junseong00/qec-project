set repo [file normalize [pwd]]
set out_dir [file join $repo report_assets vivado_schematics]
file mkdir $out_dir

proc export_rtl_schematic {name top files} {
    global repo out_dir
    puts "=== EXPORT_RTL_SCHEMATIC $name top=$top ==="
    create_project -in_memory -part xc7z020clg400-1
    foreach f $files {
        read_verilog [file join $repo $f]
    }
    synth_design -top $top -rtl -rtl_skip_mlo -part xc7z020clg400-1
    set view_name "${name}_rtl_schematic"
    show_schematic -name $view_name [get_cells]
    if {[catch {write_schematic -format svg -name $view_name -force [file join $out_dir "${name}_rtl_schematic.svg"]} err]} {
        puts "SVG_FAILED: $err"
    }
    if {[catch {write_schematic -format pdf -name $view_name -force [file join $out_dir "${name}_rtl_schematic.pdf"]} err]} {
        puts "PDF_FAILED: $err"
    }
    close_project
}

export_rtl_schematic noise_estimator_pipe noise_estimator {
    rtl/noise_estimator_pipe.v
}

export_rtl_schematic ffn_pipe ffn {
    rtl/ffn_pipe.v
}

exit

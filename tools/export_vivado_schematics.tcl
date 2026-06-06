set repo [file normalize [pwd]]
set out_dir [file join $repo report_assets vivado_schematics]
file mkdir $out_dir

proc export_one {name top files} {
    global repo out_dir
    puts "=== EXPORT_SCHEMATIC $name top=$top ==="
    create_project -in_memory -part xc7z020clg400-1
    foreach f $files {
        read_verilog [file join $repo $f]
    }
    synth_design -top $top -mode out_of_context
    opt_design
    puts "Writing schematic for $name"
    if {[catch {write_schematic -format png -force [file join $out_dir "${name}.png"]} err]} {
        puts "PNG_FAILED: $err"
    }
    if {[catch {write_schematic -format pdf -force [file join $out_dir "${name}.pdf"]} err]} {
        puts "PDF_FAILED: $err"
    }
    if {[catch {write_edif -force [file join $out_dir "${name}.edf"]} err]} {
        puts "EDIF_FAILED: $err"
    }
    close_project
}

export_one noise_estimator_pipe noise_estimator {
    rtl/noise_estimator_pipe.v
}

export_one ffn_pipe ffn {
    rtl/ffn_pipe.v
}

exit

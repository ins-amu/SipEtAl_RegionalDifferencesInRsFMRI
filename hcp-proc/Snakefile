#
# Preprocessing of Human Connectome Project data for use in whole-brain modeling
# Viktor Sip, 2020
#
#
# TODO:
#   - Proper handling of threads using Snakemake's capabilities
#   - Other parcellations
#   - Export in TVB format



import os

SUBJECTS = next(os.walk('data/Structural/'))[1]
FMRI_PHASE_ENCODINGS = ["RL", "LR"]
FMRI_REST_SESSIONS = [1, 2]
NTRACKS = "10M"
CONFS = ["all", "ctx"]

scriptdir = os.path.join(workflow.basedir, 'scripts')
etcdir = os.path.join(workflow.basedir, 'etc')


# Diffusion processing ------------------------------------------------------------------------------------------ #

rule dwi_convert:
    input:
        data="data/Diffusion/{s}/T1w/Diffusion/data.nii.gz",
        bvecs="data/Diffusion/{s}/T1w/Diffusion/bvecs",
        bvals="data/Diffusion/{s}/T1w/Diffusion/bvals"
    output: "run/{s}/dwi.mif"
    threads: 4
    shell: "mrconvert {input.data} -fslgrad {input.bvecs} {input.bvals} -datatype float32 -strides 0,0,0,1 {output[0]} -nthreads {threads}"

rule dwi2response:
    input: "run/{s}/dwi.mif"
    output:
        wm="run/{s}/response_wm.txt",
        gm="run/{s}/response_gm.txt",
        csf="run/{s}/response_csf.txt"
    threads: 4
    shell: "dwi2response dhollander {input} {output.wm} {output.gm} {output.csf} -nthreads {threads}"

rule avg_response:
    input: expand("run/{s}/response_{{tissue}}.txt", s=SUBJECTS)
    output: "run/group/response_{tissue}.txt"
    shell: "responsemean {input} {output}"

rule dwi2fod:
    input:
        dwi="run/{s}/dwi.mif",
        resp_wm="run/group/response_wm.txt",
        resp_gm="run/group/response_gm.txt",
        resp_csf="run/group/response_csf.txt",
        mask="data/Diffusion/{s}/T1w/Diffusion/nodif_brain_mask.nii.gz"
    output:
        fod_wm="run/{s}/fod_wm.mif",
        fod_gm="run/{s}/fod_gm.mif",
        fod_csf="run/{s}/fod_csf.mif"
    threads: 4
    shell:
        "dwi2fod msmt_csd {input.dwi}"
        "    {input.resp_wm} {output.fod_wm} {input.resp_gm} {output.fod_gm} {input.resp_csf} {output.fod_csf}"
        "    -mask {input.mask} -nthreads {threads}"

rule tckgen:
    input:
        fod_wm="run/{s}/fod_wm.mif",
        mask="data/Diffusion/{s}/T1w/Diffusion/nodif_brain_mask.nii.gz",
    output:
        tracks="run/{s}/tracks_all.tck"
    threads: 4
    shell:
        "tckgen {input.fod_wm} {output.tracks} -algorithm iFOD2 -mask {input.mask} -seed_image {input.mask}"
        "    -select {NTRACKS} -nthreads {threads}"

rule sift:
    input:
        tracks="run/{s}/tracks_all.tck",
        fod_wm="run/{s}/fod_wm.mif"
    output: "run/{s}/tracks_sifted.tck",
    threads: 4
    shell: "tcksift {input.tracks} {input.fod_wm} {output} -nthreads {threads}"

rule labels:
    input:
        aparc="data/Structural/{s}/T1w/aparc+aseg.nii.gz",
        inlut=os.path.join(etcdir, "FreeSurferColorLUT.txt"),
        outlut=os.path.join(etcdir, "lut.dk.{conf}.txt")
    output: "run/{s}/labels.{conf}.nii.gz"
    shell: "labelconvert {input.aparc} {input.inlut} {input.outlut} {output}"

rule conn:
    input:
        tracks="run/{s}/tracks_sifted.tck",
        labels="run/{s}/labels.{conf}.nii.gz",
    output: "run/{s}/res_{conf}/weights.txt",
    shell: "tck2connectome {input.tracks} {input.labels} {output} -symmetric -zero_diagonal"

rule region_list:
    input: os.path.join(etcdir, "lut.dk.{conf}.txt")
    output: "run/{s}/res_{conf}/region_names.txt"
    shell: "cat {input} | grep -Ev '(#.*$)|(^$)' | tr -s ' '    | cut -d' ' -f3 | tail -n +2 > {output}"
    #                   | Remove comments        | Merge delims | Third column  | Skip Unknown

rule sc:
    input: expand(["run/{s}/res_{conf}/weights.txt", "run/{s}/res_{conf}/region_names.txt"], s=SUBJECTS, conf=CONFS)

# fMRI processing ---------------------------------------------------------------------------------------------- #

rule labels_subcort:
    input: "data/Structural/{s}/MNINonLinear/ROIs/Atlas_ROIs.2.nii.gz"
    output: "run/{s}/labels_subcort.dlabel.nii"
    shell: "wb_command -cifti-create-label {output} -volume {input} {input}"

rule labels_cort:
    input: "data/Structural/{s}/MNINonLinear/fsaverage_LR32k/{s}.aparc.32k_fs_LR.dlabel.nii"
    output: "run/{s}/labels_cort.dlabel.nii"
    shell: "cp {input} {output}"

rule parcellate_fmri:
    input:
        fmri="data/rfMRI/{s}/MNINonLinear/Results/rfMRI_REST{ses}_{pe}/rfMRI_REST{ses}_{pe}_Atlas_hp2000_clean.dtseries.nii",
        labels="run/{s}/labels_{type}.dlabel.nii"
    output: "run/{s}/rfmri/REST{ses}_{pe}_{type}.ptseries.nii"
    shell: "wb_command -cifti-parcellate {input.fmri} {input.labels} COLUMN {output}"

rule merge_fmri:
    input:
        cort="run/{s}/rfmri/REST{ses}_{pe}_cort.ptseries.nii",
        subcort="run/{s}/rfmri/REST{ses}_{pe}_subcort.ptseries.nii",
        regions="run/{s}/res_{conf}/region_names.txt"
    output: "run/{s}/res_{conf}/rfMRI_REST{ses}_{pe}.npz"
    shell: "python {scriptdir}/merge_fmri.py {input.cort} {input.subcort} {input.regions} {output}"


rule fmri:
    input:
        expand("run/{s}/res_{conf}/rfMRI_REST{ses}_{pe}.npz", s=SUBJECTS, conf=CONFS,
               ses=FMRI_REST_SESSIONS, pe=FMRI_PHASE_ENCODINGS)

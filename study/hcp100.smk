
envvars: "SUBJECTS_FILE"

NRUNS = 2
NSIMULATIONS = 50

with open(os.environ["SUBJECTS_FILE"]) as fh:
    SUBJECTS = fh.read().splitlines()
    NSUBJECTS = len(SUBJECTS)

rule generate_dataset:
    input:
    output:
        data="run/hcp/{dataset}/dataset.npz",
        img=directory("run/hcp/{dataset}/img"),
        surrogates="run/hcp/{dataset}/surrogates.npz"
    threads: 1
    shell: "python generate_datasets.py {wildcards.dataset} {output.data} {output.img} {NSIMULATIONS} {output.surrogates} {threads}"


rule fit:
    input: "run/{study}/{dataset}/dataset.npz"
    output: directory("run/{study}/{dataset}/model{model}/{params}/run{run}/fit/")
    log: "logs/fit.{study}.{dataset}.{model}.{params}.{run}.log"
    threads: 36
    shell: "python util.py fit --train-ratio 0.8 {input} {wildcards.model} {wildcards.params} {wildcards.run} {output} {threads} &> {log}"



rule simulate:
    input:
        dataset="run/{study}/{dataset}/dataset.npz",
        fit_direc="run/{study}/{dataset}/model{model}/{params}/run{run}/fit/"
    output:
        parameters="run/{study}/{dataset}/model{model}/{params}/run{run}/parameters/params_{isub}.npz",
        simulations="run/{study}/{dataset}/model{model}/{params}/run{run}/simulations/simulations_{isub}.npz"
    log: "logs/simulate.{study}.{dataset}.{model}.{params}.{run}.{isub}.log"
    threads: 1,
    resources:
        mem_mb=4000,
    shell:
        "export CUDA_VISIBLE_DEVICES='';"
        "python util.py simulate --subject {wildcards.isub} --nthreads {threads}"
        "    {input.dataset} {wildcards.model} {wildcards.params} {input.fit_direc} {NSIMULATIONS}"
        "    {output.parameters} {output.simulations} &> {log}"

rule simulations:
    input: expand("run/hcp/hcp100_{{conn}}_{{preproc}}/model{{model}}/{{conf}}/run{{run}}/simulations/simulations_{isub:03d}.npz", isub=range(0,NSUBJECTS))
    output: touch("run/hcp/hcp100_{conn}_{preproc}/model{model}/{conf}/run{run}/simulations/simulations.done")


rule fc:
    input:
        dataset="run/hcp/hcp100_{conn}_{preproc}/dataset.npz",
        simfiles=expand("run/hcp/hcp100_{{conn}}_{{preproc}}/model{{model}}/{{conf}}/run{{run}}/simulations/simulations_{isub:03d}.npz", isub=range(0,NSUBJECTS))
    output: "run/hcp/hcp100_{conn}_{preproc}/model{model}/{conf}/run{run}/simulations/fc.npz"
    shell: "python util.py fc --dataset {input.dataset} --input {input.simfiles} --output {output}"




rule all:
    input: expand("run/hcp/hcp100_{conn}_{preproc}/model{model}/{conf}/run{run:02d}/simulations/fc.npz",            \
                  conn=["linw", "logw", "linwhom", "logwhom"], preproc=["dicer"], model=["AN", "AB"],               \
                  conf=["ns_3_mreg_3_msub_0_nf_32"], run=range(0, NRUNS)),                                          \
           expand("run/hcp/hcp100_linw_dicer/model{model}/{conf}/run{run:02d}/simulations/fc.npz",                  \
                  conf=["ns_1_mreg_3_msub_0_nf_32", "ns_2_mreg_3_msub_0_nf_32", "ns_3_mreg_3_msub_0_nf_32", "ns_4_mreg_3_msub_0_nf_32", "ns_5_mreg_3_msub_0_nf_32",   \
                        "ns_3_mreg_1_msub_0_nf_32", "ns_3_mreg_2_msub_0_nf_32", "ns_3_mreg_3_msub_0_nf_32", "ns_3_mreg_4_msub_0_nf_32", "ns_3_mreg_5_msub_0_nf_32"],  \
                  model=['AB'], run=range(0,NRUNS))

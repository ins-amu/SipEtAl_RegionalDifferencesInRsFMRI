

NRUNS = 1
NSIMULATIONS = 50
CONFS = ["ns_3_mreg_3_msub_3_nf_32"]
MODELS = ['AB']


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
    threads: 8
    shell: "python util.py fit --train-ratio 0.8 {input} {wildcards.model} {wildcards.params} {wildcards.run} {output} {threads}"


rule simulate:
    input:
        dataset="run/{study}/{dataset}/dataset.npz",
        fit_direc="run/{study}/{dataset}/model{model}/{params}/run{run}/fit/"
    output:
        parameters="run/{study}/{dataset}/model{model}/{params}/run{run}/parameters.npz",
        simulations="run/{study}/{dataset}/model{model}/{params}/run{run}/simulations.npz"
    threads: 8,
    shell: "python util.py simulate {input.dataset} {wildcards.model} {wildcards.params} {input.fit_direc} {NSIMULATIONS} {output.parameters} {output.simulations}"



rule all:
    input: expand("run/hcp/hcp100_linw_dicer/model{model}/{conf}/run{run:02d}/simulations.npz", \
                  model=MODELS, conf=CONFS, run=range(0, NRUNS))

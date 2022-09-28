Project-description

Libraries used:
    - torch is main machine learning library used
    - torch-geometric is used for graph learning
    - networkx is used in ortder to get 'beautiful' represenations of my graph
    - matplotlib
    - and others

Modules:
    - data: METABRIC - cancer data describing different patients with mRNA,
    CNA samples (genomic), and Clin means clinical (Booleans Yes or No questions)

    - models: machine learning models used in this project
        - CNCVAE and HVAE are re-implemented models from this paper https://www.frontiersin.org/articles/10.3389/fgene.2019.01205/full
        - all the others are combinations of different models, main being VGAE (variational graph autoencoder) and DGI (DeepGraphInfomax)

    - statistics: some statistics on the radius or hyper-parameters for the graph construction I could use

    - pictures: I have done manual assessment on the models, I plotted the obtained embeddings
    to get a better understanding of how the models work

    - sinteticdata_run_models: I have tested my models on sintetic- generated data, for normal distributions that overlap with each other

    - some_results repreesent my tentaitive to get the embeddings of some representations of the data

Description:

I tried novel approaches over MATABRIC data, they give competitive results on the ER and DR datasets.
First I tested them on the sintetic data, if they returned good results I have used them for later 

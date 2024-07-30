import os
import shutil
import argparse
import yaml
import random
import copy
import time
from typing import Tuple
import logging
from datetime import datetime
import multiprocess
import iteration_utilities

from oligo_designer_toolsuite.database import NcbiGenomicRegionGenerator, EnsemblGenomicRegionGenerator, CustomGenomicRegionGenerator, OligoDatabase, ReferenceDatabase
from oligo_designer_toolsuite.oligo_property_filter import PropertyFilter, MaskedSequences
from oligo_designer_toolsuite.oligo_specificity_filter import BlastNFilter
from oligo_designer_toolsuite.pipelines import BaseOligoDesigner
from Bio.Seq import MutableSeq, Seq
from Bio.SeqUtils import gc_fraction
from Bio.SeqUtils import MeltingTemp as mt
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import nupack
from math import log

from .generate_artificial_dataset import split_list, reverse_complement, mutate, duplexing_log_scores, base_pair


def main():
    """Generate an artificial dataset containing oligos and some hand-crafted mutations with the 
    relative mutations scores. The oligos are extracted form a given list of genes and uniformly sampled to match 
    the desidred dataset size. These oligos are then mutated by applying 0 to max_mutaions base-pairs mutations to generate potential off-targets.
    (REMARK: for each nr. of mutations we create an off-target region startic from each nucleotide of the oligo sequence
    and selecting the remaining mutated nucleotides uniformly. Therefore, from each oligo we generate 
    O(max_mutations * oligo_length) off-target regions.)

    The duplexing score is obtained from the final concentration of DNA complexes in NUPACK tube experiment simulation
    that contains the oligo sequence, the exact on-target region and the off-target. The oligo, on-target and off-target
     strands are initially set at the same concentration $C_{in}$ and we define the duplexing score as: 
    
    log( C_{oligo + off-t} /C_{oligo + off-t}  + C_{oligo + on-t}  ). 
    
    The oligos, the on-target regions and off-target regions are inserted in order to compare the amount of oligos that 
    bind to one and to the other. Additionally the log is used to sterch the scored distribution making them 
    easier to predict and a small value eps = 1e-12 is used for numerical stability.
    """

    #########################
    # read in out arguments #
    #########################

    start = time.time()
    parser = argparse.ArgumentParser(
        prog="Artificial Dataset",
        usage="generate_artificial_dataset [options]",
        description=main.__doc__,
    )
    parser.add_argument("-c", "--config", help="path to the configuration file", default="config/generate_artificial_dataset.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as handle:
        config = yaml.safe_load(handle)
    size = config["n_oligos"]*config["n_mutations_per_type"]*(config["max_mutations"] + config["max_bulges_size"])
    dataset_name = f"artificial_dataset_{config['min_length']}_{config['max_length']}_{size}"
    # set random seed for reproducibility
    random.seed(config["seed"])
    # generate directories
    os.makedirs(config["dir_output"], exist_ok=True)
    plots_dir = os.path.join(config["dir_output"], f"{dataset_name}_plots")
    os.makedirs(plots_dir, exist_ok=True)
    # nupack run
    nupack.config.threads = config["nupack_threads"] # use all cores
    nupack.config.cache = config["nupack_cache"]
    

    ##############
    # set logger #
    ##############

    timestamp = datetime.now()
    file_logger = f"log_{dataset_name}_{timestamp.year}-{timestamp.month}-{timestamp.day}-{timestamp.hour}-{timestamp.minute}.txt"
    logging.getLogger("artificial_dataset_generation")
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
        handlers=[logging.FileHandler(file_logger), logging.StreamHandler()],
    )

    ################################
    # generate the oligo sequences #
    ################################

    oligo_designer = BaseOligoDesigner(dir_output = "output_odt", log_name="database_generation")
    oligo_designer.load_annotations(source=config["source"], source_params=config["source_params"])

    oligo_database, _ = oligo_designer.create_oligo_database(
        regions=genes,
        genomic_regions=config["genomic_regions"],
        oligo_length_min=config["oligo_length_min"],
        oligo_length_max=config["oligo_length_max"],
        min_oligos_per_region=config["min_oligos_per_gene"],
        n_jobs=config["n_jobs"],
    )
    logging.info("Oligo seqeunces generated.")

    for gene in oligo_database.database.keys():
        logging.info(f"Gene {gene} has {len(oligo_database.database[gene].keys())} oligos.")

    # Property filtering
    masked_seqeunces = MaskedSequences()
    property_filter = PropertyFilter(filters=[masked_seqeunces])
    oligo_database = property_filter.apply(oligo_database=oligo_database, n_jobs=config["n_jobs"])
    logging.info("Oligo sequences filtered (property).")

    ############################
    # generate the off-targets #
    ############################

    # Run the blastn method on the reference and extract the off target sites
    # TODO

    ##############################
    # sample the oligo sequences #
    ##############################

    # original distribution of the GC content and length
    gc_content =[]
    oligo_length = []
    for gene, oligos in oligo_database.database.items():
        for oligo, features in oligos.items():
            gc_content.append([gc_fraction(features["sequence"]), "Original"])
            oligo_length.append([features["length"], "Original" ])
    # split the genes
    genes = list(oligo_database.database.keys())
    genes_train, genes_validation, genes_test = split_list(genes, config["splits_size"])
    # create list of oligos
    oligos_train = [oligo_database.database[gene][oligo_id]["sequence"] for gene in genes_train for oligo_id in oligo_database.database[gene]]
    oligos_validation = [oligo_database.database[gene][oligo_id]["sequence"] for gene in genes_validation for oligo_id in oligo_database.database[gene]]
    oligos_test = [oligo_database.database[gene][oligo_id]["sequence"] for gene in genes_test for oligo_id in oligo_database.database[gene]]
    # sample the oligos
    sample_train = round(config["splits_size"][0]*config["n_oligos"])
    sample_validation = round(config["splits_size"][1]*config["n_oligos"])
    sample_test= config["n_oligos"] - sample_train - sample_validation
    # first sample 10 times the size and get rid of duplicates
    oligos_train = random.sample(population=oligos_train, k=min(len(oligos_train), sample_train*10))
    oligos_validation = random.sample(population=oligos_validation, k=min(len(oligos_validation), sample_validation*10))
    oligos_test = random.sample(population=oligos_test, k=min(len(oligos_test), sample_test*10))
    duplicated_sequences = list(
            iteration_utilities.unique_everseen(
                iteration_utilities.duplicates(oligos_train + oligos_validation + oligos_test)
            )
        )
    for s in duplicated_sequences:
        oligos_train.remove(s)
        oligos_validation.remove(s)
        oligos_test.remove(s)
    # now sample the remaining oligos
    if len(oligos_train) < sample_train or len(oligos_validation) < sample_validation or len(oligos_test) < sample_test:
       raise Warning("Fewr oligos left to sample.")
    oligos_train = random.sample(population=oligos_train, k=min(len(oligos_train), sample_train))
    oligos_validation = random.sample(population=oligos_validation, k=min(len(oligos_validation), sample_validation))
    oligos_test = random.sample(population=oligos_test, k=min(len(oligos_test), sample_test))
    logging.info(f"Sampled {len(oligos_train)} oligos for training, {len(oligos_validation)} oligos for validation, and {len(oligos_test)} oligos for testing")
    # sampled distribution of the GC content and length
    for oligo in oligos_train:
        gc_content.append([gc_fraction(oligo), "Train"])
        oligo_length.append([len(oligo), "Train" ])
    for oligo in oligos_validation:
        gc_content.append([gc_fraction(oligo), "Validation"])
        oligo_length.append([len(oligo), "Validation" ])
    for oligo in oligos_test:
        gc_content.append([gc_fraction(oligo), "Test"])
        oligo_length.append([len(oligo), "Test" ])
    # plot the distributions
    gc_content = pd.DataFrame(data=gc_content, columns=["GC content", "Source"])
    oligo_length = pd.DataFrame(data=oligo_length, columns=["Length", "Source"])
    plt.figure(1)
    sns.violinplot(data=gc_content, y="GC content", x="Source")
    plt.title("GC content distribution")
    plt.savefig(os.path.join(plots_dir,"GC_content_distribution.pdf"))
    plt.figure(2)
    sns.violinplot(data=oligo_length, y="Length", x="Source")
    plt.title("Oligo length distribution")
    plt.savefig(os.path.join(plots_dir, "Oligo_length_distribution.pdf"))

    ################################################################
    # generate real off-targets and compute duplexing scores #
    ################################################################
    
    # train
    # train_alignments = joblib.Parallel(n_jobs=config["n_jobs"])(
    #     joblib.delayed(generate_off_targets)(
    #         oligo.upper(), config
    #     )
    #     for oligo in oligos_train
    # )
    train_alignments = [generate_off_targets(oligo.upper(), config) for oligo in oligos_train]
    train_alignments = [alignment for oligo_alignments in train_alignments for alignment in oligo_alignments] # flatten the returned structure
    # validation
    # validation_alignments = joblib.Parallel(n_jobs=config["n_jobs"])(
    #     joblib.delayed(generate_off_targets)(
    #         oligo.upper(), config
    #         )
    #     for oligo in oligos_validation
    # )
    validation_alignments = [generate_off_targets(oligo.upper(), config) for oligo in oligos_validation]
    validation_alignments = [alignment for oligo_alignments in validation_alignments for alignment in oligo_alignments] # flatten the returned structure
    # # test
    # test_alignments = joblib.Parallel(n_jobs=config["n_jobs"])(
    #     joblib.delayed(generate_off_targets)(
    #         oligo.upper(), config
    #         )
    #     for oligo in oligos_test
    # )
    test_alignments = [generate_off_targets(oligo.upper(), config) for oligo in oligos_test]
    test_alignments = [alignment for oligo_alignments in test_alignments for alignment in oligo_alignments] # flatten the returned structure
    logging.info("Generated artificial off-targets.")
    ##################
    # create dataset #
    ##################

    train_dataset = generate_dataset(train_alignments)
    file_train = os.path.join(config["dir_output"], f"{dataset_name}_train.csv")
    train_dataset.to_csv(file_train)
    validation_dataset = generate_dataset(validation_alignments)
    file_validation = os.path.join(config["dir_output"], f"{dataset_name}_validation.csv")
    validation_dataset.to_csv(file_validation)
    test_dataset = generate_dataset(test_alignments)
    file_test = os.path.join(config["dir_output"], f"{dataset_name}_test.csv")
    test_dataset.to_csv(file_test)
    logging.info(f"Dataset created and stored at: \n\t - {file_train},\n\t - {file_validation}, \n\t - {file_test}.")
    # plot distributions of the ground truths
    plt.figure(3)
    train_dataset["Source"] = "Train"
    validation_dataset["Source"] = "Validation"
    test_dataset["Source"] = "Test"
    dataset = pd.concat([train_dataset, validation_dataset, test_dataset])
    dataset["Duplexing score"] = dataset["duplexing_log_score"] # rename for better understanding
    sns.boxplot(data=dataset, x="Source", y="Duplexing score")
    plt.title("Duplexing scores distributions")
    plt.savefig(os.path.join(plots_dir,"Duplexing_scores_distribution.pdf"))
    
    logging.info(f"Computational time: {time.time() - start}")
    shutil.rmtree("output_odt") #remove oligo designer toolsuite output
    plt.show()


if __name__ == "__main__":
    main()
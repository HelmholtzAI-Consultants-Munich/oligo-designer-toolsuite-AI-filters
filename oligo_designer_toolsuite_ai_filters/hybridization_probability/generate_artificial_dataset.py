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

from oligo_designer_toolsuite.sequence_generator import NcbiGenomicRegionGenerator, EnsemblGenomicRegionGenerator, CustomGenomicRegionGenerator
from oligo_designer_toolsuite.database import OligoDatabase
from oligo_designer_toolsuite.oligo_property_filter import PropertyFilter, HardMaskedSequenceFilter
from oligo_designer_toolsuite.pipelines._base_oligo_designer import BaseOligoDesigner
from Bio.Seq import MutableSeq, Seq
from Bio.SeqUtils import gc_fraction
from Bio.SeqUtils import MeltingTemp as mt
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import nupack
from math import log



base_pair = {'A':'T', 'T':'A', 'C':'G', 'G':'C'} #, 'a':'t', 't':'a', 'c':'g', 'g':'c'}

def split_list(l: list, spilts_perc: list[float]):
    assert sum(spilts_perc) == 1, "The splits percentages must su up to 1"
    n = len(l)
    n_splits = len(spilts_perc)
    splits = [0]
    for i in range(n_splits-1):
        splits.append(round(splits[-1] + round(spilts_perc[i]*n)))
    splits.append(n)
    # randomly shuffle the list
    random.shuffle(l)
    # genrate the splits
    final_splits = []
    for i in range(n_splits):
        final_splits.append(l[splits[i]:splits[i+1]])
    return final_splits

def reverse_complement(strand: str) -> str:
    strand = list(strand)
    strand.reverse()
    for i in range(len(strand)):
        strand[i] = base_pair[strand[i]]
    return "".join(strand)

def mutate(nt: str) -> str:
    nts = ['A', 'C', 'T', 'G']
    nts.remove(nt)
    return random.sample(nts, 1)[0]


def duplexing_log_scores(oligo: str, off_target: str, model: nupack.Model, concentration: float) -> float:
     # oligo must be reversed and complemented
    oligo_strand = nupack.Strand(oligo, name="oligo")
    on_target_strand = nupack.Strand(reverse_complement(oligo), name="on_target")
    off_target_strand = nupack.Strand(reverse_complement(off_target), name="off_target")
    t = nupack.Tube(strands={oligo_strand: concentration, on_target_strand: concentration, off_target_strand: concentration}, name='t', complexes=nupack.SetSpec(max_size=2))
    tube_results = nupack.tube_analysis(tubes=[t], model=model)
    tube_concentrations = tube_results[t].complex_concentrations
    # calculate the percentage of sequences that bind to the off target region
    off_target_perc = tube_concentrations[nupack.Complex(strands=[oligo_strand,off_target_strand])] / (tube_concentrations[nupack.Complex(strands=[oligo_strand,off_target_strand])] + tube_concentrations[nupack.Complex(strands=[oligo_strand,on_target_strand])])
    return log(off_target_perc, 10) # log normalization


def generate_off_targets(sequence: Seq, config) -> list[Tuple[str,str, int, float]]:
    model = nupack.Model()
    off_target_regions = [(str(sequence), str(sequence), 0, duplexing_log_scores(str(sequence), str(sequence), model, config["concentration"]))] # include an exact match
    # single point mutations
    for i in range(1, config["max_mutations"]+1): # nr of mutations
        for _ in range(1, config["n_mutations_per_type"]+1): # nr of mutations for mutation class
            # mutate i nt
            target = MutableSeq(sequence)
            unchanged_nts = list(range(len(sequence)))
            for _ in range(i):
                k = random.sample(unchanged_nts, 1)[0]
                new_nt = mutate(target[k])
                target.pop(k)
                target.insert(k, new_nt)
                unchanged_nts.remove(k)
            off_target_regions.append((str(sequence), str(target), i, duplexing_log_scores(str(sequence), str(target), model, config["concentration"])))
    # bulges (insertions and deletions)
    for i in range(1, config["max_bulges_size"]+1): # nr of mutations
        for _ in range(1, config["n_mutations_per_type"]+1):
            # insert i nts
            target = MutableSeq(sequence)
            new_sequence = MutableSeq(sequence)
            insertion_point = random.randrange(0, len(sequence))
            for _ in range(i):
                nt = random.choice(['A', 'T', 'C', 'G'])
                target.insert(insertion_point, nt)
                new_sequence.insert(insertion_point, '-') # generate to have a correct alignement with of the sequnces (- with be encoded as a 0 vector)
            off_target_regions.append((str(new_sequence), str(target), i, duplexing_log_scores(str(sequence), str(target), model, config["concentration"])))
            # delete i nts
            target = MutableSeq(sequence)
            deletion_point = random.randint(0, len(sequence) - i) # leave the sapace to delete i nucleotides
            for _ in range(i):
                target.pop(deletion_point)
            new_target = MutableSeq(target)
            for _ in range(i):
                new_target.insert(deletion_point, '-') # generate to have a correct alignement with of the sequnces
            off_target_regions.append((str(sequence), str(new_target), i, duplexing_log_scores(str(sequence), str(target), model, config["concentration"])))
    return off_target_regions

def generate_dataset(alignments: list):
    dataset = pd.DataFrame(index=list(range(len(alignments))), columns=["query_sequence", "query_length", "query_GC_content", "off_target_sequence", "off_target_length", "off_target_GC_content", "number_mismatches", "duplexing_log_score"])
    for i, (oligo, off_target, nr_mismatches, d_log_score) in enumerate(alignments):
        dataset.loc[i] = [
            oligo, #oligo sequence
            len(oligo),# oligo length
            gc_fraction(oligo),
            off_target,
            len(off_target), # off target length
            round(gc_fraction(off_target)), # off target gc content
            nr_mismatches,
            d_log_score,
        ]
    return dataset


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
    dataset_name = f"artificial_dataset_{config['oligo_length_min']}_{config['oligo_length_max']}_{size}"
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

    with open(config["file_genes"]) as handle:
        lines = handle.readlines()
        genes = [line.rstrip() for line in lines]

    oligo_database, _ = oligo_designer.create_oligo_database(
        regions=genes,
        genomic_regions=config["genomic_regions"],
        oligo_length_min=config["oligo_length_min"],
        oligo_length_max=config["oligo_length_max"],
        min_oligos_per_region=0,
        n_jobs=config["n_jobs"],
    )
    
    logging.info("Oligo seqeunces generated.")
    for gene in oligo_database.database.keys():
        logging.info(f"Gene {gene} has {len(oligo_database.database[gene].keys())} oligos.")
    # Property filtering
    masked_seqeunces = HardMaskedSequenceFilter()
    property_filter = PropertyFilter(filters=[masked_seqeunces])
    oligo_database = property_filter.apply(oligo_database=oligo_database, n_jobs=config["n_jobs"], sequence_type="oligo")
    logging.info("Oligo sequences filtered (property).")

    ##############################
    # sample the oligo sequences #
    ##############################

    # original distribution of the GC content and length
    gc_content =[]
    oligo_length = []
    for gene, oligos in oligo_database.database.items():
        for oligo, features in oligos.items():
            gc_content.append([gc_fraction(features["oligo"]), "Original"])
            oligo_length.append([len(features["oligo"]), "Original" ])
    # split the genes
    genes = list(oligo_database.database.keys())
    genes_train, genes_validation, genes_test = split_list(genes, config["splits_size"])
    # create list of oligos
    oligos_train = [oligo_database.database[gene][oligo_id]["oligo"] for gene in genes_train for oligo_id in oligo_database.database[gene]]
    oligos_validation = [oligo_database.database[gene][oligo_id]["oligo"] for gene in genes_validation for oligo_id in oligo_database.database[gene]]
    oligos_test = [oligo_database.database[gene][oligo_id]["oligo"] for gene in genes_test for oligo_id in oligo_database.database[gene]]
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
    # generate artificial off-targets and compute duplexing scores #
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
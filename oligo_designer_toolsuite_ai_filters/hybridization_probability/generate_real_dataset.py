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
import iteration_utilities

from oligo_designer_toolsuite.sequence_generator import OligoSequenceGenerator
from oligo_designer_toolsuite.database import OligoDatabase, ReferenceDatabase
from oligo_designer_toolsuite.oligo_property_filter import PropertyFilter, HardMaskedSequenceFilter, SoftMaskedSequenceFilter
from oligo_designer_toolsuite.pipelines import GenomicRegionGenerator
from oligo_designer_toolsuite.oligo_specificity_filter import (
    BowtieFilter,
    BlastNFilter,
)
from Bio.Seq import MutableSeq, Seq
from Bio.SeqUtils import gc_fraction
from Bio.SeqUtils import MeltingTemp as mt
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import nupack
from math import log
import joblib



base_pair = {'A':'T', 'T':'A', 'C':'G', 'G':'C', 'a':'T', 't':'A', 'c':'G', 'g':'C'}

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
    reverse_strand = []
    strand = list(strand)
    strand.reverse()
    for i in strand:
        if i == "-":
            continue
        reverse_strand.append(base_pair[i])
    return "".join(reverse_strand)


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

def generate_off_targets_region(
        oligo_database: OligoDatabase, 
        alignment_method: BlastNFilter, 
        file_index: str, region_id: str, 
        file_reference: str, 
        concentration: float
    ):
    """Return a list of all the retrived off target sites in the following format:


    :param oligo_database: _description_
    :type oligo_database: OligoDatabase
    :param alignment_method: _description_
    :type alignment_method: BlastNFilter
    :param file_index: _description_
    :type file_index: str
    :param region_id: _description_
    :type region_id: str
    """

    model = nupack.Model()
    # run the filter
    table_hits = alignment_method._run_filter(
        sequence_type='oligo',
        region_id=region_id,
        oligo_database=oligo_database,
        file_index=file_index,
        consider_hits_from_input_region=True,
    )

    # add the gaps
    references = alignment_method._get_references(table_hits, file_reference, region_id)
    queries = alignment_method._get_queries('oligo', table_hits, oligo_database, region_id)
    # align the references and queries by adding gaps
    gapped_queries, gapped_references = alignment_method._add_alignment_gaps(
        table_hits=table_hits, queries=queries, references=references
    )

    # create the output
    off_targets = []
    for query, reference, gapped_query, gapped_reference in zip(queries, references, gapped_queries, gapped_references):
        n_mismatches = sum(q != r for q, r in zip(gapped_query, gapped_reference))
        off_targets.append((str(gapped_query), str(gapped_reference), n_mismatches, duplexing_log_scores(str(query), str(reference), model, concentration)))
    return off_targets


def generate_off_targets(
        oligo_database: OligoDatabase, 
        alignment_method: BlastNFilter, 
        file_index: str, 
        config: dict, 
        dataset_size: int, 
        file_reference: str
    ):
    off_target_regions = joblib.Parallel(n_jobs=config["n_jobs"])(
        joblib.delayed(generate_off_targets_region)(
            oligo_database=oligo_database,
            alignment_method=alignment_method,
            file_index=file_index,
            region_id=region_id,
            file_reference=file_reference,
            concentration=config["concentration"]
        )
        for region_id in oligo_database.database.keys()
    )
    # flatten the list
    off_target_regions = [off_target for region in off_target_regions for off_target in region]
    
    # sample
    if dataset_size > len(off_target_regions):
        logging.warning(f"Fewer oligos left to sample: {len(off_target_regions)} instead of {dataset_size}.")
    off_target_regions = random.sample(population=off_target_regions, k=min(dataset_size, len(off_target_regions)))

    # create dataset
    dataset = generate_dataset(alignments=off_target_regions)

    return dataset


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


def sample_oligos(oligo_database: OligoDatabase, oligos_per_region: int):
    for region in oligo_database.database.keys():
        oligo_ids = list(oligo_database.database[region].keys())
        if len(oligo_ids) > oligos_per_region:
            filtered_oligo_ids = random.sample(population=oligo_ids, k=len(oligo_ids) - oligos_per_region) # sample the ids to filter
            for oligo_id in filtered_oligo_ids:
                oligo_database.database[region].pop(oligo_id, None)
    return oligo_database



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
    dataset_name = f"real_dataset_{config['alignment_method']}_{config['dataset_size']}_{config['oligo_length_min']}_{config['oligo_length_max']}_{config['dataset_size']}"
    # set random seed for reproducibility
    random.seed(config["seed"])
    # generate directories
    os.makedirs(config["dir_output"], exist_ok=True)
    plots_dir = os.path.join(config["dir_output"], f"{dataset_name}_plots")
    os.makedirs(plots_dir, exist_ok=True)
    # nupack run
    nupack.config.threads = config["n_jobs"] # use all cores
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

    genomic_region_genereator = GenomicRegionGenerator(dir_output = "output_odt")
    region_generator = genomic_region_genereator.load_annotations(source=config["source"], source_params=config["source_params"])
    files_fasta = genomic_region_genereator.generate_genomic_regions(
        region_generator = region_generator,
        genomic_regions  = config["genomic_regions"],
        block_size = 0,
    )

    with open(config["file_genes"]) as handle:
        lines = handle.readlines()
        genes = [line.rstrip() for line in lines]
    genes_train, genes_validation, genes_test = split_list(genes, config["splits_size"])

    ##### creating the oligo sequences #####
    oligo_sequences = OligoSequenceGenerator(dir_output="output_odt")
    oligo_fasta_file = oligo_sequences.create_sequences_sliding_window(
        files_fasta_in=files_fasta,
        length_interval_sequences=(config["oligo_length_min"], config["oligo_length_max"]),
        region_ids=genes,
        n_jobs=config["n_jobs"],
    )

    ##### creating the oligo database #####
    # one database for train, test and validation is created
    oligo_database_train = OligoDatabase(
        min_oligos_per_region=0,
        write_regions_with_insufficient_oligos=True,
        lru_db_max_in_memory=config["n_jobs"] * 2 + 1,
        database_name="oligo_database_train",
        dir_output="output_odt",
    )
    oligo_database_train.load_database_from_fasta(
        files_fasta=oligo_fasta_file,
        sequence_type="target",
        region_ids=genes_train,
    )

    oligo_database_validation = OligoDatabase(
        min_oligos_per_region=0,
        write_regions_with_insufficient_oligos=True,
        lru_db_max_in_memory=config["n_jobs"] * 2 + 1,
        database_name="oligo_database_validation",
        dir_output="output_odt",
    )
    oligo_database_validation.load_database_from_fasta(
        files_fasta=oligo_fasta_file,
        sequence_type="target",
        region_ids=genes_validation,
    )

    oligo_database_test = OligoDatabase(
        min_oligos_per_region=0,
        write_regions_with_insufficient_oligos=True,
        lru_db_max_in_memory=config["n_jobs"] * 2 + 1,
        database_name="oligo_database_test",
        dir_output="output_odt",
    )
    oligo_database_test.load_database_from_fasta(
        files_fasta=oligo_fasta_file,
        sequence_type="target",
        region_ids=genes_test,
    )

    reference_database = ReferenceDatabase(dir_output="output_odt")
    reference_database.load_database_from_fasta(files_fasta = files_fasta)
    file_reference = reference_database.write_database_to_fasta(
            filename=f"db_reference"
        )

    # Property filtering
    masked_seqeunces = HardMaskedSequenceFilter()
    soft_masked_seqeunces = SoftMaskedSequenceFilter()
    property_filter = PropertyFilter(filters=[masked_seqeunces, soft_masked_seqeunces])
    oligo_database_train = property_filter.apply(oligo_database=oligo_database_train, n_jobs=config["n_jobs"], sequence_type="oligo")
    oligo_database_validation = property_filter.apply(oligo_database=oligo_database_validation, n_jobs=config["n_jobs"], sequence_type="oligo")
    oligo_database_test = property_filter.apply(oligo_database=oligo_database_test, n_jobs=config["n_jobs"], sequence_type="oligo")
    logging.info("Oligo sequences filtered (property).")

    # log database information
    logging.info("Oligo seqeunces generated.")
    logging.info("Training set:")
    for gene in oligo_database_train.database.keys():
        logging.info(f"Gene {gene} has {len(oligo_database_train.database[gene].keys())} oligos.")
    logging.info("Validation set:")
    for gene in oligo_database_validation.database.keys():
        logging.info(f"Gene {gene} has {len(oligo_database_validation.database[gene].keys())} oligos.")
    logging.info("Test set:")
    for gene in oligo_database_test.database.keys():
        logging.info(f"Gene {gene} has {len(oligo_database_test.database[gene].keys())} oligos.")

    # sample the oligos
    oligo_database_train = sample_oligos(oligo_database=oligo_database_train, oligos_per_region=config["oligos_per_region"])
    oligo_database_validation = sample_oligos(oligo_database=oligo_database_validation, oligos_per_region=config["oligos_per_region"])
    oligo_database_test = sample_oligos(oligo_database=oligo_database_test, oligos_per_region=config["oligos_per_region"])


    ################################################################
    # generate artificial off-targets and compute duplexing scores #
    ################################################################
    
    start_2 = time.time()
    if config["alignment_method"] == "blastn":
        alignment_method = BlastNFilter(
            search_parameters = config["search_parameters"],
            hit_parameters = config["hit_parameters"],
            names_search_output = config["names_search_output"],
            dir_output="output_odt"
        )
    elif config["alignment_method"] == "bowtie":
        alignment_method = BowtieFilter(
            search_parameters = config["search_parameters"],
            hit_parameters = config["hit_parameters"],
            dir_output="output_odt"
        )
    else:
        raise ValueError("Unknown alignment method.")
    file_index = alignment_method._create_index(file_reference=file_reference, n_jobs=config["n_jobs"])
    
    # sample the oligos
    sample_train = round(config["splits_size"][0]*config["dataset_size"])
    sample_validation = round(config["splits_size"][1]*config["dataset_size"])
    sample_test= config["dataset_size"] - sample_train - sample_validation

    # train
    train_dataset = generate_off_targets(
        oligo_database= oligo_database_train, 
        alignment_method = alignment_method, 
        file_index = file_index, 
        config=config, 
        dataset_size=sample_train, 
        file_reference=file_reference
    )

    # validation
    validation_dataset = generate_off_targets(
        oligo_database = oligo_database_validation, 
        alignment_method = alignment_method, 
        file_index = file_index, 
        config=config,
        dataset_size=sample_validation, 
        file_reference=file_reference
    )

    # test
    test_dataset = generate_off_targets(
        oligo_database = oligo_database_test, 
        alignment_method = alignment_method, 
        file_index = file_index, 
        config=config,
        dataset_size=sample_test, 
        file_reference=file_reference
    )

    logging.info("Generated artificial off-targets.")

    ##################
    # write dataset #
    ##################

    file_train = os.path.join(config["dir_output"], f"{dataset_name}_train.csv")
    train_dataset.to_csv(file_train)
    file_validation = os.path.join(config["dir_output"], f"{dataset_name}_validation.csv")
    validation_dataset.to_csv(file_validation)
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
    
    # plot distributions of the n mismatches
    plt.figure(4)
    dataset["Number Mismatches"] = dataset["number_mismatches"]
    sns.boxplot(data=dataset, x="Source", y="Number Mismatches")
    plt.title("Number Mismatches distributions")
    plt.savefig(os.path.join(plots_dir,"Number_mismatches_distribution.pdf"))
    
    logging.info(f"Computational time: {time.time() - start} (off-targets generation: {time.time() - start_2})")
    del oligo_database_train
    del oligo_database_validation
    del oligo_database_test

    shutil.rmtree("output_odt") #remove oligo designer toolsuite output

if __name__ == "__main__":
    main()
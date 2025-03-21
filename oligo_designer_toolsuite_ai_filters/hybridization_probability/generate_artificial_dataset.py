import os
import shutil
import argparse
import yaml
import random
import copy
import time
from typing import Tuple, List
import logging
from datetime import datetime
import iteration_utilities

from oligo_designer_toolsuite.sequence_generator import OligoSequenceGenerator
from oligo_designer_toolsuite.database import OligoDatabase
from oligo_designer_toolsuite.oligo_property_filter import PropertyFilter, HardMaskedSequenceFilter, SoftMaskedSequenceFilter
from oligo_designer_toolsuite.pipelines import GenomicRegionGenerator
from Bio.Seq import MutableSeq, Seq
from Bio.SeqUtils import gc_fraction
from Bio.SeqUtils import MeltingTemp as mt
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import nupack
from math import log
import joblib


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

def sample_oligos(oligo_database: OligoDatabase, oligos_per_region: int):
    for region in oligo_database.database.keys():
        oligo_ids = list(oligo_database.database[region].keys())
        if len(oligo_ids) > oligos_per_region:
            filtered_oligo_ids = random.sample(population=oligo_ids, k=len(oligo_ids) - oligos_per_region) # sample the ids to filter
            for oligo_id in filtered_oligo_ids:
                oligo_database.database[region].pop(oligo_id, None)
    return oligo_database


def reverse_complement(strand: str) -> str:
    reverse_strand = []
    strand = list(strand)
    strand.reverse()
    for i in strand:
        if i == "-":
            continue
        reverse_strand.append(base_pair[i])
    return "".join(reverse_strand)

def mutate(nt: str) -> str:
    nts = ['A', 'C', 'T', 'G']
    nts.remove(nt)
    return random.sample(nts, 1)[0]

def compute_free_energy(seq_1: str, seq_2: str, temperature: float) -> float:
    strand_1 = nupack.Strand(seq_1, name="strand_1")
    strand_2 = nupack.Strand(seq_2, name="strand_2")
    set = nupack.ComplexSet(strands=[strand_1, strand_2], complexes=nupack.SetSpec(max_size=2))
    model = nupack.Model(material="dna", celsius=temperature)
    results = nupack.complex_analysis(complexes=set, model=model, compute=['pfunc'])
    return results[nupack.Complex(strands=[strand_1, strand_2])].free_energy

def generate_datasamples(oligo: str, target: str, gap_oligo: str, gap_off_target, temperatures: List[float], nr_mismatches: int) -> Tuple[str, str, int, float]:
    """Compute a free energy for each temperature in the list of temperatures for the given oligo and target sequences."""
    data_samples = []
    for temperature in temperatures:
        data_samples.append((gap_oligo, target, nr_mismatches, temperature, compute_free_energy(oligo, reverse_complement(target), temperature)))
    return data_samples

def sample_temperatures(n: int = 1) -> List[float]:
    return [37 for _ in range(n)]


def generate_off_targets(sequence: Seq, config) -> list[Tuple[str,str, int, float]]:
    # single point mutations
    data = []
    data.extend(generate_datasamples(sequence, sequence, sequence, sequence, sample_temperatures(), 0))
    for i in range(1, config["max_mutations"]+1): # nr of mutations
        for _ in range(1, config["n_mutations_per_type"]+1): # nr of mutations for mutation class
            # mutate i nt
            off_target = MutableSeq(sequence)
            unchanged_nts = list(range(len(sequence)))
            for _ in range(i):
                k = random.sample(unchanged_nts, 1)[0]
                new_nt = mutate(off_target[k])
                off_target.pop(k)
                off_target.insert(k, new_nt)
                unchanged_nts.remove(k)
            # evaluate all the free energies and append them (make a funciton for this)
            data.extend(generate_datasamples(sequence, off_target, sequence, off_target, sample_temperatures(), i))
    # bulges (insertions and deletions)
    for i in range(1, config["max_bulges_size"]+1): # nr of mutations
        for _ in range(1, config["n_mutations_per_type"]+1):
            # insert i nts
            off_target = MutableSeq(sequence)
            gap_sequence = MutableSeq(sequence)
            insertion_point = random.randrange(0, len(sequence))
            for _ in range(i):
                nt = random.choice(['A', 'T', 'C', 'G'])
                off_target.insert(insertion_point, nt)
                gap_sequence.insert(insertion_point, '-') # generate to have a correct alignement with of the sequnces (- with be encoded as a 0 vector)
            data.extend(generate_datasamples(sequence, off_target, gap_sequence, off_target, sample_temperatures(), i))
            # delete i nts
            target = MutableSeq(sequence)
            deletion_point = random.randint(0, len(sequence) - i) # leave the sapace to delete i nucleotides
            for _ in range(i):
                target.pop(deletion_point)
            gap_target = MutableSeq(target)
            for _ in range(i):
                gap_target.insert(deletion_point, '-') # generate to have a correct alignement with of the sequnces
            data.extend(generate_datasamples(sequence, target, gap_target, sequence, sample_temperatures(), i))
    return data

def generate_dataset(alignments: list):
    dataset = pd.DataFrame(index=list(range(len(alignments))), columns=["query_sequence", "query_length", "query_GC_content", "off_target_sequence", "off_target_length", "off_target_GC_content", "number_mismatches", "temperature", "free_energy"])
    for i, (oligo, off_target, nr_mismatches, temperature, free_energy) in enumerate(alignments):
        dataset.loc[i] = [
            oligo, #oligo sequence
            len(oligo),# oligo length
            gc_fraction(oligo),
            off_target,
            len(off_target), # off target length
            round(gc_fraction(off_target)), # off target gc content
            nr_mismatches,
            temperature,
            free_energy,
        ]
    return dataset

def generate_oligos(config: dict, dir_output: str, regions: list, oligo_fasta_file: str):
    """Generate the oligo sequences.
    """

    ##### creating the oligo database #####
    # one database for train, test and validation is created
    oligo_database = OligoDatabase(
        min_oligos_per_region=0,
        write_regions_with_insufficient_oligos=True,
        lru_db_max_in_memory=config["n_jobs"] * 2 + 1,
        database_name=f"oligo_database_{str(time.time())}",
        dir_output=dir_output,
    )
    oligo_database.load_database_from_fasta(
        files_fasta=oligo_fasta_file,
        sequence_type="target",
        region_ids=regions,
        database_overwrite = True,
    )

    # Property filtering
    masked_seqeunces = HardMaskedSequenceFilter()
    soft_masked_seqeunces = SoftMaskedSequenceFilter()
    property_filter = PropertyFilter(filters=[masked_seqeunces, soft_masked_seqeunces])
    oligo_database = property_filter.apply(oligo_database=oligo_database, n_jobs=config["n_jobs"], sequence_type="oligo")
    
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
    n_genes = sum(1 for _ in open(config["file_genes"]))
    size = config["oligos_per_region"]*config["n_mutations_per_type"]*(config["max_mutations"] + config["max_bulges_size"])*n_genes
    dataset_name = f"artificial_dataset_{config['oligo_length_min']}_{config['oligo_length_max']}_{size}"
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

    dir_output = "output_odt_real_" + str(time.time())

    genomic_region_genereator = GenomicRegionGenerator(dir_output = dir_output)
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
    oligo_sequences = OligoSequenceGenerator(dir_output=dir_output)
    oligo_fasta_file = oligo_sequences.create_sequences_sliding_window(
        files_fasta_in=files_fasta,
        length_interval_sequences=(config["oligo_length_min"], config["oligo_length_max"]),
        region_ids=genes,
        n_jobs=config["n_jobs"],
    )

    oligo_database_train = generate_oligos(config, dir_output, genes_train, oligo_fasta_file)
    oligo_database_validation = generate_oligos(config, dir_output, genes_validation, oligo_fasta_file)
    oligo_database_test = generate_oligos(config, dir_output, genes_test, oligo_fasta_file)

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


    ###########################################################
    # generate artificial off-targets and compute free energy #
    ###########################################################

    start_2 = time.time()
    # train
    train_alignments = joblib.Parallel(n_jobs=config["n_jobs"])(
        joblib.delayed(generate_off_targets)(
            oligo["oligo"].upper(), config
        )
        for database_region in oligo_database_train.database.values()
        for oligo in database_region.values()
    )
    train_alignments = [alignment for oligo_alignments in train_alignments for alignment in oligo_alignments] # flatten the returned structure
    # validation
    validation_alignments = joblib.Parallel(n_jobs=config["n_jobs"])(
        joblib.delayed(generate_off_targets)(
            oligo["oligo"].upper(), config
            )
        for database_region in oligo_database_validation.database.values()
        for oligo in database_region.values()
    )
    validation_alignments = [alignment for oligo_alignments in validation_alignments for alignment in oligo_alignments] # flatten the returned structure
    # test
    test_alignments = joblib.Parallel(n_jobs=config["n_jobs"])(
        joblib.delayed(generate_off_targets)(
            oligo["oligo"].upper(), config
            )
        for database_region in oligo_database_test.database.values()
        for oligo in database_region.values()
    )
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

    logging.info(f"Computational time: {time.time() - start} (off-targets generation: {time.time() - start_2})")
    shutil.rmtree(dir_output, ignore_errors=True) #remove oligo designer toolsuite output

if __name__ == "__main__":
    main()
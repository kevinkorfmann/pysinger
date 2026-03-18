![Logo](SINGER.png)
# SINGER
SINGER stands for **S**ampling and **IN**ference of **GE**nealogies with **R**ecombination, and it is a Bayesian method to do posterior sampling of Ancestral Recombination Graph under Sequentially Markovian Coalescent. SINGER works by iterative threading one haplotype to the partially-built ARG, until the ARG for all haplotypes have been built. After initialization, MCMC will be performed to update the ARG to explore the posterior distribution. For a full description and cite our method, you can check: [Deng, Yun, Rasmus Nielsen, and Yun S. Song. "Robust and accurate bayesian inference of genome-wide genealogies for large samples." bioRxiv (2024): 2024-03.](https://www.biorxiv.org/content/10.1101/2024.03.16.585351v1.supplementary-material)


Here we maintained the version which is under active development, but you can still direclty download the binary files for all past versions. 

[We are temporarily providing beta versions of it, the official versions will be released when the preprint has been accepted for publication. You are welcome to use it, and submit bug reports at GitHub Issues. ]

## Requirements

If you want to compile the source files, then C++17 and cmake are required. Otherwise you can also used the pre-compiled binary files on various platforms. 

The analysis on the inferred ARGs will be performed using [tskit](https://tskit.dev/tskit/docs/stable/introduction.html), and you can find the installation instructions using the link. **Note that now we support the compability with tskit version 1.0 onward, which requires python 3.10 and above (from version 0.1.9).** 

## Installations

The easiser way is to directory go to the folder `releases/` and download one of the versions which work for your working platform (Linux/MacOS_Intel/MacOS_M1). After downloading, you can decompress it using:

```
tar -xvzf file_name
```

## Input and output

SINGER takes **.vcf** file and outputs a **.trees** file in tskit format. The mutations are already mapped to the branches, but non-polymorphic, multi-allelic sites and structral variants are excluded from inference. The branch length should be interpreted with units of generations, for example, for homo sapiens, you would need multiply that by 28 to convert to units of years. There will also be a **.log** file for you to check the argument you ran, and the summary statistic in MCMC iterations. 

## Basic usage

To sample ARGs with SINGER, you can run command line like shown below. 

**IMPORTANT!**:if you wish to get ARG for: **(1) a long chromosome or (2) a series of regions**, we have provided more support to help you (see the [next section: Tools](#Tools)). If you think there are other specific job pipeline which many people might want to use, please contact us and we might add it! 

```
path_to_singer/singer_master -m 1.25e-8
-vcf prefix_of_vcf_file -output prefix_of_output_file
-start 0 -end 1e6
```

This command is to get the ARG samples for a specific region in the vcf file. We specify the details of the arguments here (or you can simply type ```path_to_singer/singer_master``` to display similar information):

The required flags include (either `-m` or `-mut_map` has to be provided):

|flag|required?|details|  
|-------------------|-----|---|  
|**-m**|conditionally required|per base pair per generation mutation rate|
|**-mut_map**|conditionally required|name of the file describing the mutation rate landscape|
|**-vcf**|required|prefix of the input .vcf file name|
|**-output**|required|prefix of the output .trees file name| 
|**-start**|required|start position of the region| 
|**-end**|required|end position of the region| 

The optional flags include:

|flag|required?|details|  
|-------------------|-----|---|  
|**-Ne**|optional|the diploid effective population size, which means the haploid effective population size will be **2*Ne**|
|**-ratio**|optional|the ratio between recombination and mutation rate, default at 1|
|**-recomb_map**|optional|name of the file describing the recombination rate landscape|
|**-n**|optional|the number of posterior samples, default at 100|
|**-thin**|optional|the number of MCMC iterations between adjacent samples, default at 20|
|**-polar**|optional|the probability of correct polarization, default at 0.5 for unpolarized data, please use 0.99 for polarized data|

The output files will be:

```
prefix_of_output_files_nodes_{i}.txt, prefix_of_output_files_branches_{i}.txt, prefix_of_output_files_muts_{i}.txt, prefix_of_output_files_recombs_{i}.txt
```

with `i` from `0` to `num_samples - 1`. We recommend converting these files to tree sequence format in tskit, with this function:

```
path_to_singer/convert_to_tskit -input prefix_of_arg_files -output prefix_of_tskit_files
-start start_index -end end_index -step step_size
```

This tool will convert ARG sample with index from `start_index` to `end_index`, with interval size `step_size`. 


## Tools

### Examining the convergence of the MCMC in SINGER

SINGER is an MCMC-based sampling algorithm. To examine the convergence of it we normally examine the traces of summary statistics, and we have found that 2 summary statistics are quite good at indicating the convergence of the SINGER MCMC: fit to the diversity landscape and number of non-uniquely-mapped sites, as used in the manuscript. We have provided a python script to calculate the traces for these 2 quantities:

```
python compute_traces.py
-prefix prefix_of_tree_sequence_file -m mutation_rate
-start_index index_start_sample -end_index index_terminal_sample
-output_filename output_trace_filename
```
The script will calculate the aforementioned two statistics for all samples with index between the `start_index` and the `end_index`, and they will be output as the the 2 columns in the output file. The fit to the diversity landscape and the number of non-uniquely-mapped sites typically drops with more iterations of the MCMC, and when they reach to a stable stage that is usually a good sign for convergence. 

### Computing the pairwise coalescence times with respect to a particular haplotype

In the manuscript we used the coalescence ratio to find introgression signals, which is based on the distribution of pairwise coalescence times between one haplytype and others, in every 10kb genome windows. Here we provide a python script tools, to compute the pairwise coalescence times between one given haplotype (indiciated by leaf node index) and others, in a windowed fashion. 

```
python compute_pairwise_coalescence_times.py
--trees_file tree_sequence_filename --leaf_index index_of_leaf_node
--interval_size size_of_genome_window --output_filename output_file_name
```
Each row in the output file stands for all the pairwise coalescence times between the input leaf node index and all others. The rows are in the order of the genome windows. 

### Running SINGER for a long chromosome

Often people would like to run the ARG inference method for the entire chromosome (or even the entire genome). We recommend running ```singer_master``` for continous segments (such as 5Mb) and then use the following tool to merge them together. 

```
path_to_singer/singer_master -m 1.25e-8 -vcf prefix_of_vcf_file -output prefix_of_output_file_0 -start 0 -end 5e6
path_to_singer/singer_master -m 1.25e-8 -vcf prefix_of_vcf_file -output prefix_of_output_file_1 -start 5e6 -end 10e6
path_to_singer/singer_master -m 1.25e-8 -vcf prefix_of_vcf_file -output prefix_of_output_file_2 -start 10e6 -end 15e6
......
```

```
python merge_ARG.py --file_table sub_file_table_file --output merged_ARG_filename
```

The ```sub_file_table_file``` specifies how the inferred ARG should be pieced together with an example like this:

```
prefix_of_output_file_i_nodes_0.txt prefix_of_output_file_i_branches_0.txt prefix_of_output_file_i_muts_0.txt 0 
prefix_of_output_file_i_nodes_1.txt prefix_of_output_file_i_branches_1.txt prefix_of_output_file_i_muts_1.txt 5000000
prefix_of_output_file_i_nodes_2.txt prefix_of_output_file_i_branches_2.txt prefix_of_output_file_i_muts_2.txt 10000000
......
```

where ```i``` is the index of the ARG sample, and the index after that is the index of the genomic windows on which parallization is performed. Note that the genomic windows do not have to be all adjacent. For example, when there is a centromeric region in the middle of the chromosome, it can be skipped by only selecting windows in the left and right arms. This will be reflected in the last column of the ```sub_file_table_file```, which indicates the starting position of each genomic window. 

By doing the merging operation, you will get the ARG samples with the length of the entire chromosome rather than for each individual genomic window, which might be convenient in certain scenarios. 



## pysinger — pure-Python implementation

The `pysinger/` directory contains a readable pure-Python replica of the SINGER algorithm. It mirrors the C++ BSP + TSP threading and MCMC sampling pipeline, and can export inferred ARGs to `tskit.TreeSequence` for downstream analysis.

### Quick start

```bash
cd pysinger
uv sync --extra demo   # or: pip install -e ".[demo]"
```

```python
from pysinger import Sampler
from pysinger.io.tskit_writer import arg_to_tskit

sampler = Sampler(Ne=10000, recomb_rate=1e-8, mut_rate=1e-8)
sampler.set_seed(42)
sampler.load_vcf("data.vcf", start=0, end=1_000_000)
sampler.iterative_start()
sampler.internal_sample(num_iters=1000, spacing=1)
ts = arg_to_tskit(sampler.arg, Ne=10000)
```

See `pysinger/demo.ipynb` for a full walkthrough using a stdpopsim zigzag simulation with convergence diagnostics and validation plots.

### Package structure

```
pysinger/pysinger/
├── sampler.py           # Top-level MCMC sampler
├── data/                # ARG, Node, Branch, Tree, Recombination
├── hmm/                 # BSP (branch HMM), TSP (time HMM), emissions
├── mcmc/                # Threader (BSP + TSP threading)
├── io/                  # VCF reader, tskit writer
├── rates/               # Piecewise recombination/mutation rate maps
└── reconstruction/      # Fitch parsimony for ancestral states
```

## Suggestions from developer

1. As a Bayesian sampling method, SINGER works best when you sample some ARGs from posterior, **only using one single sample is NOT ideal**. To this point, we highly encourage specifying **-n, -thin** flags. You can find how we run SINGER on real datasets on:
2. It is of importance to carefully choose the parameters, such as -Ne, -m, and -ratio. We recommend first choosing the mutation rate m, and then based on average pairwise diversity \($\pi=4\cdot N_e \cdot m\$), you can decide the Ne parameter. If you are not super sure about the recombination rate, you can use the default ratio of 1.
3. Unfortunately for now we only support phased, high-quality genomes, and polymorphic sites with missingness will be excluded. We are working on incorporating missingness and unphased data in the near future. ARGweaver has better support in these regards.
4. By far **the most frequent bug reported** comes from using full name under the ```-vcf``` flag, note that it only accepts the prefix of the vcf file without ```.vcf```. For example, ```-vcf human_chr1.vcf``` is illegal because it will look for a file called ```human_chr1.vcf.vcf```.
5. **The second most frequent bug reported** is caused by running SINGER on essentially a region with no or very low sequencing data, such as centromeric regions. SINGER cannot infer the ARG when there is no data present, and will likely bug out due to underflow issues.

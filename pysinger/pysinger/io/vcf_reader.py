"""
VCF reader — load phased or haploid VCF into Node objects.

Mirrors Sampler::naive_read_vcf / naive_read_vcf_haploid in Sampler.cpp.

Returns a list of Node objects with mutation sites set, one node per
haplotype (2 per individual for diploid, 1 per individual for haploid).
"""
from __future__ import annotations

from typing import List, Tuple

from ..data.node import Node


def read_vcf_phased(
    vcf_file: str,
    start_pos: float = 0.0,
    end_pos: float = float("inf"),
) -> Tuple[List[Node], float]:
    """Read a phased VCF and return (nodes, sequence_length).

    One Node per haplotype (2 × n_individuals).  Mutations are stored as
    positions relative to *start_pos*.

    Mirrors Sampler::naive_read_vcf.

    Parameters
    ----------
    vcf_file:
        Path to the .vcf file.
    start_pos, end_pos:
        Genomic region to load (half-open [start_pos, end_pos)).

    Returns
    -------
    nodes : List[Node]
        Haplotype nodes (index 0 .. 2n-1).
    sequence_length : float
        end_pos - start_pos.
    """
    nodes: List[Node] = []
    prev_pos = -1
    contig_length: float = -1.0

    with open(vcf_file) as fh:
        lines = fh.readlines()

    for line in lines:
        if line.startswith("##contig"):
            # parse ##contig=<ID=...,length=NNN>
            import re
            m = re.search(r"length=(\d+)", line)
            if m:
                contig_length = float(m.group(1))
            continue
        if line.startswith("#CHROM"):
            fields = line.split()
            n_ind = len(fields) - 9
            nodes = []
            for i in range(2 * n_ind):
                n = Node(time=0.0, index=i)
                nodes.append(n)
            continue
        if line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 9:
            continue
        chrom, pos_s, id_, ref, alt = parts[0], parts[1], parts[2], parts[3], parts[4]
        pos = float(pos_s)

        if pos < start_pos:
            continue
        if pos > end_pos:
            break
        if pos == prev_pos:
            continue  # skip duplicate positions (multi-allelic)
        if len(ref) > 1 or len(alt) > 1:
            prev_pos = pos
            continue  # skip indels / structural variants

        genotypes_raw = parts[9:]
        if not nodes:
            n_ind = len(genotypes_raw)
            nodes = []
            for i in range(2 * n_ind):
                n = Node(time=0.0, index=i)
                nodes.append(n)

        gt_vals = []
        for g in genotypes_raw:
            a0 = 1 if g[0] == "1" else 0
            a1 = 1 if (len(g) > 2 and g[2] == "1") else 0
            gt_vals.append(a0)
            gt_vals.append(a1)

        gt_sum = sum(gt_vals)
        if 1 <= gt_sum < len(gt_vals):
            rel_pos = pos - start_pos
            for i, v in enumerate(gt_vals):
                if v == 1:
                    nodes[i].add_mutation(rel_pos)

        prev_pos = pos

    if end_pos < float("inf"):
        sequence_length = end_pos - start_pos
    elif contig_length >= 0:
        sequence_length = contig_length - start_pos
    else:
        sequence_length = prev_pos - start_pos
    return nodes, sequence_length


def read_vcf_haploid(
    vcf_file: str,
    start_pos: float = 0.0,
    end_pos: float = float("inf"),
) -> Tuple[List[Node], float]:
    """Read a haploid VCF and return (nodes, sequence_length).

    One Node per individual (column after FORMAT).

    Mirrors Sampler::naive_read_vcf_haploid.
    """
    nodes: List[Node] = []
    prev_pos = -1
    contig_length: float = -1.0

    with open(vcf_file) as fh:
        lines = fh.readlines()

    for line in lines:
        if line.startswith("##contig"):
            import re
            m = re.search(r"length=(\d+)", line)
            if m:
                contig_length = float(m.group(1))
            continue
        if line.startswith("#CHROM"):
            fields = line.split()
            n_ind = len(fields) - 9
            nodes = [Node(time=0.0, index=i) for i in range(n_ind)]
            continue
        if line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 9:
            continue
        pos = float(parts[1])
        ref = parts[3]
        alt = parts[4]

        if pos < start_pos:
            continue
        if pos > end_pos:
            break
        if pos == prev_pos:
            continue
        if len(ref) > 1 or len(alt) > 1:
            prev_pos = pos
            continue

        genotypes_raw = parts[9:]
        if not nodes:
            nodes = [Node(time=0.0, index=i) for i in range(len(genotypes_raw))]

        gt_vals = [1 if g[0] == "1" else 0 for g in genotypes_raw]
        gt_sum = sum(gt_vals)
        if 1 <= gt_sum < len(gt_vals):
            rel_pos = pos - start_pos
            for i, v in enumerate(gt_vals):
                if v == 1:
                    nodes[i].add_mutation(rel_pos)

        prev_pos = pos

    if end_pos < float("inf"):
        sequence_length = end_pos - start_pos
    elif contig_length >= 0:
        sequence_length = contig_length - start_pos
    else:
        sequence_length = prev_pos - start_pos
    return nodes, sequence_length

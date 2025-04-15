"""
Module for spatial LDSC genetic correlation analysis.

This module extends the Spatial LDSC functionality by estimating
genetic correlations between traits across spatial locations.
"""

import gc
import logging
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from scipy.stats import norm
from tqdm import tqdm

from gsMap.config import SpatialLDSCRgConfig
from gsMap.utils import jackknife as jk
from gsMap.utils.ldscore_regression import (Gencov, Hsq, p_z_norm)
from gsMap.utils.regression_read import (_read_ref_ld_v2, _read_sumstats,
                                         _read_w_ld, _read_M_v2)
from functools import partial
from tqdm.contrib.concurrent import thread_map, process_map

logger = logging.getLogger(__name__)


def filter_sumstats_by_chisq(sumstats, chisq_max):
    """Filter summary statistics based on chi-squared threshold."""
    before_len = len(sumstats)
    if chisq_max is None:
        chisq_max = max(0.001 * sumstats.N.max(), 80)
        logger.info(f"No chi^2 threshold provided, using {chisq_max} as default")

    sumstats["chisq"] = sumstats.Z ** 2
    sumstats = sumstats[sumstats.chisq < chisq_max]
    after_len = len(sumstats)

    if after_len < before_len:
        logger.info(
            f"Removed {before_len - after_len} SNPs with chi^2 > {chisq_max} ({after_len} SNPs remain)"
        )
    else:
        logger.info(f"No SNPs removed with chi^2 > {chisq_max} ({after_len} SNPs remain)")
    return sumstats


def _preprocess_sumstats(sumstat_file_path, baseline_and_w_ld_common_snp, chisq_max=None):
    """Preprocess a single summary statistics file."""
    sumstats = _read_sumstats(fh=sumstat_file_path, alleles=True, dropna=False)
    sumstats.set_index("SNP", inplace=True)

    # Filter by chi-square
    sumstats = filter_sumstats_by_chisq(sumstats, chisq_max)

    # Filter by common SNPs
    common_snp = baseline_and_w_ld_common_snp.intersection(sumstats.index)
    if len(common_snp) < 200000:
        logger.warning(f"WARNING: number of SNPs less than 200k in {sumstat_file_path}.")

    sumstats = sumstats.loc[common_snp]

    # Add index position
    sumstats["common_index_pos"] = pd.Index(baseline_and_w_ld_common_snp).get_indexer(sumstats.index)

    return sumstats


def merge_summary_statistics(sumstats1, sumstats2):
    """
    Merge two summary statistics files, aligning alleles.

    Parameters
    ----------
    sumstats1 : pd.DataFrame
        First summary statistics
    sumstats2 : pd.DataFrame
        Second summary statistics

    Returns
    -------
    pd.DataFrame
        Merged summary statistics
    """
    # Rename columns to avoid conflicts
    sumstats1 = sumstats1.rename(columns={"N": "N1", "Z": "Z1", "chisq": "chisq1"})
    sumstats2 = sumstats2.rename(columns={"N": "N2", "Z": "Z2", "chisq": "chisq2"})

    # Make sure we have the same index
    common_snps = sumstats1.index.intersection(sumstats2.index)
    merged = pd.merge(
        sumstats1.loc[common_snps],
        sumstats2.loc[common_snps],
        left_index=True,
        right_index=True,
        suffixes=("_1", "_2")
    )

    # Check for allele flips and fix Z-scores for flipped SNPs
    if 'A1_1' in merged.columns and 'A1_2' in merged.columns:
        is_flipped = merged.A1_1 != merged.A1_2
        merged.loc[is_flipped, "Z2"] = -merged.loc[is_flipped, "Z2"]
        logger.info(f"Flipped {is_flipped.sum()} SNPs to align effect alleles")

    logger.info(f"Merged summary statistics: {len(merged)} SNPs")
    return merged


def estimate_global_genetic_parameters(merged_sumstats, M_estimate, n_blocks=200, twostep=30):
    """
    Estimate global genetic parameters for a pair of traits.

    Parameters
    ----------
    merged_sumstats : pd.DataFrame
        Merged summary statistics with Z-scores for both traits
    M_estimate : float
        Estimate of the number of independent SNPs
    n_blocks : int
        Number of jackknife blocks

    Returns
    -------
    dict
        Dictionary with global genetic parameters
    """
    n_snp = len(merged_sumstats)
    M = np.array([[float(M_estimate)]])  # Use as scalar for all annotations

    # Convert to column vectors
    s = lambda x: np.array(x).reshape((n_snp, 1))

    if twostep is not None:
        logger.info(f"Using two-step estimator with cutoff at {twostep}.")

    # Estimate h1 using LDSC
    hsq1 = Hsq(
        y=np.square(s(merged_sumstats.Z1)),
        x=s(merged_sumstats.LD_score),
        w=s(merged_sumstats.LD_weights),
        N=s(merged_sumstats.N1),
        M=M,
        n_blocks=n_blocks,
        twostep=twostep,
    )

    # Estimate h2 using LDSC
    hsq2 = Hsq(
        y=np.square(s(merged_sumstats.Z2)),
        x=s(merged_sumstats.LD_score),
        w=s(merged_sumstats.LD_weights),
        N=s(merged_sumstats.N2),
        M=M,
        n_blocks=n_blocks,
        twostep=twostep,
    )

    # Estimate genetic covariance using LDSC
    gencov = Gencov(
        z1=s(merged_sumstats.Z1),
        z2=s(merged_sumstats.Z2),
        x=s(merged_sumstats.LD_score),
        w=s(merged_sumstats.LD_weights),
        N1=s(merged_sumstats.N1),
        N2=s(merged_sumstats.N2),
        M=M,
        hsq1=hsq1.tot,
        hsq2=hsq2.tot,
        intercept_hsq1=hsq1.intercept,
        intercept_hsq2=hsq2.intercept,
        n_blocks=n_blocks,
        twostep=twostep,
    )

    # Calculate genetic correlation
    rg_ratio = gencov.tot / np.sqrt(hsq1.tot * hsq2.tot)

    # Use ratio jackknife to get rg standard error
    rg_jknife = jk.RatioJackknife(
        np.array(rg_ratio).reshape((1, 1)),
        gencov.tot_delete_values,
        np.sqrt(np.multiply(hsq1.tot_delete_values, hsq2.tot_delete_values))
    )

    rg_se = float(rg_jknife.jknife_se)
    p, z = p_z_norm(rg_ratio, rg_se)

    return {
        "hsq1_tot": float(hsq1.tot),
        "hsq1_intercept": float(hsq1.intercept),
        "hsq1_se": float(hsq1.tot_se),
        "hsq2_tot": float(hsq2.tot),
        "hsq2_intercept": float(hsq2.intercept),
        "hsq2_se": float(hsq2.tot_se),
        "gcov_tot": float(gencov.tot),
        "gcov_intercept": float(gencov.intercept),
        "gcov_se": float(gencov.tot_se),
        "rg": float(rg_ratio),
        "rg_se": rg_se,
        "rg_p": float(p),
        "rg_z": float(z),
        "mean_chisq1": float(hsq1.mean_chisq),
        "mean_chisq2": float(hsq2.mean_chisq),
        "lambda_gc1": float(hsq1.lambda_gc),
        "lambda_gc2": float(hsq2.lambda_gc),
        "mean_z1z2": float(gencov.mean_z1z2)
    }


def compute_local_genetic_correlation(spot_id, spatial_annotation, ref_ld_baseline, merged_sumstats, w_ld,
                                      global_parameters, n_blocks, M_spatial, M_baseline, old_weights=True,
                                      use_global_parameters=False,
                                      twostep=None, ):
    """
    Compute local genetic correlation for a single spot.

    Parameters
    ----------
    spot_id : int
        Spot index
    spatial_annotation : np.ndarray
        Spatial annotation values
    ref_ld_baseline : np.ndarray
        Sum of baseline LD scores
    merged_sumstats : pd.DataFrame
        Merged summary statistics
    w_ld : pd.DataFrame
        LD weights
    global_parameters : dict
        Global genetic parameters
    n_blocks : int
        Number of jackknife blocks

    Returns
    -------
    dict
        Dictionary with local genetic correlation results
    """
    # Get spatial annotation for this spot
    spot_spatial_annotation = spatial_annotation[:, spot_id]

    # Convert to column vectors
    s = lambda x: np.array(x).reshape((-1, 1))
    n_snp = len(merged_sumstats)
    x = np.concatenate((s(spot_spatial_annotation), ref_ld_baseline), axis=1)
    M_spot= np.concatenate((M_spatial[:,spot_id:spot_id+1], M_baseline), axis=1)
    # %% TODO In the origianl rg impolementation, constrain the intercept to be 1 of hsq and intercept_genecov to be 0
    try:
        if not use_global_parameters:
        # First compute local h2 for trait 1
            hsq1 = Hsq(
                y=np.square(s(merged_sumstats.Z1)),
                x=x,
                w=s(w_ld.LD_weights),
                N=s(merged_sumstats.N1),
                M=M_spot,
                n_blocks=n_blocks,
                intercept=None,
                twostep=twostep,
                old_weights=old_weights
            )

            # Then compute local h2 for trait 2
            hsq2 = Hsq(
                y=np.square(s(merged_sumstats.Z2)),
                x=x,
                w=s(w_ld.LD_weights),
                N=s(merged_sumstats.N2),
                M=M_spot,
                n_blocks=n_blocks,
                intercept=None,
                twostep=twostep,
                old_weights=old_weights
            )

            # Then compute local genetic covariance
            gencov = Gencov(
                z1=s(merged_sumstats.Z1),
                z2=s(merged_sumstats.Z2),
                x=x,
                w=s(w_ld.LD_weights),
                N1=s(merged_sumstats.N1),
                N2=s(merged_sumstats.N2),
                M=M_spot,
                hsq1=hsq1.tot,
                hsq2=hsq2.tot,
                intercept_hsq1=hsq1.intercept,
                intercept_hsq2=hsq2.intercept,
                n_blocks=n_blocks,
                intercept_gencov=None,
                twostep=twostep,
            )
        else:
            gencov = Gencov(
                z1=s(merged_sumstats.Z1),
                z2=s(merged_sumstats.Z2),
                x=x,
                w=s(w_ld.LD_weights),
                N1=s(merged_sumstats.N1),
                N2=s(merged_sumstats.N2),
                M=M_spot,
                hsq1=global_parameters["hsq1_tot"],
                hsq2=global_parameters["hsq2_tot"],
                intercept_hsq1=global_parameters["hsq1_intercept"],
                intercept_hsq2=global_parameters["hsq2_intercept"],
                n_blocks=n_blocks,
                intercept_gencov=None,
                twostep=twostep,
            )

        # # Calculate local genetic correlation
        # if hsq1.tot <= 0 or hsq2.tot <= 0:
        #     local_rg = np.nan
        #     local_rg_se = np.nan
        #     local_rg_z = np.nan
        #     local_rg_p = np.nan
        # else:
        #     local_rg = gencov.tot / np.sqrt(hsq1.tot * hsq2.tot)
        #
        #     # Use ratio jackknife to get rg standard error
        #     local_rg_jknife = jk.RatioJackknife(
        #         np.array(local_rg).reshape((1, 1)),
        #         gencov.tot_delete_values,
        #         np.sqrt(np.multiply(hsq1.tot_delete_values, hsq2.tot_delete_values))
        #     )
        #
        #     local_rg_se = float(local_rg_jknife.jknife_se)
        #     local_rg_p, local_rg_z = p_z_norm(local_rg, local_rg_se)

        return {
            "h1_beta": float(hsq1.tot),
            "h1_se": float(hsq1.tot_se),
            "h2_beta": float(hsq2.tot),
            "h2_se": float(hsq2.tot_se),
            "gcov_beta": float(gencov.tot),
            "gcov_se": float(gencov.tot_se),
            "gcov_p": float(gencov.p),
            # "rg_beta": float(local_rg) if not np.isnan(local_rg) else np.nan,
            # "rg_se": float(local_rg_se) if not np.isnan(local_rg_se) else np.nan,
            # "rg_z": float(local_rg_z) if not np.isnan(local_rg_z) else np.nan,
            # "rg_p": float(local_rg_p) if not np.isnan(local_rg_p) else np.nan
        }

    except Exception as e:
        logger.warning(f"Error estimating local genetic correlation for spot {spot_id}: {e}")
        return {
            "h1_beta": np.nan,
            "h1_se": np.nan,
            "h2_beta": np.nan,
            "h2_se": np.nan,
            "gcov_beta": np.nan,
            "gcov_se": np.nan,
            "rg_beta": np.nan,
            "rg_se": np.nan,
            "rg_z": np.nan,
            "rg_p": np.nan
        }


# ---- Functions extracted and adapted from spatial_ldsc_multiple_sumstats.py ----

def determine_total_chunks(config):
    """Determine total number of chunks based on the ldscore save format."""
    if config.ldscore_save_format == "quick_mode":
        # For quick_mode, we'd need to instantiate the S_LDSC_Boost class
        # This is a simplification as we don't have access to the full implementation
        # of the quick_mode in this context
        logger.info("Quick mode is not fully supported in genetic correlation analysis")
        all_file = os.listdir(config.ldscore_save_dir)
        total_chunk_number_found = sum('chunk' in name for name in all_file)
    elif config.ldscore_save_format == "zarr":
        zarr_path = Path(config.ldscore_save_dir) / f"{config.sample_name}.ldscore.zarr"
        if not zarr_path.exists():
            raise FileNotFoundError(f"{zarr_path} not found")
        zarr_file = zarr.open(str(zarr_path))
        if hasattr(zarr_file, 'blocks'):
            total_chunk_number_found = zarr_file.blocks.shape[1]
        else:
            raise ValueError("Invalid zarr file structure")
    else:  # default to "feather"
        all_file = os.listdir(config.ldscore_save_dir)
        total_chunk_number_found = sum('chunk' in name for name in all_file)

    logger.info(f"Found {total_chunk_number_found} chunked files in {config.ldscore_save_dir}")
    return total_chunk_number_found


def determine_chunk_range(config, total_chunk_number_found):
    """Determine the range of chunks to process."""
    if config.all_chunk is None:
        if config.chunk_range is not None:
            if not (1 <= config.chunk_range[0] <= total_chunk_number_found) or not (
                    1 <= config.chunk_range[1] <= total_chunk_number_found
            ):
                raise ValueError("Chunk range out of bound. It should be in [1, all_chunk]")
            start_chunk, end_chunk = config.chunk_range
            logger.info(
                f"Chunk range provided, using chunked files from {start_chunk} to {end_chunk}"
            )
        else:
            start_chunk, end_chunk = 1, total_chunk_number_found
    else:
        all_chunk = config.all_chunk
        logger.info(f"Using {all_chunk} chunked files by provided argument")
        start_chunk, end_chunk = 1, all_chunk
    return start_chunk, end_chunk


def load_ldscore_chunk_from_feather(chunk_index, common_snp_among_all_sumstats_pos, config):
    """Load LD score chunk from feather format."""
    sample_name = config.sample_name
    ld_file_spatial = f"{config.ldscore_save_dir}/{sample_name}_chunk{chunk_index}/{sample_name}."
    ref_ld_spatial = _read_ref_ld_v2(ld_file_spatial)
    ref_ld_spatial = ref_ld_spatial.iloc[common_snp_among_all_sumstats_pos]
    ref_ld_spatial = ref_ld_spatial.astype(np.float32, copy=False)
    spatial_annotation_cnames = ref_ld_spatial.columns
    M_ref_ld_spatial = _read_M_v2(ld_file_spatial, len(spatial_annotation_cnames), False)
    return ref_ld_spatial.values, spatial_annotation_cnames, M_ref_ld_spatial


def load_ldscore_chunk_from_zarr(chunk_index, common_snp_among_all_sumstats_pos, zarr_file, spots_name):
    """Load LD score chunk from zarr format."""
    ref_ld_spatial = zarr_file.blocks[:, chunk_index - 1][common_snp_among_all_sumstats_pos]
    start_spot = (chunk_index - 1) * zarr_file.chunks[1]
    ref_ld_spatial = ref_ld_spatial.astype(np.float32, copy=False)
    spatial_annotation_cnames = spots_name[start_spot: start_spot + zarr_file.chunks[1]]
    return ref_ld_spatial, spatial_annotation_cnames


def load_ldscore_chunk(
        chunk_index,
        common_snp_among_all_sumstats_pos,
        config,
        zarr_file=None,
        spots_name=None
):
    assert config.ldscore_save_format == "feather", 'Only support feather in rg analysis'
    """Load LD score chunk based on save format."""
    if config.ldscore_save_format == "feather":
        return load_ldscore_chunk_from_feather(
            chunk_index, common_snp_among_all_sumstats_pos, config
        )
    elif config.ldscore_save_format == "zarr":
        if zarr_file is None or spots_name is None:
            zarr_path = Path(config.ldscore_save_dir) / f"{config.sample_name}.ldscore.zarr"
            if not zarr_path.exists():
                raise FileNotFoundError(f"{zarr_path} not found")
            zarr_file = zarr.open(str(zarr_path))
            spots_name = zarr_file.attrs["spot_names"]
        return load_ldscore_chunk_from_zarr(
            chunk_index, common_snp_among_all_sumstats_pos, zarr_file, spots_name
        )
    elif config.ldscore_save_format == "quick_mode":
        logger.warning("Quick mode is not fully supported in genetic correlation analysis")
        return load_ldscore_chunk_from_feather(
            chunk_index, common_snp_among_all_sumstats_pos, config
        )
    else:
        raise ValueError(f"Invalid ldscore_save_format: {config.ldscore_save_format}")


def run_spatial_ldsc_rg(config: SpatialLDSCRgConfig):
    """Run spatial LDSC genetic correlation analysis."""
    logger.info(f"------Running Spatial LDSC Genetic Correlation for {config.sample_name}...")
    n_blocks = config.n_blocks
    sample_name = config.sample_name

    # Load regression weights
    w_ld = _read_w_ld(config.w_file)
    w_ld.set_index("SNP", inplace=True)

    # Load baseline annotations
    ld_file_baseline = f"{config.ldscore_save_dir}/baseline/baseline."
    ref_ld_baseline = _read_ref_ld_v2(ld_file_baseline)
    baseline_and_w_ld_common_snp = ref_ld_baseline.index.intersection(w_ld.index)

    # Process GWAS summary statistics
    logger.info(f"Processing summary statistics for {config.trait1_name}...")
    sumstats1 = _preprocess_sumstats(
        config.trait1_sumstats, baseline_and_w_ld_common_snp, chisq_max=config.chisq_max
    )

    logger.info(f"Processing summary statistics for {config.trait2_name}...")
    sumstats2 = _preprocess_sumstats(
        config.trait2_sumstats, baseline_and_w_ld_common_snp, chisq_max=config.chisq_max
    )

    # Merge summary statistics
    logger.info("Merging summary statistics...")
    merged_sumstats = merge_summary_statistics(sumstats1, sumstats2)
    common_snp_among_all_sumstats = merged_sumstats.index
    common_snp_among_all_sumstats_pos = ref_ld_baseline.index.get_indexer(common_snp_among_all_sumstats)

    if len(common_snp_among_all_sumstats) < 200000:
        logger.warning(
            f"!!!!! WARNING: number of SNPs less than 200k for {sample_name}. This may lead to unreliable results."
        )

    ref_ld_baseline = ref_ld_baseline.loc[common_snp_among_all_sumstats]
    w_ld = w_ld.loc[common_snp_among_all_sumstats]

    # Load additional baseline annotations if needed
    if config.use_additional_baseline_annotation:
        logger.info("Using additional baseline annotations")
        ld_file_baseline_additional = f"{config.ldscore_save_dir}/additional_baseline/baseline."
        ref_ld_baseline_additional = _read_ref_ld_v2(ld_file_baseline_additional)
        ref_ld_baseline_additional = ref_ld_baseline_additional.loc[common_snp_among_all_sumstats]
        ref_ld_baseline = pd.concat([ref_ld_baseline, ref_ld_baseline_additional], axis=1)
        del ref_ld_baseline_additional

    # Estimate global genetic parameters
    logger.info("Estimating global genetic parameters...")
    # Determine global M_w_ld
    M_w_ld = _read_M_v2(ld_file_baseline, 2, False)
    M_base = M_w_ld[0, 1]
    merged_sumstats['LD_weights'] = w_ld.LD_weights
    merged_sumstats['LD_score'] = ref_ld_baseline['base']

    global_params = estimate_global_genetic_parameters(
        merged_sumstats, M_base, n_blocks
    )

    logger.info(
        f"Trait 1 ('{config.trait1_name}') h²: {global_params['hsq1_tot']:.4f} (SE: {global_params['hsq1_se']:.4f}, intercept: {global_params['hsq1_intercept']:.4f})")
    logger.info(
        f"Trait 2 ('{config.trait2_name}') h²: {global_params['hsq2_tot']:.4f} (SE: {global_params['hsq2_se']:.4f}, intercept: {global_params['hsq2_intercept']:.4f})")
    logger.info(
        f"Genetic correlation: {global_params['rg']:.4f} (SE: {global_params['rg_se']:.4f}, p: {global_params['rg_p']:.4e})")

    if global_params['rg_p'] > 0.05:
        logger.warning("WARNING: The global genetic correlation is not statistically significant (p > 0.05)")

    if abs(global_params['rg']) < 0.01:
        logger.warning("WARNING: The global genetic correlation is very close to zero.")

    # Save global genetic parameters
    output_dir = Path(config.rg_save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    global_params_file = output_dir / f"global_genetic_params_{config.trait1_name}_{config.trait2_name}.csv"
    pd.DataFrame([global_params]).to_csv(global_params_file, index=False)
    logger.info(f"Global genetic parameters saved to {global_params_file}")

    # Initialize zarr handling if needed
    zarr_file, spots_name = None, None
    if config.ldscore_save_format == "zarr":
        zarr_path = Path(config.ldscore_save_dir) / f"{config.sample_name}.ldscore.zarr"
        if not zarr_path.exists():
            raise FileNotFoundError(f"{zarr_path} not found, which is required for zarr format")
        zarr_file = zarr.open(str(zarr_path))
        spots_name = zarr_file.attrs["spot_names"]

    # Use the refactored functions to determine chunks
    total_chunk_number_found = determine_total_chunks(config)
    start_chunk, end_chunk = determine_chunk_range(config, total_chunk_number_found)
    running_chunk_number = end_chunk - start_chunk + 1

    # Process each chunk
    output_dict = defaultdict(list)
    trait_pair = f"{config.trait1_name}_{config.trait2_name}"
    ref_ld_baseline_column_sum = ref_ld_baseline.sum(axis=1).values

    for chunk_index in range(start_chunk, end_chunk + 1):
        logger.info(f"Processing chunk {chunk_index} of {running_chunk_number}")

        # Load spatial LD scores for this chunk using the refactored function
        ref_ld_spatial, spatial_annotation_cnames, M_ref_ld_spatial = load_ldscore_chunk(
            chunk_index,
            common_snp_among_all_sumstats_pos,
            config,
            zarr_file,
            spots_name
        )

        # Process each spot in this chunk
        chunk_size = ref_ld_spatial.shape[1]

        # Create a partial function with all arguments except spot_id
        process_spot = partial(
            compute_local_genetic_correlation,
            spatial_annotation=ref_ld_spatial,
            ref_ld_baseline=ref_ld_baseline,
            merged_sumstats=merged_sumstats,
            w_ld=w_ld,
            global_parameters=global_params,
            n_blocks=n_blocks,
            M_spatial=M_ref_ld_spatial,
            M_baseline=M_w_ld,
            old_weights=True,
            twostep=None,
            use_global_parameters=config.use_global_parameters,
        )

        # Process spots in parallel using thread_map
        spot_results_raw = thread_map(
            process_spot,
            range(chunk_size),
            max_workers=config.num_processes,
            chunksize=10,  # Process spots in small batches for better load balancing
            desc=f"Processing chunk {chunk_index}/{running_chunk_number}"
        )

        # Add spot names to results
        spot_results = []
        for spot_id, result in enumerate(spot_results_raw):
            result["spot"] = spatial_annotation_cnames[spot_id]
            spot_results.append(result)

        # Convert results to DataFrame for this chunk
        chunk_results = pd.DataFrame(spot_results)
        output_dict[trait_pair].append(chunk_results)

        # Clean up memory
        del ref_ld_spatial, spot_results
        gc.collect()

    # Combine results across chunks and save
    for trait_pair, result_chunks in output_dict.items():
        combined_results = pd.concat(result_chunks, axis=0, ignore_index=True)

        # Define output file path
        output_file = Path(config.rg_save_dir) / f"{sample_name}_{trait_pair}.csv.gz"

        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Save results
        combined_results.to_csv(output_file, index=False, compression="gzip")
        logger.info(f"Results saved to {output_file}")

    logger.info(f"------Spatial LDSC Genetic Correlation for {sample_name} finished!")


if __name__ == "__main__":
    sumstats_file1 = '/storage/yangjianLab/songliyang/GWAS_trait/LDSC/DIAMANTE_EUR_T2D_2022_NG.sumstats.gz'
    sumstats_file2 = '/storage/yangjianLab/songliyang/GWAS_trait/LDSC/BMI-GIANT_2018_cojo.sumstats.gz'
    sumstats_file1 = trait1 = '/storage/yangjianLab/songliyang/GWAS_trait/LDSC/PGC3_SCZ_wave3_public_INFO80.sumstats.gz'
    sumstats_file2 = trait2 = '/storage/yangjianLab/songliyang/GWAS_trait/LDSC/PGC_Bipolar_INFO80_2021_NatGenet.sumstats.gz'

    trait1_name = 'T2D'
    trait2_name = 'BMI'

    config = SpatialLDSCRgConfig(**{'all_chunk': None,
                                    'chisq_max': None,
                                    'chunk_range': None,
                                    'n_blocks': 200,
                                    'num_processes': 2,
                                    'sample_name': 'E16.5_E1S1.MOSTA',
                                    'trait1_name': trait1_name,
                                    'trait1_sumstats': sumstats_file1,
                                    'trait2_name': trait2_name,
                                    'trait2_sumstats': sumstats_file2,
                                    'use_additional_baseline_annotation': False,
                                    'w_file': '/storage/yangjianLab/chenwenhao/01_Project/01_Research/202312_gsMap/data/gsMap_dev_data/test_data/gsMap_resource/LDSC_resource/weights_hm3_no_hla/weights.',
                                    # 'workdir': '/storage/yangjianLab/chenwenhao/tmp/20250408_gsmap_dev_test_tmp_workdir'})
                                    'workdir': '/storage/yangjianLab/chenwenhao/pytest-39/Mouse_Embryo0'})
    logging.basicConfig(level=logging.INFO)
    run_spatial_ldsc_rg(config)

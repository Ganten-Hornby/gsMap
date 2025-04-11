import gc
import logging
import os
import re
from collections import defaultdict
from functools import partial
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import zarr
from scipy.stats import norm
from tqdm.contrib.concurrent import thread_map

import gsMap.utils.jackknife as jk
from gsMap.config import SpatialLDSCConfig
from gsMap.utils.regression_read import _read_ref_ld_v2, _read_sumstats, _read_w_ld

logger = logging.getLogger("gsMap.spatial_ldsc")


def _coef_new(jknife, Nbar):
    """Calculate coefficients adjusted by Nbar."""
    est_ = jknife.jknife_est[0, 0] / Nbar
    se_ = jknife.jknife_se[0, 0] / Nbar
    return est_, se_


def append_intercept(x):
    """Append an intercept term to the design matrix."""
    n_row = x.shape[0]
    intercept = np.ones((n_row, 1))
    x_new = np.concatenate((x, intercept), axis=1)
    return x_new


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


def aggregate(y, x, N, M, intercept=1):
    """Aggregate function used in weight calculation."""
    num = M * (np.mean(y) - intercept)
    denom = np.mean(np.multiply(x, N))
    return num / denom


def weights(ld, w_ld, N, M, hsq, intercept=1):
    """Calculate weights for regression."""
    M = float(M)
    hsq = np.clip(hsq, 0.0, 1.0)
    ld = np.maximum(ld, 1.0)
    w_ld = np.maximum(w_ld, 1.0)
    c = hsq * N / M
    het_w = 1.0 / (2 * np.square(intercept + np.multiply(c, ld)))
    oc_w = 1.0 / w_ld
    w = np.multiply(het_w, oc_w)
    return w


def get_weight_optimized(sumstats, x_tot_precomputed, M_tot, w_ld, intercept=1):
    """Optimized function to calculate initial weights."""
    tot_agg = aggregate(sumstats.chisq, x_tot_precomputed, sumstats.N, M_tot, intercept)
    initial_w = weights(
        x_tot_precomputed, w_ld.LD_weights.values, sumstats.N.values, M_tot, tot_agg, intercept
    )
    initial_w = np.sqrt(initial_w)
    return initial_w


def jackknife_for_processmap(
        spot_id,
        spatial_annotation,
        ref_ld_baseline_column_sum,
        sumstats,
        baseline_annotation,
        w_ld_common_snp,
        Nbar,
        n_blocks,
):
    """Perform jackknife resampling for a given spot."""
    spot_spatial_annotation = spatial_annotation[:, spot_id]
    spot_x_tot_precomputed = spot_spatial_annotation + ref_ld_baseline_column_sum
    initial_w = (
        get_weight_optimized(
            sumstats,
            x_tot_precomputed=spot_x_tot_precomputed,
            M_tot=10000,
            w_ld=w_ld_common_snp,
            intercept=1,
        )
        .astype(np.float32)
        .reshape((-1, 1))
    )
    initial_w_scaled = initial_w / np.sum(initial_w)
    baseline_annotation_spot = baseline_annotation * initial_w_scaled
    spatial_annotation_spot = spot_spatial_annotation.reshape((-1, 1)) * initial_w_scaled
    CHISQ = sumstats.chisq.values.reshape((-1, 1))
    y = CHISQ * initial_w_scaled
    x_focal = np.concatenate((spatial_annotation_spot, baseline_annotation_spot), axis=1)
    try:
        jknife = jk.LstsqJackknifeFast(x_focal, y, n_blocks)
    except np.linalg.LinAlgError as e:
        logger.warning(f"LinAlgError: {e}")
        return np.nan, np.nan
    return _coef_new(jknife, Nbar)


def jackknife_gencor_for_processmap(
        spot_id,
        spatial_annotation,
        ref_ld_baseline_column_sum,
        sumstats1,
        sumstats2,
        baseline_annotation,
        w_ld_common_snp,
        sqrt_N1N2,
        n_blocks,
        intercept=0,
):
    """Perform jackknife resampling for genetic correlation at a given spot."""
    spot_spatial_annotation = spatial_annotation[:, spot_id]
    spot_x_tot_precomputed = spot_spatial_annotation + ref_ld_baseline_column_sum

    # Calculate weights based on average of the two traits
    avg_chisq = (sumstats1.chisq + sumstats2.chisq) / 2
    avg_N = (sumstats1.N + sumstats2.N) / 2
    pseudo_sumstats = sumstats1.copy()
    pseudo_sumstats.chisq = avg_chisq
    pseudo_sumstats.N = avg_N

    initial_w = (
        get_weight_optimized(
            pseudo_sumstats,
            x_tot_precomputed=spot_x_tot_precomputed,
            M_tot=10000,
            w_ld=w_ld_common_snp,
            intercept=1,
        )
        .astype(np.float32)
        .reshape((-1, 1))
    )
    initial_w_scaled = initial_w / np.sum(initial_w)

    # Prepare data for genetic covariance regression
    baseline_annotation_spot = baseline_annotation * initial_w_scaled
    spatial_annotation_spot = spot_spatial_annotation.reshape((-1, 1)) * initial_w_scaled

    # Product of z-scores as genetic covariance measure
    z1z2 = (sumstats1.Z * sumstats2.Z).values.reshape((-1, 1))
    y = z1z2 * initial_w_scaled

    # Combine the focal annotation with baseline
    x_focal = np.concatenate((spatial_annotation_spot, baseline_annotation_spot), axis=1)

    try:
        # Run jackknife
        jknife = jk.LstsqJackknifeFast(x_focal, y, n_blocks)

        # Get genetic covariance estimate
        gencov_est = jknife.jknife_est[0, 0] / sqrt_N1N2
        gencov_se = jknife.jknife_se[0, 0] / sqrt_N1N2

        return gencov_est, gencov_se
    except np.linalg.LinAlgError as e:
        logger.warning(f"LinAlgError in genetic correlation calculation: {e}")
        return np.nan, np.nan


def calculate_global_genetic_correlation(
        base_ld,
        sumstats1,
        sumstats2,
        w_ld_common_snp,
        n_blocks
):
    """Calculate global genetic correlation between two traits."""
    logger.info("Calculating global genetic correlation...")

    # Calculate individual heritabilities
    h2_1 = estimate_global_heritability(base_ld, sumstats1, w_ld_common_snp, n_blocks)
    h2_2 = estimate_global_heritability(base_ld, sumstats2, w_ld_common_snp, n_blocks)

    # Calculate genetic covariance
    pseudo_sumstats = sumstats1.copy()
    avg_chisq = (sumstats1.chisq + sumstats2.chisq) / 2
    avg_N = (sumstats1.N + sumstats2.N) / 2
    pseudo_sumstats.chisq = avg_chisq
    pseudo_sumstats.N = avg_N

    # Prepare data for genetic covariance regression
    initial_w = (
        get_weight_optimized(
            pseudo_sumstats,
            x_tot_precomputed=base_ld.flatten(),
            M_tot=10000,
            w_ld=w_ld_common_snp,
            intercept=1,
        )
        .astype(np.float32)
        .reshape((-1, 1))
    )
    initial_w_scaled = initial_w / np.sum(initial_w)

    # Apply weights to baseline annotation
    baseline_annotation = base_ld.copy().astype(np.float32)
    baseline_annotation = baseline_annotation * initial_w_scaled
    baseline_annotation = append_intercept(baseline_annotation)

    # Product of z-scores for genetic covariance
    z1z2 = (sumstats1.Z * sumstats2.Z).values.reshape((-1, 1))
    y = z1z2 * initial_w_scaled

    # Run regression
    try:
        jknife = jk.LstsqJackknifeFast(baseline_annotation, y, n_blocks)
        sqrt_N1N2 = np.sqrt(sumstats1.N.mean() * sumstats2.N.mean())

        # Get genetic covariance estimate
        gencov_est = jknife.jknife_est[0, 0] / sqrt_N1N2
        gencov_se = jknife.jknife_se[0, 0] / sqrt_N1N2

        # Calculate genetic correlation
        if h2_1['h2'] <= 0 or h2_2['h2'] <= 0:
            logger.warning("Heritability estimate ≤ 0, cannot compute genetic correlation")
            return {
                'rg': np.nan,
                'rg_se': np.nan,
                'z': np.nan,
                'p': np.nan,
                'h2_1': h2_1,
                'h2_2': h2_2,
                'gcov': gencov_est,
                'gcov_se': gencov_se
            }

        # Genetic correlation = covariance / sqrt(h2_1 * h2_2)
        rg = gencov_est / np.sqrt(h2_1['h2'] * h2_2['h2'])

        # Standard error using the delta method
        rg_se = np.abs(rg) * np.sqrt(
            (gencov_se / gencov_est) ** 2 +
            (h2_1['h2_se'] / (2 * h2_1['h2'])) ** 2 +
            (h2_2['h2_se'] / (2 * h2_2['h2'])) ** 2
        )

        # Z-score and p-value
        z = rg / rg_se
        p = norm.sf(np.abs(z)) * 2  # Two-tailed test

        return {
            'rg': rg,
            'rg_se': rg_se,
            'z': z,
            'p': p,
            'h2_1': h2_1,
            'h2_2': h2_2,
            'gcov': gencov_est,
            'gcov_se': gencov_se
        }

    except np.linalg.LinAlgError as e:
        logger.error(f"LinAlgError in global genetic correlation calculation: {e}")
        return {
            'rg': np.nan,
            'rg_se': np.nan,
            'z': np.nan,
            'p': np.nan,
            'h2_1': {'h2': np.nan, 'h2_se': np.nan},
            'h2_2': {'h2': np.nan, 'h2_se': np.nan},
            'gcov': np.nan,
            'gcov_se': np.nan
        }


def estimate_global_heritability(base_ld, sumstats, w_ld_common_snp, n_blocks):
    """Estimate global heritability for a trait."""
    # Calculate weights
    initial_w = (
        get_weight_optimized(
            sumstats,
            x_tot_precomputed=base_ld.flatten(),
            M_tot=10000,
            w_ld=w_ld_common_snp,
            intercept=1,
        )
        .astype(np.float32)
        .reshape((-1, 1))
    )
    initial_w_scaled = initial_w / np.sum(initial_w)

    # Apply weights to baseline annotation
    baseline_annotation = base_ld.copy()
    baseline_annotation = baseline_annotation * initial_w_scaled
    baseline_annotation = append_intercept(baseline_annotation)

    # Prepare dependent variable
    y = sumstats.chisq.values.reshape((-1, 1)) * initial_w_scaled

    # Run regression
    try:
        jknife = jk.LstsqJackknifeFast(baseline_annotation, y, n_blocks)
        Nbar = sumstats.N.mean()

        # Get heritability estimate
        h2_est, h2_se = _coef_new(jknife, Nbar)

        return {'h2': h2_est, 'h2_se': h2_se}

    except np.linalg.LinAlgError as e:
        logger.error(f"LinAlgError in heritability calculation: {e}")
        return {'h2': np.nan, 'h2_se': np.nan}


def _preprocess_sumstats(
        trait_name, sumstat_file_path, baseline_and_w_ld_common_snp: pd.Index, chisq_max=None
):
    """Preprocess summary statistics."""
    sumstats = _read_sumstats(fh=sumstat_file_path, alleles=False, dropna=False)
    sumstats.set_index("SNP", inplace=True)
    sumstats = sumstats.astype(np.float32)
    sumstats = filter_sumstats_by_chisq(sumstats, chisq_max)
    common_snp = baseline_and_w_ld_common_snp.intersection(sumstats.index)
    if len(common_snp) < 200000:
        logger.warning(
            f"WARNING: number of SNPs less than 200k; for {trait_name} this is almost always bad."
        )
    sumstats = sumstats.loc[common_snp]
    sumstats["common_index_pos"] = pd.Index(baseline_and_w_ld_common_snp).get_indexer(
        sumstats.index
    )
    return sumstats


def _get_sumstats_with_common_snp_from_sumstats_dict(
        sumstats_config_dict: dict, baseline_and_w_ld_common_snp: pd.Index, chisq_max=None
):
    """Get summary statistics with common SNPs among all traits."""
    logger.info("Validating sumstats files...")
    for trait_name, sumstat_file_path in sumstats_config_dict.items():
        if not os.path.exists(sumstat_file_path):
            raise FileNotFoundError(f"{sumstat_file_path} not found")
    sumstats_cleaned_dict = {}
    for trait_name, sumstat_file_path in sumstats_config_dict.items():
        sumstats_cleaned_dict[trait_name] = _preprocess_sumstats(
            trait_name, sumstat_file_path, baseline_and_w_ld_common_snp, chisq_max
        )
    common_snp_among_all_sumstats = None
    for trait_name, sumstats in sumstats_cleaned_dict.items():
        if common_snp_among_all_sumstats is None:
            common_snp_among_all_sumstats = sumstats.index
        else:
            common_snp_among_all_sumstats = common_snp_among_all_sumstats.intersection(
                sumstats.index
            )
    for trait_name, sumstats in sumstats_cleaned_dict.items():
        sumstats_cleaned_dict[trait_name] = sumstats.loc[common_snp_among_all_sumstats]
    logger.info(f"Common SNPs among all sumstats: {len(common_snp_among_all_sumstats)}")
    return sumstats_cleaned_dict, common_snp_among_all_sumstats


class S_LDSC_Boost_with_pre_calculate_SNP_Gene_weight_matrix:
    """Class to handle pre-calculated SNP-Gene weight matrix for quick mode."""

    def __init__(self, config: SpatialLDSCConfig, common_snp_among_all_sumstats_pos):
        self.config = config
        mk_score = pd.read_feather(config.mkscore_feather_path).set_index("HUMAN_GENE_SYM")
        mk_score_genes = mk_score.index
        snp_gene_weight_adata = ad.read_h5ad(config.snp_gene_weight_adata_path)
        common_genes = mk_score_genes.intersection(snp_gene_weight_adata.var.index)
        # common_snps = snp_gene_weight_adata.obs.index
        self.snp_gene_weight_matrix = snp_gene_weight_adata[
            common_snp_among_all_sumstats_pos, common_genes.to_list()
        ].X
        self.mk_score_common = mk_score.loc[common_genes]
        self.chunk_starts = list(
            range(0, self.mk_score_common.shape[1], self.config.spots_per_chunk_quick_mode)
        )

    def fetch_ldscore_by_chunk(self, chunk_index):
        """Fetch LD score by chunk."""
        chunk_start = self.chunk_starts[chunk_index]
        mk_score_chunk = self.mk_score_common.iloc[
                         :, chunk_start: chunk_start + self.config.spots_per_chunk_quick_mode
                         ]
        ldscore_chunk = self.calculate_ldscore_use_SNP_Gene_weight_matrix_by_chunk(
            mk_score_chunk, drop_dummy_na=False
        )
        spots_name = self.mk_score_common.columns[
                     chunk_start: chunk_start + self.config.spots_per_chunk_quick_mode
                     ]
        return ldscore_chunk, spots_name

    def calculate_ldscore_use_SNP_Gene_weight_matrix_by_chunk(
            self, mk_score_chunk, drop_dummy_na=True
    ):
        """Calculate LD score using SNP-Gene weight matrix by chunk."""
        if drop_dummy_na:
            ldscore_chr_chunk = self.snp_gene_weight_matrix[:, :-1] @ mk_score_chunk
        else:
            ldscore_chr_chunk = self.snp_gene_weight_matrix @ mk_score_chunk
        return ldscore_chr_chunk


def load_ldscore_chunk_from_feather(chunk_index, common_snp_among_all_sumstats_pos, config):
    """Load LD score chunk from feather format."""
    sample_name = config.sample_name
    ld_file_spatial = f"{config.ldscore_save_dir}/{sample_name}_chunk{chunk_index}/{sample_name}."
    ref_ld_spatial = _read_ref_ld_v2(ld_file_spatial)
    ref_ld_spatial = ref_ld_spatial.iloc[common_snp_among_all_sumstats_pos]
    ref_ld_spatial = ref_ld_spatial.astype(np.float32, copy=False)
    spatial_annotation_cnames = ref_ld_spatial.columns
    return ref_ld_spatial.values, spatial_annotation_cnames


def run_spatial_ldsc(config: SpatialLDSCConfig):
    """Run spatial LDSC analysis."""
    logger.info(f"------Running Spatial LDSC for {config.sample_name}...")
    n_blocks = config.n_blocks
    sample_name = config.sample_name

    # Load regression weights
    w_ld = _read_w_ld(config.w_file)
    w_ld.set_index("SNP", inplace=True)

    ld_file_baseline = f"{config.ldscore_save_dir}/baseline/baseline."
    ref_ld_baseline = _read_ref_ld_v2(ld_file_baseline)
    baseline_and_w_ld_common_snp = ref_ld_baseline.index.intersection(w_ld.index)

    sumstats_cleaned_dict, common_snp_among_all_sumstats = (
        _get_sumstats_with_common_snp_from_sumstats_dict(
            config.sumstats_config_dict, baseline_and_w_ld_common_snp, chisq_max=config.chisq_max
        )
    )
    common_snp_among_all_sumstats_pos = ref_ld_baseline.index.get_indexer(
        common_snp_among_all_sumstats
    )

    if not pd.Series(common_snp_among_all_sumstats_pos).is_monotonic_increasing:
        raise ValueError("common_snp_among_all_sumstats_pos is not monotonic increasing")

    if len(common_snp_among_all_sumstats) < 200000:
        logger.warning(
            f"!!!!! WARNING: number of SNPs less than 200k; for {sample_name} this is almost always bad. Please check the sumstats files."
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

    # Initialize s_ldsc once if quick_mode
    s_ldsc = None
    if config.ldscore_save_format == "quick_mode":
        s_ldsc = S_LDSC_Boost_with_pre_calculate_SNP_Gene_weight_matrix(
            config, common_snp_among_all_sumstats_pos
        )
        total_chunk_number_found = len(s_ldsc.chunk_starts)
        logger.info(f"Split data into {total_chunk_number_found} chunks")
    else:
        total_chunk_number_found = determine_total_chunks(config)

    start_chunk, end_chunk = determine_chunk_range(config, total_chunk_number_found)
    running_chunk_number = end_chunk - start_chunk + 1

    # Load zarr file if needed
    zarr_file, spots_name = None, None
    if config.ldscore_save_format == "zarr":
        zarr_path = Path(config.ldscore_save_dir) / f"{config.sample_name}.ldscore.zarr"
        if not zarr_path.exists():
            raise FileNotFoundError(f"{zarr_path} not found, which is required for zarr format")
        zarr_file = zarr.open(str(zarr_path))
        spots_name = zarr_file.attrs["spot_names"]

    output_dict = defaultdict(list)
    for chunk_index in range(start_chunk, end_chunk + 1):
        ref_ld_spatial, spatial_annotation_cnames = load_ldscore_chunk(
            chunk_index,
            common_snp_among_all_sumstats_pos,
            config,
            zarr_file,
            spots_name,
            s_ldsc,  # Pass s_ldsc to the function
        )
        ref_ld_baseline_column_sum = ref_ld_baseline.sum(axis=1).values

        for trait_name, sumstats in sumstats_cleaned_dict.items():
            spatial_annotation = ref_ld_spatial.astype(np.float32, copy=False)
            baseline_annotation = ref_ld_baseline.copy().astype(np.float32, copy=False)
            w_ld_common_snp = w_ld.astype(np.float32, copy=False)

            baseline_annotation = (
                    baseline_annotation * sumstats.N.values.reshape((-1, 1)) / sumstats.N.mean()
            )
            baseline_annotation = append_intercept(baseline_annotation)

            Nbar = sumstats.N.mean()
            chunk_size = spatial_annotation.shape[1]

            jackknife_func = partial(
                jackknife_for_processmap,
                spatial_annotation=spatial_annotation,
                ref_ld_baseline_column_sum=ref_ld_baseline_column_sum,
                sumstats=sumstats,
                baseline_annotation=baseline_annotation,
                w_ld_common_snp=w_ld_common_snp,
                Nbar=Nbar,
                n_blocks=n_blocks,
            )

            out_chunk = thread_map(
                jackknife_func,
                range(chunk_size),
                max_workers=config.num_processes,
                chunksize=10,
                desc=f"Chunk-{chunk_index}/Total-chunk-{running_chunk_number} for {trait_name}",
            )

            out_chunk = pd.DataFrame.from_records(
                out_chunk, columns=["beta", "se"], index=spatial_annotation_cnames
            )
            nan_spots = out_chunk[out_chunk.isna().any(axis=1)].index
            if len(nan_spots) > 0:
                logger.info(
                    f"Nan spots: {nan_spots} in chunk-{chunk_index} for {trait_name}. They are removed."
                )
            out_chunk = out_chunk.dropna()
            out_chunk["z"] = out_chunk.beta / out_chunk.se
            out_chunk["p"] = norm.sf(out_chunk["z"])
            output_dict[trait_name].append(out_chunk)

            del spatial_annotation, baseline_annotation, w_ld_common_snp
            gc.collect()

    save_results(output_dict, config, running_chunk_number, start_chunk, end_chunk)
    logger.info(f"------Spatial LDSC for {sample_name} finished!")


def run_spatial_gencor(config: SpatialLDSCConfig):
    """Run spatial genetic correlation analysis between two traits."""
    logger.info(f"------Running Spatial Genetic Correlation for {config.sample_name}...")
    n_blocks = config.n_blocks
    sample_name = config.sample_name

    # Make sure we have at least 2 traits
    if len(config.sumstats_config_dict) < 2:
        raise ValueError("At least two traits are required for genetic correlation analysis")

    # Get trait names and paths
    trait_pairs = []
    trait_names = list(config.sumstats_config_dict.keys())

    # Create all pairs of traits
    for i in range(len(trait_names)):
        for j in range(i + 1, len(trait_names)):
            trait_pairs.append((trait_names[i], trait_names[j]))

    logger.info(f"Analyzing {len(trait_pairs)} trait pairs: {trait_pairs}")

    # Load regression weights
    w_ld = _read_w_ld(config.w_file)
    w_ld.set_index("SNP", inplace=True)

    ld_file_baseline = f"{config.ldscore_save_dir}/baseline/baseline."
    ref_ld_baseline = _read_ref_ld_v2(ld_file_baseline)
    baseline_and_w_ld_common_snp = ref_ld_baseline.index.intersection(w_ld.index)

    sumstats_cleaned_dict, common_snp_among_all_sumstats = (
        _get_sumstats_with_common_snp_from_sumstats_dict(
            config.sumstats_config_dict, baseline_and_w_ld_common_snp, chisq_max=config.chisq_max
        )
    )
    common_snp_among_all_sumstats_pos = ref_ld_baseline.index.get_indexer(
        common_snp_among_all_sumstats
    )

    if not pd.Series(common_snp_among_all_sumstats_pos).is_monotonic_increasing:
        raise ValueError("common_snp_among_all_sumstats_pos is not monotonic increasing")

    if len(common_snp_among_all_sumstats) < 200000:
        logger.warning(
            f"!!!!! WARNING: number of SNPs less than 200k; for {sample_name} this is almost always bad. Please check the sumstats files."
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

    # Initialize s_ldsc once if quick_mode
    s_ldsc = None
    if config.ldscore_save_format == "quick_mode":
        s_ldsc = S_LDSC_Boost_with_pre_calculate_SNP_Gene_weight_matrix(
            config, common_snp_among_all_sumstats_pos
        )
        total_chunk_number_found = len(s_ldsc.chunk_starts)
        logger.info(f"Split data into {total_chunk_number_found} chunks")
    else:
        total_chunk_number_found = determine_total_chunks(config)

    start_chunk, end_chunk = determine_chunk_range(config, total_chunk_number_found)
    running_chunk_number = end_chunk - start_chunk + 1

    # Load zarr file if needed
    zarr_file, spots_name = None, None
    if config.ldscore_save_format == "zarr":
        zarr_path = Path(config.ldscore_save_dir) / f"{config.sample_name}.ldscore.zarr"
        if not zarr_path.exists():
            raise FileNotFoundError(f"{zarr_path} not found, which is required for zarr format")
        zarr_file = zarr.open(str(zarr_path))
        spots_name = zarr_file.attrs["spot_names"]

    # First, calculate global genetic correlations for all trait pairs
    global_gencor_results = {}
    for trait1, trait2 in trait_pairs:
        logger.info(f"Calculating global genetic correlation between {trait1} and {trait2}")
        sumstats1 = sumstats_cleaned_dict[trait1]
        sumstats2 = sumstats_cleaned_dict[trait2]

        global_gencor = calculate_global_genetic_correlation(
            ref_ld_baseline['base'].values.astype(np.float32).reshape(-1, 1),
            sumstats1,
            sumstats2,
            w_ld,
            n_blocks
        )

        global_gencor_results[f"{trait1}_{trait2}"] = global_gencor

        logger.info(f"Global genetic correlation between {trait1} and {trait2}: " +
                    f"rg={global_gencor['rg']:.4f} (SE={global_gencor['rg_se']:.4f}), p={global_gencor['p']:.4e}")

    # Save global genetic correlation results
    save_global_gencor_results(global_gencor_results, config)

    # Now calculate spot-specific genetic correlations
    output_dict = defaultdict(list)
    for chunk_index in range(start_chunk, end_chunk + 1):
        ref_ld_spatial, spatial_annotation_cnames = load_ldscore_chunk(
            chunk_index,
            common_snp_among_all_sumstats_pos,
            config,
            zarr_file,
            spots_name,
            s_ldsc,
        )
        ref_ld_baseline_column_sum = ref_ld_baseline.sum(axis=1).values

        # Process each trait pair
        for trait1, trait2 in trait_pairs:
            pair_key = f"{trait1}_{trait2}"
            sumstats1 = sumstats_cleaned_dict[trait1]
            sumstats2 = sumstats_cleaned_dict[trait2]

            spatial_annotation = ref_ld_spatial.astype(np.float32, copy=False)
            baseline_annotation = ref_ld_baseline.copy().astype(np.float32, copy=False)
            w_ld_common_snp = w_ld.astype(np.float32, copy=False)

            # Scale baseline annotation by average N
            avg_N = (sumstats1.N.values + sumstats2.N.values) / 2
            baseline_annotation = (
                    baseline_annotation * avg_N.reshape((-1, 1)) / np.mean(avg_N)
            )
            baseline_annotation = append_intercept(baseline_annotation)

            sqrt_N1N2 = np.sqrt(sumstats1.N.mean() * sumstats2.N.mean())
            chunk_size = spatial_annotation.shape[1]

            # Get h2 estimates for both traits from global results
            h2_1 = global_gencor_results[pair_key]['h2_1']['h2']
            h2_2 = global_gencor_results[pair_key]['h2_2']['h2']

            # Calculate genetic covariance for each spot
            jackknife_func = partial(
                jackknife_gencor_for_processmap,
                spatial_annotation=spatial_annotation,
                ref_ld_baseline_column_sum=ref_ld_baseline_column_sum,
                sumstats1=sumstats1,
                sumstats2=sumstats2,
                baseline_annotation=baseline_annotation,
                w_ld_common_snp=w_ld_common_snp,
                sqrt_N1N2=sqrt_N1N2,
                n_blocks=n_blocks,
            )

            out_chunk = thread_map(
                jackknife_func,
                range(chunk_size),
                max_workers=config.num_processes,
                chunksize=10,
                desc=f"Chunk-{chunk_index}/{running_chunk_number} for {pair_key}",
            )

            # Process results
            out_chunk_df = pd.DataFrame.from_records(
                out_chunk, columns=["gcov", "gcov_se"], index=spatial_annotation_cnames
            )
            nan_spots = out_chunk_df[out_chunk_df.isna().any(axis=1)].index
            if len(nan_spots) > 0:
                logger.info(
                    f"Nan spots: {nan_spots} in chunk-{chunk_index} for {pair_key}. They are removed."
                )
            out_chunk_df = out_chunk_df.dropna()

            # Calculate genetic correlation for each spot
            if h2_1 <= 0 or h2_2 <= 0:
                logger.warning(
                    f"Heritability estimate ≤ 0 for traits in {pair_key}, cannot compute spot-specific genetic correlation")
                out_chunk_df["rg"] = np.nan
                out_chunk_df["rg_se"] = np.nan
                out_chunk_df["z"] = np.nan
                out_chunk_df["p"] = np.nan
            else:
                # Genetic correlation = covariance / sqrt(h2_1 * h2_2)
                out_chunk_df["rg"] = out_chunk_df.gcov / np.sqrt(h2_1 * h2_2)

                # Standard error using the delta method (simplified version)
                out_chunk_df["rg_se"] = np.abs(out_chunk_df.rg) * (out_chunk_df.gcov_se / out_chunk_df.gcov)

                # Z-score and p-value
                out_chunk_df["z"] = out_chunk_df.rg / out_chunk_df.rg_se
                out_chunk_df["p"] = 2 * norm.sf(np.abs(out_chunk_df.z))  # Two-tailed test

            output_dict[pair_key].append(out_chunk_df)

            del spatial_annotation, baseline_annotation, w_ld_common_snp
            gc.collect()

    save_gencor_results(output_dict, config, running_chunk_number, start_chunk, end_chunk)
    logger.info(f"------Spatial Genetic Correlation for {sample_name} finished!")


def determine_total_chunks(config):
    """Determine total number of chunks based on the ldscore save format."""
    if config.ldscore_save_format == "quick_mode":
        s_ldsc = S_LDSC_Boost_with_pre_calculate_SNP_Gene_weight_matrix(config, [])
        total_chunk_number_found = len(s_ldsc.chunk_starts)
        logger.info(f"Split data into {total_chunk_number_found} chunks")
    else:
        all_file = os.listdir(config.ldscore_save_dir)
        total_chunk_number_found = sum("chunk" in name for name in all_file)
        logger.info(f"Find {total_chunk_number_found} chunked files in {config.ldscore_save_dir}")
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


def load_ldscore_chunk(
        chunk_index,
        common_snp_among_all_sumstats_pos,
        config,
        zarr_file=None,
        spots_name=None,
        s_ldsc=None,
):
    """Load LD score chunk based on save format."""
    if config.ldscore_save_format == "feather":
        return load_ldscore_chunk_from_feather(
            chunk_index, common_snp_among_all_sumstats_pos, config
        )
    elif config.ldscore_save_format == "zarr":
        ref_ld_spatial = zarr_file.blocks[:, chunk_index - 1][common_snp_among_all_sumstats_pos]
        start_spot = (chunk_index - 1) * zarr_file.chunks[1]
        ref_ld_spatial = ref_ld_spatial.astype(np.float32, copy=False)
        spatial_annotation_cnames = spots_name[start_spot: start_spot + zarr_file.chunks[1]]
        return ref_ld_spatial, spatial_annotation_cnames
    elif config.ldscore_save_format == "quick_mode":
        # Use the pre-initialized s_ldsc
        if s_ldsc is None:
            raise ValueError("s_ldsc must be provided in quick_mode")
        return s_ldsc.fetch_ldscore_by_chunk(chunk_index - 1)
    else:
        raise ValueError(f"Invalid ld score save format: {config.ldscore_save_format}")


def save_results(output_dict, config, running_chunk_number, start_chunk, end_chunk):
    """Save the results to the specified directory."""
    out_dir = config.ldsc_save_dir
    for trait_name, out_chunk_list in output_dict.items():
        out_all = pd.concat(out_chunk_list, axis=0)
        sample_name = config.sample_name
        if running_chunk_number == end_chunk - start_chunk + 1:
            out_file_name = out_dir / f"{sample_name}_{trait_name}.csv.gz"
        else:
            out_file_name = (
                    out_dir / f"{sample_name}_{trait_name}_chunk{start_chunk}-{end_chunk}.csv.gz"
            )
        out_all["spot"] = out_all.index
        out_all = out_all[["spot", "beta", "se", "z", "p"]]

        # clip the p-values
        out_all["p"] = out_all["p"].clip(1e-300, 1)
        out_all.to_csv(out_file_name, compression="gzip", index=False)
        logger.info(f"Output saved to {out_file_name} for {trait_name}")


def save_global_gencor_results(global_gencor_results, config):
    """Save global genetic correlation results."""
    out_dir = config.ldsc_save_dir
    sample_name = config.sample_name
    out_file_name = out_dir / f"{sample_name}_global_gencor.csv"

    results_list = []
    for pair_key, result in global_gencor_results.items():
        trait1, trait2 = pair_key.split('_')
        results_list.append({
            'trait1': trait1,
            'trait2': trait2,
            'rg': result['rg'],
            'rg_se': result['rg_se'],
            'z': result['z'],
            'p': result['p'],
            'h2_1': result['h2_1']['h2'],
            'h2_1_se': result['h2_1']['h2_se'],
            'h2_2': result['h2_2']['h2'],
            'h2_2_se': result['h2_2']['h2_se'],
            'gcov': result['gcov'],
            'gcov_se': result['gcov_se']
        })

    pd.DataFrame(results_list).to_csv(out_file_name, index=False)
    logger.info(f"Global genetic correlation results saved to {out_file_name}")


def save_gencor_results(output_dict, config, running_chunk_number, start_chunk, end_chunk):
    """Save the genetic correlation results to the specified directory."""
    out_dir = config.ldsc_save_dir
    for pair_key, out_chunk_list in output_dict.items():
        out_all = pd.concat(out_chunk_list, axis=0)
        sample_name = config.sample_name
        if running_chunk_number == end_chunk - start_chunk + 1:
            out_file_name = out_dir / f"{sample_name}_gencor_{pair_key}.csv.gz"
        else:
            out_file_name = (
                    out_dir / f"{sample_name}_gencor_{pair_key}_chunk{start_chunk}-{end_chunk}.csv.gz"
            )
        out_all["spot"] = out_all.index
        out_all = out_all[["spot", "gcov", "gcov_se", "rg", "rg_se", "z", "p"]]

        # clip the p-values
        out_all["p"] = out_all["p"].clip(1e-300, 1)
        out_all.to_csv(out_file_name, compression="gzip", index=False)
        logger.info(f"Genetic correlation results saved to {out_file_name} for {pair_key}")
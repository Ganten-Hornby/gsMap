import logging
import shlex
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from gsMap.main import main


def parse_bash_command(command: str) -> list[str]:
    """Convert multi-line bash command to argument list for sys.argv"""
    cleaned_command = command.replace("\\\n", " ")
    cleaned_command = " ".join(cleaned_command.splitlines())
    cleaned_command = " ".join(cleaned_command.split())
    return shlex.split(cleaned_command)


@pytest.mark.real_data
@pytest.mark.parametrize("symbolic_link_results", ["rg_config"], indirect=True)
def test_spatial_ldsc_rg(symbolic_link_results, spatial_ldsc_fixture):

    logger = logging.getLogger("test_spatial_ldsc_rg")
    logger.info("Starting Spatial LDSC Genetic Correlation test")
    config = symbolic_link_results

    sumstats_file = str(config.sumstats_file)

    trait1_name = "IQ1"
    trait2_name = "IQ2"
    command = f"""
    gsmap run_spatial_ldsc_rg \
        --workdir '{config.workdir}' \
        --sample_name '{config.sample_name}' \
        --trait1_sumstats {sumstats_file} \
        --trait2_sumstats {sumstats_file} \
        --w_file '{config.w_file}' \
        --trait1_name {trait1_name} \
        --trait2_name {trait2_name} \
        --num_processes 2
    """

    with patch.object(sys, "argv", parse_bash_command(command)):
        main()

    # Verify output files were created
    rg_result_file = Path(
        config.workdir) / config.sample_name / "spatial_ldsc_rg" / f"{trait1_name}_{trait2_name}_rg.csv.gz"
    assert rg_result_file.exists(), f"Genetic correlation results file not created: {rg_result_file}"
    assert rg_result_file.stat().st_size > 0, "Genetic correlation results file is empty"

    import pandas as pd
    rg_results = pd.read_csv(rg_result_file, compression="gzip")
    assert all(rg_results['rg'] > 0.95), "Expected near-perfect correlation when using identical traits"

    logger.info("Spatial LDSC Genetic Correlation test completed successfully")
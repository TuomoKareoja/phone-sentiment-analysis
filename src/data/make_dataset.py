# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path

import click
import pandas as pd
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    logger.info("Loading the datasets")
    df_iphone = pd.read_csv(
        os.path.join("data", "external", "iphone_smallmatrix_labeled_8d.csv")
    )
    df_galaxy = pd.read_csv(
        os.path.join("data", "external", "galaxy_smallmatrix_labeled_8d.csv")
    )

    logger.info("Dropping unnecessary columns with no variation")
    # we know that the predicted columns have variance so
    # we do not need to specify them as safe
    df_iphone = df_iphone.loc[:, df_iphone.std() > 0]
    df_galaxy = df_galaxy.loc[:, df_galaxy.std() > 0]

    logger.info("Saving to data/processed")
    df_iphone.to_csv(os.path.join("data", "processed", "iphone.csv"), index=False)
    df_galaxy.to_csv(os.path.join("data", "processed", "galaxy.csv"), index=False)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

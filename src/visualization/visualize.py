# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
import matplotlib.pyplot as plt

# String constants
SSE_PLOT_FILENAME = 'sse-scores-plot.png'
MSC_PLOT_FILENAME = 'msc-scores-plot.png'


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_folder', type=click.Path())
def main(input_filepath, output_folder):
    """
    Receives the location of the SSE/MSC scores as a
    command-line Path argument.
    """
    logger = logging.getLogger(__name__)
    logger.info('Generating SSE/MSC plots')

    # Get the reports/k-means-plot-results.csv file
    dataset = pd.read_csv(input_filepath, index_col=0)
    logger.info('Loaded data file ' + input_filepath + ' with ' + str(len(dataset)) + ' rows')

    sse_df = dataset['SSE Score']
    msc_df = dataset['MSC Score']

    sse_df.plot(kind='line', title='Sum of Squared Error (SSE) Plot', ylim=(0, 1000))
    # the plot gets saved to 'reports/figures/sse-scores-plot.png'
    plt.savefig(output_folder + SSE_PLOT_FILENAME)
    logger.info('Created plot graph at ' + output_folder + SSE_PLOT_FILENAME)

    msc_df.plot(kind='line', title='Mean Silhouette Coefficient (MSC) Plot', ylim=(0, 0.1))
    # the plot gets saved to 'reports/figures/msc-scores-plot.png'
    plt.savefig(output_folder + MSC_PLOT_FILENAME)
    logger.info('Created plot graph at ' + output_folder + MSC_PLOT_FILENAME)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()

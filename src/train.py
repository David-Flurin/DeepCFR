import argparse
import logging

import yaml

from deep_crf import DeepCRF


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default='config.yaml',
        help="Path to the configuration file",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )

    config = load_config()
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)
    n_players = 2
    deep_crf = DeepCRF(config['experiment']['n_players'], 
                       10000000, config['experiment']['name'], 
                        config['cfr']['iterations'],
                        config['cfr']['sampler_iterations'],
                        config['cfr']['start_stack'],
                        config['cfr']['min_stack'],
                        config['training']['samples'], 
                        config['training']['epochs'], 
                        config['training']['batch_size'],
                        config['cfr']['max_workers'])

    deep_crf.run()

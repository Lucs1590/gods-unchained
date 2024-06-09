import sys
import argparse
import logging
import importlib

logger = logging.getLogger()
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def main():
    logger.info('Starting pipeline')
    parser = argparse.ArgumentParser(
        description='Gods Unchained Analysis Steps'
    )
    parser.add_argument(
        '--step',
        choices=['preprocessing', 'feature_engineering', 'model_building'],
        required=True
    )
    args = parser.parse_args()
    step_module = importlib.import_module(f'{args.step}')

    step_module.run()


if __name__ == "__main__":
    main()

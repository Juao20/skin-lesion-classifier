"""Boucle d'entraînement."""
import argparse

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # TODO: dataloaders, modèle, optimiseur, boucle d'entraînement
    print(cfg)


if __name__ == "__main__":
    main()

"""
Main entry point for UCI Online Retail ML pipelines.

Usage:
    # Local execution
    python -m src --pipeline preprocessing --run_locally True --environment dev
    python -m src --pipeline training --run_locally True --environment prod

    # Cloud deployment (future)
    python -m src --pipeline preprocessing --run_locally False --environment prod
    python -m src --pipeline training --run_locally False --environment prod
"""

import argparse
import sys

from src.pipelines.preprocessing_pipeline_local_runner import run_preprocessing_pipeline
from src.pipelines.training_pipeline_local_runner import run_training_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="""
            UCI Online Retail ML Pipeline Runner. \n
            Choose to run pipelines locally or deploy to Azure ML.
        """
    )

    parser.add_argument(
        "--pipelines",
        nargs="+",
        required=True,
        choices=["preprocessing", "training", "evaluation"],
        help="One or more pipelines to run (space-separated list)",
    )
    parser.add_argument(
        "--run_locally",
        type=str,
        default="True",
        choices=["True", "true", "False", "false"],
        help="Run pipeline locally using local_runner (True/False)",
    )
    parser.add_argument(
        "--environment",
        type=str,
        default="dev",
        choices=["dev", "test", "prod"],
        help="Target environment for deployment (e.g., dev, test, prod)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Local execution
    if args.run_locally.lower() == "true":
        print(f"Running pipeline(s) locally: {args.pipelines}")
        # TODO: consider: should local run support non-dev environments?

        if "preprocessing" in args.pipelines:
            config_path = f"app_config/{args.environment}/preprocessing_pipeline.yaml"
            run_preprocessing_pipeline(config_path)
        if "training" in args.pipelines:
            config_path = f"app_config/{args.environment}/training_pipeline.yaml"
            run_training_pipeline(config_path)
        if "evaluation" in args.pipelines:
            config_path = f"app_config/{args.environment}/evaluation_pipeline.yaml"
            print("****Evaluation pipeline not yet implemented****")
            sys.exit(1)

        print("Pipeline(s) completed successfully!")

    # Cloud deployment (future implementation)
    elif args.run_locally.lower() == "false":
        print(
            f"Deploying the pipelines: '{args.pipelines}' to Azure ML environment: ({args.environment})..."
        )
        print("Cloud deployment not yet implemented")
        sys.exit(1)


if __name__ == "__main__":
    main()

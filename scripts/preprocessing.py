import pandas as pd
import logging

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

def preprocess_data(input_path, output_path):
    try:
        logger.info("Loading data from %s", input_path)
        df = pd.read_csv(input_path)

        # Check and handle missing values
        if df.isnull().sum().any():
            logger.warning("Missing values detected. Applying mean imputation.")
            df = df.fillna(df.mean(numeric_only=True))
        else:
            logger.info("No missing values found.")

        # Save preprocessed data
        logger.info("Saving preprocessed data to %s", output_path)
        df.to_csv(output_path, index=False)

    except Exception as e:
        logger.error("Preprocessing failed: %s", str(e))
        raise

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess Insurance Dataset")
    parser.add_argument(
        "--input_path",
        required=True,
        help="Path to the raw dataset (e.g., Data/insurance.csv)"
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Path to save the preprocessed dataset"
    )

    args = parser.parse_args()

    logger.info("Starting preprocessing pipeline")
    preprocess_data(args.input_path, args.output_path)
    logger.info("Preprocessing completed successfully")

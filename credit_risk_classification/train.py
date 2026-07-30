import joblib
from pathlib import Path
import logging
import sys

# Add current directory to path if needed for imports
sys.path.append(str(Path(__file__).parent))

from data_preprocessing import load_credit_data, preprocess_and_prepare_features, split_data
from model_evaluation import train_decision_tree, train_random_forest, train_extra_trees, evaluate_model

# Log path Config
log_dir = Path("logs")
if not log_dir.exists():
    # Fallback if running from the inner directory
    log_dir = Path("../logs")
log_dir.mkdir(exist_ok=True, parents=True)

# Logger Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),                      
        logging.FileHandler(log_dir / "credit_risk_classification.log")
    ]
)

logger = logging.getLogger(__name__)


def main():
    logger.info("Initializing credit risk classification training pipeline...")
    
    # Path configuration depending on where script is run from
    project_root = Path(__file__).parent.parent
    data_path = project_root / "datasets" / "german_credit_data.csv"
    model_dir = project_root / "models"
    
    model_dir.mkdir(exist_ok=True, parents=True)
    logger.info("Output directory checked/created at: %s", model_dir)

    # Load data
    logger.info("Loading credit risk data from: %s", data_path)
    df = load_credit_data(str(data_path))
    logger.info("Data loaded successfully. Rows: %d, Columns: %d", df.shape[0], df.shape[1])

    # Prepare features and target
    logger.info("Extracting features, preprocessing, and saving artifacts...")
    X, y = preprocess_and_prepare_features(df, model_dir)
    
    logger.info("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    logger.info("Split complete. Train rows: %d, Test rows: %d", X_train.shape[0], X_test.shape[0])

    # Train models
    logger.info("Training Decision Tree model...")
    dt_model = train_decision_tree(X_train, y_train)
    
    logger.info("Training Random Forest model...")
    rf_model = train_random_forest(X_train, y_train)

    logger.info("Training Extra Trees model...")
    et_model = train_extra_trees(X_train, y_train)

    # Evaluate models
    logger.info("Evaluating models on test data...")
    results = []
    results.append(evaluate_model(dt_model, X_test, y_test, "Decision Tree"))
    results.append(evaluate_model(rf_model, X_test, y_test, "Random Forest"))
    results.append(evaluate_model(et_model, X_test, y_test, "Extra Trees"))

    # Select the best model (highest accuracy)
    logger.info("Comparing model evaluation metrics...")
    best_model_result = max(results, key=lambda x: x["accuracy"])
    best_model_name = best_model_result["model_name"]
    logger.info("Best model selected: %s (Accuracy: %.4f)", best_model_name, best_model_result["accuracy"])
    
    best_model = {
        "Decision Tree": dt_model,
        "Random Forest": rf_model,
        "Extra Trees": et_model
    }[best_model_name]
    
    # Save the best model
    model_path = model_dir / "best_model.pickle"
    logger.info("Saving best model artifact to: %s", model_path)
    joblib.dump(best_model, model_path)

    logger.info("Best model saved: %s", best_model_name)
    logger.info("Model path: %s", model_path)

if __name__ == "__main__":
    main()
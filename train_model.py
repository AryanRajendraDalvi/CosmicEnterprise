import pandas as pd
import numpy as np
import catboost as ctb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

print("--- Starting Model Training ---")

# --- Step 1: Load Data and Prepare for Training ---
try:
    # We only need the main training data for the environmental model.
    # The 'processed_satellite_database.csv' will be used later for pricing.
    df = pd.read_csv('model_training_data.csv', parse_dates=['datetime'], index_col='datetime')
    print("1. Data loaded successfully.")

    # Separate features (X) from the target (y)
    X = df.drop('target_class', axis=1)
    y = df['target_class']

    # CRITICAL: For time-series data, we must split chronologically, not randomly.
    # We'll use the first 80% of the data to train and the last 20% to test.
    split_index = int(len(df) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    print(f"2. Data split chronologically: {len(X_train)} training samples, {len(X_test)} testing samples.")

    # --- Step 2: Train the CatBoost Classifier ---
    print("\n--- Training CatBoost Model ---")
    # Initialize the model with good starting parameters
    model = ctb.CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function='MultiClass',
        verbose=100,  # This will print training progress every 100 iterations
        random_seed=42
    )

    # Train the model on the training data
    model.fit(X_train, y_train)
    print("3. Model training complete.")


    # --- Step 3: Evaluate Model Performance ---
    print("\n--- Evaluating Model Performance ---")
    # Make predictions on the unseen test data
    y_pred = model.predict(X_test)

    # Calculate and print accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"4. Overall Model Accuracy: {accuracy:.2%}")

    # Print a detailed classification report
    print("\n5. Classification Report:")
    # Define target names for the report
    target_names = ['Class 0 (Quiet)', 'Class 1 (Moderate)', 'Class 2 (Strong)', 'Class 3 (Severe)']
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Generate and save the confusion matrix visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.savefig('confusion_matrix.png')
    print("6. Confusion matrix plot saved as 'confusion_matrix.png'.")


    # --- Step 4: Save the Final Model ---
    model_filename = 'catboost_storm_forecaster.cbm'
    model.save_model(model_filename)
    print(f"\n7. Model successfully saved as '{model_filename}'.")
    print("\n--- Process Complete ---")

except FileNotFoundError as e:
    print(f"ERROR: Could not find a required CSV file. Make sure 'model_training_data.csv' is in the same folder.\nDetails: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
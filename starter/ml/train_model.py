import os
import logging
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from data import process_data
from model import train_model, inference, compute_model_metrics, compute_confusion_matrix, slice_metrics

def remove_file_if_exists(path):
    """
    Deletes a file if it exists.
    
    Args:
        path (str): Path to the file to be removed.
    """
    if os.path.exists(path):
        os.remove(path)


# Initialize logging
logging.basicConfig(filename='logging.log', level=logging.INFO, filemode='a', format='%(name)s - %(levelname)s - %(message)s')

# Load data
data_path = "../data/census.csv"
data = pd.read_csv(data_path)

# Split data into train and test sets
train, test = train_test_split(data, test_size=0.20, random_state=10, stratify=data['salary'])

# Define categorical features
cat_feats = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]

# Process train data
X_train, y_train, enc, lb = process_data(train, categorical_features=cat_feats, label="salary", training=True)

# Process test data
X_test, y_test, enc, lb = process_data(test, categorical_features=cat_feats, label="salary", training=False, encoder=enc, lb=lb)

# Check if a saved model already exists
save_path = '../model'
file_names = ['trained_model.pkl', 'encoder.pkl', 'labelizer.pkl']

if os.path.isfile(os.path.join(save_path, file_names[0])):
    # Load model from disk
    model = pickle.load(open(os.path.join(save_path, file_names[0]), 'rb'))
    enc = pickle.load(open(os.path.join(save_path, file_names[1]), 'rb'))
    lb = pickle.load(open(os.path.join(save_path, file_names[2]), 'rb'))
else:
    # Train a new model and save it to disk
    model = train_model(X_train, y_train)
    pickle.dump(model, open(os.path.join(save_path, file_names[0]), 'wb'))
    pickle.dump(enc, open(os.path.join(save_path, file_names[1]), 'wb'))
    pickle.dump(lb, open(os.path.join(save_path, file_names[2]), 'wb'))
    logging.info(f"Model saved to disk: {save_path}")

# Evaluate model on test set
preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)

# Log performance metrics
logging.info(f"Classification target labels: {list(lb.classes_)}")
logging.info(f"precision:{precision:.3f}, recall:{recall:.3f}, fbeta:{fbeta:.3f}")
cm = compute_confusion_matrix(y_test, preds, labels=list(lb.classes_))
logging.info(f"Confusion matrix:\n{cm}")

# Compute performance on slices for categorical features and save to file
slice_path = "./slice_output.txt"
remove_file_if_exists(slice_path)

for feat in cat_feats:
    performance_df = slice_metrics(test, feat, y_test, preds)
    performance_df.to_csv(slice_path,  mode='a', index=False)
    logging.info(f"Performance on slice {feat}")
    logging.info(performance_df)
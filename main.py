# Chronic Kidney Disease
# by Johan Moncoutie


# Global Tools
import os
import json
import argparse
import datetime
import pickle

# Math / Data Manipulation
from scipy.io import arff
import pandas as pd

# Preprocessing / Modeling
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import xgboost as xgb

current = os.path.realpath(__file__)

# Create var with the ROOT directory path
ROOT = os.path.dirname(current)


############################ Arguments Management Functions ############################

def valid_path(path: str, default_path: str, type: str = '') -> str:
    """Verify the validity of the given path and take the default value if not defined

    :param path: The given path 
    :type path: str
    :param default_path: The default path to take if 'path' is not defined
    :type default_path: str
    :param type: The path type. Can be equal to 'file', 'dir' or '', defaults to ''
    :type type: str, optional
    :raises FileNotFoundError: If the given type is equal to 'file' and the path does not exist
    :raises NotADirectoryError: If the given type is equal to 'dir' and the path does not exist
    :raises Exception: If the given type is equal to '' and the path does not exist
    :return: The validated path
    :rtype: str
    """
    if not path:
        return default_path
    if type == 'file':
        if os.path.isfile(path):
            return path
        else:
            raise FileNotFoundError(path)
    elif type == 'dir':
        if os.path.isdir(path):
            return path
        else:
            raise NotADirectoryError(path)
    else:
        if os.path.exists(path):   
            return path
        else:
            raise Exception(f'The argument "{path}" is not a valid path')

def get_args(defaults_args: dict) -> argparse.Namespace:
    """Parse and get the different program arguments 

    :param defaults_args: The dictionnary with the defaults args
    :type defaults_args: dict
    :return: The structure from argparse module with the arguments extracted inside
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(description="Main script to launch Chronic Kidney Disease process")
    
    # Set the needed parameters
    parser.add_argument('--config-path', nargs='?', type=str, metavar='file path : str', default=defaults_args["config_path"], help=f'Path to the config you want to use. By default his param is equal to "{defaults_args["config_path"]}"')

    # Parse the inline command
    args = parser.parse_args()

    # Validate the arguments values
    args.config_path = valid_path(args.config_path, defaults_args["config_path"], type='file')
        
    return args

########################################################################################


def main(config: dict[str, str]):
    """Main function to launch process

    :param config: The config used for the process
    :type config: dict[str, str]
    :raises Exception: If the dataset is not a csv or a arff file, an Exception is raised
    """
    
    path_to_dataset_file = os.path.join(ROOT, config['dataset_dir'], config['dataset_file'])

    if path_to_dataset_file.endswith('.arff'):
        data = arff.loadarff(path_to_dataset_file) # Work with advices found here : https://stackoverflow.com/questions/62653514/open-an-arff-file-with-scipy-io
        df = pd.DataFrame(data[0])
    elif path_to_dataset_file.endswith('.csv'):
        df = pd.read_csv(path_to_dataset_file)
    else:
        raise Exception(f'The dataset file format is not valid. Valid formats : (.arff, .csv)')

    print(f"Dataset Loaded, Shape : {df.shape}")

    # Decode byte to utf8
    str_df = df.select_dtypes([object])
    str_df = str_df.stack().str.decode('utf-8').unstack()
    for col in str_df:
        df[col] = str_df[col]

    # Drop empty values
    if config.get("dropna"):
        df = df.dropna()

    # Features Encoding
    target_col = config.get("target_col")
    X = df[df.columns.difference([target_col])]
    y = df[target_col]

    categorical_cols = [col for col in X.columns if X[col].dtype == 'object']
    numerical_cols = [col for col in X.columns if X[col].dtype != 'object']
    print(f"Number of Categorical Features : {len(categorical_cols)}, Number of Numerical Features : {len(numerical_cols)}")

    X = pd.get_dummies(X, columns=categorical_cols)
    y = pd.Categorical(y)
    
    print(f"After Categorical Feature Encoding, Number of Training Features {X.shape[0]}")

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y.codes, test_size=config.get("test_size", 0.25), random_state=config.get("random_seed", 42))

    # Training
    clf = xgb.XGBClassifier().fit(X_train, y_train)

    print(f"Model Trained")

    # Predictions on Test set
    y_pred = clf.predict(X_test)

    # Save Model and evaluation results (Conf matrix, classif report and training infos) in a unique training directory
    dt = datetime.datetime.now().strftime("%Y%m%d%H%M") 
    save_directory = os.path.join(ROOT, config.get("train_dir"), dt)
    os.mkdir(save_directory)
    
    # Conf Matrix
    cm = confusion_matrix(y_test, clf.predict(X_test))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=y.categories)
    disp.plot() 
    disp.ax_.set_title("Confusion Matrix")
    disp.figure_.savefig(os.path.join(save_directory, "confusion_matrix.png"))
    
    # Classif report
    dict_report = classification_report(y_test, y_pred, digits=3, zero_division=0, output_dict=True)
    report = classification_report(y_test, y_pred, digits=3, zero_division=0, output_dict=False)
    with open(os.path.join(save_directory, "classification_report.txt"), 'w') as f:
        f.write(report)
        
    # I chose f1_macro_avg to evaluate because test set is uneven
    metric_name = "f1_macro_avg"
    metric_score = dict_report["macro avg"]['f1-score']
    pickle.dump(clf, open(os.path.join(save_directory, f"model_{metric_name}_{metric_score:.3f}.pkl"),"wb"))

    # Global infos
    infos = {
        "config"       : config,
        "metric_name"  : metric_name,
        "metric_score" : metric_score
    }

    with open(os.path.join(save_directory, "infos.json"), 'w') as outfile:
        json.dump(infos, outfile)

    print(f"Model trained and results on the Test set can be found here : {save_directory}")


if __name__ == "__main__":
    
    print('------------------------- Chronic Kidney Disease Process Start -------------------------')

    try:

        defaults_args = {
            "config_path"  : os.path.join(ROOT, 'config.json'),
        }

        args = get_args(defaults_args)

        with open(args.config_path) as f:
            config = json.load(f)

        main(config)
    except Exception as e:
        print(f'Error during process : {e.with_traceback()}')
    

    print('------------------------- Chronic Kidney Disease Process End -------------------------')

import numpy as np
import pandas as pd
import argparse 
from sklearn.impute import KNNImputer
from sklearn.feature_selection import RFE 
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

def preprocess_data(df: pd.DataFrame, rules: list) -> pd.DataFrame:
    """
    Preprocess the data according to the rules. Function takes two arguments: df and rules (as a list). 
    The function returns a new dataframe with the columns representing logical values of the rules listed in the
    rules list. The function also imputes the missing values in the dataframe. If the number of missing values is less than 5% or the number of rows is less than 50,
    the mode is used to fill in the missing values. Otherwise, KNN is used to fill in the missing values.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to be preprocessed.
    rules : list
        The list of rules to be applied to the dataframe.

    Returns
    -------
    pandas.DataFrame
        The new dataframe with the columns representing logical values of the rules listed in the rules list.
    """
    df = pd.DataFrame(df)
    cleaned_rules = [i.replace(" => donor_is_old", "") for i in rules]

    #Missing values imputation
    #If the number of missing values is less than 5% or the number of rows is less than 50, I will use the mode to fill in the missing values
    #Otherwise, I will use KNN to fill in the missing values
    if (df.isna().sum().sum()/df.size) < 0.05 or len(df) < 50:
        df = df.fillna(df.mode().iloc[0])
    else:
        imputer = KNNImputer(n_neighbors=3)
        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        df = df.astype("bool")
    
    #Applying the rules
    result_df = pd.DataFrame(index=df.index)  #new dataframe to store the results of applying the rules
    
    for rule in cleaned_rules:
        conditions = rule.split(" AND ")
        condition_series = pd.Series(True, index=df.index)
        for condition in conditions:
            if condition.startswith("NOT "):
                column = condition[4:]  #deleting "NOT " form the beginning of the string
                condition_series &= ~df[column]
            else:
                column = condition
                condition_series &= df[column]
        result_df[rule] = condition_series
        
    result_df["donor_is_old"] = df["donor_is_old"]  
    
    return result_df

def feature_selection(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select the most meaningful features. Function takes one argument: df and returns a new dataframe with the selected features.
    First, the function selects the optimal number of features using crossvalidation on Recursive Feature Elimination (RFE) with Logistic Regression.
    Then, the function selects the features using the optimal number of features.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to select the features from.

    Returns
    -------
    pandas.DataFrame
        The new dataframe with the selected features.
    """
    
    X = df.drop(columns=["donor_is_old"])
    y = df["donor_is_old"]

    max_features = X.shape[1]  #maximal number of features to test
    cv_folds = 5

    model = LogisticRegression(random_state=0, penalty = "l1", solver = "liblinear") 

    feature_counts = range(1, max_features + 1)
    cv_scores = []

    for n_features in feature_counts:
        rfe = RFE(model, n_features_to_select=n_features)
        scores = cross_val_score(rfe, X, y, cv=cv_folds, scoring="accuracy")  #accuracy chosen due to balanced classes
        cv_scores.append(np.mean(scores)) 

    optimal_features = feature_counts[np.argmax(cv_scores)]
    
    rfe = RFE(model, n_features_to_select=optimal_features)
    rfe.fit(X, y)
    selected_features = X.columns[rfe.support_]
    df = df[selected_features]

    result_df = df

    return result_df

def load_file(file1: str, file2: str, result_file: str) -> None:
    """ 
    Load the data from the file1 and the rules from the file2 and save the result in the result_file. Function takes
    three arguments: file1, file2, result_file. Preferably these arguments should be the paths to the files or the names of the files 
    in the same directory as the script. If the result_file does not exist, it will be created as a new .txt file.

    Parameters
    ----------
    file1 : str
        The path to the first file. (expected format: tsv or csv)
    file2 : str
        The path to the second file. (expected format: txt)
    result_file : str
        The path to the result file. (expected format: txt)

    Returns
    -------
    None
        This function does not return a value but writes the processed data to `result_file`.
    """
    data = pd.read_csv(file1, sep="\t")
    rules = []
    with open(file2, 'r', encoding='utf-8') as file:
        for line in file:
            rules.append(line.strip())  
    
    data = preprocess_data(data, rules)
    data = feature_selection(data)

    with open(result_file, "w", encoding="utf-8") as res_file:
        for i in range(len(data.columns)):
            res_file.write(data.columns[i])
            res_file.write("\n")

def main():
    #main function to pass the arguments to the load_file function.
    parser = argparse.ArgumentParser()
    
    parser.add_argument("file1")
    parser.add_argument("file2")
    parser.add_argument("res_file")

    args = parser.parse_args()
    
    load_file(args.file1, args.file2, args.res_file)

if __name__ == "__main__":
    main()

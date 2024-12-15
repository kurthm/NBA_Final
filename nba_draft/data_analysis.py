import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn import linear_model
import statsmodels.api as sm


def knn_analysis(dframe, n=3):
    """
    Perform k-Nearest Neighbors (KNN) analysis with and without college-related predictors.

    This function trains a k-Nearest Neighbors classifier using two different sets of feature columns:
    one with college-related predictors and one without. It calculates the accuracy and cross-validation 
    score (using F1 weighted score) for both sets of features.

    Args:
        dframe (pandas.DataFrame): The input dataframe containing player data, including the target variable 'Pk' (Draft Pick).
        n (int, optional): The number of neighbors to use for the KNN classifier (default is 3).

    Returns:
        list: A list containing the accuracy for both feature sets:
              [accuracy_without_college_predictors, accuracy_with_college_predictors].

    Raises:
        Exception: If there are fewer than 3 players at each draft position in inputted dataframe, cross validation will not be calculated.
    """
    
    # Define the columns to exclude for two different feature sets: 
    # one with college-related predictors and one without
    exclude_cols1 = ['Rk', 'Player', 'Tm', 'College', 'Pk', 'WinPct_College', 'SOS_College']
    exclude_cols2 = ['Rk', 'Player', 'Tm', 'College', 'Pk']
    cols = [exclude_cols1, exclude_cols2]

    # Fill missing values in the dataframe with 0 to avoid issues during training
    dframe = dframe.fillna(0)

    # Set the target variable as 'Pk' (Draft Pick)
    y = dframe["Pk"]
    
    # Initialize empty lists to store accuracy and cross-validation scores
    accuracies = []
    scores = []

    # Loop over the two feature sets (with and without college-related predictors)
    for i in cols:
        # Select the features by dropping the columns to be excluded
        X = dframe.drop(columns=i)

        # Split the data into training and testing sets (90% train, 10% test)
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

        # Initialize the KNN classifier with the number of neighbors specified by `n`
        knn = KNeighborsClassifier(n_neighbors=n)

        # Train the KNN model on the training data
        knn.fit(x_train, y_train)

        # Make predictions on the test set
        y_pred = knn.predict(x_test)

        # Calculate the accuracy of the model on the test set
        accuracy = accuracy_score(y_test, y_pred)

        # Perform 3-fold cross-validation and compute the F1 weighted score
        if dframe['Draft_Yr'].max() - dframe['Draft_Yr'].min() >= 3:
            cv_score = cross_val_score(knn, X, y, cv = 3, scoring='f1_weighted')
        else:
            raise Exception(f"Cross validation requires at least 3 athletes in each pick position. Expand range of draft years.")


        # Append the results (accuracy and cross-validation scores) for each feature set
        accuracies.append(accuracy)
        scores.append(cv_score)
    
    # Print the results for each feature set (with and without college predictors)
    print(f"Accuracy without college predictors: {accuracies[0]:.2f}\nCV Score: {scores[0].mean():.2f}")
    print(f"Accuracy with college predictors: {accuracies[1]:.2f}\nCV Score: {scores[1].mean():.2f}")

    # Return the accuracies for both feature sets
    return accuracies




def dtree_analysis(dframe, depth=15):
    """
    Perform Decision Tree analysis with and without college predictors, including cross-validation scores and classification reports.

    This function splits the data into two sets of features: one with college-related predictors and one without.
    It then trains a Decision Tree classifier on both sets, computes the accuracy, classification report, and performs cross-validation
    to evaluate the performance of the model with different feature sets.

    Args:
        dframe (pandas.DataFrame): The input data frame containing player data, including the target variable 'Pk' (Draft Pick).
        depth (int, optional): The maximum depth of the decision tree (default is 15).

    Returns:
        list: A list containing the accuracy for both feature sets:
              [accuracy_without_college_predictors, accuracy_with_college_predictors].

    Raises:
        Exception: If there are fewer than 3 players at each draft position in inputted dataframe, cross validation will not be calculated.
    """
    
    # Define the columns to exclude for two different feature sets: one with college-related predictors and one without
    exclude_cols1 = ['Rk', 'Player', 'Tm', 'College', 'Pk', 'WinPct_College', 'SOS_College']
    exclude_cols2 = ['Rk', 'Player', 'Tm', 'College', 'Pk']
    cols = [exclude_cols1, exclude_cols2]

    # Fill missing values in the dataframe with 0
    dframe = dframe.fillna(0)

    # Set the target variable as 'Pk' (Draft Pick)
    y = dframe["Pk"]
    
    # Initialize empty lists to store accuracy, cross-validation scores, and classification reports
    accuracies = []
    class_reps = []
    scores = []

    # Loop over the two feature sets (with and without college-related predictors)
    for i in cols:
        # Select the features by dropping the columns to be excluded
        X = dframe.drop(columns=i)

        # Split the data into training and testing sets (90% train, 10% test)
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

        # Initialize the Decision Tree model with a maximum depth of `depth`
        tree = DecisionTreeClassifier(random_state=42, max_depth=depth)

        # Train the model on the training set
        tree.fit(x_train, y_train)

        # Make predictions on the test set
        y_pred_tree = tree.predict(x_test)

        # Calculate the accuracy score for the model
        accuracy = accuracy_score(y_test, y_pred_tree)

        # Generate a classification report to evaluate model performance
        class_rep = classification_report(y_test, y_pred_tree, zero_division=1)

        # Perform 3-fold cross-validation and compute the F1 weighted score
        if dframe['Draft_Yr'].max() - dframe['Draft_Yr'].min() >= 3:
            cv_score = cross_val_score(tree, X, y, cv = 3, scoring='f1_weighted')
        else:
            raise Exception(f"Cross validation requires at least 3 athletes in each pick position. Expand range of draft years.")


        # Perform 3-fold cross-validation and compute the F1 weighted score
        cv_score = cross_val_score(tree, X, y, cv = 3, scoring='f1_weighted')

        # Store the results (accuracy, classification report, and cross-validation scores)
        accuracies.append(accuracy)
        class_reps.append(class_rep)
        scores.append(cv_score)
    
    # Print the results for each feature set (with and without college predictors)
    print(f"Without college predictors:\naccuracy: {accuracies[0]:.2f}\nCV Score: {scores[0].mean():.2f}\nClassification Report: {class_reps[0]}")
    print(f"With college predictors:\naccuracy: {accuracies[1]:.2f}\nCV Score: {scores[1].mean():.2f}\nClassification Report: {class_reps[1]}")

    # Return the accuracies of both feature sets
    return accuracies




def linreg_analysis(dframe):
    """
    Performs linear regression analysis to predict the draft pick (Pk) and provides model summaries
    with and without college-related predictors.

    This function trains a linear regression model using two different sets of predictors:
    1. The first feature set excludes college-related predictors such as 'WinPct_College' and 'SOS_College'.
    2. The second feature set excludes the 'College' and 'Pk' columns.

    The model summary (coefficients, p-values, R-squared, etc.) for both sets of predictors is returned.

    Args:
        dframe (pandas.DataFrame): The input data frame containing player data, including 'Pk' and 'College' columns.

    Returns:
        list: A list containing the OLS regression R-squared value for both feature sets:
              [summary_without_college_predictors, summary_with_college_predictors].
    """
    
    # Define columns to exclude for two different feature sets
    exclude_cols1 = ['Rk', 'Player', 'Tm', 'College', 'Pk', 'WinPct_College', 'SOS_College']
    exclude_cols2 = ['Rk', 'Player', 'Tm', 'College', 'Pk']
    cols = [exclude_cols1, exclude_cols2]

    # Fill missing values in the dataframe with 0
    dframe = dframe.fillna(0)

    # Set the target variable as 'Pk' (Draft Pick)
    y = dframe["Pk"]
    
    # Initialize an empty list to store model summaries for each feature set
    summaries = []

    # Loop over the two feature sets
    for i in cols:
        
        # Select the feature set by dropping the columns to be excluded
        X = dframe.drop(columns=i)

        # Add an intercept (constant) to the model
        x_with_intercept = sm.add_constant(X)

        # Fit the OLS model using statsmodels
        model = sm.OLS(y, x_with_intercept).fit()

        # Get the model summary (coefficients, R-squared, p-values, etc.)
        mod_summary = model.rsquared

        # Append the summary to the list
        summaries.append(mod_summary)

    # # Print the summary for both feature sets
    # print(f"Summary without college predictors: {summaries[0]}")
    # print(f"Summary with college predictors: {summaries[1]}")

    # Return the list of model summaries
    return summaries

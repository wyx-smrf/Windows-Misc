params = {'dataframe':       df, 
          'cat_col_1':       'categorical_column from df', 
          'cat_col_2':       'categorical_column from df', 
          'num_col':         'numerical_column from df', 
          'impute_strategy': 'mean, median, etc.'}

def conditional_fillna(dataframe, cat_col_1, cat_col_2, num_col, impute_strategy):
    """Impute the missing values in a column using central tendency. 
       The values of the central tendency is obtained by the statistic of the grouped data

    Args:
        dataframe (DataFrame): Two-dimensional data structure stored in a tabular format.
        cat_col_1 (Series): Categorical column to be used in data grouping
        cat_col_2 (Series): Categorical column to be used in data grouping
        num_col (Series): Numerical Column to be used to compute for numeric values among different data groups
        impute_strategy (String): The method on which to impute the missing values

    Returns:
        DataFrame: Imputed missing values based on the statistic from other column/s
    """
    
    # Create a grouping for the dataset
    grouped_data = dataframe.groupby([cat_col_1, cat_col_2])[num_col]
    
    # Fill missing values in a column using the defined impute strategy
    dataframe[num_col].fillna(grouped_data.transform(impute_strategy), inplace=True)
    
    # Optional: Covert the imputed numerical values into an integer
    dataframe[num_col] = dataframe[num_col].round(0).astype(int)

    return dataframe
    
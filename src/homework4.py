import sys

# import pandas as pd UNCOMMENT AFTER COMMIT


def response_type_check(df, response):
    # Determine if response is continuous or boolean
    unique_responses = df[response].nunique()

    if unique_responses == 2:
        boolean_check = True
        print("Response is boolean\n")
    else:
        boolean_check = False
        print("Response is continuous\n")

    return boolean_check


def cat_cont_check(df, predictor):
    # Determine if the predictor is cat/cont
    if isinstance(df[predictor][0], float):
        cat_check = False
        print("Predictor {} is categorical\n".format(predictor))
    else:
        cat_check = True
        print("Predictor {} is continuous\n".format(predictor))

    return cat_check


def main():
    # Given a pandas dataframe
    # Contains both a response and predictors
    """
    df = pd.read_csv(
        "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
    ) UNCOMMENT AFTER COMMIT"""

    # Given a list of predictors and the response columns
    predictors = [
        "pclass",
        "sex",
        "age",
        "sibsp",
        "parch",
    ]  # NOTE TO SELF - Test with more predictors in the future
    # response = "survived" UNCOMMENT AFTER COMMIT

    # Determine if response is continuous or boolean
    # boolean_check = response_type_check(df, response) UNCOMMENT AFTER COMMIT

    # Loop through each predictor column
    for predictor in predictors:
        print("hello world")  # filler so commit works

        # Determine if the predictor is cat/cont
        # cat_check = cat_cont_check(df, predictor) UNCOMMENT AFTER COMMIT


if __name__ == "__main__":
    sys.exit(main())

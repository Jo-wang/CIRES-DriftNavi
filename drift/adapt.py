
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

def adapt_data_tune(df, attr):
    ros = RandomOverSampler(sampling_strategy='auto', random_state=42)
    X_res, y_res = ros.fit_resample(df[attr], df["label"])
    return X_res, y_res

def resample_data(df1, df2, attr):
    ros = RandomOverSampler(sampling_strategy='auto', random_state=42)
    X_res, y_res = ros.fit_resample(df1[attr], df1["label"])
    return X_res, y_res

def adapt_data_retrain(df1, df2, attr):
    X_res, y_res = resample_data(df1, df2, attr)
    return X_res, y_res

def smote(df, attr):
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(df[attr], df["label"])
    return X_res, y_res

def adapt_data(df, attr, op):
    if op == "tune":
        return adapt_data_tune(df, attr)
    elif op == "retrain":
        return adapt_data_retrain(df, attr)
    elif op == "smote":
        return smote(df, attr)
    else:
        raise ValueError(f"Invalid operation: {op}")
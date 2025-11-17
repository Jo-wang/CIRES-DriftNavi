import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from sklearn.svm import SVC, SVR
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor


def detect_text_columns(df, text_length_threshold=50):
    text_columns = []
    for col in df.columns:
        if df[col].dtype == 'object':  # Check if the column is of type 'object'
            # Check if the maximum length of text in the column exceeds the threshold
            if df[col].apply(lambda x: len(str(x)) if pd.notnull(x) else 0).mean() > text_length_threshold:
                text_columns.append(col)
    return text_columns


class DatasetEval:
    def __init__(self, data, label, ratio=0.5, task_type='Classification', fixed=True, text_length_threshold=50,
                 sensitive_attribute=None, model_type='SVM'):
        self.data = data
        self.target = label
        self.y_test = None
        self.y_train = None
        self.X_test = None
        self.X_train = None
        self.task_type = task_type
        self.sensitive_attribute = sensitive_attribute

        if data is None:
            raise ValueError('The dataframe is None.')

        if label not in data.columns:
            raise ValueError('The label is not in the dataset.')

        if data[label].dtype in ['float64', 'float32'] and task_type == 'Classification':
            raise ValueError('The target attribute is continuous (float) but the task is set to classification. '
                             'Consider binning the target or setting the task to regression.')

        if data[label].dtype == 'object' or data[label].dtype.name == 'bool' or data[label].dtype.name == 'category':
            if task_type == 'Regression':
                raise TypeError('The target attribute is categorical and cannot be used for regression task.')

        if task_type == 'Classification':
            self.label_encoder = LabelEncoder()
            self.label = self.label_encoder.fit_transform(data[label])
        else:
            self.label = data[label]

        long_text_columns = detect_text_columns(data, text_length_threshold)
        self.samples = data.drop(columns=long_text_columns + [label])

        self.pipline = self.preprocess(self.samples, model_type)
        self.split_dataset(data, label, ratio, fixed)

    def preprocess(self, data, model_type):
        d_copy = data.copy()
        d_copy = d_copy.drop(self.sensitive_attribute, axis=1)
        datetime_features = d_copy.select_dtypes(include=['datetime64']).columns
        for col in datetime_features:
            d_copy[col + '_year'] = d_copy[col].dt.year
            d_copy[col + '_month'] = d_copy[col].dt.month
            d_copy[col + '_day'] = d_copy[col].dt.day
            d_copy[col + '_hour'] = d_copy[col].dt.hour
            d_copy[col + '_weekday'] = d_copy[col].dt.weekday
            d_copy = d_copy.drop(columns=[col])

        numeric_features = d_copy.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = d_copy.select_dtypes(include=['object']).columns
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features),
            ],
            remainder='drop'
        )

        # Choose model based on task and user input
        if self.task_type == 'Classification':
            if model_type == 'Logistic':
                model = LogisticRegression(max_iter=500)
            elif model_type == 'SVM':
                model = SVC(max_iter=500)
            elif model_type == 'gradient_boosting':
                model = GradientBoostingClassifier()
            elif model_type == 'MLP':
                model = MLPClassifier(max_iter=500)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        else:
            if model_type == 'SVM':
                model = SVR()
            elif model_type == 'gradient_boosting':
                model = GradientBoostingRegressor()
            elif model_type == 'MLP':
                model = MLPRegressor(max_iter=500)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

        return Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

    def split_dataset(self, data, label, ratio, fixed):
        if fixed:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                data.drop(label, axis=1), self.label, test_size=ratio, random_state=42)
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                data.drop(label, axis=1), self.label, test_size=ratio)

    def train_and_test(self):
        # Train the model
        self.pipline.fit(self.X_train, self.y_train)

        if self.task_type == 'Classification':
            # Standard accuracy evaluation
            y_pred = self.pipline.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            res = f"Accuracy: {accuracy:.4f}"
            print(res)

            # If a sensitive attribute is defined, calculate disparity
            if self.sensitive_attribute:
                disparity_df = self.calculate_disparity_for_classification(y_pred)
                print("\nDisparity Results:")
                for tab in disparity_df:
                    print(tab)
                return res, disparity_df
            else:
                return res, []

        else:
            # For regression
            y_pred = self.pipline.predict(self.X_test)
            mse = mean_absolute_error(self.y_test, y_pred)
            mse = f"Model Mean absolute Error: {mse:.4f}"
            print(mse)

            # If a sensitive attribute is defined, calculate MSE disparity
            if self.sensitive_attribute:
                disparity_df = self.calculate_disparity_for_regression(y_pred)
                print("\nMSE Disparity Results:")
                for tab in disparity_df:
                    print(tab)
                return mse, disparity_df
            else:
                return mse, []

    def calculate_disparity_for_regression(self, y_pred):
        """Calculate disparity (fairness) for regression results using MSE disparity."""

        # Add the actual and predicted values to the test set
        result_df = self.X_test.copy()
        result_df['true'] = self.y_test
        result_df['pred'] = y_pred

        data_frames = []

        for sensi_attr in self.sensitive_attribute:
            disparity_scores = ['Disparity Score']
            group_counts = ['Group Count (for Test)']

            # Calculate MSE for each sensitive group
            mae_per_group = result_df.groupby(sensi_attr).apply(
                lambda group: mean_absolute_error(group['true'], group['pred']))
            mae_per_group = mae_per_group.rename('MSE')
            count_per_group = result_df.groupby(sensi_attr).size()
            # Calculate disparity score (max - min MSE across groups)
            max_mse = mae_per_group.max()
            min_mse = mae_per_group.min()
            disparity_score = max_mse - min_mse
            disparity_scores.append(disparity_score)

            # Append the results to the final DataFrame
            frame = mae_per_group.reset_index()
            frame['Group Count (for Test)'] = [int(val) for val in count_per_group.values]
            disparity_row = pd.Series(disparity_scores + [None], index=frame.columns)
            frame.loc[len(frame)] = disparity_row
            data_frames.append(frame)

        return data_frames

    def calculate_disparity_for_classification(self, y_pred):
        """Calculate disparity (fairness) for multi-class classification results with group count."""

        # Add the predicted class to the test set
        result_df = self.X_test.copy()
        result_df['predicted_class'] = self.label_encoder.inverse_transform(y_pred)

        # Get unique classes from the predicted labels
        unique_classes = result_df['predicted_class'].unique()

        disparity_results = []
        data_frames = []

        for sensi_attr in self.sensitive_attribute:
            disparity_scores = ['Disparity Score']
            group_counts = ['Group Count (for Test)']

            for cls in unique_classes:
                # Calculate the prediction rate for each group
                cls_df = result_df[result_df['predicted_class'] == cls]
                parity_df = cls_df.groupby(sensi_attr).size() / result_df.groupby(sensi_attr).size()
                parity_df = parity_df.rename(f'{cls}')
                disparity_results.append(parity_df)

                # Calculate disparity score (max - min prediction rate) for each class
                max_rate = parity_df.max()
                min_rate = parity_df.min()
                disparity_score = max_rate - min_rate
                disparity_scores.append(disparity_score)

            # Get the count of samples in each group
            count_per_group = result_df.groupby(sensi_attr).size()

            # Create final DataFrame for this sensitive attribute
            frame = pd.concat(disparity_results, axis=1).reset_index()

            # Add the group count to the DataFrame
            frame['Group Count (for Test)'] = [int(val) for val in count_per_group.values]

            # Add disparity score as a new row at the end
            disparity_row = pd.Series(disparity_scores + [None], index=frame.columns)
            frame.loc[len(frame)] = disparity_row

            # Append the frame to the results
            data_frames.append(frame)
            disparity_results = []  # Reset for the next sensitive attribute

        return data_frames

    
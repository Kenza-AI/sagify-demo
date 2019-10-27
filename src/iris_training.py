import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def train(input_file_path):
    df = pd.read_csv(
        input_file_path,
        header=None,
        names=['feature1', 'feature2', 'feature3', 'feature4', 'label']
    )

    df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)

    features_train = df_train[['feature1', 'feature2', 'feature3', 'feature4']].values
    labels_train = df_train[['label']].values.ravel()

    features_test = df_test[['feature1', 'feature2', 'feature3', 'feature4']].values
    labels_test = df_test[['label']].values.ravel()

    clf = SVC(gamma='auto')
    clf.fit(features_train, labels_train)

    test_predictions = clf.predict(features_test)
    accuracy = accuracy_score(labels_test, test_predictions)

    return clf, accuracy


if __name__ == '__main__':
    _, accuracy = train('../data/iris.data')
    print(accuracy)

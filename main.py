from classification import run_classification
from regression import run_regression
from clustering import run_clustering


def main():
    csv_path = "ObesityDataSet.csv"

    run_classification(csv_path)
    run_regression(csv_path)
    run_clustering(csv_path)


if __name__ == "__main__":
    main()

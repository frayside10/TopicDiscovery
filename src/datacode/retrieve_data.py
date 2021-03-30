from sklearn.datasets import load_files


def pull_data(path):
    """Load data from the raw files using sklearn load_files"""
    data = load_files(path)
    return data



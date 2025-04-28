import pickle

def save_index(indexer, path):
    with open(path, "wb") as f:
        pickle.dump(indexer, f)

def load_index(path):
    with open(path, "rb") as f:
        return pickle.load(f)

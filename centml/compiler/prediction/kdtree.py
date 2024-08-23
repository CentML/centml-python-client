import ast
import csv
import logging

from sklearn.neighbors import KDTree  # type: ignore

from centml.compiler.config import settings

_tree_db = None


class KDTreeWithValues:
    def __init__(self, points=None, values=None):
        self.points = points if points else []
        self.values = values if values else []
        if self.points:
            self.tree = KDTree(self.points)
        else:
            self.tree = None

    def add(self, point, value):
        self.points.append(point)
        self.values.append(value)
        self.tree = KDTree(self.points)

    def query(self, point):
        if self.tree is None:
            return None, None

        dist, idx = self.tree.query([point], k=1)
        return dist[0][0], self.values[idx[0][0]]


class TreeDB:
    def __init__(self, data_csv):
        self.db = {}
        self._populate_db(data_csv)

    def get(self, key, inp):
        if key not in self.db:
            logging.getLogger(__name__).warning(f"Key {key} not found in database")
            return float('-inf')
            # TODO: Handle the case of unfound keys better. For now, return -inf to indicate something went wrong.
            # Ideally, we shouldn't throw away a whole prediction because of one possibly insignificant node.

        _, val = self.db[key].query(inp)
        return val

    def _add_from_db(self, key, points, values):
        self.db[key] = KDTreeWithValues(points, values)

    def _populate_db(self, data_csv):
        with open(data_csv, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    key = (row['op'], int(row['dim']), row['inp_dtypes'], row['out_dtypes'], row['gpu'])
                    points = ast.literal_eval(row['points'])
                    values = ast.literal_eval(row['values'])
                    self._add_from_db(key, points, values)
                except ValueError as e:
                    logging.getLogger(__name__).exception(f"Error parsing row: {row}\n{e}")


def get_tree_db():
    global _tree_db
    if _tree_db is None:
        _tree_db = TreeDB(settings.CENTML_PREDICTION_DATA_FILE)
    return _tree_db

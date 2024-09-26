import numpy as np
import os
import itertools


def category_encode_as_int(variable):
    """
    Encodes a categorical variable as integers. 
    """
    return variable.astype('category').cat.codes


def dist_eucl(x1, x2, y1, y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)


def rmv_dupl(df, var):  # TODO document
    xp = df[var]
    idx_retained = np.where(np.abs(np.diff(xp)) > 0)
    df = df.iloc[idx_retained]
    return df


def save_obj_html(obj, directory, filename):
    """
    Saves an object as a html file.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    path = directory + '/' + filename + '.html'

    obj.save(path)


def retain_change_only(x: list) -> list:
    """
    Transforms a list of values to a list of values where all adjacent repetitions are merged into one.
    
    Example:
    > retain_change_only([0, 0, 1, 0, 2, 2])
    Output: [0, 1, 0, 2]
    """
    return [k for k, _g in itertools.groupby(x)]


def unique_lists(list_of_lists: list) -> list:
    """
    Return unique lists from a list of lists. Considers the order.
    """
    return [list(x) for x in set(tuple(x) for x in list_of_lists)]


def unique_lists(list_of_lists: list) -> list:
    """
    Return unique lists from a list of lists. Considers the order.
    """
    return [list(x) for x in set(tuple(x) for x in list_of_lists)]


def all_adjacent_pairs(list_of_lists: list) -> list:
    list_of_lists = list_subset_min_length(list_of_lists, min_length=2)
    list_of_pairs = [adjacent_pairs_all(x) for x in list_of_lists]
    adjacent_pairs = unique_tuples(flatten_list(list_of_pairs))

    return adjacent_pairs


def list_subset_min_length(list_of_lists: list, min_length: int) -> list:
    """
    Returns only lists that have a minimal length of `min_length`.
    """
    return list(np.array(list_of_lists, dtype='object')[np.array([len(x) for x in list_of_lists]) >= min_length])


def adjacent_pairs_all(nodes_list):
    """
    Returns all adjacent 2 pairs in a list. Considers the order.
    """
    return [(x, y) for x, y in zip(nodes_list, nodes_list[1:])]


def unique_tuples(tuple_list: list) -> list:
    """
    Returns unique tuples from a list of tuples.
    """
    return list(set(tuple_list))


def flatten_list(list_of_lists: list) -> list:
    """
    Flattens a nested list.
    """
    return [item for sublist in list_of_lists for item in sublist]


def list_in_list(list1, list2):
    def trans(string):
        string = ' , '.join(map(str, string))
        string = ' ' + string + ' '
        return string
    list1 = trans(list1)
    list2 = trans(list2)

    return list1 in list2


def to_array(trips):
    s_trips = np.array(trips.groupby('TripLogId').apply(lambda x:
                                                        np.array(list([np.array(x['Latitude']),
                                                                       np.array(x['Longitude']), np.array(x["Altitude"])]))))
    return s_trips


def subset_keys_dict(dictionary: dict, keys, level=0) -> dict:
    """
    Subset a dictionary to specific keys.
    """
    if not isinstance(keys, list):
        keys = [keys]
    if level == 0:
        dictionary = dict((k0, dictionary[k0])
                          for k0 in keys if k0 in dictionary)
    elif level == 1:
        dictionary = dict(
            (k0, dictionary[k0][k1]) for k0 in dictionary for k1 in keys if k1 in dictionary[k0])
    else:
        dictionary = dict()

    return dictionary

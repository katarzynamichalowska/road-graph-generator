from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
import re
import geopy as gp
import datetime
import produceGraph.data_manipulation as dm
from produceGraph.gpx_interpolate import interpolate_points_latlon
from scipy.spatial import KDTree
import random
from tqdm import tqdm
import networkx as nx
from produceGraph.DBA_multivariate import performDBA, DTW
import logging
logger = logging.getLogger('producegraph')
import sys


def find_adjacent_nodes(trips_nodelists):

    trips_nodelists_unique = dm.unique_lists(trips_nodelists)
    adjacent_nodes = dm.all_adjacent_pairs(trips_nodelists_unique)
    return adjacent_nodes


def filter_short(df, groupby, minlength):
    df_grouped = df.groupby([groupby])
    ids = list(df_grouped.size()[df_grouped.size() >= minlength].index)
    df = df[df[groupby].isin(ids)]

    return df.copy()


def format_varnames(df):
    """ Part of preprocessing. Renames some variables and columns for data processing. """
    df.columns = [re.sub('Trip\.', '', x) for x in df.columns]
    df.columns = [re.sub('Coordinate\.', '', x) for x in df.columns]
    df.rename(columns={'TimeStamp': 'Timestamp'}, inplace=True, copy=True)


def format_timevars(df):
    """ Part of preprocessing. Convert time to datetime. """
    for c in df.columns[df.columns.str.contains('time', case=False)]:
        df.loc[:, c] = pd.to_datetime(df.loc[:, c], utc=True)


def remove_incorrect_rows(df):
    """ Part of preprocessing. Removes faulty pings at latitude or longitude 0 . """
    df = df.loc[(df['Latitude'] != 0) & (df['Longitude'] != 0)]


def remove_idle_vehicles(df):
    """ Part of preprocessing. Removes idle time. """
    return df[df['Speed'] != 0].copy()


def latlon_to_xy_array(lat, lon, proj_info):
    """
    """
    x = coord_to_metres(lon, proj_info['origin'][1], proj_info['delta_x'])
    y = coord_to_metres(lat, proj_info['origin'][0], proj_info['delta_y'])

    return x, y


def latlon_to_xy(input_df, proj_info=None, latlon_varnames: list = ['Latitude', 'Longitude']):
    """
    For the latitude, longitude columns in the input data frame, return new series
    that contain x,y changes
    """
    if proj_info is None:
        origin = calculate_origin(input_df)
        start = gp.Point(origin)

        # We now set up a local Cartesian coordinate system
        meters_to_angle = 1
        d = gp.distance.geodesic(meters=meters_to_angle)
        dist = d.destination(point=start, bearing=90)
        # 1 metre in "positive" longitude direction as vector in latlon space
        delta_x = abs(start.longitude - dist.longitude)
        dist = d.destination(point=start, bearing=0)
        # 1 metre in "positive" latitude direciton as vector in latlon space
        delta_y = abs(start.latitude - dist.latitude)
        proj_info = {'delta_x': delta_x, 'delta_y': delta_y, 'origin': origin}

        logger.debug(
            f"{meters_to_angle}m corresponds to {delta_x:.3e} deg longitude and {delta_y:.3e} deg latitude")

    x, y = latlon_to_xy_array(
        input_df[latlon_varnames[0]], input_df[latlon_varnames[1]], proj_info)

    return (x, y), proj_info


def calculate_origin(latlon_df):
    """ Find a suitable point (in geocoords) to use as origin, for simplification elsewhere. 
    
    :return (lat, lon) pair defining the origin."""
    return (latlon_df['Latitude'].min(), latlon_df['Longitude'].min())


def add_meter_columns(input_df, proj_info=None, load=False, dump=False, latlon_varnames=['Latitude', 'Longitude']):
    """
    Add columns representing meter positions to match stored geocoordinate positions.

    :param input_df Original dataframe.
    :param proj_info For converting latlong to xy. If not given this object will be created.
    :param load if True, columns are added for load positions, named 'load_x' and 'load_y' using 'LoadLongitude', 'LoadLatitude'
    :param dump if True, columns are added for dump positions, named 'dump_x' and 'dump_y' using 'DumpLongitude', 'DumpLatitude'
    :return (Resulting data frame with new positions in 'x', 'y', columns, proj_info)
    """
    (x, y), proj_info = latlon_to_xy(input_df, proj_info=proj_info,
                                     latlon_varnames=latlon_varnames)

    # To make things easier to read below
    delta_x = proj_info["delta_x"]
    delta_y = proj_info["delta_y"]
    origin = proj_info["origin"]

    df = input_df.copy()
    df['x'] = x
    df['y'] = y

    if load:
        df['load_x'] = ((input_df['LoadLongitude'].to_numpy() -
                        origin[1]) / delta_x).astype(int)
        df['load_y'] = ((input_df['LoadLatitude'].to_numpy() -
                        origin[0]) / delta_y).astype(int)
    if dump:
        df['dump_x'] = ((input_df['DumpLongitude'].to_numpy() -
                        origin[1]) / delta_x).astype(int)
        df['dump_y'] = ((input_df['DumpLatitude'].to_numpy() -
                        origin[0]) / delta_y).astype(int)

    return df.copy(), proj_info


def coord_to_metres(coord, origin, delta):
    """
    Translate from latitude-longitude pairs into (x,y) coordinates in Cartesian system.

    :param coord Coordinate array
    :param origin Geo-coordinate representation of (0,0)
    :param delta Offset in geo-coordinates corresponding to one unit in Cartesian system

    :return Cartesian coordinate array.
    """
    if isinstance(np.array(coord), np.ndarray) == False:
        coord = coord.to_numpy()
    return ((coord - origin) / delta).astype(int)  # TODO why is this an int???


def divide_trips(df, thr_mins, thr_angle, thr_dist=None):
    """
    Divide trips into multiple shorter trips.
    :param thr_mins: divide into two trips if the difference in time between pings is larger than the threshold (in mins).
    :param thr_angle: divide into two trips if the difference in the angle between pings is larger than the threshold.
    :return: new trip ids.
    """

    def _eucl(x1, y1, x2, y2):
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def _new_ids(df, cumsum_var, connection):
        series_ids = df['TripLogId']
        series_nrs = df[cumsum_var].cumsum().astype('str')
        new = series_ids + connection + series_nrs
        return new

    def _apply_new_ids(df, cumsum_var, connection):
        trip_ids = np.array(df.groupby('TripLogId').apply(lambda x: _new_ids(x, cumsum_var, connection)
                                                          if x[cumsum_var].sum() > 0 else x['TripLogId']))
        return trip_ids

    df = df.copy()
    df['large_timedelta'] = np.array(
        df['timestamp_delta'] > datetime.timedelta(minutes=thr_mins))
    df['TripLogId'] = _apply_new_ids(df, 'large_timedelta', connection='_t')
    if 'course_delta' in df.columns:
        df['large_coursedelta'] = np.array(abs(df['course_delta']) > thr_angle)
        df['TripLogId'] = _apply_new_ids(
            df, 'large_coursedelta', connection='_c')
        df['course_delta'] = change_first_in_group(
            df=df, by='TripLogId', variable='course_delta', first_val=0)
    if thr_dist is not None:
        df['x_prev'] = df.groupby('TripLogId')['x'].shift(1)
        df['y_prev'] = df.groupby('TripLogId')['y'].shift(1)
        df['EuclideanDistance'] = df.apply(lambda row: _eucl(
            row['x_prev'], row['y_prev'], row['x'], row['y']) if not pd.isnull(row['x_prev']) else 0, axis=1)
        df = df.drop(columns=['x_prev', 'y_prev'])
        df['DistanceDiff'] = df.groupby('TripLogId')['Distance'].diff().abs()
        df['large_distdiff'] = df['DistanceDiff'] > thr_dist
        df['TripLogId'] = _apply_new_ids(df, 'large_distdiff', connection='_d')

    return df['TripLogId']


def add_features(df, features=['timestamp_s', 'timestamp_delta', 'timestamp_delta_s', 'course_delta', 'dist_eucl_m']):
    """ 
    Adds features to the dataframe as specified by the features list.

    :param features The features to add. Allowed values: timestamp_delta, timestamp_delta_s, course_delta, timestamp_diff_from_min, timestamp_s, dist_eucl_m
    """
    if ('timestamp_delta' in features) | ('timestamp_delta_s' in features):
        df['timestamp_delta'] = df.Timestamp.diff()
        df['timestamp_delta'] = change_first_in_group(
            df=df, by='TripLogId', variable='timestamp_delta', first_val=datetime.timedelta(0))
    if ('timestamp_delta_s' in features):
        df['timestamp_delta_s'] = list(
            map(lambda x: x.seconds, df['timestamp_delta']))
    if ('course_delta' in features) & ('Course' in df.columns):
        df['course_delta'] = df.Course.diff()
        df['course_delta'] = change_first_in_group(
            df=df, by='TripLogId', variable='course_delta', first_val=0)
    if ('timestamp_diff_from_min' in features) | ('timestamp_s' in features):
        df['timestamp_diff_from_min'] = df.Timestamp - df.Timestamp.min()
    if ('timestamp_s' in features):
        df['timestamp_s'] = list(
            map(lambda x: x.item()/1000000000, df['timestamp_diff_from_min'].to_numpy()))
    if ('dist_eucl_m' in features):
        df['dist_eucl_m'] = dm.dist_eucl_consecutive(df)

    return df


def change_first_in_group(df, by, variable, first_val=0) -> np.array:
    """
    Changes the first value in each group in the dataframe.
    """
    chunk_ids = dm.category_encode_as_int(df[by])
    new_chunk = chunk_ids.astype('float').diff() != 0
    new_chunk[0] = True
    df.loc[new_chunk, variable] = first_val

    return np.array(df[variable])


def interpolate_trips(df, proj_info, spline_deg: int, resolution_m: float, add_vars=[]):
    latlon_interp = df.groupby('TripLogId').apply(lambda x: interpolate_points_latlon(x, deg=spline_deg, res=resolution_m,
                                                                                      add_vars=add_vars))
    latlon_interp = add_meter_columns(latlon_interp, proj_info=proj_info)[0]
    latlon_interp = latlon_interp.reset_index(level=1, drop=True).reset_index()
    return latlon_interp


def compute_dist_matrix(df1_ini, df2_ini, max_distance):
    """
    Computes a matrix of distances from points in df1 to points in df2.
    Returns only those that are at least within max_distance.
    The result is in the 'dictionary of keys' format, where the multikey is iloc of df1 and df2.
    """
    df1 = df1_ini.copy()
    df2 = df2_ini.copy()
    tree1 = df_to_kdtree(df=df1, columns=["x", "y"])
    tree2 = df_to_kdtree(df=df2, columns=["x", "y"])
    mx = dict(tree2.sparse_distance_matrix(
        tree1, max_distance=max_distance, p=2))
    return mx


def df_to_kdtree(df: pd.DataFrame, columns):
    """
    Takes a df with two columns and returns them as a tree object.
    """
    tuples = [(x, y) for x, y in zip(df[columns[0]], df[columns[1]])]
    tree = KDTree(tuples)

    return tree


def preprocess_mx_dist(mx_dist, trips, cluster_info):
    """
    :param mx_dist: matrix of distances as returned by compute_dist_matrix()
    :param trips: interpolated trips
    :param cluster_info: candidate intersections
    :return mx_df: Matrix of distances from cluster centres: cl_ids, cl_x, cl_y, cl_lat, cl_lon, cl_size
             to pings in trips: ping_idx, tripLogId, ping_lat, ping_lon, ping_x, ping_y
             distance: dist
    """
    mx_df = dist_matrix_to_df(mx_dist)
    mx_df = pd.merge(trips.rename({'Latitude': 'ping_lat', 'Longitude': 'ping_lon', 'x': 'ping_x', 'y': 'ping_y'}, axis=1),
                     mx_df, left_index=True, right_on='ping_idx', how='inner')
    cluster_info = cluster_info.rename(
        {'x': 'cl_x', 'y': 'cl_y', 'Latitude': 'cl_lat', 'Longitude': 'cl_lon'}, axis=1)
    mx_df = pd.merge(cluster_info, mx_df, right_on='cl_idx',
                     left_index=True, how='inner', suffixes=["_cluster", ""])
    mx_df = mx_df.sort_index()

    return mx_df


def dist_matrix_to_df(mx):
    mx_df = pd.DataFrame.from_dict(mx, orient='index')
    mx_df_index = pd.DataFrame(mx_df.index.to_list())
    mx_df = pd.concat([mx_df_index, mx_df.reset_index(drop=True)], axis=1)
    mx_df.columns = ['ping_idx', 'cl_idx', 'dist']
    return mx_df


def cluster_info_trips(cluster_info_ini, mx_df_ini, max_dist_from_cluster):
    ''' Adds a column giving the TripLogId of all trips that passes closer than "max_dist_from_cluster" from a node'''
    cluster_info = cluster_info_ini.copy()
    mx_df = mx_df_ini.copy()
    mx_df_close_to_cluster = mx_df[mx_df['dist'] <= max_dist_from_cluster]
    trip_ids_clusters = mx_df_close_to_cluster.groupby(
        'cl_idx')['TripLogId'].apply(list)
    cluster_info['TripLogId'] = trip_ids_clusters.apply(lambda x: np.unique(x))

    return cluster_info


def find_relations(trips_ini, mx_dist_ini):
    """finds which nodes are visisted after eachother and collects the information about the edges in adjacent_nodes_info. 
    Populates the trip information with information of which nodes the trip is close to and returns it as trips annotated"""
    trips = trips_ini.copy()
    mx_dist = mx_dist_ini.copy()

    # Remove single-point trips
    trips = filter_short(df=trips, groupby='TripLogId', minlength=2)

    # make a list of lists giving the sequence of the visited nodes for each trip.
    sorted_mx_dist = mx_dist.sort_values(by=["TripLogId", "timestamp_s"])
    trips_nodelists = list(sorted_mx_dist.groupby(
        "TripLogId")["cl_idx"].apply(list))
    trips_nodelists = [dm.retain_change_only(x) for x in trips_nodelists]

    # adds information about the nodes to the trips
    trips = trips.reset_index()
    trips = trips.rename(columns={"index": "ping_idx"})
    trips_annotated = trips.merge(mx_dist[["cl_idx", "ping_idx", "dist"]], on="ping_idx", how="inner")

    adjacent_nodes = find_adjacent_nodes(trips_nodelists=trips_nodelists)
    trips = trips.merge(trips_annotated[['cl_idx', "ping_idx", "dist"]], on="ping_idx", how='outer')

    # Returns True only if the nodes are visited directly one after another in the trip
    trips_adjacent_nodes = find_trips_adjacent_nodes(
        trips_annotated, trips_nodelists, adjacent_nodes)

    adjacent_nodes_info = pd.concat([pd.Series(adjacent_nodes), pd.Series(trips_adjacent_nodes)], axis=1)
    adjacent_nodes_info['edge_id'] = ['e_' + str(i+1) for i in adjacent_nodes_info.index]
    adjacent_nodes_info.columns = ['nodes', 'trips', 'edge_id']
    adjacent_nodes_info = adjacent_nodes_info[['edge_id', 'nodes', 'trips']]
    
    return adjacent_nodes_info, trips


def find_trips_adjacent_nodes(trips, trips_nodelists, adjacent_nodes):
    """
    Find trips in which the adjacent nodes are visited directly one after another.
    :return trips_adjacent_notes: a list of lists of trip_ids, in the order as in adjacent_nodes
    """
    trips_adjacent_nodes = list(map(lambda node_pair: trips.TripLogId.unique()[(list(map(lambda trip:
                                                                                         dm.list_in_list(
                                                                                             node_pair, trip),
                                                                                         trips_nodelists)))], adjacent_nodes))
    return trips_adjacent_nodes


def cut_by_nodes_without_direction(trips_annotated, nodes):
    trips_annotated_back = trips_annotated.copy()
    trips_annotated = trips_annotated.groupby('TripLogId').apply(
        lambda trip: cut_by_nodes_one_trip(trip, nodes))
    if trips_annotated.shape[0] != 0:
        trips_annotated = trips_annotated.reset_index(level='TripLogId', drop=True)
        dmin = trips_annotated.groupby("TripLogId")["timestamp_s"].min().to_frame().to_dict()["timestamp_s"]
        trips_annotated["timestamp_s"] = trips_annotated[["TripLogId", "timestamp_s"]].apply(lambda x: x["timestamp_s"]-dmin.get(x["TripLogId"]), axis=1)
        trips_annotated["TripLogId"] = trips_annotated["TripLogId"]  # +"first"

    reverse_node = (nodes[1], nodes[0])
    trips_annotated_back = trips_annotated_back.groupby('TripLogId').apply(
        lambda trip: cut_by_nodes_one_trip(trip, reverse_node))
    if trips_annotated_back.shape[0] != 0:
        trips_annotated_back = trips_annotated_back.reset_index(
            level='TripLogId', drop=True)
        dmax = trips_annotated_back.groupby(
            "TripLogId")["timestamp_s"].max().to_frame().to_dict()["timestamp_s"]
        trips_annotated_back["timestamp_s"] = trips_annotated_back[["TripLogId", "timestamp_s"]].apply(
            lambda x: dmax.get(x["TripLogId"])-x["timestamp_s"], axis=1)
        # +"second"
        trips_annotated_back["TripLogId"] = trips_annotated_back["TripLogId"]
    trips_annotated = pd.concat(
        [trips_annotated, trips_annotated_back], ignore_index=True)
    if trips_annotated.empty:
        return trips_annotated.copy()
    return trips_annotated.copy()


def cut_by_nodes(trips_annotated, nodes):
    trips_annotated = trips_annotated.groupby('TripLogId').apply(
        lambda trip: cut_by_nodes_one_trip(trip, nodes))
    trips_annotated = trips_annotated.reset_index(level='TripLogId', drop=True)
    return trips_annotated.copy()


def cut_by_nodes_one_trip(one_trip, nodes):
    prev_idx = one_trip['cl_idx'].ffill()
    next_idx = one_trip['cl_idx'].bfill()
    one_trip = one_trip[(prev_idx == nodes[0]) & (next_idx == nodes[1])]
    return one_trip


def preprocess_data(data_ini, proj_info,
                    endpoint_threshold,
                    remove_endpoints,
                    interp_spline_deg,
                    interp_resolution_m,
                    min_trip_length,
                    divide_trip_thr_mins,
                    divide_trip_thr_angle,
                    divide_trip_thr_dist=None,
                    add_vars_trips=[],  # TODO -- This parameter doesn't really feel good
                    ):
    """
    #Does data cleaning, and interpolation. Returns a dataframe with the columns
    #['TripLogId', 'Latitude', 'Longitude', 'Speed', 'Distance','timestamp_s', 'x', 'y']

    :param data:
    :param proj_info:
    :param min_trip_length: Smallest number of pings in trip in order to keep the trip
    :param thr_mins: divide into two trips if the difference in time between pings is larger than the threshold (in mins).
    :param thr_angle: divide into two trips if the difference in the angle between pings is larger than the threshold.
    :param interp_spline_deg:
    :param interp_resolution_m:
    :param add_vars_trips:
    :return trips: dataframe with the columns
    ['TripLogId', 'Latitude', 'Longitude', 'Speed', 'Distance','timestamp_s', 'x', 'y']
    """
    data = data_ini.copy()
    # Preliminary preprocessing
    format_varnames(data)  # Rename variable names
    format_timevars(data)  # Format timestamps
    remove_incorrect_rows(data)
    data = remove_idle_vehicles(data)
    # Remove all short trips. minlength is given by number of gps pings
    data = filter_short(data, groupby='TripLogId', minlength=min_trip_length)

    df = trips_processing(data, endpoint_threshold, remove_endpoints)

    df, proj_info = add_meter_columns(df, proj_info=proj_info)
    # adding features of delta values between gps pings
    logger.debug(f'Columns:\n {df.columns}')
    df = add_features(df, features=['timestamp_delta', 'course_delta'])

    # Divide trips if a the vehicle stops or turns too suddenly.
    df['TripLogId'] = divide_trips(
        df, divide_trip_thr_mins, divide_trip_thr_angle, divide_trip_thr_dist)

    df = add_features(df, features=['timestamp_s'])
    # , 'timestamp_delta', 'timestamp_delta_s', 'course_delta', 'dist_eucl_m'])
    # Filter out short trips again since new short trips could have been made by divide trips.
    df = filter_short(df, groupby='TripLogId', minlength=min_trip_length)
    # Filter out if two consecutive variables (here Latitude and Longitude) have exactly the same value. If so, only the
    # first one is kept
    for v in add_vars_trips+['Latitude', 'Longitude']:
        df = dm.rmv_dupl(df, v)
    # Again filter short trips since after possible removing duplicates
    df = filter_short(df, groupby='TripLogId', minlength=min_trip_length)

    # Interpolation of the position using the spline method. Returns a dataframe with the columns
    # ['TripLogId', 'Latitude', 'Longitude', 'Speed', 'Distance','timestamp_s', 'x', 'y']
    trips = interpolate_trips(df, proj_info,
                              interp_spline_deg,
                              interp_resolution_m,
                              add_vars=add_vars_trips)

    trips = filter_short(trips, groupby="TripLogId", minlength=min_trip_length)

    return df, trips

def trips_processing(data, endpoint_threshold, remove_endpoints):
    """
    Initiates trip list from data, potentially removing the beginning and end of each trip.

    :param data: Input data from Ditio.
    :param endpoint_threshold: Minimum distance in m that the trip must traverse.
    :param remove_endpoints: If true, removes the ends of the trip.
    :param variables: Variables to retain in the output dataframe.
    :return: A modified data frame with trips added.
    """

    trips = data.sort_values('Timestamp').groupby(['TripLogId'])
    df_list = []
    
    
    # Wrap the trips in tqdm for progress tracking
    for _, group in tqdm(trips, desc="Processing trips: ", total=len(trips)):
        # Add cumulative distance
        group['DistanceDriven'] = group['Distance'].cumsum()

        # Endpoint removal
        if remove_endpoints:
            dist_min, dist_max = group['DistanceDriven'].min(), group['DistanceDriven'].max()
            lower = group['DistanceDriven'] > dist_min + endpoint_threshold
            upper = group['DistanceDriven'] < dist_max - endpoint_threshold
            if dist_max - endpoint_threshold < 0:
                continue  # The candidate trip was too short
            else:
                # Filter to only get internal points
                group = group[lower & upper]
        
        # Append the trip data to the data frame
        df_list.append(group)

    df = pd.concat(df_list, axis=0)

    # This initially removes all columns that are not in the variable list
    #if len(variables) < 1:
    #    variables = ['Latitude', 'Longitude', 'Altitude', 'theta', 'delta_theta', 'TripLogId', 'Timestamp', 'Speed', 'Course',
    #                 'LoadLatitude', 'LoadLongitude', 'DumpLatitude', 'DumpLongitude', 'Distance', 'DistanceDriven',
    #                 'LoadGeoFenceId', 'DumpGeoFenceId'] + ['LoadDateTime', 'DumpDateTime', 'MassTypeId', 'MassTypeName',
    #                                                        'MassTypeMaterial', 'Quantity', 'LoaderMachineName']
    #        
    #variables = list(pd.Series(variables)[list(pd.Series(variables).isin(df.columns))])
    #df = df[variables].reset_index(drop=True)

    logger.debug(
        f"Removed {len(trips) - df['TripLogId'].unique().shape[0]} trips due to endpoint being under {endpoint_threshold}m away from startpoint.")

    return df



def make_no_directional_nodes_info(adjacent_nodes_info):
    no_di_nodes = pd.DataFrame()
    t = 1
    for i, k in adjacent_nodes_info.groupby("nodes"):
        if k.shape[0] == 1:
            tr = list(k.iloc[0]["trips"])  # +list(k.iloc[1]["trips"])
        else:
            tr = list(k.iloc[0]["trips"])+list(k.iloc[1]["trips"])
        tmp_np_di_nodes = {"edge_id": [
            "e_"+str(t)], "nodes": [i], "trips": [tr]}
        tmp_np_di_nodes = pd.DataFrame.from_dict(tmp_np_di_nodes)
        no_di_nodes = pd.concat(
            [no_di_nodes, tmp_np_di_nodes], ignore_index=True)
        t += 1
    return no_di_nodes


def compute_road_trajectories(trips_annotated,
                              adjacent_nodes_info,
                              edge_type='dba',
                              max_trips_per_edge=None,
                              fast_alg=True) -> list:
    np.random.seed(10)
    random.seed(10)
    """
    Averages multiple trips that connect each pair of adjacent nodes to find the approximate road trajectory.
    :param trips_annotated: trips with nearby intersection info, as returned by find_relations().
    :param adjacent_nodes_info: as returned by find_relations().
    :param edge_type: the algorithm to use to approximate the trajectory. 
                      - 'dba' (dynamic time warping barycenter averaging)
                      - 'random' (random trip)
                      - 'pw_reg' (pairwise regression)
    :param max_trips_per_edge: nr trips to be used to approximate the trajectory between each pair of adjacent nodes.
    :return edges:
    """
    adjacent_nodes = adjacent_nodes_info['nodes']
    adjacent_nodes_trips = adjacent_nodes_info['trips']
    if max_trips_per_edge is None:
        pbar = tqdm(total=np.sum(
            [len(p) if (len(p) < 50) else 50 for p in adjacent_nodes_trips]))
    elif max_trips_per_edge < 50:
        pbar = tqdm(total=np.sum([len(p) if (len(
            p) < max_trips_per_edge) else max_trips_per_edge for p in adjacent_nodes_trips]))
    else:
        pbar = tqdm(total=np.sum(
            [len(p) if (len(p) < 50) else 50 for p in adjacent_nodes_trips]))

    edges = list()
    for i in range(len(adjacent_nodes)):
        pos_trips = list(adjacent_nodes_trips[i])
        pbar.set_description(
            "Processing edges ({0}/{1})".format(i+1, len(adjacent_nodes_trips)))
        if max_trips_per_edge is not None:
            if len(pos_trips) > max_trips_per_edge:
                pos_trips = random.sample(pos_trips, max_trips_per_edge)
        if edge_type == 'random':
            trip_id = np.random.choice(pos_trips)
            one_trip = trips_annotated[trips_annotated['TripLogId'] == trip_id]
            edge_df = cut_by_nodes_one_trip(one_trip, adjacent_nodes[i])
        else:
            relev_trips = trips_annotated[trips_annotated['TripLogId'].isin(
                pos_trips)].copy()
            relev_trips = cut_by_nodes(relev_trips, adjacent_nodes[i])
            if relev_trips.shape[0] == 0:
                lat_new = [-1]
                lon_new = [-1]
            elif edge_type == 'dba':
                if relev_trips.shape[0] > 2000 and fast_alg:
                    relev_trips = relev_trips.sample(2000)
                relev_trips_s = dm.to_array(relev_trips)
                lat_new, lon_new = performDBA(
                    relev_trips_s, n_iterations=1, pbar=pbar)

            #elif edge_type == 'pw_reg':
            #    model = PiecewiseRegressor(verbose=False, estimator=LinearRegression(),
            #                               binner=DecisionTreeRegressor(min_samples_leaf=30))
            #    model.fit(
            #        np.array(relev_trips['Longitude']).reshape(-1, 1), relev_trips['Latitude'])
            #    lon_new = np.linspace(relev_trips['Longitude'].min(
            #    ), relev_trips['Longitude'].max(), num=100)
            #    lat_new = model.predict(np.array(lon_new).reshape(-1, 1))

            edge_df = pd.DataFrame({'Latitude': lat_new, 'Longitude': lon_new})
        edge_df['edge_id'] = 'e_' + str(i+1)
        edges.append(edge_df)

    return edges


def compute_road_trajectories_without_direction(cluster_info,
                                                trips_annotated,
                                                adjacent_nodes_info,
                                                max_trips_per_edge=None,
                                                fast_alg=True) -> list:
    def _get_median_index(d):
        ranks = d.rank(pct=True)
        close_to_median = abs(ranks - 0.5)
        return close_to_median.idxmin()

    np.random.seed(10)
    random.seed(10)
    """
    Averages multiple trips that connect each pair of adjacent nodes to find the approximate road trajectory.
    :param trips_annotated: trips with nearby intersection info, as returned by find_relations().
    :param adjacent_nodes_info: as returned by find_relations().
    :param edge_type: the algorithm to use to approximate the trajectory. 
                      - 'dba' (dynamic time warping barycenter averaging)
                      - 'random' (random trip)
                      - 'pw_reg' (pairwise regression)
    :param max_trips_per_edge: nr trips to be used to approximate the trajectory between each pair of adjacent nodes.
    """
    adjacent_nodes = adjacent_nodes_info['nodes']
    adjacent_nodes_trips = adjacent_nodes_info['trips']
    edges = list()
    relev_trips_list = list()
    for i in range(len(adjacent_nodes)):
        pos_trips = list(adjacent_nodes_trips[i])
        relev_trips = trips_annotated[trips_annotated['TripLogId'].isin(
            pos_trips)].copy()
        relev_trips = cut_by_nodes_without_direction(
            relev_trips, adjacent_nodes[i])
        start_lat = cluster_info.iloc[adjacent_nodes[i][0]]["Latitude"]
        start_long = cluster_info.iloc[adjacent_nodes[i][0]]["Longitude"]
        start_z = cluster_info.iloc[adjacent_nodes[i][0]]["z"]
        end_lat = cluster_info.iloc[adjacent_nodes[i][1]]["Latitude"]
        end_long = cluster_info.iloc[adjacent_nodes[i][1]]["Longitude"]
        end_z = cluster_info.iloc[adjacent_nodes[i][1]]["z"]
        relev_trips_list.append(relev_trips)

        if relev_trips.shape[0] == 0:
            lat_new = [start_lat, end_lat]
            lon_new = [start_long, end_long]
            z_new = [start_z, end_z]
        else:
            if relev_trips.shape[0] > 1 and fast_alg:
                # shortest_trip=relev_trips.groupby("TripLogId").count()["Latitude"].idxmin()
                # Use median length instead
                trips_group = relev_trips.groupby(
                    "TripLogId").count()["Latitude"]
                median_trip = _get_median_index(d=trips_group)
                relev_trips = relev_trips[relev_trips["TripLogId"]
                                          == median_trip]
                relev_trips.sort_values(
                    by=["TripLogId", "timestamp_s"], inplace=True)
            relev_trips_s = dm.to_array(relev_trips)
            # If we want to use several tracks to estimate the track between two nodes we could use DynamicTimeWraping, but this is useless as we at the
            # moment only use one trip. The metod of using more than one trip has not been tested when including the z position.
            lat_new, lon_new, z_new = performDBA(relev_trips_s, n_iterations=1)
            lat_new = np.insert(lat_new, 0, start_lat, axis=0)
            lon_new = np.insert(lon_new, 0, start_long, axis=0)
            z_new = np.insert(z_new, 0, start_z, axis=0)
            lat_new = np.append(lat_new, end_lat)
            lon_new = np.append(lon_new, end_long)
            z_new = np.append(z_new, end_z)

        edge_df = pd.DataFrame(
            {'Latitude': lat_new, 'Longitude': lon_new, 'z': z_new})

        edge_df['edge_id'] = 'e_' + str(i+1)
        edges.append(edge_df)

    return edges, relev_trips_list


def get_cluster_to_merge(points, name, radius):
    """

    :param points: frame giving the intersection points that can be merged. name: the intersection type("load/dump/road").
    :param name:
    :param radius: the maximum distance for merging two intersection points
    :return: dataframe containing the new merge clusters and a set giving the index of the clusters that has been removed
    """
    def distance(p1, p2):
        dx = p1["x"] - p2[0]
        dy = p1["y"] - p2[1]
        return np.sqrt(dx*dx + dy*dy)

    new_clusters = pd.DataFrame()
    removed_rows = []
    for i, k in points.iterrows():  # i index, k row
        p = k[["x", "y"]].to_list()  # get position of point in row i (latlon)
        # define new column in df consisting of distances from point p (position in row i)
        points["dist" + str(i)] = points[["x", "y"]
                                         ].apply(lambda x: distance(x, p), axis=1)
        # Number of indices whose distance to p is smaller than radius
        m = points[points["dist" + str(i)] < radius].index.to_list()
        if len(m) > 1:  # if there is at least one such index
            index = points.index.max() + i  # Get index of point, for
            new_clusters = pd.concat([new_clusters, pd.DataFrame({"Latitude": points.iloc[m]["Latitude"].mean(),
                                                                 "Longitude": points.iloc[m]["Longitude"].mean(),
                                                                  "in_type": name}, index=[index])], ignore_index=True)  # Append the set m, for some reason
            removed_rows.append(m)  # m added to removed rows

        continue

    return new_clusters.drop_duplicates(), set(dm.flatten_list(removed_rows))


def get_new_points(close_points, points, name):
    """Isn't this function just taking the average over latitude and longitude for close_points?
    new_clusters = new_clusters.append(pd.DataFrame({"Latitude": points.iloc[close_points]["Latitude"].mean(),
                                                     "Longitude": points.iloc[close_points]["Longitude"].mean(),
                                                     "in_type": name)
    would do the job?
    """
    testFrame = pd.DataFrame(columns=["Latitude", "Longitude", "in_type"])
    s = len(close_points)
    la = 0
    lo = 0
    for i in close_points:
        la += points.iloc[i]["Latitude"]
        lo += points.iloc[i]["Longitude"]
    la = la*(1.0/s)
    lo = lo*(1.0/s)
    testFrame = testFrame.append(
        {"Latitude": la, "Longitude": lo, "in_type": name}, ignore_index=True)
    return testFrame


def df_turns_neighbour_similarity(df_in: pd.DataFrame, res_step,
                                  neighbour_dist, similarity_thr) -> pd.DataFrame:
    ''' Returns the cells were the track direction is different from the track direction of cells in neigbhouring cells'''
    df = df_in.copy()
    df['Course2'] = df['Course']
    df.loc[df['Course'] > 180, 'Course2'] = df['Course']-180
    df['theta2'] = np.radians(df['Course2'])
    # adding low resolution and using the median value as the aggregated value
    df["x_"+str(res_step)] = change_resolution(df["x"], step=res_step)
    df["y_"+str(res_step)] = change_resolution(df["y"], step=res_step)
    gr = df.groupby(["x_"+str(res_step), "y_"+str(res_step)])
    low_res_median = gr[['x', 'y', 'Latitude', 'Longitude',
                         'theta2', "Altitude"]].median().reset_index()
    # return low_res_median
    low_res_median['similarity_median'] = neighbour_similarity(
        low_res_median, dist=neighbour_dist)

    # Take out the cells where the median direction is different from the neigbouring cells.
    low_res_median = low_res_median[low_res_median['similarity_median']
                                    > similarity_thr]
    return low_res_median


def change_resolution(values, step):
    return round(values/step)*step


def neighbour_similarity(df, dist):
    """calculate the similarity in track direction between a cell and its neigbouring cells"""
    similarities = list()
    points = closest_points_list(df, ["x", "y"], dist)
    for i in range(len(points)):
        value1 = float(df.loc[df.index == df.index[i], "theta2"])
        values_neighbors = np.array(
            df.loc[df.index.isin(pd.Series(points[i])), "theta2"])
        dist = np.sqrt(((value1 - values_neighbors)**2).sum())
        similarities.append(dist)
    return similarities


def closest_points_list(df, geovariables, dist):
    points = list()
    if len(df) == len(df.index):
        for i in df.index:
            points.append(closest_points(
                df.iloc[i], df, geovariables, dist=dist))

    return points


def closest_points(row, df, geovariables, dist):
    cor0 = row[geovariables[0]]
    cor1 = row[geovariables[1]]
    i = row.name
    # closest_points = df[(np.abs(df[geovariables[0]] - cor0) < dist)
    #                                  & (np.abs(df[geovariables[1]] - cor1) < dist)]
    closest_points = df[dist_eucl(
        df[geovariables[0]], cor0, df[geovariables[1]], cor1) < dist]
    closest_points = closest_points[closest_points.index != i]

    return closest_points.index.values.tolist()


def dist_eucl(x1, x2, y1, y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)


def get_cluster_centres(df_ini, merge_clust_in_dist):
    ''' Merge points that are closer togheter than "merge_clust_in_dist" into one cluster using the mean value as the center'''
    df = df_ini.copy()
    df['labels'] = cluster_simple_distance(df, ["x", "y"], merge_clust_in_dist)
    df_grouped = df.groupby(['labels'])
    cluster_centres = df_grouped.mean()
    cluster_centres['cl_size'] = df_grouped.size()
    cluster_centres = cluster_centres.reset_index(drop=True)
    return cluster_centres


def cluster_simple_distance(df: pd.DataFrame, columns, max_distance):
    """
    Clusters together all points that are within a specified distance to each other.
    """
    tree = df_to_kdtree(df, columns)
    pairs = dict(tree.sparse_distance_matrix(tree, max_distance)).keys()
    G = nx.from_edgelist(pairs)
    clusters = list(nx.connected_components(G))
    series = pd.Series(index=range(len(df)), dtype='int')
    for cl_nr, idx in enumerate(clusters, 1):
        series[list(idx)] = cl_nr
    series[series == 0] = np.array(range(sum(series == 0)))+series.max()+1

    return np.array(series)


def get_center_connected_trips(mx_df, R):
    # TODO: Should this be within the function mx_extremity_clusters
    """ Returns only the gps points that are part of a trip that also has gps points within a distance R from the center of the cluser"""
    mx_df_connected = pd.DataFrame()
    for i, k in mx_df.groupby("cl_idx"):
        trips_in_center = k[k["dist"] < R]["TripLogId"].unique()
        mx_df_temp = k[k["TripLogId"].isin(trips_in_center)]
        mx_df_connected = pd.concat(
            [mx_df_connected, mx_df_temp], ignore_index=True)
    return mx_df_connected


def validate_points_in_annulus(all_points, node_center, R, L, proximity_threshold, x_var="x", y_var="y", epsilon=10, min_samples=5):
    """
    Validate points based on specified criteria and subset those within the annulus of valid clusters.

    Parameters:
    - df: DataFrame containing all points with 'x', 'y' coordinates and 'dist_to_center' pre-calculated.
    - node_center_x, node_center_y: Coordinates of the node center.
    - R: Inner radius of the annulus.
    - L: Additional distance to form the outer radius of the annulus.
    - proximity_threshold: Distance from the node center to consider a cluster as valid.

    Returns:
    - DataFrame of valid points within the annulus of valid clusters.
    """

    # Calculate distance to the node center
    node_center_x, node_center_y = node_center
    all_points['dist_to_center'] = np.sqrt(
        (all_points[x_var] - node_center_x)**2 + (all_points[y_var] - node_center_y)**2)

    # Filter points within the R+L radius
    points_within_radius = all_points[all_points['dist_to_center'] <= R + L]

    # Apply DBSCAN clustering
    # Adjust eps and min_samples as needed
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    points_within_radius['cluster'] = dbscan.fit_predict(
        points_within_radius[[x_var, y_var]])

    # Identify clusters with at least one point close to the node center
    valid_clusters = points_within_radius[points_within_radius['dist_to_center']
                                          < proximity_threshold]['cluster'].unique()

    # Subset points in valid clusters and within the annulus
    valid_points = points_within_radius[points_within_radius['cluster'].isin(
        valid_clusters) & (points_within_radius['dist_to_center'] >= R)]

    return valid_points


def mx_extremity_clusters(mx_df_ini, intersection_candidates, R, L, extremity_merging_cluster_dist, min_cl_size, max_dist_from_intersection,
                          epsilon=12, min_samples=5):
    """
    TODO: UPDATE DESCRIPTION
    Finds the subclusters that are within the radius > R and < R + L. Returns a fram with on row per ping inside this area, indicating its subcluseter. 
    @param extremity_merging_cluster_dist: max distance to merge points in extremities into one cluster
    @param min_cl_size: minimal size of extremity cluster to be counted
    @param nr_trips: nr trips to use for each cluster to find the extremity clusters (do not compute all) If -1, then all are computed.
    @max_dist_from_intersection: The radius around an intersection candidate that we require a track to pass by in o for its gps points in the extremity to be considered.
    """
    mx_df = mx_df_ini.copy()
    list_mx_df_extremities = []
    for cl_idx in range(len(intersection_candidates)):
        print("Verifying candidate: " + str(cl_idx) +
              "/" + str(len(intersection_candidates)))
        mx_df_subset = mx_df.loc[mx_df["cl_idx"] == cl_idx]
        cl = intersection_candidates.iloc[cl_idx]
        mx_df_subset_val = validate_points_in_annulus(all_points=mx_df_subset, node_center=(cl["x"], cl["y"]),
                                                      R=R, L=L,
                                                      proximity_threshold=max_dist_from_intersection,
                                                      x_var="ping_x", y_var="ping_y", epsilon=epsilon, min_samples=min_samples)

        if not mx_df_subset_val.empty:
            mx_df_subset_val["subcluster"] = cluster_simple_distance(mx_df_subset_val, ['ping_x', 'ping_y'],
                                                                     extremity_merging_cluster_dist)

            list_mx_df_extremities.append(mx_df_subset_val)

    mx_df_extremities = pd.concat(list_mx_df_extremities, axis=0)

    mx_df_extremities['subcluster'] = [str(i) + '_' + str(j) for i, j in zip(mx_df_extremities['cl_idx'],
                                                                             mx_df_extremities['subcluster'])]
    subcluster_sizes = mx_df_extremities.groupby(['cl_idx', 'subcluster']).size()
    subcluster_sizes = subcluster_sizes[subcluster_sizes >= min_cl_size]
    subcluster_sizes = subcluster_sizes.reset_index(level=1)
    mx_df_extremities = mx_df_extremities[mx_df_extremities['subcluster'].isin(subcluster_sizes['subcluster'])]

    return mx_df_extremities


def update_frames(cluster_info_ini, mx_df_extremities_ini):
    """
    Helper function updating nr_roads in cluster_info, and returning only information about candidate intersections with nr_roads>=3
    """
    cluster_info = cluster_info_ini.copy()
    mx_df_extremities = mx_df_extremities_ini.copy()
    cluster_info['nr_roads'] = mx_df_extremities.groupby('cl_idx')[
        'subcluster'].nunique()
    cluster_info = cluster_info[cluster_info['nr_roads'] >= 3]
    mx_df_extremities = mx_df_extremities[mx_df_extremities['cl_idx'].isin(
        cluster_info.index)]

    return cluster_info, mx_df_extremities


def kdtree_neighbors(df1, df2, r=25, p=2, eps=0):
    ''' Takes in two dataframes, returns a list with length of df2. The elements in the list is a list of the indexes of the elements in df1 that is closter than r'''
    tree1 = df_to_kdtree(df=df1, columns=['x', 'y'])
    tree2 = df_to_kdtree(df=df2, columns=['x', 'y'])
    neighbors = tree2.query_ball_tree(tree1, r=r)

    return neighbors

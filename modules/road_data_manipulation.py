from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
import re
import geopy as gp
from geopy.distance import geodesic
import datetime
import modules.data_manipulation as dm
from modules.gpx_interpolate import interpolate_points_latlon
from scipy.spatial import KDTree
from tqdm import tqdm
import networkx as nx
import logging
logger = logging.getLogger('producegraph')


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
    df.columns = [re.sub(r'Trip\.', '', x) for x in df.columns]
    df.columns = [re.sub(r'Coordinate\.', '', x) for x in df.columns]
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
        d = geodesic(meters=meters_to_angle)
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
    return (coord - origin) / delta


def divide_trips(df, thr_minutes, thr_degrees, thr_metres=None, trip_id='TripLogId'):
    """
    Divide trips into multiple shorter trips.
    :param thr_mins: divide into two trips if the difference in time between pings is larger than the threshold (in mins).
    :param thr_angle: divide into two trips if the difference in the angle between pings is larger than the threshold.
    :return: new trip ids.
    """

    def _eucl(x1, y1, x2, y2):
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def _new_ids(df, cumsum_var, connection):
        series_ids = df[trip_id]
        series_nrs = df[cumsum_var].cumsum().astype('str')
        new = series_ids + connection + series_nrs
        return new

    def _apply_new_ids(df, cumsum_var, connection):
        trip_ids = np.array(df.groupby(trip_id).apply(lambda x: _new_ids(x, cumsum_var, connection)
                                                          if x[cumsum_var].sum() > 0 else x[trip_id]))
        return trip_ids

    df = df.copy()
    df['large_timedelta'] = np.array(
        df['timestamp_delta'] > datetime.timedelta(minutes=thr_minutes))
    df[trip_id] = _apply_new_ids(df, 'large_timedelta', connection='_t')
    if thr_degrees>0:        
        df['large_coursedelta'] = np.array(abs(df['course_delta']) > thr_degrees)
        df[trip_id] = _apply_new_ids(df, 'large_coursedelta', connection='_c')
        df['course_delta'] = change_first_in_group(df=df, by=trip_id, variable='course_delta', first_val=0)
    if (thr_metres is not None) and (thr_metres > 0):
        df['x_prev'] = df.groupby(trip_id)['x'].shift(1)
        df['y_prev'] = df.groupby(trip_id)['y'].shift(1)
        df['EuclideanDistance'] = df.apply(lambda row: _eucl(row['x_prev'], row['y_prev'], row['x'], row['y']) if not pd.isnull(row['x_prev']) else 0, axis=1)
        df = df.drop(columns=['x_prev', 'y_prev'])
        df['DistanceDiff'] = df.groupby(trip_id)['Distance'].diff().abs()
        df['large_distdiff'] = df['DistanceDiff'] > thr_metres
        df[trip_id] = _apply_new_ids(df, 'large_distdiff', connection='_d')

    return df[trip_id]


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


def compute_dist_matrix(df1_ini, df2_ini, max_dist_trip_from_node):
    """
    Computes a matrix of distances from points in df1 to points in df2.
    Returns only those that are at least within max_distance.
    The result is in the 'dictionary of keys' format, where the multikey is iloc of df1 and df2.
    """
    df1 = df1_ini.copy()
    df2 = df2_ini.copy()
    tree1 = df_to_kdtree(df=df1, columns=["x", "y"])
    tree2 = df_to_kdtree(df=df2, columns=["x", "y"])
    distance_matrix = dict(tree2.sparse_distance_matrix(tree1, max_distance=max_dist_trip_from_node, p=2))
    return distance_matrix


def df_to_kdtree(df: pd.DataFrame, columns):
    """
    Takes a df with two columns and returns them as a tree object.
    """
    tuples = [(x, y) for x, y in zip(df[columns[0]], df[columns[1]])]
    tree = KDTree(tuples)

    return tree


def preprocess_distance_matrix(mx_dist, trips, cluster_info):
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


def assign_triplog_to_clusters(cluster_info_ini, mx_df_ini, max_dist_from_cluster):
    ''' 
    Assigns a list of unique TripLogIds to clusters that have trips passing within the specified proximity threshold. 
    The proximity threshold defines the maximum distance from a cluster node that a trip must pass to be considered.
    '''    
    cluster_info = cluster_info_ini.copy()
    mx_df = mx_df_ini.copy()
    mx_df_close_to_cluster = mx_df[mx_df['dist'] <= max_dist_from_cluster]
    trip_ids_clusters = mx_df_close_to_cluster.groupby(
        'cl_idx')['TripLogId'].apply(list)
    cluster_info['TripLogId'] = trip_ids_clusters.apply(lambda x: np.unique(x))

    return cluster_info


def identify_adjacent_node_trips(trips_ini, mx_dist_ini):
    """finds which nodes are visisted after eachother and collects the information about the edges in adjacent_nodes_info. 
    Populates the trip information with information of which nodes the trip is close to and returns it as trips annotated"""
    trips = trips_ini.copy()
    mx_dist = mx_dist_ini.copy()

    # Remove single-point trips
    trips = filter_short(df=trips, groupby='TripLogId', minlength=2)

    # make a list of lists giving the sequence of the visited nodes for each trip.
    sorted_mx_dist = mx_dist.sort_values(by=["TripLogId", "timestamp_s"])
    trips_nodelists = list(sorted_mx_dist.groupby("TripLogId")["cl_idx"].apply(list))
    trips_nodelists = [dm.retain_change_only(x) for x in trips_nodelists]
    adjacent_nodes = find_adjacent_nodes(trips_nodelists=trips_nodelists)


    # adds information about the nodes to the trips
    trips = trips.reset_index()
    trips = trips.rename(columns={"index": "ping_idx"})
    trips_with_node_info = trips.merge(mx_dist[["cl_idx", "ping_idx", "dist"]], on="ping_idx", how="inner")
    trips = trips.merge(trips_with_node_info[['cl_idx', "ping_idx", "dist"]], on="ping_idx", how='outer')

    # Returns True only if the nodes are visited directly one after another in the trip
    trips_adjacent_nodes = list_tripids_between_adjacent_nodes(trips_with_node_info, trips_nodelists, adjacent_nodes)

    adjacent_nodes_info = pd.concat([pd.Series(adjacent_nodes), pd.Series(trips_adjacent_nodes)], axis=1)
    adjacent_nodes_info['edge_id'] = ['e_' + str(i+1) for i in adjacent_nodes_info.index]
    adjacent_nodes_info.columns = ['nodes', 'trips', 'edge_id']
    adjacent_nodes_info = adjacent_nodes_info[['edge_id', 'nodes', 'trips']]
    
    return adjacent_nodes_info, trips


def list_tripids_between_adjacent_nodes(trips, trips_nodelists, adjacent_nodes):
    """
    Find trips in which the adjacent nodes are visited directly one after another.
    :return trips_adjacent_nodes: a list of lists of trip_ids, in the order as in adjacent_nodes
    """
    trips_adjacent_nodes = list(map(lambda node_pair: trips.TripLogId.unique()[(list(map(lambda trip:
                                                                                         dm.list_in_list(
                                                                                             node_pair, trip),
                                                                                         trips_nodelists)))], adjacent_nodes))
    return trips_adjacent_nodes


def cut_by_nodes_without_direction(trips_with_node_info, nodes):
    trips_with_node_info_reverse = trips_with_node_info.copy()
    trips_with_node_info = trips_with_node_info.groupby('TripLogId').apply(
        lambda trip: extract_trip_segments_between_node_pairs_one_trip(trip, nodes))
    if trips_with_node_info.shape[0] != 0:
        trips_with_node_info = trips_with_node_info.reset_index(level='TripLogId', drop=True)
        dmin = trips_with_node_info.groupby("TripLogId")["timestamp_s"].min().to_frame().to_dict()["timestamp_s"]
        trips_with_node_info["timestamp_s"] = trips_with_node_info[["TripLogId", "timestamp_s"]].apply(lambda x: x["timestamp_s"]-dmin.get(x["TripLogId"]), axis=1)
        trips_with_node_info["TripLogId"] = trips_with_node_info["TripLogId"]

    reverse_node = (nodes[1], nodes[0])
    trips_with_node_info_reverse = trips_with_node_info_reverse.groupby('TripLogId').apply(
        lambda trip: extract_trip_segments_between_node_pairs_one_trip(trip, reverse_node))
    
    if trips_with_node_info_reverse.shape[0] != 0:
        trips_with_node_info_reverse = trips_with_node_info_reverse.reset_index(level='TripLogId', drop=True)
        dmax = trips_with_node_info_reverse.groupby("TripLogId")["timestamp_s"].max().to_frame().to_dict()["timestamp_s"]
        trips_with_node_info_reverse["timestamp_s"] = trips_with_node_info_reverse[["TripLogId", "timestamp_s"]].apply(lambda x: dmax.get(x["TripLogId"])-x["timestamp_s"], axis=1)
        trips_with_node_info_reverse["TripLogId"] = trips_with_node_info_reverse["TripLogId"]
    trips_with_node_info = pd.concat([trips_with_node_info, trips_with_node_info_reverse], ignore_index=True)

    if trips_with_node_info.empty:
        return trips_with_node_info.copy()
    return trips_with_node_info.copy()


def extract_trip_segments_between_node_pairs(trips_with_node_info, nodes):
    """
    Extracts segments of trips that lie between specified pairs of nodes.

    Parameters:
    - trips_with_node_info (DataFrame): A DataFrame containing trip information with 
      GPS points and their corresponding node indices.
    - node_pairs (list of tuples): A list of tuples, where each tuple contains two 
      node indices representing the start and end of the segment to extract.

    Returns:
    - DataFrame: A DataFrame containing only the trip segments that lie between the specified 
      node pairs.
    """
    trips_with_node_info = trips_with_node_info.groupby('TripLogId').apply(
        lambda trip: extract_trip_segments_between_node_pairs_one_trip(trip, nodes))
    trips_with_node_info = trips_with_node_info.reset_index(level='TripLogId', drop=True)
    return trips_with_node_info.copy()


def extract_trip_segments_between_node_pairs_one_trip(one_trip_with_node_info, nodes):
    """
    Extracts the segment of a trip that is located between the specified pair of nodes.

    Parameters:
    - trip (DataFrame): A DataFrame containing the details of a single trip, including 
      a column 'cl_idx' representing indices of nodes that are nearby.
    - node_pair (tuple): A tuple containing two node indices, where the first node is 
      the starting point and the second is the endpoint.

    Returns:
    - DataFrame: A DataFrame of the trip segment that is directly between the specified 
      nodes. Segments are retained only if the preceding node is equal to the first node 
      and the following node is equal to the second node.
    """
    prev_idx = one_trip_with_node_info['cl_idx'].ffill()                                   # Series with sprevious "closest node" to the end of the trip
    next_idx = one_trip_with_node_info['cl_idx'].bfill()                                   # Series with next "closest node" to the start of the trip
    one_trip_with_node_info = one_trip_with_node_info[(prev_idx == nodes[0]) & (next_idx == nodes[1])]    # Keep only the segments that are between the specified nodes
    return one_trip_with_node_info


def preprocess_data(data_ini, proj_info,
                    dist_endpoints_trim,
                    remove_endpoints,
                    interpolation_spline_degrees,
                    interpolation_resolution_metres,
                    min_nr_points_trip,
                    divide_trip_threshold_minutes,
                    divide_trip_threshold_degrees,
                    divide_trip_threshold_metres=None,
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
    data = filter_short(data, groupby='TripLogId', minlength=min_nr_points_trip)

    df = trips_processing(data, dist_endpoints_trim, remove_endpoints)

    df, proj_info = add_meter_columns(df, proj_info=proj_info)
    # adding features of delta values between gps pings
    logger.debug(f'Columns:\n {df.columns}')
    df = add_features(df, features=['timestamp_delta', 'course_delta'])

    # Divide trips if a the vehicle stops or turns too suddenly.
    df['TripLogId'] = divide_trips(
        df, divide_trip_threshold_minutes, divide_trip_threshold_degrees, divide_trip_threshold_metres)

    df = add_features(df, features=['timestamp_s'])
    # , 'timestamp_delta', 'timestamp_delta_s', 'course_delta', 'dist_eucl_m'])
    # Filter out short trips again since new short trips could have been made by divide trips.
    df = filter_short(df, groupby='TripLogId', minlength=min_nr_points_trip)
    # Filter out if two consecutive variables (here Latitude and Longitude) have exactly the same value. If so, only the
    # first one is kept
    for v in add_vars_trips+['Latitude', 'Longitude']:
        df = dm.rmv_dupl(df, v)
    # Again filter short trips since after possible removing duplicates
    df = filter_short(df, groupby='TripLogId', minlength=min_nr_points_trip)
    df['theta2'] = np.radians(np.where(df['Course'] > 180, df['Course'] - 180, df['Course']))


    trips = interpolate_trips(df, proj_info,
                              interpolation_spline_degrees,
                              interpolation_resolution_metres,
                              add_vars=add_vars_trips)

    trips = filter_short(trips, groupby="TripLogId", minlength=min_nr_points_trip)

    return df, trips

def metres_to_coord(metres, origin, delta):
    return metres * delta + origin

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
    for _, group in tqdm(trips, desc="Processing trips", total=len(trips)):
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
                group = group[lower & upper]        
        df_list.append(group)

    df = pd.concat(df_list, axis=0)
    logger.debug(
        f"Removed {len(trips) - df['TripLogId'].unique().shape[0]} trips due to endpoint being under {endpoint_threshold}m away from startpoint.")

    return df


def calculate_neighbour_similarity(df_in: pd.DataFrame, res_step,neighbour_dist, similarity_thr) -> pd.DataFrame:
    ''' Returns the cells were the track direction is different from the track direction of cells in neigbhouring cells'''

    df = df_in.copy()

    # adding low resolution and using the median value as the aggregated value
    df["x_"+str(res_step)] = change_resolution(df["x"], step=res_step)
    df["y_"+str(res_step)] = change_resolution(df["y"], step=res_step)
    gr = df.groupby(["x_"+str(res_step), "y_"+str(res_step)])
    histogram_median_directions = gr[['x', 'y', 'Latitude', 'Longitude','theta2', "Altitude"]].median().reset_index()
    histogram_median_directions['similarity_median'] = neighbour_similarity(histogram_median_directions, dist=neighbour_dist)

    # Take out the cells where the median direction is different from the neigbouring cells.
    intersection_candidates = histogram_median_directions[histogram_median_directions['similarity_median'] > similarity_thr]
    return intersection_candidates, histogram_median_directions


def change_resolution(values, step):
    return round(values/step)*step


def neighbour_similarity(df, dist):
    """calculate the similarity in track direction between a cell and its neigbouring cells"""
    similarities = list()
    points = closest_points_list(df, ["x", "y"], dist)
    for i in range(len(points)):
        value1 = float(df.loc[df.index == df.index[i], "theta2"])
        values_neighbors = np.array(df.loc[df.index.isin(pd.Series(points[i])), "theta2"])
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


def merge_nearby_cluster_centres(df_ini, distance, coordinates=["x", "y"], min_size=1):
    ''' Merge points that are closer together than "distance" into one cluster using the mean value as the center'''
    df = df_ini.copy()
    df['labels'] = cluster_simple_distance(df, coordinates, distance)
    cluster_centres = df.groupby(['labels']).mean()
    cluster_centres['cl_size'] = df.groupby(['labels']).size()
    cluster_centres.reset_index(drop=True, inplace=True)

    # Subset if the cluster is too small
    cluster_centres = cluster_centres[cluster_centres['cl_size'] > min_size].reset_index(drop=True)

    return cluster_centres

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
    for i,k in points.iterrows(): #i index, k row
        p = k[["x", "y"]].to_list() #get position of point in row i (latlon)
        points["dist" + str(i)] = points[["x", "y"]].apply(lambda x: distance(x, p), axis=1) #define new column in df consisting of distances from point p (position in row i)
        m = points[points["dist" + str(i)] < radius].index.to_list() # Number of indices whose distance to p is smaller than radius
        if len(m) > 1: # if there is at least one such index
            index = points.index.max() + i # Get index of point, for
            new_clusters = pd.concat([new_clusters,pd.DataFrame({"Latitude": points.iloc[m]["Latitude"].mean(),
                                                             "Longitude": points.iloc[m]["Longitude"].mean(),
                                                             "in_type": name}, index=[index])],ignore_index=True) #Append the set m, for some reason
            removed_rows.append(m)#m added to removed rows

        continue

    return new_clusters.drop_duplicates(), set(dm.flatten_list(removed_rows))


def cluster_simple_distance(df: pd.DataFrame, columns, max_distance):
    """
    Cluster points that are within a specified distance to each other.
    """
    kd_tree = df_to_kdtree(df, columns)
    point_pairs = dict(kd_tree.sparse_distance_matrix(kd_tree, max_distance)).keys()
    graph = nx.from_edgelist(point_pairs)
    clusters = list(nx.connected_components(graph))
    cluster_labels = pd.Series(index=range(len(df)), dtype='int')
    for cluster_nr, idx in enumerate(clusters, 1):
        cluster_labels[list(idx)] = cluster_nr
    cluster_labels[cluster_labels==0] = np.array(range(sum(cluster_labels==0))) + cluster_labels.max() + 1

    return np.array(cluster_labels)


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


def validate_points_in_annulus(all_points, node_center, R, L, max_dist_from_intersection, x_var="x", y_var="y", epsilon=10, min_samples=5,
                               max_nr_points=1000):
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
    all_points['dist_to_center'] = np.sqrt((all_points[x_var] - node_center_x)**2 + (all_points[y_var] - node_center_y)**2)

    # Filter points within the R+L radius
    points_within_radius = all_points[all_points['dist_to_center'] <= R + L]
    if max_nr_points is not None:
        points_within_radius = points_within_radius.sample(n=min(max_nr_points, len(points_within_radius)), replace=False)

    # Apply DBSCAN clustering
    # Adjust eps and min_samples as needed
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    points_within_radius['cluster'] = dbscan.fit_predict(points_within_radius[[x_var, y_var]])

    # Identify clusters with at least one point close to the node center
    valid_clusters = points_within_radius[points_within_radius['dist_to_center'] < max_dist_from_intersection]['cluster'].unique()

    # Subset points in valid clusters and within the annulus
    valid_points = points_within_radius[points_within_radius['cluster'].isin(valid_clusters) & (points_within_radius['dist_to_center'] >= R)]

    return valid_points


def cluster_extremities(distance_df, intersection_candidates, R, L, extremity_merging_cluster_dist, min_cl_size, 
                                max_dist_from_intersection, epsilon=12, min_samples=5, max_nr_points=1000):
    """
    TODO: UPDATE DESCRIPTION
    Finds the subclusters that are within the radius > R and < R + L. Returns a frame with one row per ping inside this area, indicating its subcluster. 
    @param extremity_merging_cluster_dist: max distance to merge points in extremities into one cluster
    @param min_cl_size: minimal size of extremity cluster to be counted
    @param max_dist_from_intersection: The radius around an intersection candidate that we require a track to pass by in order for its GPS points in the extremity to be considered.
    """
    distance_df = distance_df.copy()
    extremity_clusters_list = []
    
    for cl_idx in tqdm(range(len(intersection_candidates)), desc="Verifying candidates", unit="candidate"):
        distance_df_subset = distance_df.loc[distance_df["cl_idx"] == cl_idx]
        cl = intersection_candidates.iloc[cl_idx]
        distance_df_subset_validated = validate_points_in_annulus(all_points=distance_df_subset, node_center=(cl["x"], cl["y"]),
                                                                          R=R, L=L,
                                                                          max_dist_from_intersection=max_dist_from_intersection,
                                                                          x_var="ping_x", y_var="ping_y", epsilon=epsilon, 
                                                                          min_samples=min_samples, max_nr_points=max_nr_points)

        if not distance_df_subset_validated.empty:
            distance_df_subset_validated["subcluster"] = cluster_simple_distance(distance_df_subset_validated, ['ping_x', 'ping_y'],
                                                                     extremity_merging_cluster_dist)

            extremity_clusters_list.append(distance_df_subset_validated)

    extremity_clusters_df = pd.concat(extremity_clusters_list, axis=0)

    extremity_clusters_df['subcluster'] = [str(i) + '_' + str(j) for i, j in zip(extremity_clusters_df['cl_idx'],
                                                                             extremity_clusters_df['subcluster'])]
    subcluster_sizes = extremity_clusters_df.groupby(['cl_idx', 'subcluster']).size()
    subcluster_sizes = subcluster_sizes[subcluster_sizes >= min_cl_size]
    subcluster_sizes = subcluster_sizes.reset_index(level=1)
    extremity_clusters_df = extremity_clusters_df[extremity_clusters_df['subcluster'].isin(subcluster_sizes['subcluster'])]

    return extremity_clusters_df



def filter_out_non_intersections(nodes_info, mx_df_extremities_ini):
    """
    Update nr_roads in cluster_info and return only candidate intersections with nr_roads>=3
    """
    mx_df_extremities = mx_df_extremities_ini.copy()
    nodes_info['nr_roads'] = mx_df_extremities.groupby('cl_idx')['subcluster'].nunique()
    nodes_info = nodes_info[nodes_info['nr_roads'] >= 3]
    mx_df_extremities = mx_df_extremities[mx_df_extremities['cl_idx'].isin(nodes_info.index)]
    nodes_info["in_type"] = "intersection"

    return nodes_info, mx_df_extremities


def kdtree_neighbors(df1, df2, r=25, p=2, eps=0):
    ''' Takes in two dataframes, returns a list with length of df2. The elements in the list is a list of the indexes of the elements in df1 that is closter than r'''
    tree1 = df_to_kdtree(df=df1, columns=['x', 'y'])
    tree2 = df_to_kdtree(df=df2, columns=['x', 'y'])
    neighbors = tree2.query_ball_tree(tree1, r=r)

    return neighbors

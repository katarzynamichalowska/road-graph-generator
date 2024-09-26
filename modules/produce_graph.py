from produceGraph.wp3.dtos.loader_site import LoaderSites
import logging
from math import cos, asin, sqrt
from produceGraph.wp3.dtos.loader import Loaders
from produceGraph.converter import convert
from produceGraph.wp3.dtos.vehicle import Vehicles
from produceGraph.wp3.schema import loaders_schema, vehicles_schema, loader_side_schema
from produceGraph.wp3.dtos.output import Wp3Output
import pandas as pd
import numpy as np
from scipy.spatial import distance
import produceGraph.road_data_manipulation as rdm
import produceGraph.data_manipulation as dm
import json
import importlib
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')


logger = logging.getLogger('producegraph')

# reloading rdm in order not to restart notebook when changing functions


# TODO: Document according to reStructuredText https://realpython.com/documenting-python-code/
"""Gets and prints the spreadsheet's header columns

:param file_loc: The file location of the spreadsheet
:type file_loc: str
:param print_cols: A flag used to print the columns to the console
    (default is False)
:type print_cols: bool
:returns: a list of strings representing the header columns
:rtype: list
"""

# TODO: Format code according to pep8
# = in definitions have space around
# = in arguments have no spaces
# : and , are usually is followed by space
# Line length
# remove ;

# TODO:
# Collect ALL variables in settings file instead of having some hardcoded

# Functions to find the altitude of the closest point of the found dump points. Based on this: https://stackoverflow.com/questions/41336756/find-the-closest-latitude-and-longitude
# The Haversine formula is used for finding the distance between points


def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295
    hav = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p) * \
        cos(lat2*p) * (1-cos((lon2-lon1)*p)) / 2
    return 12742 * asin(sqrt(hav))


import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

def cluster_and_categorize_points(mx_df, intersection_candidates, epsilon, min_samples, R, L, proximity_threshold, x_var="ping_x", y_var="ping_y"):
    """
    Cluster GPS points around intersection candidates and categorize them into different groups.

    Parameters:
        mx_df (DataFrame): DataFrame containing the GPS points with clustering information.
        intersection_candidates (DataFrame): DataFrame containing intersection candidate points.
        epsilon (float): The epsilon parameter for DBSCAN clustering.
        min_samples (int): The minimum number of samples required to form a cluster in DBSCAN.
        L (float): The offset distance for filtering points in the annulus.
        proximity_threshold (float): The maximum distance from the intersection to consider a point valid.
        x_var (str): Column name for x coordinates in mx_df.
        y_var (str): Column name for y coordinates in mx_df.
        R2 (float): Inner radius for extremity cluster filtering.

    Returns:
        tuple: A tuple containing three DataFrames:
            - points_df: All points grouped by clusters.
            - valid_points_df: Points that are valid according to the proximity threshold.
            - extremity_clusters_df: Points categorized as extremity clusters.
    """
    # Initialize lists to store results
    points_list, valid_points_list, extremity_clusters_list = [], [], []
    
    # Iterate through each intersection candidate
    for cl_idx in range(len(intersection_candidates)):
        mx_df_subset = mx_df[mx_df["cl_idx"] == cl_idx]
        cl = intersection_candidates.iloc[cl_idx]
        node_center_x, node_center_y = cl["x"], cl["y"]
        
        # Calculate distance to center
        mx_df_subset['dist_to_center'] = np.sqrt((mx_df_subset[x_var] - node_center_x)**2 + (mx_df_subset[y_var] - node_center_y)**2)

        # Filter points within the R + L radius
        points_within_radius = mx_df_subset[mx_df_subset['dist_to_center'] <= R + L]

        # Apply DBSCAN clustering
        dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
        points_within_radius['cluster'] = dbscan.fit_predict(points_within_radius[[x_var, y_var]])

        # Store all points in a cluster
        points_list.append(points_within_radius)

        # Filter valid clusters
        valid_clusters = points_within_radius[points_within_radius['dist_to_center'] < proximity_threshold]['cluster'].unique()
        valid_points = points_within_radius[points_within_radius['cluster'].isin(valid_clusters)]
        valid_points_list.append(valid_points)

        # Filter extremity clusters
        extremity_clusters = valid_points[valid_points['dist_to_center'] >= R]
        extremity_clusters_list.append(extremity_clusters)

    # Concatenate all lists into DataFrames
    points_df = pd.concat(points_list, ignore_index=True)
    valid_points_df = pd.concat(valid_points_list, ignore_index=True)
    extremity_clusters_df = pd.concat(extremity_clusters_list, ignore_index=True)

    return points_df, valid_points_df, extremity_clusters_df


def get_center_connected_trips(mx_df, R, L):
    """ 
    Returns only the GPS points that are part of a trip which also has at least one of the 50 neighboring points 
    around the part in the annulus L+R in the center of the cluster.
    """
    mx_df_connected = pd.DataFrame()

    for i, k in mx_df.groupby("cl_idx"):
        # Identify trips with points in the center
        trips_in_center = k[k["dist"] < R]["TripLogId"].unique()

        for trip_id in trips_in_center:
            # Filter the trip
            trip_points = k[k["TripLogId"] == trip_id]

            # Check for points within the annulus L+R
            points_in_annulus = trip_points[(trip_points["dist"] >= L) & (
                trip_points["dist"] <= L + R)]

            # For each point in the annulus, check the 50 neighboring points
            for idx, point in points_in_annulus.iterrows():
                # Define the range for neighboring points
                start_idx = max(idx - 50, 0)
                end_idx = min(idx + 50, len(trip_points) - 1)

                # Extract neighboring points
                neighbors = trip_points.iloc[start_idx:end_idx + 1]

                # Check if any neighboring points are in the center
                if any(neighbors["dist"] < R):
                    # If a point is found, add the entire trip to the connected dataframe
                    mx_df_connected = pd.concat(
                        [mx_df_connected, trip_points], ignore_index=True)
                    break  # No need to check further points for this trip

    return mx_df_connected


def closest(data, v):
    m = min(data, key=lambda p: distance(
        v['Latitude'], v['Longitude'], p['Latitude'], p['Longitude']))
    return data[data.index(m)]["Altitude"]


# Adding altitude and longitude for the load and dump positions by finding the altitude of the closest timestamp in the tracking data
def add_altitude_load_dump(raw_data_init):
    df = raw_data_init.copy()

    def add_altitude(x):
        x.set_index("Timestamp", inplace=True)
        dump_time = x.iloc[0]["DumpDateTime"]
        load_time = x.iloc[0]["LoadDateTime"]
        x = x[~x.index.duplicated(keep='first')]
        dump_id = x.index.get_indexer([dump_time], method='nearest')
        load_id = x.index.get_indexer([load_time], method='nearest')
        dump_altitude = x.iloc[dump_id[0]]["Altitude"]
        load_altitude = x.iloc[load_id[0]]["Altitude"]
        x["LoadAltitude"] = load_altitude
        x["DumpAltitude"] = dump_altitude
        x.reset_index(inplace=True)
        return x
    df = df.groupby("TripLogId").apply(
        lambda x: add_altitude(x)).reset_index(drop=True)
    return df


def make_mass_transport_problem(events, tracking, config, fast_alg=False):
    """ Reads in tracking and events file and produces the graph for the mass transportation problem. 
    :param events: Dataframe with one row per trip, and colums:
    "Id", "LoadDateTime", "DumpDateTime", "LoadLongitude", "LoadLatitude", "DumpLongitude",      "DumpLatitude", "LoaderMachineName", "DumperMachineName"
    :type events: pd.DataFrame
    :param tracking. Dataframe with one line per gps point, and columns "Id", "TripLogId", "Coordinate.Latitude", "Coordinate.Longitude",  
    "Speed", "Distance", "Course". The index should be the timestamp of the gps point, and be of the type datetimeindex. 
    :type trackin: pd.DataFrame
    :param config: Config for algorithimc parameters that are tunable. 
    :type config:configparser.ConfigParser
    :return mass_transporation_problem: Definition of the problem that is used by the planner
    :type mass_transportation: json string
    At the moment the function also returns graph,proj_info and adjacent_nodes_info in order to easily being able to visualize the graph. 
    This should be removed when development and tuning is done """
    raw_data, proj_info = read_data(events, tracking)
    raw_data = add_altitude_load_dump(raw_data)
    df, trips = rdm.preprocess_data(raw_data, proj_info,
                                    config["graph"].getfloat(
                                        'endpoint_threshold'),
                                    config["graph"].getboolean(
                                        'remove_endpoints'),
                                    config["graph"].getint(
                                        'interp_spline_deg'),
                                    config["graph"].getfloat(
                                        'interp_resolution_m'),
                                    config["graph"].getfloat(
                                        'min_trip_length'),
                                    config["graph"].getfloat(
                                        'divide_trip_thr_mins'),
                                    config["graph"].getfloat(
                                        'divide_trp_thr_angle'),
                                    add_vars_trips=['Speed', 'Distance', 'timestamp_s', "x", "y", "Altitude"])
    nodes_info = get_nodes(raw_data, df, trips, proj_info, config)
    # TODO determine a home for this drop
    raw_data = raw_data.drop(columns=["Course"])
    nodes_info, adjacent_nodes_info, edges, trips_annotated, mx_dist, trips, mx_df, relevant_trips = produce_graph_elements(trips, nodes_info, proj_info,
                                                                                                                            config['graph'],
                                                                                                                            fast_alg=fast_alg)
    graph = produce_graph(proj_info, nodes_info, adjacent_nodes_info, edges)
    loaders = produce_loaders(nodes_info)
    loaders = Loaders.from_dict(loaders_schema.validate(json.dumps(loaders)))
    vehicles = produce_vehicles(raw_data, proj_info)
    vehicles = Vehicles.from_dict(
        vehicles_schema.validate(json.dumps(vehicles)))
    loaders_sites_and_task = produce_load_and_tasks(nodes_info)[0]
    loaders_sites_and_task = LoaderSites.from_dict(
        loader_side_schema.validate(json.dumps(loaders_sites_and_task)))
    wp3_inputs = Wp3Output(
        loaders,
        vehicles,
        loaders_sites_and_task,
        graph)
    mass_transportation_problem = convert(wp3_inputs).to_json()
    return mass_transportation_problem, proj_info, adjacent_nodes_info, nodes_info, raw_data, trips, mx_df, relevant_trips


def read_data(events, tracking):
    """
    Reads in the data as downloaded from Ditio.
    :return data: Pandas dataframe containing the combined data of gps signals and metadata for the trips.
    :return proj_info: The proj_info dictionary gives the conversions between the xy coordinate system and the longitude
    latitude. It is important to always use the the same proj_info after a xy coordinate system has been defined.
    """
    tracking = tracking.drop('Id', axis=1).reset_index()
    data = tracking.merge(events, how='inner',
                          left_on='TripLogId', right_on='Id')
    data["Date"] = data["Timestamp"].apply(lambda x: x.date())
    data.rename(columns={"Coordinate.Latitude": "Latitude",
                "Coordinate.Longitude": "Longitude"}, inplace=True)
    data, proj_info = rdm.add_meter_columns(data, load=True, dump=True)
    return data, proj_info


def produce_graph_elements(trips: pd.DataFrame,
                           cluster_info: dict,
                           proj_info: dict,
                           config,
                           fast_alg=True):
    """
    Produce graph elements such as intersections, dump and load points.

    :param trips: Pandas dataframe containing the combined data of GPS signals and metadata for the trips.
    :type trips: pd.DataFrame
    :param cluster_info: Dictionary containing information about the clusters.
    :type cluster_info: dict
    :param proj_info: Dictionary containing the conversions between the xy coordinate system and the longitude latitude.
    :type proj_info: dict
    :param config: Configuration object containing various settings.
    :type config: object
    :param fast_alg: Flag indicating whether to use the fast algorithm for computing road trajectories.
    :type fast_alg: bool, optional
    :return: Tuple containing the cluster information, adjacent nodes information, edges, annotated trips,
             distance matrix, original trips dataframe, mx_df, and relevant trips.
    :rtype: tuple
    """
    max_distance = config.getint('max_distance')

    logger.debug('Preprocessing data')

    trips.reset_index(drop=True, inplace=True)

    # The function compute the distances between the center of the clusters and the datapoint in trips. The structure of
    # the return is a dictionary on the form (iloc trips,cluster number):distance. Only those points closer than
    # max_distance from the cluster is considered.
    mx_dist = rdm.compute_dist_matrix(cluster_info, trips, max_distance)

    # Takes the mx_dist dict and makes a dataframe of information about the points that are close to a node.
    # Each row represent a trip point that is within the max_distance of a node. The columns of the
    # output frame are: 'cl_lat', 'cl_lon', 'cl_x', 'cl_y', 'in_type', 'Name','TripLogId', 'ping_lat', 'ping_lon',
    # 'Speed', 'Distance', 'timestamp_s','ping_x', 'ping_y', 'ping_idx', 'cl_idx', 'dist'. Everything cl + (in_type,Name)
    # refers to the node, while the rest refers to the trip point within the radius of the node. The
    # variable dist is the distance that the trip point is from the decision point.
    mx_df = rdm.preprocess_mx_dist(mx_dist, trips, cluster_info)

    # Returns the dataframe with the decision points where for each decision point the trip log id of all the trips
    # passing by that datapoint is added.
    cluster_info = rdm.cluster_info_trips(cluster_info, mx_df, max_distance)

    # Return:
    # TODO: Move to documentation of function
    # adjacent_nodes_info: Every set of two nodes (decision points) that are connected by one or more trips gets an id.
    # A frame with columns edge_id,(from_node, to_node), tripLogId
    # trips annotated: Same frame as trips, but for each row it is added [cl_idx] which node it is close to. If it not
    # within the max_distance of any node, cl_idx takes the value nan.
    # mx_dist. It takes in mx_df and returns it as mx_dist. This is a little strange??
    # TODO: Agree, this is strange use of notation, clean up!

    adjacent_nodes_info, trips_annotated = rdm.find_relations(trips,
                                                              cluster_info,
                                                              mx_df,
                                                              max_distance)

    # To make the code run faster we only use 3 trips to calculate the trajectory between two nodes
    # TODO: Do we use three or one?
    if config.getboolean('without_direction'):
        # Here we change the adjacent_nodes info such that it is independent of direction. Entries that represent
        # the same edge but with opposite direction are merged.
        adjacent_nodes_info["nodes"] = adjacent_nodes_info["nodes"].apply(
            lambda x: tuple(sorted(x)))
        adjacent_nodes_info = rdm.make_no_directional_nodes_info(
            adjacent_nodes_info.copy())
        # WARNING: At the moment we are only using the fast algorithm, it is not verified and tested without. The fast
        # algorithm just uses the shortest trip between two nodes to describe the edge.
        # edges is a list of dataframe with columns edge_id, latitude, longitude. It describes the positions along the
        # edge. The first and last position on the edge is always in the nodes.
        edges, relevant_trips = rdm.compute_road_trajectories_without_direction(cluster_info,
                                                                                trips_annotated=trips_annotated,
                                                                                adjacent_nodes_info=adjacent_nodes_info,
                                                                                fast_alg=fast_alg)

    # WARNING: This option where we care about the direction between the nodes is currently not used, I (Helga) have not
    # tested this option after changes have been made in other places in the code.
    else:
        edges = rdm.compute_road_trajectories(trips_annotated=trips_annotated,
                                              adjacent_nodes_info=adjacent_nodes_info,
                                              edge_type='dba',
                                              max_trips_per_edge=3)

    # mx_dist.drop(columns=["id"], inplace=True) #because id and cl_idx are the same
    return cluster_info, adjacent_nodes_info, edges, trips_annotated, mx_dist, trips, mx_df, relevant_trips


def produce_graph(proj_info, cluster_info_truth, adjacent_nodes_info, edges):
    """
    Creates a graph from the data that can be written to file.

    :param proj_info: Dictionary providing conversion from cartesian coordinates to geographical.
    :param cluster_info_truth: TODO Note that we change this parameter in the function!
    :param adjacent_nodes_info: TODO
    :param edges: TODO

    :return Graph dictionary.
    """

    def _make_coordinate(t):
        """Get coordinate from a point t."""
        return [round(t["Longitude"], 10), round(t["Latitude"], 10), round(t["z"], 10)]

    # Set up geographical coordinates in the cluster_info_truth dataframe
    cluster_info_truth["coordinates"] = cluster_info_truth[["Longitude", "Latitude", "z"]].apply(_make_coordinate, axis=1)

    # Use these to set up the nodes in the graph.
    out = {}
    out["Nodes"] = cluster_info_truth[["id", "coordinates"]].to_dict("records")

    # Now we will define the edges
    for_input = adjacent_nodes_info.copy()[["nodes", "edge_id"]]
    for_input.set_index("edge_id", drop=False, inplace=True)

    edges_d = []
    for i in edges:
        r = {}
        if len(i) == 0:
            continue
        e = i["edge_id"].unique()[0]
        if e not in for_input.index:
            continue
        r["id"] = e
        i = rdm.add_meter_columns(i, proj_info=proj_info)[0]
        i["coordinates"] = i.apply(_make_coordinate, axis=1)
        r.update(i[["coordinates"]].to_dict("list"))
        r["description"] = "road between two nodes"
        r["Node1"] = {"nodeId": for_input.loc[e]["nodes"][0]}
        r["Node2"] = {"nodeId": for_input.loc[e]["nodes"][1]}
        edges_d.append(r)

    out["Edges"] = edges_d
    out["id"] = "MC1Network"
    out["coordinate_system"] = {
        "type": "Cartesian2D", "origoInLongitudeLatitude": list(proj_info["origin"])}
    out["coordinate_system"]["origoInLongitudeLatitude"][0] = proj_info["origin"][1]
    out["coordinate_system"]["origoInLongitudeLatitude"][1] = proj_info["origin"][0]
    graph = {}
    graph["Network"] = out

    return graph

    # with open("exampleNetwork/Network.json", 'w') as file:
    #    file.write(json.dumps(graph, indent=2))


def get_nodes(data_ini, df_ini, trips_ini, proj_info, config):
    """
    Obtain a dataframe with decision points based on a set of tracks.

    :param data: dataframe containing raw data from file
    :param proj_info: conversion between xy and latlong
    :param config: Configuration for running the algorithm
    :param max_dump_pos: The maximum dump positions that can be found. This will be the number of clusters that we
    start the kmeans with before removing to small clusters and merging clusters that are to close
    :param merge_dump_pos: If two dump positions are closer in radius than this radius they are merge
    :param min_dump_points: the minimum amount of dumps that has to happen for it to be defined as a dump point
    :return cluster_info: dataframe containing all decisions points for the graph.
    """

    data = data_ini.copy()
    df = df_ini.copy()
    trips = trips_ini.copy()

    if config["io"].getboolean('manual_intersection'):
        cluster_info = pd.read_csv(config["io"]['intersection_file_input']).reset_index(
        ).rename(columns={"index": "id"})
    else:
        config_inter = config["intersections"]
        intersection_candidates = rdm.df_turns_neighbour_similarity(df,
                                                                    config_inter.getint(
                                                                        'res_step'),
                                                                    config_inter.getfloat(
                                                                        'neighbour_dist'),
                                                                    config_inter.getfloat('similarity_thre'))
        intersection_candidates = rdm.get_cluster_centres(intersection_candidates,
                                                          config_inter.getint('max_merging_cluster_dist'))
        intersection_candidates = intersection_candidates[intersection_candidates['cl_size'] > 1].reset_index(
            drop=True)
        # Compute the distances from intersection candidates to neighboring points of interpolated trips. Returns a dictonary, mx_dist,  with keys being the index of the two frames
        max_distance = config_inter.getfloat("R") + config_inter.getfloat("L")
        # Returns a frame containg all points that are within the max_distance from canditate clusters
        mx_dist = rdm.compute_dist_matrix(intersection_candidates, trips[["x", "y"]], max_distance)
        mx_df = rdm.preprocess_mx_dist(mx_dist, trips, intersection_candidates)
        # Compute the number of extremity clusters around each candidate intersection
        # TODO: This function returns faulty intersections with parallel roads.
        mx_df_extremities = rdm.mx_extremity_clusters(mx_df, config_inter.getfloat('R'),
                                                      config_inter.getfloat('L'),
                                                      config_inter.getfloat('extremity_merging_cluster_dist'),
                                                      config_inter.getint('max_extremity_cluster_size'),
                                                      config_inter.getint('nr_trips'),
                                                      config_inter.getfloat('max_dist_from_intersection'))
        intersection_candidates, mx_df_extremities = rdm.update_frames(intersection_candidates, mx_df_extremities)
        cluster_info = intersection_candidates[["x", "y", "Longitude", "Latitude", "Altitude"]].reset_index().rename(columns={"index": "id"})
        cluster_info.rename(columns={"Altitude": "z"}, inplace=True)

    cluster_info["in_type"] = "road"
    cluster_info["Name"] = "Road"
    # First obtain the load and dump points.
    load = get_load_points(data, proj_info, radius=config["graph"].getint('merge_dump_pos'),)
    dump = get_dropoff_points_db(data, proj_info, radius=config["graph"].getint('merge_dump_pos'), eps=0.001)
    cluster_info = pd.concat([cluster_info, dump], ignore_index=True)
    cluster_info = pd.concat([cluster_info, load], ignore_index=True)
    cluster_info = cluster_info[["Latitude", "Longitude",
                                 "z", "in_type", "Name"]].reset_index(drop=True)
    # Only copying lat, lon, and then adding meter information making sure to use the right proj_info
    cluster_info, _ = rdm.add_meter_columns(cluster_info, proj_info=proj_info)
    cluster_info["id"] = cluster_info.index
    return cluster_info


def produce_load_and_tasks(cluster_info):
    """

    :param cluster_info:
    :return:
    """
    loads_nodes = cluster_info[cluster_info["in_type"]== "load"][["id", "Name"]]
    dump_nodes = cluster_info[cluster_info["in_type"]== "dump"][["id", "Name"]]
    # right now there is only one loader per loading site. This dict is a temp solution to keep track of the ids
    d_to_loaders = {}

    from datetime import datetime
    # TODO: Any reason for not importing at the top?
    out = []
    idcount = 0
    task = 100
    for i, row in loads_nodes.iterrows():
        d = {}
        d["Id"] = idcount
        d["Location"] = {"Node": {"Id": row["id"]}}
        d["Type"] = "loading"
        d["Tasks"] = [{"Id": task, "Product": "Sand", "StartLevelMass": 10000, "TargetLevelMass": 0,
                       "PlannedCompletionTime": datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}]
        out.append(d)
        idcount += 1
        d_to_loaders[row["Name"]] = idcount
    for i, row in dump_nodes.iterrows():
        d = {}
        d["Id"] = idcount
        d["Location"] = {"Node": {"Id": row["id"]}}
        d["Type"] = "unloading"
        d["Tasks"] = [{"Id": task, "Product": "Sand", "StartLevelMass": 0, "TargetLevelMass": 10000,
                       "PlannedCompletionTime": datetime(2017, 11, 28, 23, 55, 59).strftime("%Y-%m-%dT%H:%M:%S")}]
        out.append(d)
        idcount += 1
        task += 1

    graph = {}
    graph["LoaderSites"] = out
    return graph, d_to_loaders


def produce_loaders(cluster_info):
    """Not used at the moment?"""

    d_to_loaders = produce_load_and_tasks(cluster_info)[1]
    loads_nodes = cluster_info[cluster_info["in_type"]
                               == "load"][["id", "Name"]]
    out = []
    for i, row in loads_nodes.iterrows():
        d = {}
        d["Id"] = row["Name"]
        d["MaxWeight"] = 500
        d["WeightIndependentLoadingTime"] = "00:03:00"
        d["MassLoadingSpeed"] = 165
        d["VolumeLoadingSpeed"] = 105
        d["LoaderSiteId"] = d_to_loaders.get(row["Name"])
        out.append(d)
    graph = {}
    graph["Loaders"] = out
    return graph


def produce_vehicles(data, proj_info):
    """Not used at the moment?"""

    def coordinate(t):
        return [t["Longitude"], t["Latitude"]]
    data = rdm.add_meter_columns(data, proj_info=proj_info)[0]
    data["x"] = data["x"].astype(float)
    data["y"] = data["y"].astype(float)
    data["coordinates"] = data[["Longitude", "Latitude"]].apply(
        coordinate, axis=1)

    out = []
    for i, k in data.groupby("DumperMachineName"):
        d = {}
        d["Id"] = i
        k.sort_values(by="Timestamp", inplace=True)
        k.iloc[0][["coordinates"]]

        d["WeightCapacity"] = 30  # In standard mass units.
        d["VolumeCapacity"] = 30  # In standard mass units.
        d["Speed"] = 16  # In m/s. This is the default speed when not feeding or brushing
        d["ParkingPosition"] = k.iloc[0]["coordinates"]
        d["Length"] = 1000  # //cm
        d["Width"] = 500  # , //cm
        d["WorkingHoursStart"] = (
            k["Timestamp"].min().strftime("%Y-%m-%dT%H:%M:%S"))
        d["WorkingHoursEnd"] = k["Timestamp"].max().strftime("%Y-%m-%dT%H:%M:%S")
        out.append(d)
        graph = {}
        graph["Vehicles"] = out
    return graph


def get_load_points(data, proj_info, radius):
    """ Takes in the combined data frame and returns a frame containing the load positions. A load position is defined
    by the trimmed average position of a loader in the input data. 
    
    WARNING: This might be a problem if one loader is placed 
    in more than one position during the day with lots of loads in each position or if the input data covers more than one day.
    
    :param data: The data frame containing raw data. Contains a LoaderMachineName and load_x and load_y columns.
    :param proj_info: conversion between xy and latlong
    :return: data frame with load positions and names
    """

    loads = pd.DataFrame()
    xload = []
    yload = []
    zload = []

    # The below loop might look nicer in numpy-ish, but it's probably less important
    for i, frame in data.groupby(["LoaderMachineName", "TaskId"], dropna=False):
        # Lists of all the unique positions for each dumper.
        x, y, z = list(set(frame["load_x"])), list(
            set(frame["load_y"])), list(set(frame["Altitude"]))
        if len(x) < 3:
            logger.debug(
                f'Skipping loader machine {i} - less than 10 positions available')
            continue

        # Find x and y positions that are within one standard deviation of the average x and y positions
        xav, yav, zav = np.mean(x), np.mean(y), np.mean(z)
        xstd, ystd, zav = np.std(x), np.std(y), np.mean(z)
        trimmed_points = [p for p in zip(x, y, z) if (
            abs(xav-p[0]) <= xstd) and (abs(yav-p[1]) <= ystd)]

        # Now represent the loader position with the average of those positions
        xload.append(np.mean([p[0] for p in trimmed_points]))
        yload.append(np.mean([p[1] for p in trimmed_points]))
        zload.append(np.mean([p[2] for p in trimmed_points]))
    loads["x"] = xload
    loads["y"] = yload
    loads["z"] = zload
    # Project to latlong and store in the data frame
    loads["Latitude"] = loads["y"].apply(rdm.metres_to_coord,
                                         args=(proj_info.get("origin")[0],
                                               proj_info.get("delta_y")))
    loads["Longitude"] = loads["x"].apply(rdm.metres_to_coord,
                                          args=(proj_info.get("origin")[1],
                                                proj_info.get("delta_x")))

    new_loads, close_points = rdm.get_cluster_to_merge(
        loads.copy(), "loads", radius=radius)

    new_loads["Name"] = new_loads.index
    new_loads["Name"] = new_loads["Name"].apply(lambda x: str(x))
    new_loads["in_type"] = "load"
    logger.debug(f"Computed load points.")
    # Adding the altitude to the load points
    data_list = data[["Longitude", "Latitude", "Altitude"]].to_dict('records')
    new_loads = rdm.add_meter_columns(new_loads, proj_info)[0]
    new_loads["z"] = new_loads.apply(lambda x: closest(data_list, x), axis=1)

    return new_loads


def get_dropoff_points(data, proj_info, n_cluster, radius, min_dump_points):
    """
    Obtain the dump points for a set of tracks. The points are clustered and cluster centers are merged if closer to
    each other than a given radius.

    :param data: dataframe containing intersections found manually
    :param proj_info: conversion between xy and latlong
    :param n_cluster: The number of clusters kmeans should find before removing small clusters and merging clusters
    that are to close each other
    :param radius: If two dump positions are closer in radius than this radius they are merged
    :param min_dump_points: the minimum amount of dumps that has to happen for it to be defined as a dump point
    :return dump: data frame containing the dump positions
    """

    # Functions to find the altitude of the closest point of the found dump points. Based on this: https://stackoverflow.com/questions/41336756/find-the-closest-latitude-and-longitude
    # The Haversine formula is used for finding the distance between points
    def distance(lat1, lon1, lat2, lon2):
        p = 0.017453292519943295
        hav = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p) * \
            cos(lat2*p) * (1-cos((lon2-lon1)*p)) / 2
        return 12742 * asin(sqrt(hav))

    def closest(data, v):
        m = min(data, key=lambda p: distance(
            v['Latitude'], v['Longitude'], p['Latitude'], p['Longitude']))
        return data[data.index(m)]["Altitude"]

    # Here we make a list of all the locations of the dump points. There is one dump point per trip
    trips = data.groupby("TripLogId")[
        ["DumpLatitude", "DumpLongitude"]].first()
    X = trips.to_numpy()
    # TODO: Consider trying some other clustering algorithm that doesn't assume sphericity or fixed number of clusters
    # Spectral?
    kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(
        X)  # TODO re-enable randomization?
    center = kmeans.cluster_centers_
    cluster_map = pd.DataFrame()
    cluster_map['data_index'] = trips.index.values
    cluster_map['cluster'] = kmeans.labels_

    point_count = cluster_map["cluster"].value_counts(
    ).reset_index().sort_values(by="index").reset_index()["cluster"]
    dropoff = pd.DataFrame(columns=["Latitude", "Longitude", "in_type"])
    dropoff["Latitude"] = center[:, 0]
    dropoff["Longitude"] = center[:, 1]
    dropoff["in_type"] = "dump"
    dropoff["Name"] = "Dump"
    dropoff["count"] = point_count

    dropoff = dropoff[dropoff["count"] > min_dump_points].reset_index()
    dropoff = rdm.add_meter_columns(dropoff, proj_info)[0]
    new_dropoff, close_points = rdm.get_cluster_to_merge(dropoff.copy(), "dump", radius=radius)
    new_dropoff = rdm.add_meter_columns(new_dropoff, proj_info)[0]

    # Adding the altitude to the dump points
    data_list = data[["Longitude", "Latitude", "Altitude"]].to_dict('records')
    new_dropoff["z"] = new_dropoff.apply(lambda x: closest(data_list, x), axis=1)

    return new_dropoff


def get_dropoff_points_db(data, proj_info, radius, eps=0.0001):
    """
    Obtain the dump points for a set of tracks. The points are clustered and cluster centers are merged if closer to
    each other than a given radius.

    :param data: dataframe containing intersections found manually
    :param proj_info: conversion between xy and latlong
    :param n_cluster: The number of clusters kmeans should find before removing small clusters and merging clusters
    that are to close each other
    :param radius: If two dump positions are closer in radius than this radius they are merged
    :param min_dump_points: the minimum amount of dumps that has to happen for it to be defined as a dump point
    :return dump: data frame containing the dump positions
    """

    # Functions to find the altitude of the closest point of the found dump points. Based on this: https://stackoverflow.com/questions/41336756/find-the-closest-latitude-and-longitude
    # The Haversine formula is used for finding the distance between points
    def distance(lat1, lon1, lat2, lon2):
        p = 0.017453292519943295
        hav = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p) * \
            cos(lat2*p) * (1-cos((lon2-lon1)*p)) / 2
        return 12742 * asin(sqrt(hav))

    def closest(data, v):
        m = min(data, key=lambda p: distance(
            v['Latitude'], v['Longitude'], p['Latitude'], p['Longitude']))
        return data[data.index(m)]["Altitude"]

    # Here we make a list of all the locations of the drop-off points. There is one drop-off point per trip
    trips = data.groupby("TripLogId")[
        ["DumpLatitude", "DumpLongitude"]].first()
    X = trips.to_numpy()
    db = DBSCAN(eps=eps).fit(X)
    trips["labels"] = db.labels_
    dropoff = pd.DataFrame(columns=["Latitude", "Longitude"])

    for i, k in trips.groupby("labels"):
        latitude = k["DumpLatitude"].mean()
        longitude = k["DumpLongitude"].mean()
        dropoff.loc[len(dropoff.index)] = [latitude, longitude]

    dropoff["in_type"] = "dump"
    dropoff["Name"] = "Dump"
    dropoff = rdm.add_meter_columns(dropoff, proj_info)[0]
    new_dropoff = dropoff
    new_dropoff = rdm.add_meter_columns(new_dropoff, proj_info)[0]
    data_list = data[["Longitude", "Latitude", "Altitude"]].to_dict('records')

    new_dropoff["z"] = new_dropoff.apply(lambda x: closest(data_list, x), axis=1)
    return new_dropoff

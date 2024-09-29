import logging
from math import cos, asin, sqrt
import pandas as pd
import numpy as np
from scipy.spatial import distance
import modules.road_data_manipulation as rdm
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')
logger = logging.getLogger('producegraph')


def distance_haversine(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295
    hav = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p) * \
        cos(lat2*p) * (1-cos((lon2-lon1)*p)) / 2
    return 12742 * asin(sqrt(hav))



def intersection_validation_cluster_points(mx_df, intersection_candidates, epsilon, min_samples, R, L, proximity_threshold, x_var="ping_x", y_var="ping_y"):
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


def closest(data, v):
    m = min(data, key=lambda p: distance_haversine(v['Latitude'], v['Longitude'], p['Latitude'], p['Longitude']))
    return data[data.index(m)]["Altitude"]


def add_altitude_load_dump(raw_data_init):
    """Adding altitude and longitude for the load and drop-off positions by finding the altitude of the closest timestamp in the tracking data"""

    df = raw_data_init.copy()

    def add_altitude(x):
        x.set_index("Timestamp", inplace=True)
        dropoff_time = x.iloc[0]["DumpDateTime"]
        load_time = x.iloc[0]["LoadDateTime"]
        x = x[~x.index.duplicated(keep='first')]
        dropoff_id = x.index.get_indexer([dropoff_time], method='nearest')
        load_id = x.index.get_indexer([load_time], method='nearest')
        dump_altitude = x.iloc[dropoff_id[0]]["Altitude"]
        load_altitude = x.iloc[load_id[0]]["Altitude"]
        x["LoadAltitude"] = load_altitude
        x["DumpAltitude"] = dump_altitude
        x.reset_index(inplace=True)
        return x
    
    df = df.groupby("TripLogId").apply(lambda x: add_altitude(x)).reset_index(drop=True)
    return df


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
        x, y, z = list(set(frame["load_x"])), list(set(frame["load_y"])), list(set(frame["Altitude"]))

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

    new_loads["in_type"] = "load"
    logger.debug(f"Computed load points.")
    data_list = data[["Longitude", "Latitude", "Altitude"]].to_dict('records')
    new_loads = rdm.add_meter_columns(new_loads, proj_info)[0]
    new_loads["Altitude"] = new_loads.apply(lambda x: closest(data_list, x), axis=1)

    return new_loads


def get_dropoff_points_dbscan(data, proj_info, eps=0.001):
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

    # Functions to find the altitude of the closest point of the found drop-off points. Based on this: https://stackoverflow.com/questions/41336756/find-the-closest-latitude-and-longitude

    def closest(data, v):
        m = min(data, key=lambda p: distance_haversine(v['Latitude'], v['Longitude'], p['Latitude'], p['Longitude']))
        return data[data.index(m)]["Altitude"]

    # Here we make a list of all the locations of the drop-off points. There is one drop-off point per trip
    trips = data.groupby("TripLogId")[["DumpLatitude", "DumpLongitude"]].first()
    X = trips.to_numpy()
    db = DBSCAN(eps=eps).fit(X)
    trips["labels"] = db.labels_
    dropoff = pd.DataFrame(columns=["Latitude", "Longitude"])

    for i, k in trips.groupby("labels"):
        latitude = k["DumpLatitude"].mean()
        longitude = k["DumpLongitude"].mean()
        dropoff.loc[len(dropoff.index)] = [latitude, longitude]

    dropoff["in_type"] = "dropoff"
    dropoff = rdm.add_meter_columns(dropoff, proj_info)[0]
    new_dropoff = dropoff
    new_dropoff = rdm.add_meter_columns(new_dropoff, proj_info)[0]
    data_list = data[["Longitude", "Latitude", "Altitude"]].to_dict('records')

    #new_dropoff["z"] = new_dropoff.apply(lambda x: closest(data_list, x), axis=1)
    new_dropoff["Altitude"] = new_dropoff.apply(lambda x: closest(data_list, x), axis=1)
    return new_dropoff


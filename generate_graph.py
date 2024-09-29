import modules.config as config
from modules.config import parse_list
import modules.road_data_manipulation as rdm
import modules.produce_graph as produce_graph
import modules.plotting_functions as pl
import modules.edge_inference as edgei
import pandas as pd
import os


script_dir = os.path.dirname(os.path.realpath(__file__))
config = config.read_settings(os.path.join(script_dir, 'config.ini'))
folder = config["io"]["data_dir"]

output_dir = config["io"]["output_dir"]
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Read and preprocess the data
events = pd.read_hdf(os.path.join(folder, config["io"]["events_data"]+".h5"))
events = events[["Id", "LoadDateTime", "DumpDateTime", "LoadLongitude", "LoadLatitude", "DumpLongitude", "DumpLatitude", "LoaderMachineName", "TaskId", "TaskDescription"]]
tracking = pd.read_hdf(os.path.join(folder, config["io"]["tracking_data"]+".h5"))
tracking = tracking[["Id", "TripLogId", "Latitude", "Longitude", "Speed", "Distance", "Course", "Altitude", "Type"]]
tracking.reset_index(inplace=True)

raw_data, proj_info = produce_graph.read_data(events, tracking)
raw_data = produce_graph.add_altitude_load_dump(raw_data)


df, trips = rdm.preprocess_data(raw_data, proj_info,
                                config["preprocessing"].getfloat('dist_endpoints_trim_metres'),
                                config["preprocessing"].getboolean('remove_endpoints'),
                                config["preprocessing"].getint('interpolation_spline_degree'),
                                config["preprocessing"].getfloat('interpolation_resolution_metres'),
                                config["preprocessing"].getfloat('min_nr_points_trip'),
                                config["preprocessing"].getfloat('divide_trip_threshold_minutes'),
                                config["preprocessing"].getfloat('divide_trip_threshold_degrees'),
                                config["preprocessing"].getfloat('divide_trip_threshold_metres'),
                                add_vars_trips=['Speed', 'Distance', 'timestamp_s', "x", "y", "Altitude"])

R1, R2 = parse_list(config["intersection_validation"].get("R"), float, sep=',')
L = config["intersection_validation"].getfloat("L")

print("\nSTEP 1 & 2: Generating 2D histograms of heading directions and identifying candidate intersections")

intersection_candidates, histogram_median_directions = rdm.calculate_neighbour_similarity(df, config["intersection_candidates"].getint('resolution_2d_histogram_metres'),
                                                                             config["intersection_candidates"].getfloat('dist_neighbour_metres'),
                                                                             config["intersection_candidates"].getfloat('similarity_threshold_neighbour_degrees'))


intersection_candidates = rdm.merge_nearby_cluster_centres(intersection_candidates, coordinates=["x", "y"], 
                                                           distance=config["intersection_candidates"].getint('dist_intersection_cluster_metres'))


# Returns a frame containg all points that are within the max_distance from canditate clusters
distance_matrix = rdm.compute_distance_matrix(intersection_candidates, trips[["x", "y"]], L+max(R1,R2))
distance_df = rdm.preprocess_distance_matrix(distance_matrix, trips, intersection_candidates)

print("\nSTEP 3: Validating candidate intersections")

# Validating intersections with R1
extremity_clusters = rdm.cluster_extremities(distance_df=distance_df, intersection_candidates=intersection_candidates,
                                              R=R1, L=L,
                                              extremity_merging_cluster_dist=config["intersection_validation"].getfloat("dist_extremity_cluster_metres"),
                                              min_cl_size=config["intersection_validation"].getint("max_extremity_cluster_size"),
                                              max_dist_from_intersection=config["intersection_validation"].getfloat("max_dist_from_intersection"),
                                              epsilon=config["intersection_validation"].getfloat("dbscan_epsilon_metres"),
                                              min_samples=config["intersection_validation"].getint("dbscan_min_samples"),
                                              max_nr_points=config["intersection_validation"].getint("max_nr_points"))

confirmed_intersections1, extremity_clusters = rdm.keep_candidates_minimum_three_roads(intersection_candidates, extremity_clusters)

# Validating intersections with R2
extremity_clusters2 = rdm.cluster_extremities(distance_df=distance_df, intersection_candidates=intersection_candidates,
                                              R=R2, L=L,
                                              extremity_merging_cluster_dist=config["intersection_validation"].getfloat("dist_extremity_cluster_metres"),
                                              min_cl_size=config["intersection_validation"].getint("max_extremity_cluster_size"),
                                              max_dist_from_intersection=config["intersection_validation"].getfloat("max_dist_from_intersection"),
                                              epsilon=config["intersection_validation"].getfloat("dbscan_epsilon_metres"),
                                              min_samples=config["intersection_validation"].getint("dbscan_min_samples"),
                                              max_nr_points=config["intersection_validation"].getint("max_nr_points"))

confirmed_intersections2, extremity_clusters2 = rdm.keep_candidates_minimum_three_roads(intersection_candidates, extremity_clusters2)

confirmed_intersections = pd.concat([confirmed_intersections1, confirmed_intersections2])
filtered_df = confirmed_intersections.drop_duplicates(subset=['Latitude', 'Longitude'])

print(confirmed_intersections.head())
confirmed_intersections = confirmed_intersections[["x", "y", "Longitude", "Latitude", "Altitude", "in_type"]]


print("\nSTEP 4: Identifying load and drop-off locations")

load = produce_graph.get_load_points(raw_data, proj_info, radius=config["load_dropoff"].getint('dist_merge_metres'),)
dropoff = produce_graph.get_dropoff_points_dbscan(raw_data, proj_info, eps=0.001)
nodes_info = pd.concat([confirmed_intersections, dropoff, load]).reset_index(drop=True)
nodes_info["id"] = nodes_info.index

nodes_info.to_csv(os.path.join(output_dir, f'cluster_info.csv'))
extremity_clusters.to_csv(os.path.join(output_dir, f'mx_df_extremities.csv'))

print("\nSTEP 5: Road inference")

distance_matrix = rdm.compute_distance_matrix(nodes_info, trips, config["road_inference"].getfloat('dist_node'))
distance_df = rdm.preprocess_distance_matrix(distance_matrix, trips, nodes_info)

nodes_info = rdm.assign_triplog_to_clusters(nodes_info, distance_df, config["road_inference"].getfloat('dist_node'))
adjacent_nodes_info, trips_with_node_info = rdm.identify_adjacent_node_trips(trips_ini=trips, mx_dist_ini=distance_df)
trips_with_node_info = edgei.mark_trips_closest_to_intersection(trips_with_node_info=trips_with_node_info)
segments_df = edgei.divide_trips_at_intersections(trips_with_node_info)

node_segment_summary = edgei.assign_segmentids_to_nodes(segments_df)
node_segment_summary = edgei.remove_duplicate_segment_ids(node_segment_summary)

segments_cluster_df = edgei.cluster_segments_connected_by_nodes(segments_df, node_segment_summary, 
                                                                epsilon=config["road_inference"].getfloat('dbscan_epsilon_metres'), 
                                                                min_samples=config["road_inference"].getint('dbscan_min_samples'))
edges_list = edgei.generate_edges(segments_cluster_df, segments_df, min_segment_length=1)

print("\nSTEP 6: Plotting the graph")

pl.plot_graph(edges_list, proj_info, nodes_info, trip=pd.DataFrame(), distance_df=pd.DataFrame(), dump=False, point=None, savename=os.path.join(output_dir, "graph"))
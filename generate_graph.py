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
events = events[["Id", "LoadDateTime", "DumpDateTime", "LoadLongitude", "LoadLatitude", "DumpLongitude", "DumpLatitude", "LoaderMachineName","DumperMachineName", "TaskId", "TaskDescription", "Comment", "Origin", "Quantity", "DumpGeoFenceLongitudes", "LoadGeoFenceLongitudes"]]
tracking = pd.read_hdf(os.path.join(folder, config["io"]["tracking_data"]+".h5"))
tracking = tracking[["Id", "TripLogId", "Coordinate.Latitude", "Coordinate.Longitude", "Speed", "Distance", "Course", "Altitude", "Type"]]
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

R, R2 = parse_list(config["intersection_validation"].get("R"), float, sep=',')
L = config["intersection_validation"].getfloat("L")

print("\nSTEP 1 & 2: Generating 2D histograms of heading directions and identifying candidate intersections")

intersection_candidates, histogram_median_directions = rdm.calculate_neighbour_similarity(df, config["intersection_candidates"].getint('resolution_2d_histogram_metres'),
                                                                             config["intersection_candidates"].getfloat('dist_neighbour_metres'),
                                                                             config["intersection_candidates"].getfloat('similarity_threshold_neighbour_degrees'))

if config["plotting"].getboolean('generate_plots'):
    x_lim = parse_list(config["plotting"].get("x_lim_plot1"), int)
    y_lim = parse_list(config["plotting"].get("y_lim_plot1"), int)
    pl.plot_candidates_detection(histogram_median_directions, raw_data, y_lim=y_lim, x_lim=x_lim, savename="candidate_intersection_heatmap.pdf")


intersection_candidates = rdm.merge_nearby_cluster_centres(intersection_candidates, coordinates=["x", "y"], 
                                                           distance=config["intersection_candidates"].getint('dist_intersection_cluster_metres'))

if config["plotting"].getboolean('generate_plots'):
    x_lim = parse_list(config["plotting"].get("x_lim_plot2"), int)
    y_lim = parse_list(config["plotting"].get("y_lim_plot2"), int)
    pl.plot_intersections_with_radii(trips=trips, mx_df_extremities=None, intersection_candidates=intersection_candidates, radii_list=[R, R+L],
                                    title=f"Initial intersection candidates with $R={R}$ and $L={L}$",
                                    y_lim=y_lim, x_lim=x_lim, savename=os.path.join(output_dir, "candidate_intersections_radii.pdf"))

# Returns a frame containg all points that are within the max_distance from canditate clusters
mx_dist = rdm.compute_dist_matrix(intersection_candidates, trips[["x", "y"]], L+max(R,R2))
mx_df = rdm.preprocess_mx_dist(mx_dist, trips, intersection_candidates)

print("\nSTEP 3: Validating candidate intersections")

# Verifying candidates is here:
mx_df_extremities = rdm.mx_extremity_clusters(mx_df_ini=mx_df, intersection_candidates=intersection_candidates,
                                              R=R, L=L,
                                              extremity_merging_cluster_dist=config["intersection_validation"].getfloat("dist_extremity_cluster_metres"),
                                              min_cl_size=config["intersection_validation"].getint("max_extremity_cluster_size"),
                                              max_dist_from_intersection=config["intersection_validation"].getfloat("max_dist_from_intersection"),
                                              epsilon=config["intersection_validation"].getfloat("dbscan_epsilon"),
                                              min_samples=config["intersection_validation"].getint("dbscan_min_samples"))

confirmed_intersections, mx_df_extremities = rdm.filter_out_non_intersections(intersection_candidates, mx_df_extremities)

nodes_info = confirmed_intersections[["x", "y", "Longitude", "Latitude", "Altitude"]].reset_index().rename(columns={"index": "id"})
nodes_info.rename(columns={"Altitude": "z"}, inplace=True)

nodes_info["in_type"] = "road"
nodes_info = nodes_info[["Latitude", "Longitude","z", "in_type"]].reset_index(drop=True)
print("\nSTEP 4: Identifying load and drop-off locations")

load = produce_graph.get_load_points(raw_data, proj_info, radius=config["load_dropoff"].getint('dist_merge_metres'),)
dropoff = produce_graph.get_dropoff_points_dbscan(raw_data, proj_info, radius=config["load_dropoff"].getint('dist_merge_metres'), eps=0.001)

nodes_info = pd.concat([nodes_info, dropoff], ignore_index=True)
nodes_info = pd.concat([nodes_info, load], ignore_index=True)
nodes_info = nodes_info[["Latitude", "Longitude", "z", "in_type"]].reset_index(drop=True)

# Only copying lat, lon, and then adding meter information making sure to use the right proj_info
nodes_info, _ = rdm.add_meter_columns(nodes_info, proj_info=proj_info)
nodes_info["id"] = nodes_info.index

nodes_info.to_csv(os.path.join(output_dir, f'cluster_info.csv'))
mx_df_extremities.to_csv(os.path.join(output_dir, f'mx_df_extremities.csv'))


print("\nSTEP 5: Road inference")

# Increased distance because sometimes a trip is very far and we dont want to duplicate edges
trips.reset_index(drop=True, inplace=True)
mx_dist = rdm.compute_dist_matrix(nodes_info, trips, config["road_inference"].getfloat('dist_node'))
mx_df = rdm.preprocess_mx_dist(mx_dist, trips, nodes_info)

nodes_info = rdm.assign_triplog_to_clusters(nodes_info, mx_df, config["road_inference"].getfloat('dist_node'))
adjacent_nodes_info, trips_with_node_info = rdm.identify_adjacent_node_trips(trips_ini=trips, mx_dist_ini=mx_df)
trips_with_node_info = edgei.mark_trips_closest_to_intersection(trips_with_node_info=trips_with_node_info)
segments_df = edgei.divide_trips_at_intersections(trips_with_node_info)

node_segment_summary = edgei.assign_segmentids_to_nodes(segments_df)
node_segment_summary = edgei.remove_duplicate_segment_ids(node_segment_summary)

segments_cluster_df = edgei.cluster_segments_connected_by_nodes(segments_df, node_segment_summary, 
                                                                epsilon=config["road_inference"].getfloat('dbscan_epsilon'), 
                                                                min_samples=config["road_inference"].getint('dbscan_min_samples'))
edges_list = edgei.generate_edges(segments_cluster_df, segments_df, min_segment_length=1)

print("\nSTEP 6: Plotting the graph")

pl.plot_graph(edges_list, proj_info, nodes_info, trip=pd.DataFrame(), mx_df=pd.DataFrame(), dump=False, point=None, savename=os.path.join(output_dir, "graph"))
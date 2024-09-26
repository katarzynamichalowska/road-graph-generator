import modules.config as config
import modules.road_data_manipulation as rdm
import modules.produce_graph as produce_graph
import modules.plotting_functions as pl
import modules.edge_inference as edgei
import pandas as pd
import os


config = config.read_settings('config.ini')
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
                                config["graph"].getfloat('endpoint_threshold'),
                                config["graph"].getboolean('remove_endpoints'),
                                config["graph"].getint('interp_spline_deg'),
                                config["graph"].getfloat('interp_resolution_m'),
                                config["graph"].getfloat('min_trip_length'),
                                config["graph"].getfloat('divide_trip_thr_mins'),
                                config["graph"].getfloat('divide_trp_thr_angle'),
                                config["graph"].getfloat('divide_trp_thr_distance'),
                                add_vars_trips=['Speed', 'Distance', 'timestamp_s', "x", "y", "Altitude"])


config_inter = config["intersections"]
R, R2 = [float(item.strip()) for item in config_inter.get("R").split(',')]
L = config_inter.getfloat("L")
res_step = config_inter.getint('res_step')
neighbour_dist = config_inter.getfloat('neighbour_dist')

config_inter = config["intersections"]

print("\nSTEP 1 & 2: Generating 2D histograms of heading directions and identifying candidate intersections")

intersection_candidates, low_res_median = rdm.calculate_neighbour_similarity(df, config_inter.getint('res_step'),
                                                                             config_inter.getfloat('neighbour_dist'),
                                                                             config_inter.getfloat('similarity_thre'))

pl.plot_candidates_detection(low_res_median, raw_data, y_lim=(15850, 16000), x_lim=(9970, 10370),
                             savename="initial_candidates_detection.pdf")

# Merging the cluster centres
intersection_candidates = rdm.get_cluster_centres(intersection_candidates,
                                                  config_inter.getint('max_merging_cluster_dist'))
intersection_candidates = intersection_candidates[intersection_candidates['cl_size'] > 1].reset_index(drop=True)


# Step 1: Candidates
y_lim, x_lim = (15800, 16100), (9680, 10400)
pl.plot_intersections_with_radii(trips=trips, mx_df_extremities=None, intersection_candidates=intersection_candidates, radii_list=[R, R + config_inter.getfloat("L")],
                                 y_lim=y_lim, x_lim=x_lim, savename=os.path.join(output_dir, "intersection_candidates.pdf"))

max_distance = L + max(R,R2)
# Returns a frame containg all points that are within the max_distance from canditate clusters
mx_dist = rdm.compute_dist_matrix(intersection_candidates, trips[["x", "y"]], max_distance)
mx_df = rdm.preprocess_mx_dist(mx_dist, trips, intersection_candidates)

epsilon = config_inter.getfloat("dbscan_epsilon_validate_connected_roads")
min_samples = config_inter.getint("dbscan_min_samples_validate_connected_roads")
proximity_threshold = config_inter.getfloat("max_dist_from_intersection")
points_list, valid_points_list, extremity_clusters_list = [], [], []
extremity_clusters_list_2 = []

print("\nSTEP 3: Validating candidate intersections")

points_df, valid_points_df, extremity_clusters_df = produce_graph.cluster_and_categorize_points(mx_df, intersection_candidates, epsilon=config_inter.getfloat("dbscan_epsilon_validate_connected_roads"), 
                                            min_samples=config_inter.getint("dbscan_min_samples_validate_connected_roads"), 
                                            R=R, L=L, proximity_threshold=config_inter.getfloat("max_dist_from_intersection"), 
                                            x_var="ping_x", y_var="ping_y")

points_df, valid_points_df, extremity_clusters_df2 = produce_graph.cluster_and_categorize_points(mx_df, intersection_candidates, epsilon=config_inter.getfloat("dbscan_epsilon_validate_connected_roads"), 
                                            min_samples=config_inter.getint("dbscan_min_samples_validate_connected_roads"), 
                                            R=R2, L=L, proximity_threshold=config_inter.getfloat("max_dist_from_intersection"), 
                                            x_var="ping_x", y_var="ping_y")
    
pl.plot_validating_intersections_schema(trips, valid_points_df=valid_points_df, intersection_candidates=intersection_candidates, 
                             extremity_clusters_df=extremity_clusters_df, R=R, L=L, 
                             max_dist_from_intersection=config_inter.getfloat("max_dist_from_intersection"), 
                             tick_resolution=50, marker_size=3, y_lim=(15845, 16000), x_lim=(9970, 10370),
                             savename=os.path.join(output_dir, "validating_intersections_schema.pdf"))

y_lim, x_lim = (14300, 15450), (11300, 12850)
confirmed_intersections = intersection_candidates.loc[((intersection_candidates["x"] > 11650) & (intersection_candidates["x"] < 11800)) | (
    (intersection_candidates["y"] > 14650) & (intersection_candidates["y"] < 14800))]
pl.plot_intersections_with_radii(trips=trips,  # mx_df_extremities=valid_points_df,
                                 extremity_clusters=extremity_clusters_df, intersection_candidates=intersection_candidates,
                                 confirmed_intersections=confirmed_intersections,
                                 radii_list=[R + L],
                                 y_lim=y_lim, x_lim=x_lim, marker_size=1, title=f"Validation with $R={R}$", savename=os.path.join(output_dir, f"validation_R_{R}.pdf"))

confirmed_intersections = intersection_candidates.loc[((intersection_candidates["x"] > 11500) & (intersection_candidates["x"] < 11800) & (
    intersection_candidates["y"] < 15100)) | ((intersection_candidates["y"] > 14650) & (intersection_candidates["y"] < 14800))]
pl.plot_intersections_with_radii(trips=trips,  # mx_df_extremities=valid_points_df,
                                 extremity_clusters=extremity_clusters_df2, intersection_candidates=intersection_candidates,
                                 confirmed_intersections=confirmed_intersections,
                                 radii_list=[R2, R2 + L],
                                 y_lim=y_lim, x_lim=x_lim, marker_size=1, title=f"Validation with $R={R2}$", savename=os.path.join(output_dir, f"validation_R_{R2}.pdf"))

df_bounded_region = low_res_median[(low_res_median['x_5'] >= x_lim[0]) & (low_res_median['x_5'] + 5 <= x_lim[1]) & (low_res_median['y_5'] >= y_lim[0]) & (low_res_median['y_5'] + 5 <= y_lim[1])]
confirmed_intersections = intersection_candidates.loc[intersection_candidates["x"] > 10200]


pl.plot_intersections(df_bounded_region=df_bounded_region, trips=trips, extremity_clusters_df=extremity_clusters_df, extremity_clusters_df2=extremity_clusters_df2,
                       confirmed_intersections1=confirmed_intersections, confirmed_intersections2=confirmed_intersections, 
                       intersection_candidates=intersection_candidates, x_lim=x_lim, y_lim=y_lim, R=30, R2=100, L=20, 
                       savename=os.path.join(output_dir, "intersection_candidates_example.pdf"))

# Verifying candidates is here:
mx_df_extremities = rdm.mx_extremity_clusters(mx_df_ini=mx_df, intersection_candidates=intersection_candidates,
                                              R=R, L=L,
                                              extremity_merging_cluster_dist=config_inter.getfloat("extremity_merging_cluster_dist"),
                                              min_cl_size=config_inter.getint("max_extremity_cluster_size"),
                                              max_dist_from_intersection=config_inter.getfloat("max_dist_from_intersection"),
                                              epsilon=config_inter.getfloat("dbscan_epsilon_validate_connected_roads"),
                                              min_samples=config_inter.getint("dbscan_min_samples_validate_connected_roads"))

intersection_candidates_2, mx_df_extremities = rdm.update_frames(intersection_candidates, mx_df_extremities)


cluster_info = intersection_candidates_2[["x", "y", "Longitude", "Latitude", "Altitude"]].reset_index().rename(columns={"index": "id"})
cluster_info.rename(columns={"Altitude": "z"}, inplace=True)

cluster_info["in_type"] = "road"
cluster_info["Name"] = "Road"
cluster_info = cluster_info[["Latitude", "Longitude","z", "in_type", "Name"]].reset_index(drop=True)
# Only copying lat, lon, and then adding meter information making sure to use the right proj_info
cluster_info, _ = rdm.add_meter_columns(cluster_info, proj_info=proj_info)
cluster_info["id"] = cluster_info.index

cluster_info.to_csv(os.path.join(output_dir, f'cluster_info_R_{R}_L_{L}.csv'))
mx_df_extremities.to_csv(os.path.join(output_dir, f'mx_df_extremities_R_{R}_L_{L}.csv'))

print("\nSTEP 4: Identifying load and drop-off locations")

load = produce_graph.get_load_points(raw_data, proj_info, radius=config["graph"].getint('merge_dump_pos'),)
dropoff = produce_graph.get_dropoff_points_db(raw_data, proj_info, radius=config["graph"].getint('merge_dump_pos'), eps=0.001)

cluster_info = pd.concat([cluster_info, dropoff], ignore_index=True)
cluster_info = pd.concat([cluster_info, load], ignore_index=True)
cluster_info = cluster_info[["Latitude", "Longitude", "z", "in_type", "Name"]].reset_index(drop=True)

# Only copying lat, lon, and then adding meter information making sure to use the right proj_info
cluster_info, _ = rdm.add_meter_columns(cluster_info, proj_info=proj_info)
cluster_info["id"] = cluster_info.index

nodes_info = cluster_info


print("\nSTEP 5: Road inference")

# Increased distance because sometimes a trip is very far and we dont want to duplicate edges
max_distance = config["graph"].getfloat('max_distance') # TODO: Change for a more understandable parameter name
cluster_info = nodes_info
trips.reset_index(drop=True, inplace=True)
mx_dist = rdm.compute_dist_matrix(cluster_info, trips, max_distance)
mx_df = rdm.preprocess_mx_dist(mx_dist, trips, cluster_info)

cluster_info = rdm.cluster_info_trips(cluster_info, mx_df, max_distance)


adjacent_nodes_info, trips_annotated = rdm.find_relations(trips_ini=trips, mx_dist_ini=mx_df)
trips_annotated = edgei.mark_trips_closest_to_intersection(trips_annotated=trips_annotated)
segments_df = edgei.divide_trips_at_intersections(trips_annotated)

node_segment_summary = edgei.summarize_segments_and_nodes(segments_df)
node_segment_summary = edgei.filter_node_summary_duplicates(node_segment_summary)
segments_cluster_df = edgei.cluster_all_connected_roads(segments_df, node_segment_summary, epsilon=15, min_samples=5)
edges_list = edgei.generate_edges(segments_cluster_df, segments_df, min_segment_length=1)

print("\nSTEP 6: Plotting the graph")

pl.plot_graph(edges_list, proj_info, nodes_info, trip=pd.DataFrame(), mx_df=pd.DataFrame(), dump=False, point=None, savename=os.path.join(output_dir, "graph"))
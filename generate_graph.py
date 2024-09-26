import modules.config as config
import modules.road_data_manipulation as rdm
import modules.produce_graph as produce_graph
import modules.plotting_functions as pl
import modules.edge_inference as edgei
from sklearn.cluster import DBSCAN
import json
import numpy as np
import pandas as pd
import os
import plotly.io as pio
pio.renderers.default = "iframe"

config = config.read_settings('config.ini')
folder = config["io"]["folder"]

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

df['theta2'] = np.radians(np.where(df['Course'] > 180, df['Course'] - 180, df['Course']))

print("Data preprocessed")

config_inter = config["intersections"]
res_step = config_inter.getint('res_step')
neighbour_dist = config_inter.getfloat('neighbour_dist')
df = df.copy()
# adding low resolution and using the median value as the aggregated value
df["x_"+str(res_step)] = rdm.change_resolution(df["x"], step=res_step)
df["y_"+str(res_step)] = rdm.change_resolution(df["y"], step=res_step)
gr = df.groupby(["x_"+str(res_step), "y_"+str(res_step)])
low_res_median = gr[['x', 'y', 'Latitude', 'Longitude', 'theta2', "Altitude"]].median().reset_index()
low_res_median['similarity_median'] = rdm.neighbour_similarity(
    low_res_median, dist=neighbour_dist)

print("Low resolution computed")

pl.plot_candidates_detection(low_res_median, raw_data, y_lim=(15850, 16000), x_lim=(9970, 10370),
                             savename="initial_candidates_detection.pdf")

print("Candidates plotted")

config_inter = config["intersections"]
intersection_candidates = rdm.df_turns_neighbour_similarity(df, config_inter.getint('res_step'),
                                                            config_inter.getfloat('neighbour_dist'),
                                                            config_inter.getfloat('similarity_thre'))
# Merging the cluster centres
intersection_candidates = rdm.get_cluster_centres(intersection_candidates,
                                                  config_inter.getint('max_merging_cluster_dist'))
intersection_candidates = intersection_candidates[intersection_candidates['cl_size'] > 1].reset_index(drop=True)
print("Intersection candidates computed")
print(intersection_candidates.head())

R, R2 = [float(item.strip()) for item in config_inter.get("R").split(',')]

# Step 1: Candidates
y_lim, x_lim = (15800, 16100), (9680, 10400)
pl.plot_intersections_with_radii(trips=trips, mx_df_extremities=None, intersection_candidates=intersection_candidates, radii_list=[R, R + config_inter.getfloat("L")],
                                 y_lim=y_lim, x_lim=x_lim, savename="candidates.pdf")
print("Candidates with radii plotted")


max_distance = 125  # config_inter.getfloat("R") + config_inter.getfloat("L")
# Returns a frame containg all points that are within the max_distance from canditate clusters
mx_dist = rdm.compute_dist_matrix(intersection_candidates, trips[["x", "y"]], max_distance)
mx_df = rdm.preprocess_mx_dist(mx_dist, trips, intersection_candidates)
print("Distances computed")

x_var = "ping_x"
y_var = "ping_y"

epsilon = config_inter.getfloat("dbscan_epsilon_validate_connected_roads")
min_samples = config_inter.getint("dbscan_min_samples_validate_connected_roads")
L = config_inter.getfloat("L")
proximity_threshold = config_inter.getfloat("max_dist_from_intersection")
points_list, valid_points_list, extremity_clusters_list = [], [], []
extremity_clusters_list_2 = []

points_df, valid_points_df, extremity_clusters_df = produce_graph.cluster_and_categorize_points(mx_df, intersection_candidates, epsilon=config_inter.getfloat("dbscan_epsilon_validate_connected_roads"), 
                                            min_samples=config_inter.getint("dbscan_min_samples_validate_connected_roads"), 
                                            R=R, L=config_inter.getfloat("L"), proximity_threshold=config_inter.getfloat("max_dist_from_intersection"), 
                                            x_var="ping_x", y_var="ping_y")

points_df, valid_points_df, extremity_clusters_df2 = produce_graph.cluster_and_categorize_points(mx_df, intersection_candidates, epsilon=config_inter.getfloat("dbscan_epsilon_validate_connected_roads"), 
                                            min_samples=config_inter.getint("dbscan_min_samples_validate_connected_roads"), 
                                            R=R2, L=config_inter.getfloat("L"), proximity_threshold=config_inter.getfloat("max_dist_from_intersection"), 
                                            x_var="ping_x", y_var="ping_y")

print("Points clustered and categorized")
    
pl.plot_validating_intersections_schema(trips, valid_points_df=valid_points_df, intersection_candidates=intersection_candidates, 
                             extremity_clusters_df=extremity_clusters_df, R=R, L=config_inter.getfloat("L"), 
                             max_dist_from_intersection=config_inter.getfloat("max_dist_from_intersection"), 
                             tick_resolution=50, marker_size=3, y_lim=(15845, 16000), x_lim=(9970, 10370),
                             savename="validating_intersections_schema.pdf")

print("Validating intersections schema plotted")

y_lim, x_lim = (14300, 15450), (11300, 12850)
confirmed_intersections = intersection_candidates.loc[((intersection_candidates["x"] > 11650) & (intersection_candidates["x"] < 11800)) | (
    (intersection_candidates["y"] > 14650) & (intersection_candidates["y"] < 14800))]
pl.plot_intersections_with_radii(trips=trips,  # mx_df_extremities=valid_points_df,
                                 extremity_clusters=extremity_clusters_df, intersection_candidates=intersection_candidates,
                                 confirmed_intersections=confirmed_intersections,
                                 radii_list=[R + L],
                                 y_lim=y_lim, x_lim=x_lim, marker_size=1, title="Validation with $R=30$", savename="validation_R_30.pdf")
print("Validation with R=30 plotted")

confirmed_intersections = intersection_candidates.loc[((intersection_candidates["x"] > 11500) & (intersection_candidates["x"] < 11800) & (
    intersection_candidates["y"] < 15100)) | ((intersection_candidates["y"] > 14650) & (intersection_candidates["y"] < 14800))]
pl.plot_intersections_with_radii(trips=trips,  # mx_df_extremities=valid_points_df,
                                 extremity_clusters=extremity_clusters_df2, intersection_candidates=intersection_candidates,
                                 confirmed_intersections=confirmed_intersections,
                                 radii_list=[R2, R2 + L],
                                 y_lim=y_lim, x_lim=x_lim, marker_size=1, title="Validation with $R=100$", savename="validation_R_100.pdf")

print("Validation with R=100 plotted")

df_bounded_region = low_res_median[(low_res_median['x_5'] >= x_lim[0]) & (low_res_median['x_5'] + 5 <= x_lim[1]) &
                             (low_res_median['y_5'] >= y_lim[0]) & (low_res_median['y_5'] + 5 <= y_lim[1])]
confirmed_intersections = intersection_candidates.loc[intersection_candidates["x"] > 10200]

print("Bounded region computed")

pl.plot_intersections(df_bounded_region=df_bounded_region, trips=trips, extremity_clusters_df=extremity_clusters_df, extremity_clusters_df2=extremity_clusters_df2,
                       confirmed_intersections1=confirmed_intersections, confirmed_intersections2=confirmed_intersections, 
                       intersection_candidates=intersection_candidates, x_lim=x_lim, y_lim=y_lim, R=30, R2=100, L=20, savename="bounded_region.pdf")
print("Intersections plotted")

mx_df_extremities = rdm.mx_extremity_clusters(mx_df_ini=mx_df, intersection_candidates=intersection_candidates,
                                              R=R, L=L,
                                              extremity_merging_cluster_dist=config_inter.getfloat("extremity_merging_cluster_dist"),
                                              min_cl_size=config_inter.getint("max_extremity_cluster_size"),
                                              max_dist_from_intersection=config_inter.getfloat("max_dist_from_intersection"),
                                              epsilon=config_inter.getfloat("dbscan_epsilon_validate_connected_roads"),
                                              min_samples=config_inter.getint("dbscan_min_samples_validate_connected_roads"))

intersection_candidates_2, mx_df_extremities = rdm.update_frames(intersection_candidates, mx_df_extremities)

print("Extremities computed")

cluster_info = intersection_candidates_2[["x", "y", "Longitude", "Latitude", "Altitude"]].reset_index().rename(columns={"index": "id"})
cluster_info.rename(columns={"Altitude": "z"}, inplace=True)

cluster_info["in_type"] = "road"
cluster_info["Name"] = "Road"
cluster_info = cluster_info[["Latitude", "Longitude","z", "in_type", "Name"]].reset_index(drop=True)
# Only copying lat, lon, and then adding meter information making sure to use the right proj_info
cluster_info, _ = rdm.add_meter_columns(cluster_info, proj_info=proj_info)
cluster_info["id"] = cluster_info.index

cluster_info.to_csv(f'cluster_info_R_{R}_L_{L}.csv')

print("Cluster info saved")
mx_df_extremities.to_csv(f'mx_df_extremities_R_{R}_L_{L}.csv')


load = produce_graph.get_load_points(raw_data, proj_info, radius=config["graph"].getint('merge_dump_pos'),)
dump = produce_graph.get_dump_points_db(raw_data, proj_info, radius=config["graph"].getint('merge_dump_pos'), eps=0.001)

print("Load and dump points computed")
cluster_info = pd.concat([cluster_info, dump], ignore_index=True)
cluster_info = pd.concat([cluster_info, load], ignore_index=True)
cluster_info = cluster_info[["Latitude", "Longitude", "z", "in_type", "Name"]].reset_index(drop=True)
# Only copying lat, lon, and then adding meter information making sure to use the right proj_info
cluster_info, _ = rdm.add_meter_columns(cluster_info, proj_info=proj_info)
cluster_info["id"] = cluster_info.index

print("Load and dump points added to cluster info")
nodes_info = cluster_info

# Increased distance because sometimes a trip is very far and we dont want to duplicate edges
max_distance = config["graph"].getfloat('max_distance') # TODO: Change for a more understandable parameter name
cluster_info = nodes_info
trips.reset_index(drop=True, inplace=True)
mx_dist = rdm.compute_dist_matrix(cluster_info, trips, max_distance)
mx_df = rdm.preprocess_mx_dist(mx_dist, trips, cluster_info)

cluster_info = rdm.cluster_info_trips(cluster_info, mx_df, max_distance)

print("Cluster info and trips clustered")


adjacent_nodes_info, trips_annotated = rdm.find_relations(trips,
                                                          cluster_info,
                                                          mx_df,
                                                          max_distance)

print("Relations found")

trips_annotated = edgei.mark_trips_closest_to_intersection(trips_annotated=trips_annotated)
print("Trips marked")

segments_df = edgei.divide_trips_at_intersections(trips_annotated)

node_segment_summary = edgei.summarize_segments_and_nodes(segments_df)
node_segment_summary = edgei.filter_node_summary_duplicates(node_segment_summary)
print("Segments divided and summarized")

segments_cluster_df = edgei.cluster_all_connected_roads(segments_df, node_segment_summary, epsilon=15, min_samples=5)
print("Segments clustered")

edges_list = edgei.generate_edges(segments_cluster_df, segments_df, min_segment_length=1)
print("Edges created")


# This is the part from mass transport problem
nodes_info, adjacent_nodes_info, edges, trips_annotated, mx_dist, trips, mx_df, relevant_trips = produce_graph.produce_graph_elements(trips, nodes_info, proj_info,
                                                                                                                        config['graph'],
                                                                                                                        fast_alg=False)

print("Graph elements produced")
graph = produce_graph(proj_info, nodes_info, adjacent_nodes_info, edges)
print("Graph created")
print(graph)
#loaders = produce_loaders(nodes_info)
#loaders = Loaders.from_dict(loaders_schema.validate(json.dumps(loaders)))
#vehicles = produce_vehicles(raw_data, proj_info)
#vehicles = Vehicles.from_dict(
#    vehicles_schema.validate(json.dumps(vehicles)))
#loaders_sites_and_task = produce_load_and_tasks(nodes_info)[0]
#loaders_sites_and_task = LoaderSites.from_dict(
#    loader_side_schema.validate(json.dumps(loaders_sites_and_task)))
#wp3_inputs = Wp3Output(
#    loaders,
#    vehicles,
#    loaders_sites_and_task,
#    graph)
#mass_transportation_problem = convert(wp3_inputs).to_json()



#mass_transport_problem, proj_info, adjacent_nodes_info, nodes_info, raw_data, trips, mx_df, relevant_trips = produce_graph.make_mass_transport_problem(
#    events, tracking, config, fast_alg=True)


#m = json.loads(mass_transport_problem)

#m["SpatialGraphContainer"]["Network"]["Edges"] = edges_list


#pl.plot_mass_transport_problem_paper(m["SpatialGraphContainer"], m,proj_info,nodes_info, trip=trips, zoom_start=230, location=[proj_info["origin"][0], proj_info["origin"][1]],
#                                      dump=False, point=None)
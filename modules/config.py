import logging
logger = logging.getLogger('producegraph')

import configparser

def parse_list(config_subsection, datatype, sep=','):
	return [datatype(item.strip()) for item in config_subsection.split(sep)]


def default_settings():
	""" Sets up a default configuration."""
	config = configparser.ConfigParser()
	config.read_dict({
	'intersections': {
		'res_step' : 5,
		'neighbour_dist' : 20,
		'similarity_thre' : 1.4,
		'L' : 25, 
		'R' : 100, 
		'nr_trips' : 50,
		'max_merging_cluster_dist' : 15,
		'extremity_merging_cluster_dist' : 15,
		'max_extremity_cluster_size' : 2,
		'max_dist_from_intersection' : 30
				},
	'graph': {
		'max_dump_pos' : 20,
		'merge_dump_pos' : 500,
		'min_dump_points' : 5,
		'max_distance' : 30,
		'without_direction' : "yes",
		'interp_spline_deg' : 1,
		'interp_resolution_m' : 10,
		'endpoint_threshold' : 100,
		'remove_endpoints' : "yes",
		'min_trip_length' : 4,
		'divide_trip_thr_mins' : 5,
		'divide_trip_thr_angle' : 180
		},
		'io': {
			'manual_intersection':False
		}

	})
	return config 


def read_settings(filename):
	""" Sets up a configuration with default values overridden where given in the file."""
	config = default_settings()
	config.read(filename)
	return config
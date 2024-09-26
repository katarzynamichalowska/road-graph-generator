
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle, Circle
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle
import folium
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
from matplotlib.legend_handler import HandlerBase
from matplotlib.text import Text
from matplotlib.lines import Line2D
from IPython.display import display

def plot_graph(edges_list, proj_info, nodes_info, trip=pd.DataFrame(), mx_df=pd.DataFrame(), dump=False, point=None, savename="graph"):

    mapit = folium.Map(location=[proj_info["origin"][0], proj_info["origin"][1]],
                       zoom_start=3, control_scale=False, tiles="cartodb positron")

    if trip.shape[0] > 0:
        for _, k in trip.groupby("TripLogId"):
            points = [(lat, lon)
                      for lat, lon in zip(k["Latitude"], k["Longitude"])]
            folium.PolyLine(points, color='grey', weight=2,
                            opacity=1).add_to(mapit)

    for k in edges_list:
        points = k["coordinates"]
        points = [(t[1], t[0]) for t in points]
        folium.PolyLine(points, color="blue", weight=4, opacity=1).add_to(mapit)

    color_mapping = {
        "load": "green",
        "dump": "black",
        "road": "red"
    }

    for node_type, color in color_mapping.items():
        for _, k in nodes_info[nodes_info["in_type"] == node_type].iterrows():
            p = [k["Latitude"], k["Longitude"]]
            folium.CircleMarker(location=p, color=color, fill_color=color, radius=5).add_to(mapit)


    if mx_df.shape[0] > 0:
        for i, k in mx_df.iterrows():
            p = [k["ping_lat"], k["ping_lon"]]
            folium.CircleMarker(location=p, fill=True, fill_color='yellow',
                                color='yellow', radius=3, fill_opacity=1, weight=2).add_to(mapit)
    if dump:
        for i, k in trip.groupby("TripLogId"):
            p = [k.iloc[0]["LoadLatitude"], k.iloc[0]["LoadLongitude"]]
            folium.CircleMarker(location=p, fill=True, fill_color='pink',
                                color='pink', radius=6, fill_opacity=1, weight=2).add_to(mapit)

    if point != None:
        folium.CircleMarker(location=point, fill=True, fill_color='pink',
                            color='pink', radius=20, fill_opacity=1, weight=2).add_to(mapit)

    mapit.fit_bounds(mapit.get_bounds(), padding=(20, 10))

    mapit.save(f"{savename}.html")
    display(mapit)



def plot_intersections_with_radii(trips, mx_df_extremities=None, intersection_candidates=None, confirmed_intersections=None,
                                  extremity_clusters=None, radii_list=None, y_lim=None, x_lim=None,
                                  title="", marker_size=1, savename=None):
    plt.figure(figsize=(5, 6))

    plt.scatter(trips["x"], trips["y"], c="gray", s=marker_size, rasterized=True)
    if mx_df_extremities is not None:
        plt.scatter(mx_df_extremities["ping_x"], mx_df_extremities["ping_y"], c="darkorange", s=marker_size, rasterized=True)

    if extremity_clusters is not None:
        plt.scatter(extremity_clusters["ping_x"], extremity_clusters["ping_y"], c="blue", s=marker_size, rasterized=True)

    if radii_list is not None:
        for index, row in intersection_candidates.iterrows():
            for radius in radii_list:
                circle = Circle((row['x'], row['y']),
                                radius, color='green', fill=False)
                plt.gca().add_patch(circle)
    if intersection_candidates is not None:
        plt.scatter(intersection_candidates["x"], intersection_candidates["y"], c="red", s=50)
        plt.scatter(intersection_candidates["x"], intersection_candidates["y"], c="white", marker="$?$", s=35)

    if confirmed_intersections is not None:
        plt.scatter(confirmed_intersections["x"], confirmed_intersections["y"], c="darkorange", s=50)
        plt.scatter(confirmed_intersections["x"], confirmed_intersections["y"], c="white", marker="x", s=35)

    if x_lim is not None:
        plt.xlim(x_lim)
    if y_lim is not None:
        plt.ylim(y_lim)
    plt.title(title)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.gca().set_aspect('equal', adjustable='box')
    if savename is not None:
        plt.savefig(savename, bbox_inches='tight')
    plt.close()


class ConfIntHandler(HandlerBase):
    def create_artists(self, legend, orig_handle, x0, y0, width, height, fontsize, trans):
        center = x0 + width / 2., y0 + height / 2.
        patch = Circle(center, radius=width / 5., color='red', transform=trans)
        text = Text(x=center[0], y=center[1], text='x', color='white',
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=fontsize, fontweight='bold', transform=trans)

        return [patch, text]


class CandidateIntHandler(HandlerBase):
    def create_artists(self, legend, orig_handle, x0, y0, width, height, fontsize, trans):
        center = x0 + width / 2., y0 + height / 2.
        patch = Circle(center, radius=width / 5., color='red', transform=trans)
        text = Text(x=center[0], y=center[1], text='?', color='white',
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=fontsize, fontweight='bold', transform=trans)

        return [patch, text]
    

# Custom legend handlers
#class ConfIntHandler(HandlerBase):
#    def create_artists(self, legend, orig_handle, x0, y0, width, height, fontsize, trans):
#        center = x0 + width / 2., y0 + height / 2.
#        patch = Circle(center, radius=width / 5.,
#                        color='red', transform=trans)
#        text = Text(x=center[0]-0.2, y=center[1]+0.6, text='x', color='white',
#                    horizontalalignment='center', verticalalignment='center',
#                    fontsize=9.5, fontweight='bold', transform=trans)
#        return [patch, text]

#class CandidateIntHandler(HandlerBase):
#    def create_artists(self, legend, orig_handle, x0, y0, width, height, fontsize, trans):
#        center = x0 + width / 2., y0 + height / 2.
#        patch = Circle(center, radius=width / 5.,
#                        color='red', transform=trans)
#        text = Text(x=center[0]-0.2, y=center[1]-0.3, text='?', color='white',
#                    horizontalalignment='center', verticalalignment='center',
#                    fontsize=9.5, fontweight='bold', transform=trans)
#        return [patch, text]

def plot_candidates_detection(low_res_median, raw_data, y_lim=(15850, 16000), x_lim=(9970, 10370), marker_size=3,
                              savename=None):
    """
    Plots GPS paths with similarity and direction dissimilarity.

    Parameters:
        low_res_median (DataFrame): Filtered data with median similarity.
        raw_data (DataFrame): Raw data with direction and position information.
        y_lim (tuple): Y-axis limits.
        x_lim (tuple): X-axis limits.
        marker_size (int): Size of scatter plot markers.
    """

    # Create subplots
    fig, ax = plt.subplots(2, 1, figsize=(
        7, 5), sharex=True, layout='constrained')

    # Filter DataFrame based on your limits
    df_filtered = low_res_median[(low_res_median['x_5'] >= x_lim[0]) & (low_res_median['x_5'] + 5 <= x_lim[1]) &
                                 (low_res_median['y_5'] >= y_lim[0]) & (low_res_median['y_5'] + 5 <= y_lim[1])]

    # Normalize similarity_median for coloring
    norm = plt.Normalize(df_filtered['similarity_median'].min(
    ), df_filtered['similarity_median'].max())

    # Process raw data for direction calculation
    raw_data['Course2'] = raw_data['Course']
    raw_data.loc[raw_data['Course'] > 180,
                 'Course2'] = raw_data['Course'] - 180
    raw_data['theta2'] = np.radians(raw_data['Course2'])
    norm2 = plt.Normalize(raw_data['theta2'].min(), raw_data['theta2'].max())

    # First plot: scatter of raw data with directions
    ax[0].scatter(raw_data["x"], raw_data["y"], c=raw_data['theta2'],
                  s=marker_size, cmap='viridis', rasterized=True)

    # Second plot: rectangles for directional dissimilarity
    for _, row in df_filtered.iterrows():
        rect = Rectangle((row['x_5'], row['y_5']), 5, 5,
                         color=plt.cm.plasma(norm(row['similarity_median'])))
        ax[1].add_patch(rect)

    # Configure ticks
    major_x_ticks = np.arange(x_lim[0], x_lim[1] + 50, 50)
    minor_x_ticks = np.arange(x_lim[0], x_lim[1] + 5, 5)
    major_y_ticks = np.arange(y_lim[0], y_lim[1] + 50, 50)
    minor_y_ticks = np.arange(y_lim[0], y_lim[1] + 5, 5)

    # First colorbar (for similarity)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax[1], location='right', pad=0.01,
                 label="Directional dissimilarity $\Delta\phi_\mathrm{i, j}$")

    # Second colorbar (for directions)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm2)
    sm.set_array([])
    cbar2 = fig.colorbar(sm, ax=ax[0], location='right',
                         pad=0.01, label="Movement directions $\Phi$ (rad)")

    # Ensure integer ticks for the second colorbar
    cbar2.locator = ticker.MaxNLocator(integer=True)
    cbar2.update_ticks()

    # Set axis limits, ticks, and labels
    for ax_i in ax:
        ax_i.set_xlim(x_lim)
        ax_i.set_ylim(y_lim)
        ax_i.set_xticks(major_x_ticks)
        ax_i.set_xticks(minor_x_ticks, minor=True)
        ax_i.set_yticks(major_y_ticks)
        ax_i.set_yticks(minor_y_ticks, minor=True)
        ax_i.set_xlabel("x (m)")
        ax_i.set_ylabel("y (m)")
        ax_i.set_aspect('equal', adjustable='box')

    # Add gridlines to the second plot
    ax[1].grid(True, which='both', linestyle='--', linewidth=0.4, zorder=-10)
    ax[1].set_axisbelow(True)

    # Save figure
    if savename is not None:
        plt.savefig(savename, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_validating_intersections_schema(trips, valid_points_df, intersection_candidates, extremity_clusters_df,
                                         R, L, max_dist_from_intersection, tick_resolution=50, marker_size=3,
                                         y_lim=(15845, 16000), x_lim=(9970, 10370),
                                         savename=None):
    """
    Plot schema for validating intersections with candidate points, clusters, and radius overlays.
    
    Parameters:
        trips (DataFrame): DataFrame containing trip data with x and y coordinates.
        valid_points_df (DataFrame): DataFrame containing valid points.
        intersection_candidates (DataFrame): DataFrame with intersection candidate points.
        extremity_clusters_df (DataFrame): DataFrame containing extremity clusters.
        tick_resolution (int): Spacing for major ticks.
        marker_size (int): Size of scatter plot markers.
        y_lim (tuple): Limits for y-axis.
        x_lim (tuple): Limits for x-axis.
        save_path (str): File path to save the plot as PDF.
    """

    fig, ax = plt.subplots(2, 1, figsize=(7, 5), sharex=True, layout='constrained')

    # Plot the first subplot with trips and valid points
    ax[0].scatter(trips["x"], trips["y"], c="gray", s=marker_size, label="All trips", rasterized=True)
    ax[0].scatter(valid_points_df["ping_x"], valid_points_df["ping_y"], c="darkorange", s=marker_size, label="Valid points", rasterized=True)

    # Plot intersection candidates, if available
    if intersection_candidates is not None:
        ax[0].scatter(intersection_candidates["x"], intersection_candidates["y"], c="red", s=80, label="Intersection candidates")
        ax[0].scatter(intersection_candidates["x"], intersection_candidates["y"], c="white", marker="$?$")

        for _, row in intersection_candidates.iterrows():
            outer_circle = Circle((row['x'], row['y']), R+L, color='green', fill=False, label="$R+L$ radius")
            inner_circle = Circle((row['x'], row['y']), max_dist_from_intersection, color='green', fill=False, label="Inner radius")
            ax[0].add_patch(outer_circle)
            ax[0].add_patch(inner_circle)

    # Plot second subplot with clusters and filled annulus for intersection candidates
    ax[1].scatter(trips["x"], trips["y"], c="gray", s=marker_size, rasterized=True)
    ax[1].scatter(valid_points_df["ping_x"], valid_points_df["ping_y"], c="darkorange", s=marker_size, rasterized=True)

    if intersection_candidates is not None:
        ax[1].scatter(intersection_candidates["x"],
                      intersection_candidates["y"], c="red", s=80)
        ax[1].scatter(intersection_candidates["x"],
                      intersection_candidates["y"], c="white", marker="$?$")

        for _, row in intersection_candidates.iterrows():
            wedge = Wedge((row['x'], row['y']), R + L,
                          0, 360, width=L, color='green', alpha=0.3, label="Filled annulus")
            ax[1].add_patch(wedge)

    ax[1].scatter(extremity_clusters_df["ping_x"], extremity_clusters_df["ping_y"],
                  c="blue", s=marker_size, label="Only clusters", rasterized=True)

    # Mark confirmed intersections
    confirmed_intersections = intersection_candidates.loc[intersection_candidates["x"] > 10200]
    ax[1].scatter(confirmed_intersections["x"], confirmed_intersections["y"],
                  c="red", s=80, label="Confirmed intersections")
    ax[1].scatter(confirmed_intersections["x"],
                  confirmed_intersections["y"], c="white", marker="x")

    # Set axis limits and ticks
    for ax_i in ax:
        ax_i.set_xlim(x_lim)
        ax_i.set_ylim(y_lim)
        ax_i.set_xlabel("x (m)")
        ax_i.set_ylabel("y (m)")
        ax_i.set_aspect('equal', adjustable='box')

        # Set major ticks
        start_xtick = np.ceil(x_lim[0] / tick_resolution) * tick_resolution
        end_xtick = np.floor(x_lim[1] / tick_resolution) * tick_resolution
        ax_i.set_xticks(np.arange(start_xtick, end_xtick + 1, tick_resolution))

        start_ytick = np.ceil(y_lim[0] / tick_resolution) * tick_resolution
        end_ytick = np.floor(y_lim[1] / tick_resolution) * tick_resolution
        ax_i.set_yticks(np.arange(start_ytick, end_ytick + 1, tick_resolution))

    dots = lambda color: Line2D([0], [0], marker='o', color=color, linestyle='None', markersize=3)
    conf_int = cand_int = Line2D([0], [0], linestyle="none", c='b', marker='o')
    green_line = Line2D([0], [0], marker='_', color='green', linestyle='None', linewidth=1)

    all_handles = [conf_int, cand_int, green_line, dots('darkorange'), dots('blue'), dots('gray')]


    fig.legend(handles=all_handles, labels=['Confirmed intersections', 'Candidate intersections', 'Subsetting radii',
                                            'Valid GPS points', 'Extremity clusters', 'Other GPS points'],
               handler_map={conf_int: ConfIntHandler(), cand_int: CandidateIntHandler()},
               loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.13))

    # Save figure
    if savename is not None:
        fig.savefig(savename, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def create_subplot(ax, plot_type, **kwargs):
    if plot_type == 'rectangles':
        x_lim, y_lim, df_filtered = kwargs['x_lim'], kwargs['y_lim'], kwargs['df_filtered']
        norm = Normalize(df_filtered['similarity_median'].min(
        ), df_filtered['similarity_median'].max() - 8)

        for _, row in df_filtered.iterrows():
            rect = Rectangle((row['x_5'], row['y_5']), 15, 15, color=cm.plasma(
                norm(row['similarity_median'])))
            ax.add_patch(rect)

        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect('equal', adjustable='box')

    elif plot_type == 'intersections':
        trips = kwargs['trips']
        extremity_clusters = kwargs.get('extremity_clusters')
        intersection_candidates = kwargs.get('intersection_candidates')
        confirmed_intersections = kwargs.get('confirmed_intersections')
        radii_list = kwargs.get('radii_list')
        x_lim = kwargs['x_lim']
        y_lim = kwargs['y_lim']

        ax.scatter(trips["x"], trips["y"], c="gray", s=1, rasterized=True)

        if extremity_clusters is not None:
            ax.scatter(
                extremity_clusters["ping_x"], extremity_clusters["ping_y"], c="blue", s=1, rasterized=True)

        if intersection_candidates is not None:
            for index, row in intersection_candidates.iterrows():
                for radius in radii_list:
                    circle = Circle((row['x'], row['y']),
                                    radius, color='green', fill=False)
                    ax.add_patch(circle)

        if confirmed_intersections is not None:
            ax.scatter(
                confirmed_intersections["x"], confirmed_intersections["y"], c="red", s=50)
            ax.scatter(
                confirmed_intersections["x"], confirmed_intersections["y"], c="white", marker="x", s=35)

        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect('equal', adjustable='box')



def plot_intersections(df_bounded_region, trips, extremity_clusters_df, extremity_clusters_df2,
                       confirmed_intersections1, confirmed_intersections2, 
                       intersection_candidates, x_lim, y_lim, R=30, R2=100, L=20, savename=None):
    
    fig, axs = plt.subplots(2, 2, figsize=(8, 6), sharex=True, sharey=True, gridspec_kw={'wspace': 0.05, 'hspace': 0.2})

    # First subplot with rectangles
    create_subplot(axs[0, 0], 'rectangles',
                   df_filtered=df_bounded_region, x_lim=x_lim, y_lim=y_lim)

    # Second subplot with intersections
    create_subplot(axs[0, 1], 'intersections', trips=trips, extremity_clusters=extremity_clusters_df,
                   intersection_candidates=intersection_candidates, confirmed_intersections=confirmed_intersections1,
                   radii_list=[R+L], x_lim=x_lim, y_lim=y_lim)

    # Third subplot with intersections
    create_subplot(axs[1, 0], 'intersections', trips=trips, extremity_clusters=extremity_clusters_df2,
                   intersection_candidates=intersection_candidates, confirmed_intersections=confirmed_intersections2,
                   radii_list=[R2, R2+L], x_lim=x_lim, y_lim=y_lim)

    # Fourth subplot (only trips and confirmed intersections)
    axs[1, 1].scatter(trips["x"], trips["y"], c="gray", s=1, rasterized=True)
    axs[1, 1].scatter(confirmed_intersections2["x"], confirmed_intersections2["y"], c="red", s=50, label='Confirmed Intersections')
    axs[1, 1].scatter(confirmed_intersections2["x"], confirmed_intersections2["y"], c="white", marker="x", s=35)

    # Adjusting subplots and setting aspect ratio
    for ax in axs.flat:
        ax.set_aspect('equal', adjustable='box')

    axs[0, 0].set_title("Directional dissimilarity")
    axs[0, 1].set_title("Intersection verification ($R=30$)")
    axs[1, 0].set_title("Intersection verification ($R=100$)")
    axs[1, 1].set_title("Confirmed intersections")
    axs[1, 0].set_xlabel("x (m)")
    axs[1, 1].set_xlabel("x (m)")
    axs[0, 0].set_ylabel("y (m)")
    axs[1, 0].set_ylabel("y (m)")

    # Custom legend handlers
    conf_int = Line2D([0], [0], linestyle="none", c='b', marker='o')
    cand_int = Line2D([0], [0], linestyle="none", c='b', marker='o')
    grey_dot = Line2D([0], [0], marker='o', color='gray',
                      linestyle='None', markersize=3)
    blue_dot = Line2D([0], [0], marker='o', color='blue',
                      linestyle='None', markersize=3)
    green_line = Line2D([0], [0], marker='_', color='green',
                        linestyle='None', linewidth=1)

    all_handles = [conf_int, cand_int, green_line, blue_dot, grey_dot]

    # Adding the custom legend entries
    fig.legend(handles=all_handles, labels=['Confirmed intersections', 'Candidate intersections', 'Extremity radii',
                                            'Extremity clusters', 'Other GPS points'],
               loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.07))
    
    if savename is not None:
        fig.savefig(savename, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


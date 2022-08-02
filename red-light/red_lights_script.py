import json
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Point
import argparse
pd.options.mode.chained_assignment = None


def get_track(track_name):
    track = pd.read_json(track_name)
    return track

def get_road_graph(road_graph_name):
    with open(road_graph_name) as f:
        graph = json.load(f)
        f.close()
    return graph

def get_light(path_to_lights, light_id):
    light = pd.read_csv(f'light_id_{light_id}.csv')
    return light

def save_output(file):
    file.to_csv('output_red_light.csv', sep=',', header=True, index=False)

def lights_signals(graph, sjoin_nearest_table):

  signals = []
  for i in range(len(sjoin_nearest_table)):
    node_ = sjoin_nearest_table['node_id'][i]

    for u in graph['nodes'][node_]['neighbours']:

      delta = 3
      delta_min, delta_max = u['azimuth'] - delta, u['azimuth'] + delta

      if delta_min <= sjoin_nearest_table['azimuth'][i] <= delta_max:
        signals.append(u['traffic_lights'][0])

  return signals

def driving_on_light(route_name, graph_name, path_to_lights_working_tables, max_distance_to_graph):
      
    output = get_track(route_name)
    graph = get_road_graph(graph_name)
    
    output['geometry'] = list(zip(output.longitude, output.latitude))
    output['geometry'] = output['geometry'].apply(Point)
    geo_output = gpd.GeoDataFrame(output[['timestamp', 'azimuth', 'geometry']], geometry = 'geometry', crs='epsg:4326')
    geo_output = geo_output.to_crs(32639)
    
    table = []
    
    for i in graph['nodes']:
        if i['traffic_light']==True:    
            table.append( list((i['longitude'], i['latitude'], i['node_id'])) )
        
    geo_graph = pd.DataFrame(table, columns=['longitude', 'latitude', 'node_id'])
    geo_graph['geometry'] = list(zip(geo_graph.longitude, geo_graph.latitude))
    geo_graph['geometry'] = geo_graph['geometry'].apply(Point)
    geo_graph = gpd.GeoDataFrame(geo_graph[['geometry', 'node_id']], geometry = 'geometry', crs='epsg:4326')
    geo_graph = geo_graph.to_crs(32639)
    
    nearest = gpd.sjoin_nearest(geo_graph, geo_output, max_distance=10, distance_col='distances')
    nearest = nearest.to_crs(epsg=4326)
    signals = lights_signals(graph = graph, sjoin_nearest_table = nearest)
    
    for i in signals[0].keys():
        nearest[i] = np.nan

    for i, singl in enumerate(signals):
        sign = list(singl.values())
        nearest['id_traffic_light'][i] = int(sign[0])
        nearest['signal_group'][i] = int(sign[1])


    lines = []

    for i in zip(nearest['id_traffic_light'], nearest['signal_group']):
        for u in graph['traffic_lights']:
            for group in u['signal_groups']:
                if i[0]==u['id'] and group['id']==i[1]:
                    lines.append( group['stop_line'] )
    
    data_line = pd.DataFrame(columns=['lat_start', 'lng_start', 'lat_end', 'lng_end'])
    
    for i in range(len(lines)):
        data_line.loc[i] = [ lines[i][0]['latitude'], lines[i][0]['longitude'], lines[i][1]['latitude'], lines[i][1]['longitude'] ]
    
    data_line['geom_start'] = list(zip( data_line['lng_start'], data_line['lat_start'] ))
    data_line['geom_end'] = list(zip( data_line['lng_end'],data_line['lat_end'] ))
    data_line['geometry'] = list(zip( data_line['geom_start'],data_line['geom_end'] ))
    data_line['geometry'] = data_line['geometry'].apply(LineString)
    geo_data_line = gpd.GeoDataFrame(data_line[['geometry']], geometry='geometry', crs='epsg:4326')
    geo_data_line = geo_data_line.to_crs(32639)
    
    stop_lines_nearest = gpd.sjoin_nearest(geo_data_line, geo_output, max_distance=max_distance_to_graph, distance_col='distances')
    stop_lines_nearest = stop_lines_nearest.to_crs(epsg=4326)
    
    for i in range(len(stop_lines_nearest)):
        if stop_lines_nearest['distances'][i] >= max_distance_to_graph:
            stop_lines_nearest.drop(axis=0, index=i, inplace=True)
            
    near = pd.concat([nearest[['node_id', 'id_traffic_light', 'signal_group']], stop_lines_nearest[['azimuth', 'timestamp', 'index_right', 'distances']]], axis=1)

    result_table = pd.DataFrame()
    
    for i in range(len(near)):
    
        id = near['node_id'][i]
        signalgroup = int(near['signal_group'][i])
        light_open = get_light(path_to_lights_working_tables, light_id = id)

        for u in range(len(light_open)):
            if pd.to_datetime(light_open['time'][u]) == pd.to_datetime(near['timestamp'][i]):
                
                check_table = light_open.iloc[u].to_frame().T.copy()

                if light_open[f'signal_group_{int(signalgroup)}'][u]==False:

                    check_table['light_id'] = int(id)
                    check_table['result'] = 'Red light traffic is not recorded'
                    result_table = pd.concat([result_table, check_table])
                            
                else:

                    check_table['light_id'] = int(id)
                    check_table['result'] = 'Red light traffic is recorded'
                    result_table = pd.concat([result_table, check_table])  
            
    result_table.drop_duplicates(inplace=True)
    result_table.reset_index(drop=False, inplace=True)
    result_table.rename(columns={'index':'track_moment'}, inplace=True)
   
    return save_output(result_table) 

def main():
    parser = argparse.ArgumentParser(description='Generating')
    parser.add_argument('route_name', type=str, help='Getting route of car')
    parser.add_argument('graph_name', type=str, help='Getting graph of road')
    parser.add_argument('path_to_lights_working_tables', type=str, help='Getting tables of traffic lights working time')
    parser.add_argument('max_distance_to_graph', type=int, nargs='?', default=5, help='Setting max distance to graph')
    args = parser.parse_args()
    driving_on_light(args.route_name, args.graph_name, args.path_to_lights_working_tables, args.max_distance_to_graph)
    
if __name__ == "__main__":
        main()

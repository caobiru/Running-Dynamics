import geopandas as gpd
import pandas as pd
import os
from tqdm import tqdm

def encoding_and_save(grid,thu,geojson_path, output_csv_path):
    """
    Perform geometric operations on the geojson file 
    and save the encoding result as a CSV file.
    """

    gdf_keep = gpd.read_file(geojson_path)

    sjoin_intersects = gpd.sjoin(gdf_keep, thu, predicate='intersects')
    sjoin_intersects['length'] = sjoin_intersects['geometry'].to_crs('4509').length
    sjoin_intersects = sjoin_intersects[['start_time', 'end_time', 'start', 'end', 'length', 'geometry']]

    # 初始化 grid 中的列
    grid['keep_length'] = 0.0
    grid['length_average'] = 0.0
    grid['keep_num'] = 0.0

    for i in grid.index:
        clip = gpd.overlay(sjoin_intersects, grid[i:i+1], how='intersection').to_crs('4509')

        if len(clip)>=1:
            grid.loc[i, 'length_average'] += clip['length'].sum()
            grid.loc[i, 'keep_length'] += clip['geometry'].length.sum()
            grid.loc[i, 'keep_num'] += len(clip)

    # save the result as a CSV file
    pd.DataFrame(grid[['keep_length', 'length_average', 'keep_num']]).to_csv(output_csv_path, index=False)

def process_directories(root_dir, grid, thu):
    """
    遍历根目录中的所有子目录，并对每个子目录调用 `calculate_and_save` 函数。

    :param root_dir: 根目录路径
    :param grid: 基准 GeoDataFrame, 包含网格
    :param thu: 基准 GeoDataFrame, 包含 Tsinghua 的形状
    """
    for subdir in tqdm(os.listdir(root_dir)):
        subdir_path = os.path.join(root_dir, subdir)

        if os.path.isdir(subdir_path):
            geojson_path = os.path.join(subdir_path, 'new_keep_gdf_all.geojson')

            if os.path.exists(geojson_path):
                output_csv_path = os.path.join(output_dir, f'{subdir}.csv')
                encoding_and_save(grid, thu, geojson_path, output_csv_path)

    print('All done!')

root_dir = '../03_input/keep_data_bj_2022'
thu = gpd.read_file('../03_input/GIS/tsinghua.shp')
thu.crs = 'epsg:4326'
grid = gpd.read_file('../03_input/grid.geojson')

output_dir = '../03_input/input_data_2022'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

process_directories(root_dir, grid, thu)

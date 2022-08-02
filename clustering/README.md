# To run script use:

python clustering.py ['https://cms.dtp-stat.ru/media/opendata/samarskaia-oblast.geojson'] [2021] [False] [True] [0] [0]

## Input parameters:

1)  url - url to geojson file with crashes from https://dtp-stat.ru/opendata
2)  _year_to_filter - year to clustering
3)  _all_time_period - bool, if True: all data for clustering
4)  _year_period - bool, if True: period year from _year_to_filter for clustering
5)  _quarter_year_period - int, from 1 to 4, quarter of year from _year_to_filter
6)  _month_year_period - int, from 1 to 12, month of year from _year_to_filter

## Output files:
### 6 csv files - clustering by filters:

1)  cluster_all_crashes_{year}_{period} - clustering by all crashes
2)  cluster_injured_{year}_{period} - clustering by injured>0
3)  cluster_dead_{year}_{period} - clustering by all dead>0
4)  cluster_all_road_conditions_{year}_{period} - clustering by different types of road
5)  cluster_all_weather_{year}_{period} - clustering by different types of weather
6)  cluster_all_light_{year}_{period} - clustering by different types of light

# Example of output file:

```CSV
labels,number_of_dead,number_of_crashes,center_lat,center_lng
0,3,3,53.186099,50.182778
2,2,2,53.290071499999996,51.6578865
134,3,3,53.092788,50.158099666666665
8,2,2,53.4067175,50.334999499999995
```



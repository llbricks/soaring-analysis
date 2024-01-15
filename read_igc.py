import re
import os
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta, time
import numpy as np
import argparse
from scipy.ndimage import uniform_filter1d

# function used for plt later
def on_click(event): plt.close()  # Close the plot window

def read_igc_file(file_path):
    """ Read an IGC file and parse its lines. """
    parsed_data = []
    with open(file_path, 'r') as file:
        for line in file:
            parsed_line = parse_igc_line(line.strip())
            if parsed_line:
                parsed_data.append(parsed_line)
    data = pd.DataFrame(parsed_data)
    data = convert_utc_to_pt_nodate(data)
    data = add_normalized_time_column(data)

    return data

def parse_igc_line(line):
    """ Parse an individual IGC line and return a dictionary with the extracted data. """
    if line.startswith('B'):
        time_str = line[1:7]
        lat_str = line[7:15]
        lon_str = line[15:24]
        gps_altitude = line[25:30]
        baro_altitude = line[30:35]

        # Convert time
        time = datetime.strptime(time_str, '%H%M%S').time()

        # Convert latitude and longitude to decimal degrees
        lat = convert_to_decimal(lat_str[:2], lat_str[2:7], lat_str[7])
        lon = convert_to_decimal(lon_str[:3], lon_str[3:8], lon_str[8])

        return {
            'time': time,
            'latitude': lat,
            'longitude': lon,
            'gps_altitude': int(gps_altitude),
            'baro_altitude': int(baro_altitude)
        }
    return None

def convert_utc_to_pt_nodate(df, time_column='time'):
    # Define a typical offset for Pacific Time (e.g., -8 hours for UTC-8)
    offset = timedelta(hours=-8)

    # Function to convert time
    def convert_time(t):
        # Convert to a full datetime, apply the offset, and then extract the time
        full_datetime = datetime.combine(datetime.today(), t) + offset
        return full_datetime.time()

    # Apply the conversion to the time column
    df['time_pt'] = df[time_column].apply(convert_time)

    return df

def add_normalized_time_column(df, time_column='time_pt'):
    def time_to_seconds(time_points):
        datetime_points = [datetime.combine(datetime.min, t) for t in time_points]
        min_time = min(datetime_points)
        return [(dt - min_time).total_seconds() for dt in datetime_points]

    # Check if the time column exists in the DataFrame
    if time_column in df.columns:
        # Apply the conversion
        df['normalized_time'] = time_to_seconds(df[time_column])
    else:
        raise ValueError(f"Time column '{time_column}' not found in DataFrame")
    
    # also add the fractional second time assuming even spacing between time points
    nt = df['normalized_time']
    df['normalized_time_fractional'] = np.linspace(min(nt),max(nt),len(nt))

    return df

def convert_to_decimal(degrees, minutes, direction):
    """ Convert degrees and minutes to decimal format. """
    decimal = float(degrees) + float(minutes) / 60000
    if direction in ['S', 'W']:
        decimal = -decimal
    return decimal

def main(file_path,plot_results):

    file_title, _ = os.path.splitext(os.path.basename(file_path))
    data = read_igc_file(file_path)[:-50]

    # isolate time and altitude vectors from the dataframe
    time_v = data['normalized_time_fractional']
    data['altitude_grad'] = np.gradient(data['gps_altitude'])
    data['altitude_grad'] = data['altitude_grad'].rolling(window=5).mean()

    # Create bins
    num_bins = 30 
    altitude_bins = np.linspace(min(data['gps_altitude']), max(data['gps_altitude']), num_bins)
    data['altitude_bin'] = pd.cut(data['gps_altitude'], bins=altitude_bins)

    # Calculate the average gradient for each bin
    max_grad = data.groupby('altitude_bin')['altitude_grad'].max()
    def top_20_percent_avg(group):
        values = group['altitude_grad'].nlargest(int(len(group) * 0.2))
        return values.mean()

    def bottom_20_percent_avg(group):
        values = group['altitude_grad'].nsmallest(int(len(group) * 0.2))
        return values.mean()

    # Group by altitude_bin and apply the custom function
    average_top_grad = data.groupby('altitude_bin').apply(top_20_percent_avg)
    average_bottom_grad = data.groupby('altitude_bin').apply(bottom_20_percent_avg)
    average_points = data.groupby('altitude_bin').size()

    if plot_results:
        # Create a figure and a set of subplots
#        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 3]})

        plt.subplots_adjust(hspace=0.35)  # Adjust horizontal space

        # First subplot: Altitude over time
        ax1.plot(time_v, data['gps_altitude'], label='Altitude')
        ax1.set_title('File:{} Altitude Over Time'.format(file_title))
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Altitude (m)')

        # Third subplot: Derivative of altitude over time
        ax2.plot(altitude_bins[:-1], max_grad, label='Max Derivative')
        ax2.plot(altitude_bins[:-1], average_top_grad, label='Average top 20%')
        ax2.plot(altitude_bins[:-1], average_bottom_grad, label='Average bottom 20%')
        ax2.set_xlabel('Altitude (m)')
        ax2.set_ylabel('Mean Derivative of Altitude (m/s)')
        ax2.legend()

        # 2nd axis, plot the number in each bin
        ax2_2 = ax2.twinx()
        bin_widths = [(altitude_bins[i+1] - altitude_bins[i])*0.85 for i in range(len(altitude_bins)-1)]
        ax2_2.bar(altitude_bins[:-1], average_points, width=bin_widths, align='edge', alpha=0.3, color='b')

        ax2_2.set_ylabel('Count/Bin', color='b')
        plt.title('Average Derivative and Count by Altitude Bin')

        plt.gcf().canvas.mpl_connect('button_press_event', on_click)
        plt.show()

if __name__ == '__main__':

    default_file = '/Users/llbricks/flight_analysis/flight_IGCs/2024-01-01-XFH-000-01.IGC' 
    parser = argparse.ArgumentParser(description='Process a file.')
    parser.add_argument('--file_path', type=str, default = default_file, help='The path to the file')
    args = parser.parse_args()

    main(args.file_path,plot_results = True)

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

import time
import random
import datetime as dt
import matplotlib.dates as md

from concurrent.futures import ThreadPoolExecutor


def animate(i, data_list, file_obj, headers, ax, max_points, debug):
    '''
    Assumes file is csv
    first col is x axis
    all other cols are y axis
    '''
    line = file_obj.readline()
    if(debug):
        print(str(i) + ' line: ' + line)
    line = line.strip()
    num_cols = len(headers)
    if not line == "":
        
        data = line.split(',')
        
        try:
            assert len(data) == num_cols
            for x in range(num_cols):
                data_list[x].append(float(data[x]))
                if len(data_list[x]) > max_points:
                    data_list[x] = data_list[x][len(data_list[x])-max_points:]
        except AssertionError:
            print("Data does not match header: " + str(data))
            print("header: " + str(headers))
        except ValueError:
            print("Data is not float: " + str(data[x]))
        
    #print('plotting')
    ax.clear()
    for x in range(1, num_cols):
        dates=[dt.datetime.fromtimestamp(ts) for ts in data_list[0]]
        datenums = md.date2num(dates)
        ax.plot(datenums, data_list[x], label=headers[x])
    #print('past plotting')
    xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
    #print('done formattign')
    ax.xaxis.set_major_formatter(xfmt)
    #print('done setting')
    ax.tick_params(labelrotation=15)
    ax.set_xlabel(headers[0])
    ax.set_ylabel('Value')
    ax.set_title('Some title')
    ax.legend()

def plot_file_live(filename, refresh=200, debug=False, max_points = 50, figsize=(6,6)):
    '''
    Assumes file is comma separated with header

    '''
    print('Plotting data')
    fig, ax = plt.subplots(1,1, figsize=figsize)
    with open(filename, 'r') as file_obj:
        header_line = file_obj.readline().strip()
        headers = header_line.split(',')
        data_list = []
        for i in range(len(headers)):
            data_list.append([])
        ani = animation.FuncAnimation(fig, animate, fargs=(data_list, file_obj, headers, ax, max_points, debug), interval=refresh)
        plt.show()
    print('Done plotting')
    
def write_headers(filename, headers):
    '''
    writes headers to a file
    '''
    with open(filename, 'w') as f:
        f.write(','.join(headers) + '\n')
    print('Done writing headers: ' + str(headers))


def generate_rand_data(filename, headers, data_range, refresh = 100):
    '''
    assumes first entry is timestamps
    '''
    print('Generating data')
    max_gen = 1000
    with open(filename, 'a') as f:
        
        for i in range((max_gen)):
            time.sleep(refresh/1000)
            time1 = time.time()
            data = [str(time1)]
            for x in range(1, len(headers)):
                rand = random.uniform(data_range[0], data_range[1])
                data.append(str(rand))
            f.write(','.join(data) + '\n')
            f.flush()
    print('Done generating data')


def main():
    filename = 'example.txt'
    
    refresh1 = 50
    refresh2 = 50
    debug = False
    headers = ['timestamp', 'v1', 'v2', 'v3']
    data_range=[0,10]
    max_points = 50
    write_headers(filename, headers)
    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.submit(plot_file_live, filename, refresh=refresh2, max_points = max_points, debug=debug)
        executor.submit(generate_rand_data, filename, headers, data_range, refresh=refresh1)
        
    '''
    
    print('Generating Data')
    generate_rand_data(filename, headers, data_range, refresh=refresh1)
    print('Plotting data')
    plot_file_live(filename, refresh=refresh2, max_points = max_points, debug=debug)
    
    '''

if __name__== '__main__':
    main()

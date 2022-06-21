import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import numpy as np 

import time
import random
import datetime as dt
import pyvisa 
import matplotlib.dates as md

import threading
from concurrent.futures import ThreadPoolExecutor

global_dict={'stop':False}
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

def plot_file_live(filename, refresh=200, debug=False, figsize=(6,6)):
    '''
    Assumes file is comma separated with header

    '''
    print('Plotting data')
    shown_points = 100
    fig, ax = plt.subplots(1,1, figsize=figsize)
    with open(filename, 'r') as file_obj:
        header_line = file_obj.readline().strip()
        headers = header_line.split(',')
        data_list = []
        for i in range(len(headers)):
            data_list.append([])
        ani = animation.FuncAnimation(fig, animate, fargs=(data_list, file_obj, headers, ax, shown_points, debug), interval=refresh)
        plt.show()
    print('Done plotting')
    
def write_headers(filename, headers):
    '''
    writes headers to a file
    '''
    with open(filename, 'w') as f:
        f.write(','.join(headers) + '\n')
    print('Done writing headers: ' + str(headers))

def read_data_stream(device, filename, headers, channels, commands, refresh, readtime):
    '''
    read data from device and append to filename
    only write channels given with given commands
    refresh in milliseconds
    readtime in seconds
    '''
    print('Reading data from ' + str(device) + ' to ' + str(filename))
    max_points = int(readtime*1000/refresh)
    write_headers(filename, headers)
    with open(filename, 'a') as f:
        for i in range(max_points):
            if(global_dict['stop']):
                print('Stopping data reading at input: ' + str(i))
                break
            time.sleep(refresh/1000)
            time1 = time.time()
            data = [str(time1)]
            for command in commands:
                res = device.query(command).split(',')
                res = np.array(res)[channels]
                data.extend(res)
            f.write(','.join(data) + '\n')
            f.flush()

def generate_rand_data(filename, headers, data_range, readtime, refresh = 100):
    '''
    assumes first entry is timestamps
    '''
    print('Generating data')
    max_gen = int(readtime*1000/refresh)
    write_headers(filename, headers)
    with open(filename, 'a') as f:
        
        for i in range((max_gen)):
            if(global_dict['stop']):
                print('Stopping data reading at input: ' + str(i))
                break
            time.sleep(refresh/1000)
            time1 = time.time()
            data = [str(time1)]
            for x in range(1, len(headers)):
                rand = random.uniform(data_range[0], data_range[1])
                data.append(str(rand))
            f.write(','.join(data) + '\n')
            f.flush()
    print('Done generating data')


def get_gpb_device(device_name = 'LSCI'):
    '''
    opens the first devie that has device name
    '''
    rm = pyvisa.ResourceManager()
    rlist = rm.list_resources()
    for r in rlist:
        dev = rm.open_resource(r)
        try:
            dev_name = dev.query('IDN?')
        
        except pyvisa.errors.VisaIOError:
            print('Device timeout: ' + str(r))
            continue
        if(dev_name.split(',')[0] == device_name):
            return dev

def read_temperature_data(filename, refresh, channels, commands, headers,readtime, rand=False, data_range=[0,1], debug=False):
    '''
    '''
    
    
    if(rand):
        rand_data_thread = threading.Thread(target=generate_rand_data, args=[filename, headers, data_range, readtime, refresh]) 
        rand_data_thread.start()
    else:
        device = get_gpb_device()
        if(device is None):
            print('Could not find device')
            return
        read_data_thread = threading.Thread(target=read_data_stream, args=[ device, filename, headers, channels, commands, refresh, readtime])
        read_data_thread.start()
    plot_file_live(filename, refresh=refresh, debug=debug)
    print('Press any key to stop script')
    x=input()
    if(x):
        global_dict['stop'] = True 

def main():
    filename = 'example.txt'
    
    refresh = 500
    debug = False
    channels = [0, 2]
    headers = ['t'+str(c) for c in channels]
    headers2 = ['v'+str(c) for c in channels]
    headers.insert(0, 'timestamp')
    headers.extend(headers2)
    commands = ['CRDG?', 'SRDG?']
    readtime = 1000
    data_range=[0,10]
    max_points = 100
    read_temperature_data(filename, refresh, channels, commands, headers,readtime, rand=False)

if __name__== '__main__':
    main()

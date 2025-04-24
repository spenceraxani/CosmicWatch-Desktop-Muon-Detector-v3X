# This script requires one library:
# pyserial
# to install, type: >> pip install pyserial

from __future__ import print_function
import serial 
import time
import glob
import sys
import os
import os.path
import signal
from datetime import datetime
from multiprocessing import Process
import numpy as np
import math
import random
import platform

print('Operating System: ',platform.system())

def signal_handler(signal, frame):
    print('You pressed Ctrl+C!')
    ComPort.close()     
    file.close() 
    sys.exit(0)

def serial_ports():
    if sys.platform.startswith('win'):
        ports = ['COM%s' % (i + 1) for i in range(256)]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        # this excludes your current terminal "/dev/tty"
        ports = glob.glob('/dev/tty[A-Za-z]*')
    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/tty.*')
    else:
        raise EnvironmentError('Unsupported platform')
        sys.exit(0)
    result = []
    for port in ports:
        try: 
            s = serial.Serial(port)
            s.close()
            result.append(port)
        except (OSError, serial.SerialException):
            pass
    return result

t1 = time.time()
port_list = serial_ports()
if (time.time()-t1)>2:
    print('Listing ports is taking unusually long...')

print('\nWhich ports do you want to read from?')
for i in range(len(port_list)):
    print('  ['+str(i+1)+'] ' + str(port_list[i]))




# Account for Python 2 and Python 3 syntax
if sys.version_info[:3] > (3,0):
    ArduinoPort = input("Select port: ")
    ArduinoPort = ArduinoPort.split(',')

elif sys.version_info[:3] > (2,5,2):
    ArduinoPort = raw_input("Select port(s): ")
    ArduinoPort = ArduinoPort.split(',')

nDetectors = len(ArduinoPort)


port_name_list = []
for i in range(len(ArduinoPort)):
	port_name_list.append(str(port_list[int(ArduinoPort[i])-1]))

# Ask for file name:
cwd = os.getcwd()
print('')
default_fname = cwd+"/CW_data.txt"
if sys.version_info[:3] > (3,0):
    fname = input("Enter file name (press Enter for default: "+default_fname+"):")
elif sys.version_info[:3] > (2,5,2):
    fname = raw_input("Enter file name (press Enter for default: "+default_fname+"):")
detector_name_list = []
if fname == '':
    fname = default_fname
# If the input is just a file name (no path separators), prepend cwd
elif '/' not in fname and '\\' not in fname:
    fname = os.path.join(cwd, fname)
print(' -- Saving data to: '+fname)


print()
for i in range(nDetectors):
    time.sleep(0.1)
    port = port_name_list[i]
    baudrate = 115200
    globals()['Det%s' % str(i)] = serial.Serial(port,baudrate)
    time.sleep(0.1)
file = open(fname, "w")

# Get list of names, using 5 seconds of data.
'''
print('')
print('Acquiring detector names')
det_names = []
t1 = time.time()
while (time.time()-t1) < 5:
    for i in range(nDetectors):
        if globals()['Det%s' % str(i)].inWaiting():
            data = globals()['Det%s' % str(i)].readline().decode().replace('\r\n','')    # Wait and read data 
            data = data.split("\t")
            det_names.append(data[-1])
            
#print("\nHere is a list of the detectors I see:")
det_names = list(set(det_names))
print(det_names)
for i in range(len(det_names)):
    print("  "+str(i+1)+') '+det_names[i])
'''
# Start recording data to file.
print("\nTaking data ...")
if platform.system() == "Windows":
    print("ctrl+break to termiante process")
else:
    print("Press ctl+c to terminate process")

while True:
    for i in range(nDetectors):
        if globals()['Det%s' % str(i)].inWaiting():
            
            data = globals()['Det%s' % str(i)].readline().decode().replace('\r\n','')    # Wait and read data 
            #print(data)
            data = data.split("\t")
            
            ti = str(datetime.now()).split(" ")
            comp_time = ti[-1]
            data.append(comp_time)
            #data[1] = comp_time
            comp_date = ti[0].split('-')
            data.append(comp_date[2] + '/' +comp_date[1] + '/' + comp_date[0]) #ti[0].replace('-','/')
            for j in range(len(data)):
                #print(data[j])
                file.write(data[j]+'\t')
            file.write("\n")
            #print(str(i+'\t') for i in data)
            #print(*data, sep='\t')
            event_number = int(data[0])
            if event_number % 1 ==0:
                file.flush() 


#for i in range(nDetectors):
globals()['Det%s' % str(0)].close()     
file.close()  




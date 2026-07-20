#imports
import sys
#from matplotlib.backends.backend_qt5 import NavigationToolbar2QT
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout,
    QHBoxLayout, QGridLayout, QSlider, QTextEdit, QGroupBox, QProgressBar,
    QFileDialog, QComboBox, QRadioButton, QButtonGroup, QToolButton, QCheckBox, QFileDialog, QLineEdit
)
from PyQt5.QtCore import (
    Qt,QSize, Qt, QObject, pyqtSignal, QTimer
)
from PyQt5.QtGui import (
    QPixmap,QIcon, QFont
)
import sys
import time
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.figure import Figure
import serial.tools.list_ports
import glob
import serial 
import glob
import os
import os.path
import signal
from datetime import datetime
from multiprocessing import Process
import math
import random
import platform
import threading
import math
import pyqtgraph.opengl as gl
from stl import mesh
import sys, os, time, warnings, argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
from scipy.stats import poisson

from PIL import Image, ImageOps
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="libpng")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def img(rel_path):
    return os.path.join(BASE_DIR, rel_path)

input_images = [img('Images/refresh_white.png'), img('Images/upload_white.png'), img('Images/save_white.png'), img('Images/play_white.png'), img('Images/pause_white.png'), img('Images/logo_white.png')]
_devnull = open(os.devnull, 'w')
for input_image in input_images:
    out_image = input_image.replace('white','black')
    if os.path.exists(out_image):
        continue
    # Suppress C-level libpng stderr warnings during open
    _old_stderr = os.dup(2)
    os.dup2(_devnull.fileno(), 2)
    try:
        img = Image.open(input_image).convert("RGBA")
    finally:
        os.dup2(_old_stderr, 2)
        os.close(_old_stderr)
    # Strip ICC profile to avoid future iCCP warnings
    img.info.pop('icc_profile', None)
    r, g, b, a = img.split()
    rgb_image = Image.merge("RGB", (r, g, b))
    inverted_rgb = ImageOps.invert(rgb_image)
    inverted_img = Image.merge("RGBA", (*inverted_rgb.split(), a))
    inverted_img.save(input_image.replace('white','black'))
_devnull.close()

class CustomNavigationToolbar(NavigationToolbar):
    def __init__(self, canvas, parent=None, get_filename_func=None, plot_type="plot"):
        super().__init__(canvas, parent)
        self.get_filename_func = get_filename_func
        self.plot_type = plot_type

    def save_figure(self, *args):
        # --- 1. Get input file if available ---
        try:
            input_file = self.get_filename_func() if self.get_filename_func else "plot"
        except Exception:
            input_file = "plot"

        base = os.path.splitext(os.path.basename(input_file))[0]

        # --- 2. Determine save directory ---
        if input_file and input_file != "plot":
            base_dir = os.path.dirname(input_file)
        else:
            base_dir = os.getcwd()

        figures_dir = os.path.join(base_dir, "Figures")
        os.makedirs(figures_dir, exist_ok=True)   # ✅ auto-create if missing

        suggested = os.path.join(figures_dir, f"{base}_{self.plot_type}.pdf")

        # --- 3. Open save dialog ---
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Figure", suggested, 
            "PDF (*.pdf);;PNG (*.png);;All Files (*)"
        )

        if path:
            self.canvas.figure.savefig(path, format=os.path.splitext(path)[1][1:])

font = {
    "family": "serif",
    "serif": "Computer Modern Roman",
    "weight": 200,
    "size": 15
}
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

# Define your own color palette
mycolors = ['#c70039','#ff5733','#ff8d1a','#ffc300','#eddd53','#add45c','#57c785',
               '#00baad','#2a7b9b','#3d3d6b','#511849','#900c3f','#900c3f']

mpl.rcParams['savefig.format'] = 'pdf'


class CWClass():
    def __init__(self,file_name, bin_size, feed_box, adc_min=0, adc_max=4095):
        self.bin_size = bin_size
        self.feed_box = feed_box
        self.file_path = file_name
        self.name = file_name.split('/')[-1]
        fileHandle = open(file_name,"r" )
        lineList = fileHandle.readlines()
        fileHandle.close()
        header_lines = 0
        last_line_of_header = 0
        for i in range(min(len(lineList),1000)):
            if "#" in lineList[i]:
                last_line_of_header = i+1
        # Determine number of columns from a line near the end of the file. Scan backward
        # (skipping the very last line, which may still be mid-write) for one that matches
        # a known-good format, rather than trusting a single line — a file being actively
        # appended to (especially on a cloud-synced folder like Google Drive) can carry an
        # occasional short/malformed row without the rest of the recording being invalid.
        # Recorded lines end with a trailing tab before the newline, which would otherwise
        # split into a spurious extra (empty/"\n") column here.
        number_of_columns = 0
        scan_start = len(lineList) - 2
        for i in range(scan_start, max(scan_start - 200, -1), -1):
            n = len(lineList[i].rstrip('\t\n').split("\t"))
            if n in (10, 13):
                number_of_columns = n
                break
        if number_of_columns == 0:
            number_of_columns = len(lineList[scan_start].rstrip('\t\n').split("\t"))
        column_array = range(0,number_of_columns)


        def split_xyz(str_array):
            """Split 'X:Y:Z' strings (accel/gyro columns) into three float arrays."""
            x, y, z = [], [], []
            for s in str_array:
                parts = s.split(':')
                x.append(parts[0])
                y.append(parts[1])
                z.append(parts[2])
            return np.asarray(x).astype(float), np.asarray(y).astype(float), np.asarray(z).astype(float)

        file_from_computer = False
        file_from_sdcard   = False
        if number_of_columns == 13:
            file_from_computer = True
            data = np.genfromtxt(file_name, dtype = str, delimiter='\t', usecols=column_array, invalid_raise=False, skip_header=header_lines)
            event_number = data[:,0].astype(float) #first column of data
            PICO_timestamp_s = data[:,1].astype(float)
            coincident = np.array([s.strip() not in ('0', 'False', 'false', '') for s in data[:,2]])
            adc = data[:,3].astype(int)
            sipm = data[:,4].astype(float)
            deadtime = data[:,5].astype(float)
            temperature = data[:,6].astype(float)
            pressure = data[:,7].astype(float)
            accel_x, accel_y, accel_z = split_xyz(data[:,8].astype(str))
            gyro_x, gyro_y, gyro_z = split_xyz(data[:,9].astype(str))
            detName = data[:,10]
            comp_time = data[:,11]
            comp_date = data[:,12]

        elif number_of_columns == 10:
            file_from_sdcard = True
            self.feed_box.append('File from MicroSD Card')
            data = np.genfromtxt(file_name, dtype = str, delimiter='\t', usecols=column_array, invalid_raise=False, skip_header=header_lines)
            event_number = data[:,0].astype(float)#first column of data
            PICO_timestamp_s = data[:,1].astype(float)
            coincident = np.array([s.strip() not in ('0', 'False', 'false', '') for s in data[:,2]])
            adc = data[:,3].astype(int)
            sipm = data[:,4].astype(float)
            deadtime = data[:,5].astype(float)
            temperature = data[:,6].astype(float)
            pressure = data[:,7].astype(float)
            accel_x, accel_y, accel_z = split_xyz(data[:,8].astype(str))
            gyro_x, gyro_y, gyro_z = split_xyz(data[:,9].astype(str))
        else:
            msg = f"Incorrect number of columns in file: {number_of_columns}"
            self.feed_box.append(msg)
            raise ValueError(msg)

        if adc_min > 0 or adc_max < 4095:
            adc_mask = (np.asarray(adc) >= adc_min) & (np.asarray(adc) <= adc_max)
            event_number      = np.asarray(event_number)[adc_mask]
            PICO_timestamp_s  = np.asarray(PICO_timestamp_s)[adc_mask]
            coincident        = np.asarray(coincident)[adc_mask]
            adc               = np.asarray(adc)[adc_mask]
            sipm              = np.asarray(sipm)[adc_mask]
            deadtime          = np.asarray(deadtime)[adc_mask]
            temperature       = np.asarray(temperature)[adc_mask]
            pressure          = np.asarray(pressure)[adc_mask]
            accel_x           = np.asarray(accel_x)[adc_mask]
            accel_y           = np.asarray(accel_y)[adc_mask]
            accel_z           = np.asarray(accel_z)[adc_mask]
            gyro_x            = np.asarray(gyro_x)[adc_mask]
            gyro_y            = np.asarray(gyro_y)[adc_mask]
            gyro_z            = np.asarray(gyro_z)[adc_mask]
            if file_from_computer:
                comp_time = np.asarray(comp_time)[adc_mask]
                comp_date = np.asarray(comp_date)[adc_mask]
                detName   = np.asarray(detName)[adc_mask]

        if file_from_computer:
            time_stamp = []
            for i in range(len(comp_date)):
                day  = int(comp_date[i].split('/')[0])
                month = int(comp_date[i].split('/')[1])
                year   = int(comp_date[i].split('/')[2])
                hour  = int(comp_time[i].split(':')[0])
                mins  = int(comp_time[i].split(':')[1])
                sec   = int(np.floor(float(comp_time[i].split(':')[2])))
                try:  
                    decimal = float('0.'+str(comp_time[i].split('.')[-1]))
                except:
                    decimal = 0.0
                time_stamp.append(float(time.mktime((year, month, day, hour, mins, sec, 0, 0, 0))) + decimal) 
            self.time_stamp_s     = np.asarray(time_stamp) -  min(np.asarray(time_stamp))       # The absolute time of an event in seconds
            self.time_stamp_ms    = np.asarray(time_stamp -  min(np.asarray(time_stamp)))*1000  # The absolute time of an event in miliseconds   
            self.total_time_s     = max(time_stamp) -  min(time_stamp)     # The absolute time of an event in seconds
            self.detector_name    = detName                                
            self.n_detector       = len(set(detName))
        # deadtime is cumulative since the detector started, not since this recording started,
        # so zero it relative to this file's first event before diffing — otherwise the first
        # entry picks up the entire pre-recording cumulative deadtime as a single huge spike.
        event_deadtime_s = np.diff(np.append([0], deadtime - min(deadtime)))
        self.PICO_timestamp_s       = PICO_timestamp_s
        self.PICO_total_time_s = max(self.PICO_timestamp_s) - min(self.PICO_timestamp_s)
        self.PICO_total_time_ms= self.PICO_total_time_s * 1000.
        self.event_number     = np.asarray(event_number)  # an arrray of the event numbers
        self.total_counts     = max(event_number.astype(int)) - min(event_number.astype(int))
        self.select_coincident        = coincident         # an arrray of the measured event ADC value

        self.adc              = adc         # an arrray of the measured event ADC value
        self.sipm             = sipm        # an arrray of the measured event SiPM value
        
        self.temperature      = temperature         # an arrray of the measured event ADC value
        self.pressure        = pressure         # an arrray of the measured event ADC value

        self.accel_x        = accel_x         # an arrray event acceleration x data
        self.accel_y        = accel_y        # an arrray event acceleration x data
        self.accel_z        = accel_z        # an arrray event acceleration x data

        self.gyro_x = gyro_x
        self.gyro_y = gyro_y
        self.gyro_z = gyro_z
        

        self.event_deadtime_s   = event_deadtime_s     # an array of the measured event deadtime in seconds
        self.event_deadtime_ms  = event_deadtime_s*1000            # an array of the measured event deadtime in miliseconds
        self.total_deadtime_s   = max(deadtime) - min(deadtime)       # an array of the measured event deadtime in miliseconds
        self.total_deadtime_ms  = self.total_deadtime_s*1000. # The total deadtime in seconds
        # Same issue as deadtime above: PICO_timestamp_s is the raw device clock, not zero-based
        # at the start of this recording, so zero it first to avoid a huge spurious first entry.
        self.PICO_event_livetime_s = np.diff(np.append([0], self.PICO_timestamp_s - min(self.PICO_timestamp_s))) - self.event_deadtime_s
        def round(x, err):
            """Round x and err based on the first significant digit of err."""
            if err == 0:
                return x, err  # Avoid division by zero
            # Find order of magnitude of error
            order_of_magnitude = int(np.floor(np.log10(err)))
            # Find the first significant digit of err
            first_digit = int(err / (10 ** order_of_magnitude))
            # Round both values to the first significant digit of err
            rounded_x = np.round(x, -order_of_magnitude+1)
            rounded_err = np.round(err, -order_of_magnitude+1)#first_digit * (10 ** (order_of_magnitude)) 
            return rounded_x, rounded_err

        if file_from_computer:
            self.live_time        = (self.total_time_s - self.total_deadtime_s)
            self.live_time_s      = self.live_time
            self.weights          = np.ones(len(event_number)) / self.live_time
            self.count_rate, self.count_rate_err = round(
                    self.total_counts/self.live_time,
                    np.sqrt(self.total_counts)/self.live_time)

            bins = range(0,int(max(self.time_stamp_s)), self.bin_size)
            counts, binEdges       = np.histogram(self.time_stamp_s, bins = bins)
            bin_livetime, binEdges = np.histogram(self.time_stamp_s, bins = bins, weights = self.PICO_event_livetime_s)

            self.binned_counts     = counts
            self.binned_counts_err = np.sqrt(counts)
            safe_livetime = np.where(bin_livetime > 0, bin_livetime, np.nan)
            self.binned_count_rate = counts/safe_livetime
            self.binned_count_rate_err = np.sqrt(counts)/safe_livetime

            counts_coincident, _ = np.histogram(self.time_stamp_s[self.select_coincident], bins = bins)
            counts_non_coincident, _ = np.histogram(self.time_stamp_s[~self.select_coincident], bins = bins)

            self.total_coincident = len(self.time_stamp_s[self.select_coincident])
            self.count_rate_coincident, self.count_rate_err_coincident = round(
                    self.total_coincident/self.live_time,
                    np.sqrt(self.total_coincident)/self.live_time)

            self.binned_counts_coincident     = counts_coincident
            self.binned_counts_err_coincident = np.sqrt(counts_coincident)
            # Use the same per-bin livetime as the all-events rate so coincident + non-coincident == all,
            # even in the first/last bin where a bin doesn't span the full nominal bin_size.
            self.binned_count_rate_coincident = counts_coincident/safe_livetime
            self.binned_count_rate_err_coincident = np.sqrt(counts_coincident)/safe_livetime

            self.total_non_coincident = len(self.time_stamp_s[~self.select_coincident])
            self.count_rate_non_coincident, self.count_rate_err_non_coincident = round(
                    self.total_non_coincident/self.live_time,
                    np.sqrt(self.total_non_coincident)/self.live_time)

            self.binned_counts_non_coincident     = counts_non_coincident
            self.binned_counts_err_non_coincident = np.sqrt(counts_non_coincident)
            self.binned_count_rate_non_coincident = counts_non_coincident/safe_livetime
            self.binned_count_rate_err_non_coincident = np.sqrt(counts_non_coincident)/safe_livetime

            # Bin the pressure by taking the average pressure in each bin
            sum_pressure, _ = np.histogram(self.time_stamp_s, bins=bins, weights=self.pressure)
            count_pressure, _ = np.histogram(self.time_stamp_s, bins=bins)
            self.binned_pressure = sum_pressure / np.maximum(count_pressure, 1)
            good_p = count_pressure > 0
            if good_p.sum() > 1:
                self.binned_pressure = np.interp(np.arange(len(self.binned_pressure)), np.where(good_p)[0], self.binned_pressure[good_p])

            # Bin the temperature by taking the average temperature in each bin
            sum_temperature, _ = np.histogram(self.time_stamp_s, bins=bins, weights=self.temperature)
            count_temperature, _ = np.histogram(self.time_stamp_s, bins=bins)
            self.binned_temperature = sum_temperature / np.maximum(count_temperature, 1)
            good_t = count_temperature > 0
            if good_t.sum() > 1:
                self.binned_temperature = np.interp(np.arange(len(self.binned_temperature)), np.where(good_t)[0], self.binned_temperature[good_t])

            # Bin the temperature by taking the average temperature in each bin
            def interp_bin(ts, bins, weights):
                s, _ = np.histogram(ts, bins=bins, weights=weights)
                c, _ = np.histogram(ts, bins=bins)
                v = s / np.maximum(c, 1)
                good = c > 0
                if good.sum() > 1:
                    v = np.interp(np.arange(len(v)), np.where(good)[0], v[good])
                return v

            self.binned_accel_x = interp_bin(self.time_stamp_s, bins, self.accel_x)
            self.binned_accel_y = interp_bin(self.time_stamp_s, bins, self.accel_y)
            self.binned_accel_z = interp_bin(self.time_stamp_s, bins, self.accel_z)
            self.binned_gyro_x  = interp_bin(self.time_stamp_s, bins, self.gyro_x)
            self.binned_gyro_y  = interp_bin(self.time_stamp_s, bins, self.gyro_y)
            self.binned_gyro_z  = interp_bin(self.time_stamp_s, bins, self.gyro_z)

            bin_deadtime, _ = np.histogram(self.time_stamp_s, bins=bins, weights=self.event_deadtime_s)
            self.binned_deadtime_pct = bin_deadtime / self.bin_size * 100

        elif file_from_sdcard:
            self.live_time_s        = (self.PICO_total_time_s - self.total_deadtime_s)
            self.live_time_ms        = (self.PICO_total_time_ms - self.total_deadtime_ms)/1000.

            self.weights          = np.ones(len(event_number)) / self.live_time_s

            n = 4
            self.feed_box.append(
                f"    -- Total Count Rate: {np.round(self.total_counts/self.live_time_s, n)} +/- "
                f"{np.round(np.sqrt(self.total_counts)/self.live_time_s, n)} Hz"
            )

            self.count_rate, self.count_rate_err = round(
                    self.total_counts/self.live_time_s, 
                    np.sqrt(self.total_counts)/self.live_time_s)
            

            bins = range(int(min(self.PICO_timestamp_s)),int(max(self.PICO_timestamp_s)),max(1, int(self.bin_size)))
            counts, binEdges = np.histogram(self.PICO_timestamp_s, bins = bins)
            bin_livetime, binEdges = np.histogram(self.PICO_timestamp_s, bins = bins, weights = self.PICO_event_livetime_s)

            self.bin_size          = bin_size
            self.binned_counts     = counts
            self.binned_counts_err = np.sqrt(counts)
            safe_livetime = np.where(bin_livetime > 0, bin_livetime, np.nan)
            self.binned_count_rate = counts/safe_livetime
            self.binned_count_rate_err = np.sqrt(counts)/safe_livetime

            counts_coincident, binEdges      = np.histogram(self.PICO_timestamp_s[self.select_coincident], bins = bins)

            self.total_coincident = len(self.PICO_timestamp_s[self.select_coincident])

            self.feed_box.append(
                f"    -- Count Rate Coincident (coincident): {np.round(self.total_coincident/self.live_time_s, n)} +/- "
                f"{np.round(np.sqrt(self.total_coincident)/self.live_time_s, n)} Hz"
            )

            self.count_rate_coincident, self.count_rate_err_coincident = round(
                    self.total_coincident/self.live_time_s,
                    np.sqrt(self.total_coincident)/self.live_time_s)




            self.binned_counts_coincident     = counts_coincident
            self.binned_counts_err_coincident = np.sqrt(counts_coincident)
            # Use the same per-bin livetime as the all-events rate so coincident + non-coincident == all,
            # even in the first/last bin where a bin doesn't span the full nominal bin_size.
            self.binned_count_rate_coincident = counts_coincident/safe_livetime
            self.binned_count_rate_err_coincident = np.sqrt(counts_coincident)/safe_livetime



            counts_non_coincident, binEdges      = np.histogram(self.PICO_timestamp_s[~self.select_coincident], bins = bins)
            self.total_non_coincident = len(self.PICO_timestamp_s[~self.select_coincident])
            self.binned_counts_non_coincident     = counts_non_coincident
            self.binned_counts_err_non_coincident = np.sqrt(counts_non_coincident)
            self.binned_count_rate_non_coincident = counts_non_coincident/safe_livetime
            self.binned_count_rate_err_non_coincident = np.sqrt(counts_non_coincident)/safe_livetime
            self.feed_box.append(
                f"    -- Count Rate Non-Coincident: {np.round(self.total_non_coincident/self.live_time_s, n)} +/- "
                f"{np.round(np.sqrt(self.total_non_coincident)/self.live_time_s, n)} Hz"
            )

            self.count_rate_non_coincident, self.count_rate_err_non_coincident = round(
                    self.total_non_coincident/self.live_time_s, 
                    np.sqrt(self.total_non_coincident)/self.live_time_s)

            sum_pressure, _ = np.histogram(self.PICO_timestamp_s, bins=bins, weights=self.pressure)
            count_pressure, _ = np.histogram(self.PICO_timestamp_s, bins=bins)
            self.binned_pressure = sum_pressure / np.maximum(count_pressure, 1)
            good_p = count_pressure > 0
            if good_p.sum() > 1:
                self.binned_pressure = np.interp(np.arange(len(self.binned_pressure)), np.where(good_p)[0], self.binned_pressure[good_p])

            # Bin the temperature by taking the average temperature in each bin
            sum_temperature, _ = np.histogram(self.PICO_timestamp_s, bins=bins, weights=self.temperature)
            count_temperature, _ = np.histogram(self.PICO_timestamp_s, bins=bins)
            self.binned_temperature = sum_temperature / np.maximum(count_temperature, 1)
            good_t = count_temperature > 0
            if good_t.sum() > 1:
                self.binned_temperature = np.interp(np.arange(len(self.binned_temperature)), np.where(good_t)[0], self.binned_temperature[good_t])

            # Bin the temperature by taking the average temperature in each bin
            sum_accel_x, _ = np.histogram(self.PICO_timestamp_s, bins=bins, weights=self.accel_x)
            count_accel_x, _ = np.histogram(self.PICO_timestamp_s, bins=bins)
            self.binned_accel_x = sum_accel_x / np.maximum(count_accel_x, 1)  # Avoid division by zero

            def interp_bin(ts, bins, weights):
                s, _ = np.histogram(ts, bins=bins, weights=weights)
                c, _ = np.histogram(ts, bins=bins)
                v = s / np.maximum(c, 1)
                good = c > 0
                if good.sum() > 1:
                    v = np.interp(np.arange(len(v)), np.where(good)[0], v[good])
                return v

            self.binned_accel_x = interp_bin(self.PICO_timestamp_s, bins, self.accel_x)
            self.binned_accel_y = interp_bin(self.PICO_timestamp_s, bins, self.accel_y)
            self.binned_accel_z = interp_bin(self.PICO_timestamp_s, bins, self.accel_z)
            self.binned_gyro_x  = interp_bin(self.PICO_timestamp_s, bins, self.gyro_x)
            self.binned_gyro_y  = interp_bin(self.PICO_timestamp_s, bins, self.gyro_y)
            self.binned_gyro_z  = interp_bin(self.PICO_timestamp_s, bins, self.gyro_z)

            bin_deadtime, _ = np.histogram(self.PICO_timestamp_s, bins=bins, weights=self.event_deadtime_s)
            self.binned_deadtime_pct = bin_deadtime / self.bin_size * 100

            # Coincident binned data
        bincenters = 0.5*(binEdges[1:]+ binEdges[:-1])
        self.binned_time_s     = bincenters
        self.binned_time_m     = bincenters/60.


        
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

def plusSTD(n,array):
    xh = np.add(n,np.sqrt(np.abs(array)))
    return xh

def subSTD(n,array):
    xl = np.subtract(n,np.sqrt(np.abs(array)))
    return xl
def fill_between_steps(x, y1, y2=0, h_align='mid', ax=None, lw=2, **kwargs):
    # If no Axes object given, grab the current one:
    if ax is None:
        ax = plt.gca()
    
    # First, duplicate the x values
    xx = np.ravel(np.column_stack((x, x)))[1:]
    
    # Now: calculate the average x bin width
    xstep = np.ravel(np.column_stack((x[1:] - x[:-1], x[1:] - x[:-1])))
    xstep = np.concatenate(([xstep[0]], xstep, [xstep[-1]]))
    
    # Now: add one step at the end of the row
    xx = np.append(xx, xx.max() + xstep[-1])

    # Adjust step alignment
    if h_align == 'mid':
        xx -= xstep / 2.
    elif h_align == 'right':
        xx -= xstep

    # Also, duplicate each y coordinate in both arrays
    y1 = np.ravel(np.column_stack((y1, y1)))
    if isinstance(y2, np.ndarray):
        y2 = np.ravel(np.column_stack((y2, y2)))

    # Plotting
    ax.fill_between(xx, y1, y2=y2, lw=lw, **kwargs)
    return ax







#GUI Display
class FuturisticDashboard(QWidget):
    data_ready = pyqtSignal(str)
    data_ready2 = pyqtSignal(str)
    data_ready3 = pyqtSignal(str)
    data_ready4 = pyqtSignal(str)
        
    def default_data_dir(self):
        """Data folder inside the GUI installation location."""
        gui_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(gui_dir, "Data")
        os.makedirs(data_dir, exist_ok=True)
        return data_dir

    def handle_port_selected(self, selected_port):
        time.sleep(0.1)
        if selected_port in ("Select Port", "No ports found", ""):
            return
        baudrate = 115200
        self.serial_connection = serial.Serial(selected_port, baudrate)
        time.sleep(0.1)
        if self.serial_connection.is_open:
            self.feed_box.append(f"Connected to {selected_port}.")
            filename = f"CW_data_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
            filepath = os.path.join(self.default_data_dir(), filename)
            self.record_file(filepath=filepath)
        else:
            self.feed_box.append(f"Failed to connect to {selected_port}.")

        return
    def refresh_ports(self):
        if hasattr(self, 'serial_connection') and self.serial_connection and self.serial_connection.is_open:
            self.feed_box.append(f"Disconnected from serial port {self.serial_connection.port}.")
            self.serial_connection.close()
        self.port_dropdown.clear()
        self.port_dropdown.addItem("Select Port") 
        time.sleep(0.1)
        t1 = time.time()
        ports = serial_ports()
        if (time.time()-t1)>6:
            self.feed_box.append('Listing ports is taking unusually long...')
        for port in ports:
            self.port_dropdown.addItem(port)
        self.feed_box.append(f"Serial port refreshed.")
        if not ports:
            self.port_dropdown.addItem("No ports found")
        return
        
    
    #creating window
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CosmicWatch Dashboard")
        screen = QApplication.primaryScreen().availableGeometry()
        self._screen_h = screen.height()
        self.setStyleSheet("background-color: #0e1a2b; color: white;")
        self.init_ui()
        self.apply_theme()
        self.read_serial_active = False
        self.current_plot_func = self.run_rate  # tracks the last-shown plot
        self.data_ready.connect(self.feed_box.append)
        # self.data_ready3.connect(self.avg_stat_2.setText)
        # self.data_ready4.connect(self.avg_stat_3.setText)

        
        

    #creating layout/funcitons
    def init_ui(self):
        self.detector_name = "AxLab"
        self.time_stamp = "   "
        self.rate = "Unknown"
        self.rate_error = " "
        self.coincidence_rate = "Unknown"
        self.coincidence_rate_error = " "
        self.binning_selected = False

        self.themes = {
            "dark": {
                "bg": "#0e1a2b",
                "fg": "white",
                "accent": "#0e1a2b",
                "panel": "#1c2b3a",
                "button_background": "#122c3d",   # ← dark button
                "button_hover": "#0e1a2b",  # dark button hover
                "button_border": "white",      # ← light butto
                "button_text": "white"  # dark button hover
            },
            "light": {
                "bg": "white",
                "fg": "#202020",
                "accent": "#0066cc",
                "panel": "#f0f0f0",
                "button_background": "#122c3d",      # ← light button
                "button_border": "black",      # ← light butto
                "button_hover": "#0e1a2b",  # dark button hover
                "button_text": "white"  # dark button hover
            }
        }
        self.current_theme = "dark"
        
        #setting side lengths of right and left panel
        main_layout = QGridLayout()
        main_layout.setColumnStretch(0, 18)  # Left side ~72%
        main_layout.setColumnStretch(1, 7)  # Right side ~28%
        
        #left Panel/top buttons
        left_panel = QVBoxLayout()
        control_btns = QHBoxLayout()
        button_style = """
            QPushButton, QToolButton {
                background-color: #122c3d;
                color: white;              /* keep cyan text/icons */
                border: 1px solid white;     /* <-- white border */
                border-radius: 8px;
                padding: 6px;
            }

            QPushButton:hover, QToolButton:hover {
                background-color: #0e1a2b;
                border: 1px solid white;     /* <-- white border */
                color: #00ffff;
            }

            QPushButton:pressed, QToolButton:pressed {
                background-color: #0b1a29;
                border: 1px solid white;     /* <-- white border */
                color: #00cccc;
            }
            """
        

        #load button
        load_btn = QToolButton()
        load_btn.setIcon(QIcon(img("Images/upload_white.png")))
        load_btn.setIconSize(QSize(20, 20))
        load_btn.setStyleSheet(button_style)
        load_btn.clicked.connect(self.load_file)
        control_btns.addWidget(load_btn)

        left_panel.addLayout(control_btns)
        # STOP button
        stop_btn = QToolButton()
        stop_btn.setIcon(QIcon(img("Images/pause_white.png")))
        stop_btn.setIconSize(QSize(20, 20))
        stop_btn.setStyleSheet(button_style)
        stop_btn.clicked.connect(self.stop_file)
        control_btns.addWidget(stop_btn)
        #record
        record_btn = QToolButton()
        record_btn.setIcon(QIcon(img("Images/play_white.png")))
        record_btn.setIconSize(QSize(20, 20))
        record_btn.setStyleSheet(button_style)
        record_btn.clicked.connect(self.record_file)
        control_btns.addWidget(record_btn)
        
        #portselection
        self.port_dropdown = QComboBox()
        self.port_dropdown.setStyleSheet("""
            QComboBox {
                background-color: #122c3d;
                color: #eee;
                border: 1px solid white;
                border-radius: 4px;
                padding: 4px;
                min-height: 20px;
            }
            QComboBox QAbstractItemView {
                background-color: #0e1a2b;
                color: white;
                selection-background-color: white;
            }
        """)
                

        #bin time buttons
        radio_style = """
            QRadioButton {
                color: #eee;
                background-color: #122c3d;
                border-radius: 4px;
                padding: 4px;
            }
            QRadioButton::indicator {
                width: 16px;
                height: 16px;
            }
            QRadioButton::indicator::checked {
                background-color: white;
                border: 1px solid white;
                border-radius: 8px;
            }
        """

        binning_layout = QHBoxLayout()

        self.binning_label = QLabel("Binning Time:")
        self.binning_label.setStyleSheet("""
            font-family: 'Times New Roman', Times, serif;
            color: #eee;
            padding-right: 1px;
            font-size: 18px;
            font-weight: bold;
        """)
        binning_layout.addWidget(self.binning_label)
        

        self.bin_button_30 = QRadioButton("30s")
        self.bin_button_60 = QRadioButton("60s")
        self.bin_button_120 = QRadioButton("120s")
        self.bin_button_180 = QRadioButton("180s")
        self.bin_240s_btn = QRadioButton("240s")
        self.bin_600s_btn = QRadioButton("600s")
        
        self.bin_button_30.setChecked(True)
        self.selected_bin_time = 30
        self.binning_selected = True

        binning_layout.addWidget(self.bin_button_30)
        binning_layout.addWidget(self.bin_button_60)
        binning_layout.addWidget(self.bin_button_120)
        binning_layout.addWidget(self.bin_button_180)
        binning_layout.addWidget(self.bin_240s_btn)
        binning_layout.addWidget(self.bin_600s_btn)
        self.bin_button_30.clicked.connect(self.select_bin_30)
        self.bin_button_60.clicked.connect(self.select_bin_60)
        self.bin_button_120.clicked.connect(self.select_bin_120)
        self.bin_button_180.clicked.connect(self.select_bin_180)
        self.bin_240s_btn.clicked.connect(self.select_bin_240)
        self.bin_600s_btn.clicked.connect(self.select_bin_600)

        self.custom_bin_input = QLineEdit()
        self.custom_bin_input.setPlaceholderText("custom s")
        self.custom_bin_input.setFixedWidth(72)
        self.custom_bin_input.setStyleSheet("""
            QLineEdit {
                background-color: #122c3d;
                color: #eee;
                border: 1px solid white;
                border-radius: 4px;
                padding: 3px 5px;
                font-size: 12px;
            }
        """)
        self.custom_bin_input.returnPressed.connect(self.select_bin_custom)
        binning_layout.addWidget(self.custom_bin_input)
        left_panel.addLayout(binning_layout)

        # ADC threshold filter
        adc_filter_layout = QHBoxLayout()

        self.adc_filter_label = QLabel("ADC Min:")
        self.adc_filter_label.setStyleSheet("color: #eee; font-size: 14px; font-weight: bold;")
        adc_filter_layout.addWidget(self.adc_filter_label)

        self.adc_slider = QSlider(Qt.Horizontal)
        self.adc_slider.setMinimum(0)
        self.adc_slider.setMaximum(4095)
        self.adc_slider.setValue(0)
        self.adc_slider.setTickInterval(256)
        self.adc_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 3px;
                background: #3a4a5a;
                border-radius: 1px;
            }
            QSlider::handle:horizontal {
                background: white;
                border: 1px solid white;
                width: 10px;
                height: 10px;
                margin: -4px 0;
                border-radius: 5px;
            }
            QSlider::sub-page:horizontal {
                background: white;
                border-radius: 1px;
                height: 3px;
            }
            QSlider::add-page:horizontal {
                background: #3a4a5a;
                border-radius: 1px;
                height: 3px;
            }
        """)
        adc_filter_layout.addWidget(self.adc_slider)

        self.adc_slider_label = QLabel("0")
        self.adc_slider_label.setStyleSheet("color: #eee; font-size: 14px; min-width: 40px;")
        adc_filter_layout.addWidget(self.adc_slider_label)

        self.adc_update_btn = QPushButton("Update")
        adc_filter_layout.addWidget(self.adc_update_btn)

        self.adc_slider.valueChanged.connect(lambda v: self.adc_slider_label.setText(str(v)))
        self.adc_slider.valueChanged.connect(self._update_adc_cut_line)
        self.adc_update_btn.clicked.connect(self.recalc_cw)

        left_panel.addLayout(adc_filter_layout)

        # ADC max filter
        adc_max_layout = QHBoxLayout()

        self.adc_max_filter_label = QLabel("ADC Max:")
        self.adc_max_filter_label.setStyleSheet("color: #eee; font-size: 14px; font-weight: bold;")
        adc_max_layout.addWidget(self.adc_max_filter_label)

        self.adc_max_slider = QSlider(Qt.Horizontal)
        self.adc_max_slider.setMinimum(0)
        self.adc_max_slider.setMaximum(4095)
        self.adc_max_slider.setValue(4095)
        self.adc_max_slider.setTickInterval(256)
        self.adc_max_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 3px;
                background: #3a4a5a;
                border-radius: 1px;
            }
            QSlider::handle:horizontal {
                background: white;
                border: 1px solid white;
                width: 10px;
                height: 10px;
                margin: -4px 0;
                border-radius: 5px;
            }
            QSlider::sub-page:horizontal {
                background: white;
                border-radius: 1px;
                height: 3px;
            }
            QSlider::add-page:horizontal {
                background: #3a4a5a;
                border-radius: 1px;
                height: 3px;
            }
        """)
        adc_max_layout.addWidget(self.adc_max_slider)

        self.adc_max_slider_label = QLabel("4095")
        self.adc_max_slider_label.setStyleSheet("color: #eee; font-size: 14px; min-width: 40px;")
        adc_max_layout.addWidget(self.adc_max_slider_label)

        self.adc_max_update_btn = QPushButton("Update")
        adc_max_layout.addWidget(self.adc_max_update_btn)

        self.adc_max_slider.valueChanged.connect(lambda v: self.adc_max_slider_label.setText(str(v)))
        self.adc_max_slider.valueChanged.connect(self._update_adc_max_cut_line)
        self.adc_max_update_btn.clicked.connect(self.recalc_cw)

        left_panel.addLayout(adc_max_layout)













        # graph
        canvas_h = max(280, int(self._screen_h * 0.44))
        self.static_canvas = FigureCanvas(Figure(figsize=(4, 3)))
        self.static_canvas.setMinimumHeight(canvas_h)
        self.static_canvas.setSizePolicy(
            self.static_canvas.sizePolicy().horizontalPolicy(),
            QtWidgets.QSizePolicy.Expanding
        )
        self.static_canvas.setStyleSheet("border: 1px solid #1f2f46; padding: 10px;")


        left_panel.addWidget(self.static_canvas)
     
        # Create toolbar
        #toolbar_static = NavigationToolbar(self.static_canvas, self)
        #toolbar_static = CustomNavigationToolbar(
        #    self.static_canvas,
        #    self,
        #    get_filename_func=lambda: getattr(self, "cw", None) and getattr(self.cw, "file_path", "plot"),
        #    plot_type="rate.pdf"   # change dynamically when different run_* methods are called
        #)
        self.toolbar = CustomNavigationToolbar(
            self.static_canvas,
            self,
            get_filename_func=lambda: getattr(self.cw, "file_path", "plot"),
            plot_type="plot"   # default; updated in run_* methods
        )

        #toolbar_static = CustomNavigationToolbar(self.static_canvas, self, plot_type="rate")

        # Suppress built-in coordinate readout
        def suppress_message(*args, **kwargs):
            pass
        self.toolbar.set_message = suppress_message

        # Center the toolbar
        toolbar_layout = QHBoxLayout()
        toolbar_layout.addStretch()
        #toolbar_layout.addWidget(toolbar_static)
        toolbar_layout.addWidget(self.toolbar)
        toolbar_layout.addStretch()
        left_panel.addLayout(toolbar_layout)

        # --- Add custom coordinate display ---
        self.coord_label = QLabel("x: -, y: -")
        self.coord_label.setStyleSheet("color: white;")   # will adapt to theme later
        left_panel.addWidget(self.coord_label, alignment=Qt.AlignLeft)

        # --- Connect mpl event for mouse motion ---
        def update_coords(event):
            if event.inaxes:  # inside the plot
                x, y = event.xdata, event.ydata
                self.coord_label.setText(f"x: {x:.2f}, y: {y:.2f}")
            else:  # outside axes
                self.coord_label.setText("x: -, y: -")

        self.static_canvas.mpl_connect("motion_notify_event", update_coords)

        # --- Optional: style toolbar buttons ---
        self.toolbar.setStyleSheet("""
            QToolBar { border-radius: 8px; }
            QToolButton {
                background-color: white;
                border: 2px solid black;
                border-radius: 8px;
                color: white;
            }
            QToolButton:pressed {
                background-color: grey;
            }

        """)

        

        
        self.static_ax = self.static_canvas.figure.subplots()
        
        for spine in self.static_ax.spines.values():
            spine.set_edgecolor("white")

        self.static_ax.set_facecolor("#0e1a2b")  # dark dashboard color
        self.static_canvas.figure.set_facecolor("#0e1a2b")  # whole figure background
        # 4️⃣ Make all text white
        self.static_ax.title.set_color('white')
        self.static_ax.xaxis.label.set_color('white')
        self.static_ax.yaxis.label.set_color('white')
        self.static_ax.tick_params(axis='x', colors='white')
        self.static_ax.tick_params(axis='y', colors='white')
        # 5️⃣ Add grid with white lines
        self.static_ax.grid(True, color='white', linestyle='--', alpha=0.3)

        #Bottom buttons
        scan_btns = QHBoxLayout()

        # Temperature button
        self.rate_btn = QPushButton("Rate")
        self.rate_btn.clicked.connect(lambda: self._plot_or_correlate(
            'Coincident Rate [Hz]',
            lambda: [(self.cw.binned_count_rate_coincident, mycolors[1], 'Coincident')],
            self.run_rate, 'binned', self.rate_btn))
        scan_btns.addWidget(self.rate_btn)

        # ADC button
        self.adc_btn = QPushButton("ADC")
        self.adc_btn.clicked.connect(lambda: self._plot_or_correlate(
            'ADC [0-4095]', lambda: [(self.cw.adc, mycolors[7], 'All Events')],
            self.run_adc, 'per_event', self.adc_btn))
        scan_btns.addWidget(self.adc_btn)

        # SiPM button
        self.SiPM_btn = QPushButton("SiPM")
        self.SiPM_btn.clicked.connect(lambda: self._plot_or_correlate(
            'SiPM [mV]', lambda: [(self.cw.sipm, mycolors[7], 'All Events')],
            self.run_voltage, 'per_event', self.SiPM_btn))
        scan_btns.addWidget(self.SiPM_btn)

        # Pressure button
        self.pressure_btn = QPushButton("Pressure")
        self.pressure_btn.clicked.connect(lambda: self._plot_or_correlate(
            'Pressure [Pa]', lambda: [(self.cw.binned_pressure, mycolors[6], 'Pressure')],
            self.run_pressure, 'binned', self.pressure_btn))
        scan_btns.addWidget(self.pressure_btn)

        # Temperature button
        self.temperature_btn = QPushButton("Temperature")
        self.temperature_btn.clicked.connect(lambda: self._plot_or_correlate(
            'Temperature [C]', lambda: [(self.cw.binned_temperature, mycolors[5], 'Temperature')],
            self.run_temperature, 'binned', self.temperature_btn))
        scan_btns.addWidget(self.temperature_btn)

        # Linear Acceleration button
        self.acc_btn = QPushButton("Linear Acceleration")
        self.acc_btn.clicked.connect(lambda: self._plot_or_correlate(
            'Accel magnitude [g]',
            lambda: [(np.sqrt(self.cw.binned_accel_x**2 + self.cw.binned_accel_y**2 + self.cw.binned_accel_z**2), mycolors[2], 'Accel')],
            self.run_acc, 'binned', self.acc_btn))
        scan_btns.addWidget(self.acc_btn)

        # Angular Velocity button
        self.gyro_btn = QPushButton("Angular velocity")
        self.gyro_btn.clicked.connect(lambda: self._plot_or_correlate(
            'Gyro magnitude [deg/s]',
            lambda: [(np.sqrt(self.cw.binned_gyro_x**2 + self.cw.binned_gyro_y**2 + self.cw.binned_gyro_z**2), mycolors[2], 'Gyro')],
            self.run_gyro, 'binned', self.gyro_btn))
        scan_btns.addWidget(self.gyro_btn)

        self.deadtime_btn = QPushButton("Deadtime")
        self.deadtime_btn.clicked.connect(lambda: self._plot_or_correlate(
            'Deadtime [%]', lambda: [(self.cw.binned_deadtime_pct, mycolors[9], 'Deadtime')],
            self.run_deadtime, 'binned', self.deadtime_btn))
        scan_btns.addWidget(self.deadtime_btn)

        self.rate_dist_btn = QPushButton("Rate Distribution")
        self.rate_dist_btn.clicked.connect(lambda: self._plot_or_correlate(
            None, None, self.run_rate_dist, 'none', self.rate_dist_btn))
        scan_btns.addWidget(self.rate_dist_btn)

        self.interevent_btn = QPushButton("Δt Distribution")
        self.interevent_btn.clicked.connect(lambda: self._plot_or_correlate(
            None, None, self.run_interevent, 'none', self.interevent_btn))
        scan_btns.addWidget(self.interevent_btn)

        # Add to layout — scan_btns spans both columns (row 1)
        main_layout.addLayout(left_panel, 0, 0)
        main_layout.addLayout(scan_btns, 1, 0, 1, 2)


        # right panel text display
        right_panel = QGridLayout()
        right_panel.setColumnStretch(0, 1)
        right_panel.setColumnStretch(0, 1)

        self.stats_container = QWidget()
        self.stats_layout = QVBoxLayout(self.stats_container)
        self.stats_layout.setContentsMargins(0, 0, 0, 0)
        self.stats_layout.setSpacing(12)  # Slightly more breathing room between boxes

        # New improved style for the QTextEdit "content box"
        stats_style = """
            background-color: #1c2b3a;
            color: white;          /* Neon green text */
            border-radius: 8px;
            font-family: 'Consolas', 'Courier New', monospace;  /* Monospaced font for techy look */
            font-size: 14px;
            font-weight: bold;
            background-image: url('""" + img('Images/refresh_white.png').replace('\\', '/') + """');
            background-repeat: no-repeat;
            background-position: center;
            background-color: #1c2b3a;
            color: white;        /* <--- text color */
            border-radius: 12px;

            letter-spacing: 1px;    /* space between letters */
        """

        # A separate style just for the LABELS
        #font-family: 'Times New Roman', Times, serif;
        label_style = """
            color: #FFFFFF;
            font-size: 13px;
            font-weight: bold;
            padding-left: 2px;
        """

        def make_labeled_box(label_text, content_text):
            container = QWidget()
            vbox = QVBoxLayout(container)
            vbox.setContentsMargins(6, 4, 6, 4)
            vbox.setSpacing(4)

            label = QLabel(label_text)
            
            

            textedit = QTextEdit()
            textedit.setReadOnly(True)
            textedit.setFixedHeight(max(150, int(self._screen_h * 0.24)))
            textedit.setText(content_text)
            textedit.setStyleSheet("""
                border: none;               /* ✅ no border */
                background: transparent;    /* optional, blends with parent */
            """)

            vbox.addWidget(label)
            vbox.addWidget(textedit)

            return container, label, textedit

        # Create all 3 labeled boxes
        

        box1, self.status_label, self.avg_stat1 = make_labeled_box("Status:", "")
        #self.stats_layout.addWidget(box1)
        self.stats_layout.addWidget(box1)
        right_panel.addWidget(self.stats_container, 1, 0, alignment=Qt.AlignTop)
        self.data_ready2.connect(self.avg_stat1.setHtml)

        # CAD viewer container
        # --- CAD viewer container
        image_container = QWidget()
        image_container_h = max(200, int(self._screen_h * 0.32))
        image_container.setFixedWidth(250)
        image_container.setFixedHeight(image_container_h)

        image_container.setStyleSheet("background: transparent; border: none;")

        image_layout = QVBoxLayout(image_container)
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.setSpacing(4)
        #image_container.setStyleSheet("border: 1px solid black; padding: 10px; border-radius: 8px;")
        image_container.setStyleSheet("background: transparent; border: none;")
        # --- PNG image ABOVE the STL widget ---
        self.logo_label = QLabel()
        self.logo_label.setAlignment(Qt.AlignCenter)
        self.logo_label.setStyleSheet("background: transparent;")
        self.logo_label.setAttribute(Qt.WA_TranslucentBackground, True)
        self.logo_label.setAttribute(Qt.WA_OpaquePaintEvent, False)
        self.logo_label.setStyleSheet("background-color: rgba(0,0,0,0); border: none;")

        logo_path = img("Images/logo_white.png")
        pix = QPixmap(logo_path)
        if not pix.isNull():
            dpr = QApplication.primaryScreen().devicePixelRatio()
            logical_w = 230
            physical_w = int(logical_w * dpr)
            physical_h = int(physical_w * pix.height() / pix.width())
            scaled = pix.scaled(physical_w, physical_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            scaled.setDevicePixelRatio(dpr)
            self.logo_label.setPixmap(scaled)
        else:
            self.logo_label.setText("logo_white.png not found")

        image_layout.addWidget(self.logo_label, alignment=Qt.AlignHCenter | Qt.AlignTop)

        gl_h = max(140, image_container_h - 80)
        self.gl_widget = gl.GLViewWidget()
        self.gl_widget.setFixedSize(250, gl_h)
        self.gl_widget.setCameraPosition(distance=100, elevation=20)

        self.gl_widget.setBackgroundColor((14, 26, 43, 255))
        your_mesh = mesh.Mesh.from_file(img("Images/bare_assembly.stl"))
        vertices = your_mesh.vectors.reshape(-1, 3)
        faces = np.arange(len(vertices)).reshape(-1, 3)

        image_layout.addWidget(self.gl_widget, alignment=Qt.AlignHCenter | Qt.AlignTop)
        image_layout.addStretch(1)
        image_layout.setContentsMargins(0, 0, 0, 130)

        mesh_data = gl.MeshData(vertexes=vertices, faces=faces)
        mesh_item = gl.GLMeshItem(
            meshdata=mesh_data, smooth=False, drawFaces=True, drawEdges=True,
            edgeColor=(0.3, 0.3, 0.3, 1),
            color=(0.9, 0.9, 0.9, 1)
        )
        mesh_item.setGLOptions('opaque')
        self.gl_widget.addItem(mesh_item)

        
        right_panel.addWidget(image_container, 0, 0, alignment=Qt.AlignHCenter | Qt.AlignTop)

        main_layout.addLayout(right_panel, 0, 1, alignment=Qt.AlignHCenter | Qt.AlignTop)
       
        

        #texbox bottom
        bottom_panel = QHBoxLayout()
        self.bottom_widget = QWidget()
        self.bottom_widget.setLayout(bottom_panel)
        #self.bottom_widget.setStyleSheet("background-color:  white; color: white;  border-radius: 3px")
        self.bottom_widget.setStyleSheet("""
            background-color: #1c2b3a;
            color: white;
            border-radius: 3px;
        """)
        log_h = max(80, int(self._screen_h * 0.14))
        self.bottom_widget.setMinimumHeight(log_h)

        self.feed_box = QTextEdit()
        self.feed_box.setReadOnly(True)
        self.feed_box.setStyleSheet("background-color: #1c2b3a; color: #FFFFFF; border-radius: 3px; font-family: 'Menlo', 'SF Mono', 'Consolas', monospace; font-size: 11px;")
        self.feed_box.setTabStopDistance(60)  # pixels per tab stop
        self.feed_box.setMinimumHeight(log_h)
        bottom_panel.addWidget(self.feed_box)

        # Add bottom panel to row 2, spanning 2 columns; let it grow vertically
        main_layout.addWidget(self.bottom_widget, 2, 0, 1, 2)
        main_layout.setRowStretch(2, 1)
        self.setLayout(main_layout)
        #self.feed_box.setText(f'Welcome to CosmicDAQ GUI')
        self.feed_box.setText(f' -- Welcome to CosmicDAQ Graphical User Interface -- \nOperating System: {platform.system()}')


        self.port_dropdown.addItem("Select Port") 
        
        t1 = time.time()
        ports = serial_ports()
        if (time.time()-t1)>6:
           self.feed_box.setText('Listing ports is taking unusually long...')
        for port in ports:
            self.port_dropdown.addItem(port)
        if not ports:
            self.port_dropdown.addItem("No ports found")
        control_btns.addWidget(self.port_dropdown)

        

        self.port_dropdown.activated[str].connect(self.handle_port_selected)
        

        refresh_btn = QToolButton()
        refresh_btn.setIcon(QIcon(img("Images/refresh_white.png")))
        refresh_btn.setIconSize(QSize(20, 20))
        refresh_btn.setStyleSheet(button_style)
        control_btns.addWidget(refresh_btn)
        refresh_btn.clicked.connect(self.refresh_ports)   

        self.theme_toggle = QToolButton()
        self.theme_toggle.setText("☀️ / 🌙")
        self.theme_toggle.setStyleSheet(button_style)
        self.theme_toggle.clicked.connect(self.toggle_theme)
        control_btns.addWidget(self.theme_toggle)  
        

    def apply_theme(self):
        t = self.themes[self.current_theme]

        # Window
        self.setStyleSheet(f"background-color: {t['bg']}; color: {t['fg']};")

        # Feed box + stats
        self.feed_box.setStyleSheet(f"background-color: {t['panel']}; color: {t['fg']}; border-radius: 3px; font-family: 'Menlo', 'SF Mono', 'Consolas', monospace; font-size: 11px;")
        
        #self.bottom_widget.setStyleSheet(
        #    f"""
        #    background-color: {t['panel']};
        #    color: {t['fg']};
        #    border: 1px solid black;
        #    border-radius: 3px;
        #    """
        #)
        '''
        QPushButton, QToolButton {
                background-color: #122c3d;
                color: white;              /* keep cyan text/icons */
                border: 1px solid white;     /* <-- white border */
                border-radius: 8px;
                padding: 6px;
            }

            QPushButton:hover, QToolButton:hover {
                background-color: #0e1a2b;
                border: 1px solid white;     /* <-- white border */
                color: #00ffff;
            }

            QPushButton:pressed, QToolButton:pressed {
                background-color: #0b1a29;
                border: 1px solid white;     /* <-- white border */
                color: #00cccc;
            }
            """
        

        button_style = f"""
            QPushButton {{
                background-color: {t['button']};
                color: {t['fg']};
                border: 1px solid {t['fg']};
                border-radius: 6px;
                padding: 4px;
            }}
            QPushButton:hover {{
                background-color: {t['bg']};
                color: {t['fg']};
                border: 1px solid {t['fg']};
                border-radius: 6px;
                padding: 4px;
            }}
        """
        '''
        button_style = f"""
            QPushButton {{
                background-color: {t['button_background']};
                color: {t['button_text']};
                border: 1px solid {t['button_border']};
                border-radius: 6px;
                padding: 4px;
            }}
            QPushButton:hover {{
                background-color: {t['button_hover']};
                color: {t['button_text']};
                border: 1px solid {t['button_border']};
                border-radius: 6px;
                padding: 4px;
            }}
        """

        for btn in [self.rate_btn, self.adc_btn, self.pressure_btn,
            self.temperature_btn, self.acc_btn, self.gyro_btn, self.SiPM_btn,
            self.deadtime_btn, self.rate_dist_btn, self.interevent_btn,
            self.adc_update_btn, self.adc_max_update_btn]:
            btn.setStyleSheet(button_style)
        # Re-apply cyan to the active plot button and any pending correlation selection
        active = getattr(self, '_active_plot_btn', None)
        if active:
            active.setStyleSheet(self._CORR_SELECTED_STYLE)
        pending = getattr(self, '_corr_pending', None)
        if pending and pending[3] and pending[3] is not active:
            pending[3].setStyleSheet(self._CORR_SELECTED_STYLE)

        # Apply to all bottom buttons
        #for btn in [self.rate_btn, self.adc_btn, self.pressure_btn,
        #            self.temp_btn, self.acc_btn, self.gyro_btn]:
        #    btn.setStyleSheet(button_style)

        self.avg_stat1.setStyleSheet(f"background-color: {t['panel']}; color: {t['accent']}; border-radius: 8px;")
        self.binning_label.setStyleSheet(f"""
            color: {t['fg']};
            padding-right: 10px;
            font-size: 18px;
            font-weight: bold;
        """)
        
        self.coord_label.setStyleSheet(f"color: {t['fg']};")

        # ADC filter row
        self.adc_filter_label.setStyleSheet(
            f"color: {t['fg']}; font-size: 14px; font-weight: bold;")
        self.adc_slider_label.setStyleSheet(
            f"color: {t['fg']}; font-size: 14px; min-width: 40px;")
        handle_color = "#333333" if self.current_theme == "light" else "white"
        groove_color = "#aaaaaa" if self.current_theme == "light" else "#3a4a5a"
        self.adc_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                height: 3px;
                background: {groove_color};
                border-radius: 1px;
            }}
            QSlider::handle:horizontal {{
                background: {handle_color};
                border: 1px solid {handle_color};
                width: 10px;
                height: 10px;
                margin: -4px 0;
                border-radius: 5px;
            }}
            QSlider::sub-page:horizontal {{
                background: {handle_color};
                border-radius: 1px;
                height: 3px;
            }}
            QSlider::add-page:horizontal {{
                background: {groove_color};
                border-radius: 1px;
                height: 3px;
            }}
        """)

        self.status_label.setStyleSheet(f"""
            font-family: 'Times New Roman', Times, serif;
            font-size: 16pt;
            font-weight: bold;
            color: {t['fg']};
            """)
        self.avg_stat1.setStyleSheet(f"""
            background-color: {t['panel']};
            color: {t['fg']};
            border-radius: 8px;
        """)
        
        if self.current_theme == "dark":
            bg = (14, 26, 43, 255)   # dark blue
        else:
            bg = (255, 255, 255, 255)  # pure white
        self.gl_widget.setBackgroundColor(bg)
        
        #logo_file = "Images/logo_white.png" if self.current_theme == "dark" else "Images/logo_black.png"
        # ✅ Update logo depending on theme
        logo_file = img("Images/logo_white.png") if self.current_theme == "dark" else img("Images/logo_black.png")
        pix = QPixmap(logo_file)
        if not pix.isNull():
            dpr = QApplication.primaryScreen().devicePixelRatio()
            logical_w = 230
            physical_w = int(logical_w * dpr)
            physical_h = int(physical_w * pix.height() / pix.width())
            scaled = pix.scaled(physical_w, physical_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            scaled.setDevicePixelRatio(dpr)
            self.logo_label.setPixmap(scaled)
        else:
            self.logo_label.setText(f"{logo_file} not found")

        for spine in self.static_ax.spines.values():
            spine.set_edgecolor(t["fg"])
        self.static_ax.grid(True, color=t["fg"], linestyle='--', alpha=0.3)

        # ✅ Legend (if one exists)
        leg = self.static_ax.get_legend()
        if leg:
            for text in leg.get_texts():
                text.set_color(t["fg"])

        for line in self.static_ax.get_lines():
            line.set_markerfacecolor(t["fg"])
            line.set_markeredgecolor(t["fg"])
            
        # Matplotlib figure
        self.static_ax.set_facecolor(t["bg"])
        self.static_canvas.figure.set_facecolor(t["bg"])
        self.static_ax.title.set_color(t["fg"])
        self.static_ax.xaxis.label.set_color(t["fg"])
        self.static_ax.yaxis.label.set_color(t["fg"])
        self.static_ax.tick_params(axis='x', colors=t["fg"])
        self.static_ax.tick_params(axis='y', colors=t["fg"])
        self.static_ax.grid(True, color=t["fg"], linestyle='--', alpha=0.3)
        self.static_canvas.draw()


    def _status_html(self, run_time_s, deadtime_s, live_time_s, events, rate, rate_err,
                      coincidences, coinc_rate, coinc_rate_err, avg_temp, avg_pressure):
        """Build the summary-statistics table, shared by the loaded-file view and the live serial view."""
        def row(label, value):
            return f"<tr><td>{label}</td><td align='right' nowrap>{value}</td></tr>"
        return (
            "<table width='100%' cellspacing='3'>"
            + row("Run Time:",    f"{run_time_s:,.2f} s")
            + row("Deadtime:",    f"{deadtime_s:,.2f} s")
            + row("Live Time:",   f"{live_time_s:,.2f} s")
            + "<tr><td colspan='2'>&nbsp;</td></tr>"
            + row("Events:",      f"{events:,.0f}")
            + row("Rate:",        f"{rate:,.4f} ± {rate_err:,.4f} Hz")
            + row("Coincidences:", f"{coincidences:,.0f}")
            + row("Rate:",        f"{coinc_rate:,.4f} ± {coinc_rate_err:,.4f} Hz")
            + "<tr><td colspan='2'>&nbsp;</td></tr>"
            + row("Avg Temperature:", f"{avg_temp:,.2f} °C")
            + row("Avg Pressure:",    f"{avg_pressure:,.0f} Pa")
            + "</table>"
        )

    def _emit_status(self):
        """Emit updated statistics HTML to the status panel."""
        html = self._status_html(
            self.cw.PICO_total_time_s, self.cw.total_deadtime_s, self.cw.live_time_s,
            self.cw.total_counts, self.cw.count_rate, self.cw.count_rate_err,
            self.cw.total_coincident, self.cw.count_rate_coincident, self.cw.count_rate_err_coincident,
            np.mean(self.cw.temperature), np.mean(self.cw.pressure),
        )
        self.data_ready2.emit(html)

    def _update_adc_cut_line(self, value):
        """Draw/update a vertical dashed line on the ADC plot showing the current cut."""
        if not hasattr(self, 'toolbar') or self.toolbar.plot_type != "ADC":
            return
        if hasattr(self, '_adc_cut_line') and self._adc_cut_line is not None:
            try:
                self._adc_cut_line.remove()
            except Exception:
                pass
        self._adc_cut_line = self.static_ax.axvline(
            x=value, color='white', linestyle='--', linewidth=1.5, alpha=0.8
        )
        self.static_ax.figure.canvas.draw_idle()

    def _update_adc_max_cut_line(self, value):
        """Draw/update a vertical dashed line on the ADC plot showing the current max cut."""
        if not hasattr(self, 'toolbar') or self.toolbar.plot_type != "ADC":
            return
        if hasattr(self, '_adc_max_cut_line') and self._adc_max_cut_line is not None:
            try:
                self._adc_max_cut_line.remove()
            except Exception:
                pass
        self._adc_max_cut_line = self.static_ax.axvline(
            x=value, color='white', linestyle='--', linewidth=1.5, alpha=0.8
        )
        self.static_ax.figure.canvas.draw_idle()

    def recalc_cw(self):
        """Recalculate CWClass when binning or ADC threshold changes."""
        if hasattr(self, "cw"):
            file_name = self.cw.file_path
            adc_min = self.adc_slider.value()
            adc_max = self.adc_max_slider.value()
            try:
                self.cw = CWClass(file_name, self.selected_bin_time, self.feed_box, adc_min=adc_min, adc_max=adc_max)
            except Exception as e:
                self.feed_box.append(f"Recalculation failed: {e}")
                return
            self.feed_box.append(f"Recalculated with bin size = {self.selected_bin_time}s, ADC min = {adc_min}, ADC max = {adc_max}")
            self._emit_status()
            self.current_plot_func()  # re-run whichever plot was last shown
            self.apply_theme()
            
    def select_bin_30(self):
        self.binning_selected = True
        self.selected_bin_time = 30
        self.feed_box.append("Selected bin time: 30")
        self.recalc_cw()
        
        
    def toggle_theme(self):
        self.current_theme = "light" if self.current_theme == "dark" else "dark"
        self.apply_theme()
        
        
    def select_bin_30(self):
        self.binning_selected = True
        self.selected_bin_time = 30
        self.feed_box.append("Selected bin time: 30")
        self.recalc_cw()
        
        

    def select_bin_60(self):
        self.binning_selected = True
        self.selected_bin_time = 60
        self.feed_box.append("Selected bin time: 60")
        self.recalc_cw()

    def select_bin_120(self):
        self.selected_bin_time = 120
        self.binning_selected = True
        self.feed_box.append("Selected bin time: 120")
        self.recalc_cw()

    def select_bin_180(self):
        self.selected_bin_time = 180
        self.binning_selected = True
        self.feed_box.append("Selected bin time: 180")
        self.recalc_cw()

    def auto_select_bin(self, file_name):
        """Pick the available bin time closest to total_time / 200."""
        available = [30, 60, 120, 180, 240, 600]
        btn_map = {
            30: self.bin_button_30, 60: self.bin_button_60,
            120: self.bin_button_120, 180: self.bin_button_180,
            240: self.bin_240s_btn, 600: self.bin_600s_btn,
        }
        try:
            timestamps = np.genfromtxt(file_name, comments='#', delimiter='\t',
                                       usecols=1, invalid_raise=False)
            timestamps = timestamps[~np.isnan(timestamps)]
            total_time = float(np.max(timestamps) - np.min(timestamps))
        except Exception:
            return  # leave current selection unchanged
        ideal = total_time / 200.0
        best = min(available, key=lambda b: abs(b - ideal))
        self.selected_bin_time = best
        self.binning_selected = True
        btn_map[best].setChecked(True)

    def load_file(self):
        options = QFileDialog.Options()
        default_dir = os.path.join(os.getcwd(), "ExampleData")
        if not os.path.isdir(default_dir):
            default_dir = os.getcwd()
        file_name, _ = QFileDialog.getOpenFileName(self, "Load File", default_dir, "Text Files (*.txt);;All Files (*)", options=options)
        if file_name:
            self.auto_select_bin(file_name)
            adc_min = self.adc_slider.value()
            adc_max = self.adc_max_slider.value()
            try:
                self.cw = CWClass(file_name, self.selected_bin_time, self.feed_box, adc_min=adc_min, adc_max=adc_max)
            except Exception as e:
                self.feed_box.append(f"Failed to load file: {e}")
                return
            self.cw.file_path = file_name
            self.current_plot_func = self.run_rate
            self.run_rate()
            self._emit_status()

    def select_bin_240(self):
        self.binning_selected = True
        self.selected_bin_time = 240
        self.feed_box.append("Selected bin time: 240")
        self.recalc_cw()

    def select_bin_600(self):
        self.binning_selected = True
        self.selected_bin_time = 600
        self.feed_box.append("Selected bin time: 600")
        self.recalc_cw()

    def select_bin_custom(self):
        text = self.custom_bin_input.text().strip()
        try:
            value = float(text)
            if value <= 0:
                raise ValueError
        except ValueError:
            self.feed_box.append(f"Invalid bin time: '{text}'. Enter a positive number.")
            return
        # Deselect all radio buttons
        for btn in [self.bin_button_30, self.bin_button_60, self.bin_button_120,
                    self.bin_button_180, self.bin_240s_btn, self.bin_600s_btn]:
            btn.setChecked(False)
        self.binning_selected = True
        self.selected_bin_time = value
        self.feed_box.append(f"Selected bin time: {value}s")
        self.recalc_cw()

    _CORR_SELECTED_STYLE = """
        QPushButton { background-color: #003333; color: #00ffff; border: 1px solid #00ffff; border-radius: 6px; padding: 4px; }
        QPushButton:hover { background-color: #004444; color: #00ffff; border: 1px solid #00ffff; border-radius: 6px; padding: 4px; }
        QPushButton:pressed { background-color: #002222; color: #00ffff; border: 1px solid #00ffff; border-radius: 6px; padding: 4px; }
    """

    def _plot_or_correlate(self, label, data_fn, plot_fn, kind='binned', btn=None):
        """Normal click → plot (highlights btn). Cmd+click → pick variable for correlation."""
        if label is not None and QApplication.keyboardModifiers() & Qt.ControlModifier:
            self._corr_select(label, data_fn, kind, btn)
        else:
            self._active_plot_btn = btn
            plot_fn()

    def _corr_select(self, label, data_fn, kind, btn=None):
        """Track up to two Cmd+clicked variables, then run correlation."""
        if not hasattr(self, '_corr_pending'):
            self._corr_pending = None
        mod_key = "Cmd" if platform.system() == "Darwin" else "Ctrl"
        if self._corr_pending is None:
            self._corr_pending = (label, data_fn, kind, btn)
            if btn:
                btn.setStyleSheet(self._CORR_SELECTED_STYLE)
            self.feed_box.append(f"Correlation: '{label}' selected. {mod_key}+click a second variable.")
        elif self._corr_pending[0] == label:
            self._corr_pending = None
            self.feed_box.append("Correlation: selection cleared.")
            self.apply_theme()
        else:
            if self._corr_pending[2] != kind:
                first_label = self._corr_pending[0]
                self._corr_pending = None
                self.feed_box.append(
                    f"Correlation: can't mix '{first_label}' (time-binned) "
                    f"with '{label}' (per-event). Selection cleared.")
                self.apply_theme()
                return
            self.run_correlation(self._corr_pending, (label, data_fn, kind, btn))
            self._corr_pending = None

    def run_correlation(self, a, b):
        """Scatter plot of two variables with Pearson r. Supports multi-series data."""
        if not hasattr(self, 'cw'):
            self.feed_box.append("Load a file first.")
            return
        self.current_plot_func = lambda: self.run_correlation(a, b)

        def to_series(data, default_label):
            """Normalize to list of (array, color, label)."""
            if isinstance(data, list):
                return data
            return [(data, mycolors[7], default_label)]

        series_a = to_series(a[1](), a[0])
        series_b = to_series(b[1](), b[0])

        self.static_ax.cla()

        if len(series_a) > 1 and len(series_b) == 1:
            # e.g. Rate (3 series) vs Temperature (1 series)
            xb_arr = np.asarray(series_b[0][0], dtype=float)
            for xa_arr, color, lbl in series_a:
                xa_arr = np.asarray(xa_arr, dtype=float)
                n = min(len(xa_arr), len(xb_arr))
                xa_t, xb_t = xa_arr[:n], xb_arr[:n]
                mask = np.isfinite(xa_t) & np.isfinite(xb_t)
                if mask.sum() < 2:
                    continue
                r = np.corrcoef(xa_t[mask], xb_t[mask])[0, 1]
                self.static_ax.scatter(xb_t[mask], xa_t[mask], s=10, alpha=0.7,
                                       color=color, label=f'{lbl}  r={r:.3f}')
            self.static_ax.set_xlabel(b[0])
            self.static_ax.set_ylabel(a[0])
        elif len(series_b) > 1 and len(series_a) == 1:
            xa_arr = np.asarray(series_a[0][0], dtype=float)
            for xb_arr, color, lbl in series_b:
                xb_arr = np.asarray(xb_arr, dtype=float)
                n = min(len(xa_arr), len(xb_arr))
                xa_t, xb_t = xa_arr[:n], xb_arr[:n]
                mask = np.isfinite(xa_t) & np.isfinite(xb_t)
                if mask.sum() < 2:
                    continue
                r = np.corrcoef(xa_t[mask], xb_t[mask])[0, 1]
                self.static_ax.scatter(xa_t[mask], xb_t[mask], s=10, alpha=0.7,
                                       color=color, label=f'{lbl}  r={r:.3f}')
            self.static_ax.set_xlabel(a[0])
            self.static_ax.set_ylabel(b[0])
        else:
            # Both single series
            xa_arr = np.asarray(series_a[0][0], dtype=float)
            xb_arr = np.asarray(series_b[0][0], dtype=float)
            n = min(len(xa_arr), len(xb_arr))
            xa_t, xb_t = xa_arr[:n], xb_arr[:n]
            mask = np.isfinite(xa_t) & np.isfinite(xb_t) & (xa_t > 0) & (xb_t > 0)
            if mask.sum() < 2:
                self.feed_box.append("Correlation: not enough data.")
                return
            r = np.corrcoef(xa_t[mask], xb_t[mask])[0, 1]
            color_a, color_b = series_a[0][1], series_b[0][1]
            color = color_a if color_a != mycolors[7] else color_b
            self.static_ax.scatter(xa_t[mask], xb_t[mask], s=10, alpha=0.7,
                                   color=color, label=f'r={r:.3f}')
            self.static_ax.set_xlabel(a[0])
            self.static_ax.set_ylabel(b[0])

        self.static_ax.set_title(f"{a[0]} vs {b[0]}")
        self.static_ax.legend(fontsize=10, frameon=False)
        self.static_canvas.figure.tight_layout()
        self.static_canvas.draw()
        self.apply_theme()
        self.feed_box.append(f"Correlation: {a[0]} vs {b[0]}")

    def save_serial(self):
        """Save the contents of the serial readout to a text file."""
        # Suggest a filename but open dialog in user's Documents folder
        if hasattr(self, 'data_file') and not self.data_file.closed:
            suggested_name = os.path.basename(self.data_file.name)
        else:
            suggested_name = f"CW_data_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
        suggested = os.path.join(
            os.path.expanduser("~/Documents"),
            suggested_name
        )
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Serial Log", suggested,
            "Text Files (*.txt);;All Files (*)"
        )
        if path:
            with open(path, 'w') as f:
                f.write(self.feed_box.toPlainText())
            self.feed_box.append(f"Serial log saved to: {path}")

    class NPlot():
        def __init__(self, 
                    data,
                    weights,
                    colors,
                    labels,
                    xmin,xmax,ymin,ymax,
                    ax,
                    figsize = [8,6],fontsize = 11,nbins = 101, alpha = 0.85,fit_gaussian=False,
                    xscale = 'log',yscale = 'log',xlabel = '',loc = 1,pdf_name='',lw=2, title=''):
            #
            # fg = "#ffffff" if self.parent.current_theme == "dark" else "#000000"
            self.static_ax = ax
            if xscale == 'log':
                xmin = max(xmin, 1e-10)
                bins = np.logspace(np.log10(xmin),np.log10(xmax),nbins)
            if xscale == 'linear':
                bins = np.linspace(xmin,xmax,nbins)
            self.static_ax.clear()
            #self.static_ax.set_xlabel(xlabel, size=fontsize, color=fg)
            
            self.static_ax.set_axisbelow(True)
            self.static_ax.grid(which='both', linestyle='--', alpha=0.5, zorder=0)
            #self.static_ax.set_title(title, fontsize=fontsize + 1, color = "white")
            #self.static_ax.title.set_color("white")

            hist_data = []
            std = []
            bin_centers = []

            # Define the Gaussian function
            def gaussian(x, a, mu, sigma):
                return a * np.exp(-(x - mu)**2 / (2 * sigma**2))
            
            for i in range(len(data)):
                valid_data = data[i][~np.isnan(data[i])]
                valid_weights = weights[i][~np.isnan(weights[i])]

                counts, bin_edges = np.histogram(valid_data, bins=bins, weights=valid_weights)
                bin_center = 0.5 * (bin_edges[1:] + bin_edges[:-1])

                sum_weights_sqrd, _ = np.histogram(valid_data, bins=bins, weights=np.power(valid_weights, 2))

                hist_data.append(counts)
                upper_value = plusSTD(counts,sum_weights_sqrd)
                lower_value = subSTD(counts,sum_weights_sqrd)
                std.append([upper_value,lower_value])
                bin_centers.append(bin_center)
                fill_between_steps(bin_center, upper_value,lower_value,  color = colors[i],alpha = alpha,lw=lw,ax=self.static_ax)
                self.static_ax.plot([1e14,1e14], label = labels[i],color = colors[i],alpha = alpha,linewidth = 2)
                    

            self.static_ax.set_yscale(yscale)
            self.static_ax.set_xscale(xscale)
            self.static_ax.legend(fontsize=10, loc='upper right', fancybox=False, frameon=False)
            legend = self.static_ax.legend(fontsize=10, loc='upper right', fancybox=False, frameon=False)

            #for text in legend.get_texts():
            #    text.set_color('white')
            self.static_ax.set_ylabel(r'Rate/bin [s$^{-1}$]', size=10)
            self.static_ax.set_xlabel(xlabel, size=10)
            self.static_ax.set_xlim(xmin, xmax)
            self.static_ax.set_ylim(ymin, ymax)


            self.static_ax.tick_params(axis='both', which='major', labelsize=10)
            self.static_ax.tick_params(axis='both', which='minor', labelsize=10) 


            

            self.static_ax.figure.tight_layout()
            self.static_ax.figure.canvas.draw()   
            
    class ratePlot():
        def __init__(self,
                    time,
                    count_rates,
                    count_rates_err,
                    colors,
                    labels,
                    xmin,xmax,ymin,ymax,
                    ax,
                    figsize = [8,8],fontsize = 12, alpha = 0.9,
                    xscale = 'linear',yscale = 'linear',
                    xlabel = '',ylabel = '',
                    loc = 2, pdf_name='',title = ''):
            
            #fg = "#ffffff" if self.parent.current_theme == "dark" else "#000000"
            self.static_ax = ax
            self.static_ax.set_axisbelow(True)
            self.static_ax.clear()
            self.static_ax.grid(which='both', linestyle='--', alpha=0.5, zorder=0)
            
            for i in range(len(count_rates)):
                self.static_ax.errorbar(time[i], 
                            count_rates[i],
                            xerr=0, yerr=count_rates_err[i],
                            fmt='o',label = labels[i], linewidth = 2, ecolor = colors[i], markersize = 2)             # theme-dependent)

            self.static_ax.set_yscale(yscale)
            self.static_ax.set_xscale(xscale)
            self.static_ax.set_ylabel(ylabel, size=10, color='white')
            self.static_ax.set_xlabel(xlabel, size=10, color='white')
            # Guard against xmin == xmax (e.g. only one bin of data so far), which would
            # otherwise trigger a "singular transform" warning from matplotlib.
            if xmin == xmax:
                pad = 0.5 if xmin == 0 else abs(xmin) * 0.05
                xmin, xmax = xmin - pad, xmax + pad
            if ymin == ymax:
                pad = 0.5 if ymin == 0 else abs(ymin) * 0.05
                ymin, ymax = ymin - pad, ymax + pad
            self.static_ax.set_xlim(xmin, xmax)
            self.static_ax.set_ylim(ymin, ymax)

            self.static_ax.tick_params(axis='both', which='major', labelsize=10)
            self.static_ax.tick_params(axis='both', which='minor', labelsize=10)
            self.static_ax.xaxis.labelpad = 0

            self.static_ax.legend(fontsize=10, loc='upper right', fancybox=False, frameon=False)
            legend = self.static_ax.legend(fontsize=10, loc='upper right', fancybox=False, frameon=False)

            for text in legend.get_texts():
                text.set_color('white')

            self.static_ax.figure.tight_layout()
            self.static_ax.figure.canvas.draw()   
            

    def run_adc(self):
        if not hasattr(self, 'cw'): self.feed_box.append("Load a file first."); return
        self.current_plot_func = self.run_adc
        self.toolbar.plot_type = "ADC"
        self._adc_cut_line = None
        self._adc_max_cut_line = None
        f1 = self.cw
        xmin = max(np.min(f1.adc), 1)
        xmax = np.max(f1.adc)
        bins = np.logspace(np.log10(xmin), np.log10(xmax), 101)
        counts, _ = np.histogram(f1.adc, bins=bins, weights=f1.weights)
        nonzero = counts[counts > 0]
        ymax = np.max(counts) * 20  # headroom for legend on log scale
        ymin = np.min(nonzero) / 5 if len(nonzero) > 0 else 1e-4

        c = self.NPlot(
        data=[f1.adc, f1.adc[~f1.select_coincident], f1.adc[f1.select_coincident]],
        weights=[f1.weights, f1.weights[~f1.select_coincident], f1.weights[f1.select_coincident]],
        colors=[mycolors[7], mycolors[3], mycolors[1]],
        labels=[r'All Events:  ' + str(f1.count_rate) + '+/-' + str(f1.count_rate_err) +' Hz',
                r'Non-Coincident:  ' + str(f1.count_rate_non_coincident) + '+/-' + str(f1.count_rate_err_non_coincident) +' Hz',
                r'Coincident: ' + str(f1.count_rate_coincident) + '+/-' + str(f1.count_rate_err_coincident) +' Hz'],
        ax=self.static_ax,
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, nbins=101,
        xlabel='ADC value [0-4095]',
        pdf_name='_ADC.pdf', title='ADC Measurement')
        self.apply_theme()

    def run_temperature(self):
        if not hasattr(self, 'cw'): self.feed_box.append("Load a file first."); return
        self.current_plot_func = self.run_temperature
        self.toolbar.plot_type = "temperature"
        f1=self.cw
        y_min = np.min(f1.binned_temperature)
        y_max = np.max(f1.binned_temperature)
        span = y_max - y_min
        padding = 0.1 * span   # 5% of the span


        c = self.ratePlot(time = [f1.binned_time_m,],
        count_rates = [f1.binned_temperature],
        count_rates_err = [np.ones(len(f1.binned_temperature))*0.1], # Uncertainty on pressure is 0.1C
        colors =[mycolors[5]],
        ax = self.static_ax,
        xmin = min(f1.binned_time_m),xmax = max(f1.binned_time_m),ymin = y_min - padding,ymax = y_max + padding,
        figsize = [7,5],
        fontsize = 12,alpha = 1,labels=['Temperature Data'],
        xscale = 'linear',yscale = 'linear',xlabel = 'Time [min]',ylabel = r'Temperature [$^{\circ}$C]',
        loc = 4,pdf_name='_temperature.pdf',title = 'Temperature Measurement')
        self.apply_theme()
    
    def run_gyro(self):
        if not hasattr(self, 'cw'): self.feed_box.append("Load a file first."); return
        self.current_plot_func = self.run_gyro
        self.toolbar.plot_type = "gyro"
        f1 = self.cw
        all_y = np.concatenate([
            f1.binned_gyro_x,
            f1.binned_gyro_y,
            f1.binned_gyro_z
            ])

        err = 1.0  # 1 deg/s error bars
        upper = np.max(all_y) + err
        lower = np.min(all_y) - err
        span = upper - lower
        padding = 0.2 * span

        ymin = lower - padding
        ymax = upper + padding

        c = self.ratePlot(
            time=[f1.binned_time_m, f1.binned_time_m, f1.binned_time_m],
            count_rates=[f1.binned_gyro_x, f1.binned_gyro_y, f1.binned_gyro_z],
            count_rates_err=[
                np.ones(len(f1.binned_gyro_x)) * 1,
                np.ones(len(f1.binned_gyro_y)) * 1,
                np.ones(len(f1.binned_gyro_z)) * 1,
            ],
            colors=[mycolors[2], mycolors[3], mycolors[4]],
            labels=[r'Angular velocity X', r'Angular velocity Y', r'Angular velocity Z'],
            ax=self.static_ax,
            xmin=min(f1.binned_time_m),
            xmax=max(f1.binned_time_m),
            ymin=ymin,
            ymax=ymax,
            figsize=[7, 5],
            fontsize=12,
            alpha=1,
            xscale='linear',
            yscale='linear',
            xlabel='Time [min]',
            ylabel=r'Angular velocity [deg/s]',  # ← your original 'ylabel' string was cut off
        )
        self.apply_theme()

    def run_deadtime(self):
        if not hasattr(self, 'cw'): self.feed_box.append("Load a file first."); return
        self.current_plot_func = self.run_deadtime
        self.toolbar.plot_type = "deadtime"
        f1 = self.cw
        err = 0.0  # no uncertainty on deadtime fraction
        upper = np.max(f1.binned_deadtime_pct) + err
        lower = np.min(f1.binned_deadtime_pct) - err
        span = upper - lower
        padding = max(0.15 * span, 1.0)  # at least 1% headroom

        c = self.ratePlot(
            time=[f1.binned_time_m],
            count_rates=[f1.binned_deadtime_pct],
            count_rates_err=[np.zeros(len(f1.binned_deadtime_pct))],
            colors=[mycolors[9]],
            labels=[r'Deadtime fraction'],
            ax=self.static_ax,
            xmin=min(f1.binned_time_m),
            xmax=max(f1.binned_time_m),
            ymin=max(0, lower - padding),
            ymax=upper + padding,
            figsize=[7, 5],
            fontsize=12,
            alpha=1,
            xscale='linear',
            yscale='linear',
            xlabel='Time [min]',
            ylabel=r'Deadtime / bin width [%]',
        )
        self.apply_theme()

    def run_acc(self):
        if not hasattr(self, 'cw'): self.feed_box.append("Load a file first."); return
        self.current_plot_func = self.run_acc
        self.toolbar.plot_type = "linear_acceleration"
        f1 = self.cw
        all_y = np.concatenate([
            f1.binned_accel_x,
            f1.binned_accel_y,
            f1.binned_accel_z
            ])

        err = 0.1  # 0.1 g error bars
        upper = np.max(all_y) + err
        lower = np.min(all_y) - err
        span = upper - lower
        padding = 0.2 * span

        ymin = lower - padding
        ymax = upper + padding
        
        #ymin = np.min(all_y) - 0.3
        #ymax = np.max(all_y) + 0.3
        c = self.ratePlot(
            time=[f1.binned_time_m, f1.binned_time_m, f1.binned_time_m],
            count_rates=[f1.binned_accel_x, f1.binned_accel_y, f1.binned_accel_z],
            count_rates_err=[
                np.ones(len(f1.binned_accel_x)) * 0.1,
                np.ones(len(f1.binned_accel_y)) * 0.1,
                np.ones(len(f1.binned_accel_z)) * 0.1,
            ],
            colors=[mycolors[2], mycolors[3], mycolors[4]],
            labels=[r'Acceleration X', r'Acceleration Y', r'Acceleration Z'],
            ax=self.static_ax,
            xmin=min(f1.binned_time_m),
            xmax=max(f1.binned_time_m),
            ymin=ymin,
            ymax=ymax,
            figsize=[7, 5],
            fontsize=12,
            alpha=1,
            xscale='linear',
            yscale='linear',
            xlabel='Time [min]',
            ylabel=r'Acceleration [g]',  # ← your original 'ylabel' string was cut off
        )
        self.apply_theme()
    
    def run_pressure(self):
        if not hasattr(self, 'cw'): self.feed_box.append("Load a file first."); return
        self.current_plot_func = self.run_pressure
        self.toolbar.plot_type = "pressure"
        f1 = self.cw
        y_min = np.min(f1.binned_pressure)
        y_max = np.max(f1.binned_pressure)
        span = y_max - y_min
        #padding = 0.05 * span   # 5% of the span
        padding = max(0.05 * span, 1000)   # 5% of span OR 1000

        c = self.ratePlot(time = [f1.binned_time_m,],
        count_rates = [f1.binned_pressure],
        count_rates_err = [np.ones(len(f1.binned_pressure)) * 500], # Uncertainty on pressure is 100 Pa
        colors =[mycolors[6]],
        xmin = min(f1.binned_time_m),xmax = max(f1.binned_time_m),ymin =y_min - padding,ymax =y_max + padding,
        figsize = [7,5],labels=['Pressure Data'],
        fontsize = 12,alpha = 1,
        ax = self.static_ax,
        xscale = 'linear',yscale = 'linear',xlabel = 'Time [min]',ylabel = r'Pressure [Pa]',
        loc = 4,pdf_name='_pressure.pdf',title = 'Pressure Measurement')
        self.apply_theme()

    def run_voltage(self):
        if not hasattr(self, 'cw'): self.feed_box.append("Load a file first."); return
        self.current_plot_func = self.run_voltage
        self.toolbar.plot_type = "SiPM_pulse_height"
        f1 = self.cw
        xmin = max(np.min(f1.sipm), 1e-10) * 0.95
        xmax = np.max(f1.sipm) * 1.05
        bins = np.logspace(np.log10(xmin), np.log10(xmax), 101)
        counts, _ = np.histogram(f1.sipm, bins=bins, weights=f1.weights)
        nonzero = counts[counts > 0]
        ymax = np.max(counts) * 20  # headroom for legend on log scale
        ymin = np.min(nonzero) / 5 if len(nonzero) > 0 else 1e-4

        c = self.NPlot(
        data=[f1.sipm, f1.sipm[~f1.select_coincident], f1.sipm[f1.select_coincident]],
        weights=[f1.weights, f1.weights[~f1.select_coincident], f1.weights[f1.select_coincident]],
        colors=[mycolors[7], mycolors[3], mycolors[1]],
        labels=[r'All Events:  ' + str(f1.count_rate) + '+/-' + str(f1.count_rate_err) +' Hz',
                r'Non-Coincident:  ' + str(f1.count_rate_non_coincident) + '+/-' + str(f1.count_rate_err_non_coincident) +' Hz',
                r'Coincident: ' + str(f1.count_rate_coincident) + '+/-' + str(f1.count_rate_err_coincident) +' Hz'],
        ax=self.static_ax,
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, xscale='log',
        xlabel='SiPM [mV]', fit_gaussian=True,
        pdf_name='_SiPM_peak_voltage.pdf', title='SiPM Peak Voltage Measurement')
        self.apply_theme()

    def run_rate(self):
        if not hasattr(self, 'cw'): self.feed_box.append("Load a file first."); return
        self.current_plot_func = self.run_rate
        self.toolbar.plot_type = "rate"
        f1 = self.cw
        # Derive the axis range from the same (already zero-shifted) array that gets plotted,
        # rather than the raw device timestamps, so the limits always match the data.
        run_duration_min = max(f1.binned_time_m)
        if run_duration_min > 1440:#2880:  # > 2 days — switch to days
            time_scale = 1440.0
            xlabel_rate = 'Time [days]'
        else:
            time_scale = 1.0
            xlabel_rate = 'Time [min]'
        binned_time_scaled = f1.binned_time_m / time_scale
        xmin_rate = min(f1.binned_time_m) / time_scale
        xmax_rate = run_duration_min / time_scale
        c = self.ratePlot(time = [binned_time_scaled, binned_time_scaled, binned_time_scaled],
        count_rates = [f1.binned_count_rate,f1.binned_count_rate_non_coincident,f1.binned_count_rate_coincident],
        count_rates_err = [f1.binned_count_rate_err,f1.binned_count_rate_err_non_coincident,f1.binned_count_rate_err_coincident],
        colors=[mycolors[7], mycolors[3], mycolors[1]],
        labels=[r'All Events: ' + str(f1.count_rate) + '+/-' + str(f1.count_rate_err) +' Hz',
                r'Non-Coincident:  ' + str(f1.count_rate_non_coincident) + '+/-' + str(f1.count_rate_err_non_coincident) +' Hz',
                r'Coincident:  ' + str(f1.count_rate_coincident) + '+/-' + str(f1.count_rate_err_coincident) +' Hz'],
        ax = self.static_ax,
        xmin=xmin_rate, xmax=xmax_rate, ymin=0, ymax=1.3*max(f1.binned_count_rate),
        figsize=[7,5],
        fontsize=12, alpha=1,
        xscale='linear', yscale='linear', xlabel=xlabel_rate, ylabel=r'Rate [s$^{-1}$]',
        loc=1, pdf_name='_rate.pdf', title='Detector Count Rate')
        #print(f1.binned_count_rate_err_non_coincident)
        self.apply_theme()

    def run_rate_dist(self):
        if not hasattr(self, 'cw'): self.feed_box.append("Load a file first."); return
        self.current_plot_func = self.run_rate_dist
        self.toolbar.plot_type = "rate_distribution"
        f1 = self.cw

        self.static_ax.clear()
        self.static_ax.set_axisbelow(True)
        self.static_ax.grid(which='both', linestyle='--', alpha=0.5, zorder=0)

        datasets = [
            (f1.binned_count_rate,               mycolors[7], 'All Events'),
            (f1.binned_count_rate_non_coincident, mycolors[3], 'Non-Coincident'),
            (f1.binned_count_rate_coincident,     mycolors[1], 'Coincident'),
        ]

        # Shared bins: global range across all three datasets
        all_rates = np.concatenate([
            d[np.isfinite(d) & (d > 0)]
            for d, _, _ in datasets
            if len(d[np.isfinite(d) & (d > 0)]) > 0
        ])
        if len(all_rates) == 0:
            return

        # Rates are counts/livetime — the underlying counts are integers, so
        # the rate values are quantized in steps of 1/bin_size.  Place bin
        # edges at (n + 0.5)/bin_size so each bar captures exactly one integer
        # count level, eliminating the "binning of a binning" artefact.
        step = 1.0 / f1.bin_size
        count_min = int(np.floor(all_rates.min() * f1.bin_size))
        count_max = int(np.ceil(all_rates.max() * f1.bin_size))
        shared_edges = (np.arange(count_min, count_max + 2) - 0.5) * step
        bin_width = step

        for rates, color, label in datasets:
            rates = rates[np.isfinite(rates) & (rates > 0)]
            if len(rates) == 0:
                continue

            counts, _ = np.histogram(rates, bins=shared_edges)
            bin_centers = 0.5 * (shared_edges[1:] + shared_edges[:-1])
            self.static_ax.bar(bin_centers, counts, width=bin_width * 0.9,
                               color=color, alpha=0.4, zorder=2)

            # Poisson fit: λ = mean rate * bin_size
            lam = np.mean(rates) * f1.bin_size
            if lam <= 0 or max(counts) == 0:
                continue
            sig = np.sqrt(lam)
            # Dense continuous x for smooth curve
            k_dense = np.linspace(max(0, lam - 4 * sig), lam + 4 * sig, 500)
            rate_vals = k_dense / f1.bin_size
            from scipy.stats import norm
            pmf_vals = norm.pdf(k_dense, lam, sig)

            # Anchor amplitude to observed peak so fit always aligns with data
            amplitude = max(counts) / max(pmf_vals) if max(pmf_vals) > 0 else 1
            expected = amplitude * pmf_vals

            # Chi-squared / dof
            exp_per_bin = amplitude * np.array([
                poisson.pmf(int(round(c * f1.bin_size)), lam) for c in bin_centers
            ])
            mask = exp_per_bin > 0
            chi2 = np.sum((counts[mask] - exp_per_bin[mask])**2 / exp_per_bin[mask])
            dof = max(1, mask.sum() - 1)

            self.static_ax.plot(rate_vals, expected, color=color, linewidth=2,
                                alpha=0.9, label=f'{label}  ($\\chi^2$/dof={chi2/dof:.3g})', zorder=3)

        self.static_ax.set_xlabel(r'Rate [s$^{-1}$]', size=11)
        self.static_ax.set_ylabel('Counts', size=11)
        self.static_ax.set_xlim(left=0)
        self.static_ax.legend(fontsize=10, loc='upper right', fancybox=False, frameon=False)
        self.static_ax.tick_params(axis='both', which='major', labelsize=10)
        self.static_ax.figure.tight_layout()
        self.static_ax.figure.canvas.draw()
        self.apply_theme()

    def run_interevent(self):
        if not hasattr(self, 'cw'): self.feed_box.append("Load a file first."); return
        self.current_plot_func = self.run_interevent
        self.toolbar.plot_type = "interevent"
        f1 = self.cw

        self.static_ax.clear()
        self.static_ax.set_axisbelow(True)
        self.static_ax.grid(which='both', linestyle='--', alpha=0.5, zorder=0)

        ts_all  = f1.PICO_timestamp_s
        ts_non  = f1.PICO_timestamp_s[~f1.select_coincident]
        ts_coin = f1.PICO_timestamp_s[f1.select_coincident]

        datasets = [
            (ts_all,  mycolors[7], 'All Events'),
            (ts_non,  mycolors[3], 'Non-Coincident'),
            (ts_coin, mycolors[1], 'Coincident'),
        ]

        # Shared linear bins covering the full range across all three datasets
        t_max = 0
        for ts, _, _ in datasets:
            dt = np.diff(np.sort(ts))
            dt = dt[dt > 0]
            if len(dt) > 0:
                t_max = max(t_max, dt.max())
        if t_max == 0:
            self.feed_box.append("No inter-event data available.")
            return
        t_min = 0.01
        n_bins = 101
        shared_bins = np.logspace(np.log10(t_min), np.log10(t_max), n_bins + 1)

        max_counts = 0
        for ts, color, label in datasets:
            dt = np.diff(np.sort(ts))
            dt = dt[dt > 0]
            if len(dt) < 2:
                continue

            counts, bin_edges = np.histogram(dt, bins=shared_bins)
            if max(counts) > max_counts:
                max_counts = max(counts)
            bin_widths = np.diff(bin_edges)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

            self.static_ax.bar(bin_centers, counts, width=bin_widths * 0.9,
                               color=color, alpha=0.4, zorder=2)

            # Exponential fit: MLE λ = 1 / mean(dt)
            lam = 1.0 / np.mean(dt)
            t_fit = np.logspace(np.log10(t_min), np.log10(t_max), 300)
            expected_fit = len(dt) * lam * np.exp(-lam * t_fit) * t_fit * np.log(10) * (np.log10(t_max) - np.log10(t_min)) / n_bins

            # Chi-squared / dof
            exp_per_bin = len(dt) * lam * np.exp(-lam * bin_centers) * bin_widths
            mask = exp_per_bin > 0
            chi2 = np.sum((counts[mask] - exp_per_bin[mask])**2 / exp_per_bin[mask])
            dof = max(1, mask.sum() - 1)

            self.static_ax.plot(t_fit, expected_fit, color=color, linewidth=2,
                                alpha=0.9, zorder=3,
                                label=f'{label}  ($\\chi^2$/dof={chi2/dof:.3g})')

        self.static_ax.set_xscale('log')
        self.static_ax.set_yscale('log')
        self.static_ax.set_xlim(t_min, t_max * 1.3)
        self.static_ax.set_ylim(0.5, 3.0 * max_counts)
        self.static_ax.set_xlabel(r'$\Delta t$ [s]', size=11)
        self.static_ax.set_ylabel('Counts', size=11)
        self.static_ax.legend(fontsize=10, loc='upper right', fancybox=False, frameon=False)
        self.static_ax.tick_params(axis='both', which='major', labelsize=10)
        self.static_ax.figure.tight_layout()
        self.static_ax.figure.canvas.draw()
        self.apply_theme()

    def stop_file(self):
        if hasattr(self, 'live_plot_timer'):
            self.live_plot_timer.stop()
        self.serial_connection.close()
        self.feed_box.append("Serial connection closed.")
        self.data_file.close()
        self.feed_box.append("Data file closed.")
        self.read_serial_active = False
        self.feed_box.append("Recording stopped and file closed.")

    def refresh_live_plot(self, file_name):
        """Reload the in-progress recording file and plot it the same way load_file does."""
        if not self.read_serial_active or not os.path.exists(file_name):
            return
        try:
            self.auto_select_bin(file_name)
            adc_min = self.adc_slider.value()
            adc_max = self.adc_max_slider.value()
            self.cw = CWClass(file_name, self.selected_bin_time, self.feed_box, adc_min=adc_min, adc_max=adc_max)
            self.cw.file_path = file_name
            self.current_plot_func = self.run_rate
            self.run_rate()
            self._emit_status()
        except Exception as e:
            self.feed_box.append(f"Live plot refresh skipped: {e}")

    def record_file(self, filepath=None):
        if not hasattr(self, 'serial_connection') or not self.serial_connection.is_open:
            self.feed_box.append("Select a valid port or connect first.")
            return
        if self.serial_connection.is_open:
            self.start_timestamp_dt = datetime.now()
            self.start_timestamp_str = self.start_timestamp_dt.strftime("%Y-%m-%d_%H-%M-%S")
            if filepath is None:
                # Manual record button: ask the user where to save, defaulting to the Data folder
                filename = f"CW_data_{self.start_timestamp_str}.txt"
                suggested = os.path.join(self.default_data_dir(), filename)
                filepath, _ = QFileDialog.getSaveFileName(
                    self, "Save Data File", suggested,
                    "Text Files (*.txt);;All Files (*)"
                )
                if not filepath:
                    return  # user cancelled
            self.read_serial_active = True
            # Open the file for writing and save the file handle
            self.data_file = open(filepath, "w")
            # Log to your feed_box GUI element
            self.feed_box.append(f"Saving data to: {filepath}")
            header_lines = [
        "###########################################################################################################################################################",
        "#                                                          CosmicWatch: The Desktop Muon Detector v3X",
        "#                                                                   Questions? saxani@udel.edu",
        "# Event  Timestamp[s]  Coincident[bool]  ADC[0-4095]  SiPM[mV]  Deadtime[s]  Temp[C]  Pressure[Pa]  Accel(X:Y:Z)[g]  Gyro(X:Y:Z)[deg/sec]  Name  Time  Date",
        "###########################################################################################################################################################"
    ]      
            
            for line in header_lines:
                self.data_file.write(line + "\n")
                self.data_ready.emit(line)
                self.read_serial_active = True 
                self.time_stamp = 0
            def read_serial_data():
                self.first_event_time = None
                self.last_screen_update_time = time.time()  # Track last screen update time
                self.coincidence_counter = 0
                deadtime = 0
                livetime = 0 
                self.events = 0
                self.rate = 0
                self.rate_error = 0
                self.coincidence_rate = 0
                self.coincidence_rate_error = 0
                temp_sum = 0.0
                temp_count = 0
                pressure_sum = 0.0
                pressure_count = 0

                while self.read_serial_active:
                  try:
                    if self.serial_connection.inWaiting():
                        raw = self.serial_connection.readline().decode(errors='replace').replace('\r\n','')
                        data = raw.split("\t")

                        # Skip malformed lines (need at least event + timestamp as valid floats)
                        try:
                            float(data[0]); float(data[1])
                        except (ValueError, IndexError):
                            continue

                        if len(data) > 2 and data[2].strip() == '1':
                            self.coincidence_counter += 1

                        ti = str(datetime.now()).split(" ")
                        comp_time = ti[-1]
                        data.append(comp_time)
                        comp_date = ti[0].split('-')
                        data.append(comp_date[2] + '/' + comp_date[1] + '/' + comp_date[0])

                        # Find deadtime column: first plain float after SiPM (col 4),
                        # skipping accel/gyro (colon-separated) and date (slash-separated)
                        deadtime_col = 5  # fallback
                        for _i in range(5, len(data) - 2):
                            _v = data[_i].strip()
                            if ':' not in _v and '/' not in _v:
                                try:
                                    float(_v)
                                    deadtime_col = _i
                                    break
                                except ValueError:
                                    pass

                        # Write the complete row in one call so a stats-parsing error below
                        # can never leave a truncated/partial line in the file.
                        self.data_file.write('\t'.join(data) + '\n')

                        try:
                            self.detector_name = data[-3]
                            if len(data) > 7:
                                try:
                                    temp_sum += float(data[6])
                                    temp_count += 1
                                    pressure_sum += float(data[7])
                                    pressure_count += 1
                                except ValueError:
                                    pass
                            if self.first_event_time is None:
                               self.first_event_time = float(data[1])
                               self.first_event = float(data[0])
                               self.first_dead_time = float(data[deadtime_col])
                            self.time_stamp = float(data[1]) - self.first_event_time
                            deadtime = float(data[deadtime_col]) - self.first_dead_time
                            livetime = (self.time_stamp) - deadtime
                            if livetime <= 0:
                                livetime = 1e-8
                            self.events = (float(data[0])-(self.first_event))
                            self.rate = self.events/livetime
                            self.rate_error = math.sqrt(self.rate) / livetime
                            self.coincidence_rate = self.coincidence_counter / livetime
                            self.coincidence_rate_error = math.sqrt(self.coincidence_counter) / livetime
                            avg_temp = temp_sum / temp_count if temp_count else 0.0
                            avg_pressure = pressure_sum / pressure_count if pressure_count else 0.0
                            self.data_ready2.emit(self._status_html(
                                self.time_stamp, deadtime, livetime,
                                self.events, self.rate, self.rate_error,
                                self.coincidence_counter, self.coincidence_rate, self.coincidence_rate_error,
                                avg_temp, avg_pressure,
                            ))
                            self.last_screen_update_time = time.time()
                        except (ValueError, IndexError):
                            pass  # skip stats update for lines that can't be fully parsed

                        #print(str(i+'\t') for i in data)
                        #print(*data, sep='\t')
                        event_number = int(data[0])
                        if event_number % 1 ==0:
                            self.data_file.flush()
    
                        col_widths = [6, 12, 3, 5, 7, 12, 6, 10, 19, 19, 7, 13, 10]
                        line_str = '  '.join(
                            str(item).strip()[:w].ljust(w)
                            for item, w in zip(data, col_widths)
                        )
                        self.data_ready.emit(line_str)
                    else:
                        if time.time() - self.last_screen_update_time >= 1.0:
                            self.time_stamp += 1.0  # increment displayed time by 1 second
                            avg_temp = temp_sum / temp_count if temp_count else 0.0
                            avg_pressure = pressure_sum / pressure_count if pressure_count else 0.0
                            self.data_ready2.emit(self._status_html(
                                self.time_stamp, deadtime, livetime,
                                self.events, self.rate, self.rate_error,
                                self.coincidence_counter, self.coincidence_rate, self.coincidence_rate_error,
                                avg_temp, avg_pressure,
                            ))
                            self.last_screen_update_time = time.time()
                  except OSError:
                      self.read_serial_active = False
                      self.data_ready.emit("Serial connection lost.")
                      break



        # Create and start the thread
        self.read_thread = threading.Thread(target=read_serial_data, daemon=True)
        self.read_thread.start()

        # Periodically reload and plot the file being recorded, same as load_file()
        self.live_plot_timer = QTimer(self)
        self.live_plot_timer.timeout.connect(lambda: self.refresh_live_plot(filepath))
        self.live_plot_timer.start(60000)  # every 1 minute
         
        
        

# If the input is just a file name (no path separators), prepend cwd



            

if __name__ == '__main__':
    print("Welcome to the CosmicWatch GUI")
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(img("Images/logo_white.png")))
    dashboard = FuturisticDashboard()
    screen = QApplication.primaryScreen().availableGeometry()
    w = int(screen.width() * 0.65)
    h = int(screen.height() * 1.00)
    x = screen.x()
    y = screen.y() + (screen.height() - h) // 2
    dashboard.setGeometry(x, y, w, h)
    dashboard.show()
    sys.exit(app.exec_())
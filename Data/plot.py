#***********************************************************************************
# Master import
#***********************************************************************************

import sys, os, time, warnings, argparse
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np


warnings.filterwarnings('ignore')


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

class CWClass():
    def __init__(self,fname, bin_size = 60):
        self.name = fname.split('/')[-1]
        self.bin_size = bin_size
        
        fileHandle = open(fname,"r" )
        lineList = fileHandle.readlines()
        fileHandle.close()
        header_lines = 0
        
        # Look through the first 1000 lines for the word "Device". Everything prior is considered part of the header.
        
        last_line_of_header=0
        for i in range(min(len(lineList),1000)):
            if "#" in lineList[i]:
                last_line_of_header = i+1
  
        #Determine number of columns by looking at the second last line in the file.
        number_of_columns = len(lineList[len(lineList)-2].split("\t"))
        column_array = range(0,number_of_columns)
        
        
        file_from_computer = False
        file_from_sdcard   = False
        
        
        if number_of_columns == 13:
            file_from_computer = True  
            print('File from Computer')
            data = np.genfromtxt(fname, dtype = str, delimiter='\t', usecols=column_array, invalid_raise=False, skip_header=header_lines)
            
            event_number = data[:,0].astype(float) #first column of data
            PICO_timestamp_s = data[:,1].astype(float)
            adc = data[:,2].astype(int)
            sipm = data[:,4].astype(float)
            pressure = data[:,5].astype(float)
            temperature = data[:,6].astype(float)
            deadtime = data[:,7].astype(float)
            coincident = data[:,8].astype(bool)
            detName = data[:,9]
            comp_time = data[:,10]
            comp_date = data[:,11]
        
        elif number_of_columns == 8:

            file_from_sdcard = True 
            print('File from MicroSD Card')
            data = np.genfromtxt(fname, dtype = str, delimiter='\t', usecols=column_array, invalid_raise=False, skip_header=header_lines)
            event_number = data[:,0].astype(float)#first column of data
            PICO_timestamp_s = data[:,1].astype(float)
            coincident = data[:,2].astype(bool)
            adc = data[:,3].astype(int)
            sipm = data[:,4].astype(float)
            deadtime = data[:,5].astype(float)
            temperature = data[:,6].astype(float)
            pressure = data[:,7].astype(float)
            
        else: 
            print('Incorrect number of collumns in file: %1u' %number_of_columns)

        # Convert the computer time to an absolute time (MJD).
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

        # Convert the cumulative deadtime to the deadtime between events
        # The detector starts at time 0, so append a zero.
        event_deadtime_s = np.diff(np.append([0],deadtime))

        # The RP Pico absolute time isn't great. Over the course of a few hours, it will be off by several seconds. 
        # The computer will give you accurate time down to about 1ms. Reading from the serial port has ~ms scale uncertainty.
        # The RP Pico can give you a precise measurement (down to 1us), but the absolute time will drift. Expect it to be off by roughly 1min per day.
        #self.PICO_time_ms      = PICO_time_ms
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

        self.event_deadtime_s   = event_deadtime_s     # an array of the measured event deadtime in seconds
        self.event_deadtime_ms  = event_deadtime_s*1000            # an array of the measured event deadtime in miliseconds
        self.total_deadtime_s   = max(deadtime) - min(deadtime)       # an array of the measured event deadtime in miliseconds
        self.total_deadtime_ms  = self.total_deadtime_s*1000. # The total deadtime in seconds
                
         
        # The time between events is well described by the PICO timestamp. 
        # The 'diff' command takes the difference between each element in the array.
        self.PICO_event_livetime_s = np.diff(np.append([0],self.PICO_timestamp_s)) - self.event_deadtime_s
        
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
            self.weights          = np.ones(len(event_number)) / self.live_time
            self.count_rate       = self.total_counts/self.live_time 
            self.count_rate_err   = np.sqrt(self.total_counts)/self.live_time 

            bins = range(0,int(max(self.time_stamp_s)), self.bin_size)
            counts, binEdges       = np.histogram(self.time_stamp_s, bins = bins)
            bin_livetime, binEdges = np.histogram(self.time_stamp_s, bins = bins, weights = self.PICO_event_livetime_s)
        
            # Bin the pressure by taking the average pressure in each bin
            sum_pressure, _ = np.histogram(self.time_stamp_s, bins=bins, weights=self.pressure)
            count_pressure, _ = np.histogram(self.time_stamp_s, bins=bins)
            self.binned_pressure = sum_pressure / np.maximum(count_pressure, 1)  # Avoid division by zero

            # Bin the temperature by taking the average temperature in each bin
            sum_temperature, _ = np.histogram(self.time_stamp_s, bins=bins, weights=self.temperature)
            count_temperature, _ = np.histogram(self.time_stamp_s, bins=bins)
            self.binned_temperature = sum_temperature / np.maximum(count_temperature, 1)  # Avoid division by zero


        elif file_from_sdcard:
            self.live_time_s        = (self.PICO_total_time_s - self.total_deadtime_s)
            self.live_time_ms        = (self.PICO_total_time_ms - self.total_deadtime_ms)/1000.

            self.weights          = np.ones(len(event_number)) / self.live_time_s
            #self.count_rate       = np.round(self.total_counts/self.live_time_s ,3)
            #self.count_rate_err   = np.round(np.sqrt(self.total_counts)/self.live_time_s ,3)
            n = 4
            print("    -- Total Count Rate: ", np.round(self.total_counts/self.live_time_s,n),"+/-",
                    np.round(np.sqrt(self.total_counts)/self.live_time_s,n),"Hz")

            self.count_rate, self.count_rate_err = round(
                    self.total_counts/self.live_time_s, 
                    np.sqrt(self.total_counts)/self.live_time_s)
            

            bins = range(int(min(self.PICO_timestamp_s)),int(max(self.PICO_timestamp_s)),self.bin_size)
            counts, binEdges = np.histogram(self.PICO_timestamp_s, bins = bins)
            bin_livetime, binEdges = np.histogram(self.PICO_timestamp_s, bins = bins, weights = self.PICO_event_livetime_s)

            self.bin_size          = bin_size
            self.binned_counts     = counts
            self.binned_counts_err = np.sqrt(counts)
            self.binned_count_rate = counts/bin_livetime
            self.binned_count_rate_err = np.sqrt(counts)/bin_livetime

            counts_coincident, binEdges      = np.histogram(self.PICO_timestamp_s[self.select_coincident], bins = bins)
            bin_deadtime, binEdges      = np.histogram(self.PICO_timestamp_s, bins = bins, weights = self.event_deadtime_s)

            self.total_coincident = len(self.PICO_timestamp_s[self.select_coincident])
            
            print("    -- Count Rate Coincident (coincident): ",np.round(self.total_coincident/self.live_time_s,n),"+/-" ,
                        np.round(np.sqrt(self.total_coincident)/self.live_time_s,n),"Hz")

            self.count_rate_coincident, self.count_rate_err_coincident = round(
                    self.total_coincident/self.live_time_s, 
                    np.sqrt(self.total_coincident)/self.live_time_s)
            
            
            
            
            self.binned_counts_coincident     = counts_coincident
            self.binned_counts_err_coincident = np.sqrt(counts_coincident)
            self.binned_count_rate_coincident = counts_coincident/(bin_size-bin_deadtime)
            self.binned_count_rate_err_coincident = np.sqrt(counts_coincident)/(bin_size-bin_deadtime)



            counts_non_coincident, binEdges      = np.histogram(self.PICO_timestamp_s[~self.select_coincident], bins = bins)
            bin_deadtime, binEdges      = np.histogram(self.PICO_timestamp_s, bins = bins, weights = self.event_deadtime_s)
            self.total_non_coincident = len(self.PICO_timestamp_s[~self.select_coincident])
            self.binned_counts_non_coincident     = counts_non_coincident
            self.binned_counts_err_non_coincident = np.sqrt(counts_non_coincident)
            self.binned_count_rate_non_coincident = counts_non_coincident/(bin_size-bin_deadtime)
            self.binned_count_rate_err_non_coincident = np.sqrt(counts_non_coincident)/(bin_size-bin_deadtime)

            print("    -- Count Rate Non-Coincident: ",np.round(self.total_non_coincident/self.live_time_s,n),"+/-",
                        np.round(np.sqrt(self.total_non_coincident)/self.live_time_s,n),"Hz")

            self.count_rate_non_coincident, self.count_rate_err_non_coincident = round(
                    self.total_non_coincident/self.live_time_s, 
                    np.sqrt(self.total_non_coincident)/self.live_time_s)

            sum_pressure, _ = np.histogram(self.PICO_timestamp_s, bins=bins, weights=self.pressure)
            count_pressure, _ = np.histogram(self.PICO_timestamp_s, bins=bins)
            self.binned_pressure = sum_pressure / np.maximum(count_pressure, 1)  # Avoid division by zero

            # Bin the temperature by taking the average temperature in each bin
            sum_temperature, _ = np.histogram(self.PICO_timestamp_s, bins=bins, weights=self.temperature)
            count_temperature, _ = np.histogram(self.PICO_timestamp_s, bins=bins)
            self.binned_temperature = sum_temperature / np.maximum(count_temperature, 1)  # Avoid division by zero

            # Coincident binned data

        else:
            print('Error')
        
        bincenters = 0.5*(binEdges[1:]+ binEdges[:-1])
        self.binned_time_s     = bincenters
        self.binned_time_m     = bincenters/60.
        self.weights           = np.ones(len(event_number)) / self.live_time_s  

 
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



class NPlot():
    def __init__(self, 
                 data,
                 weights,
                 colors,
                 labels,
                 xmin,xmax,ymin,ymax,
                 figsize = [8,6],fontsize = 15,nbins = 101, alpha = 0.85,fit_gaussian=False,
                 xscale = 'log',yscale = 'log',xlabel = '',loc = 1,pdf_name='',lw=2, title=''):

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True, 
                                       gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.1})  # `hspace=0` removes space

        if xscale == 'log':
            bins = np.logspace(np.log10(xmin),np.log10(xmax),nbins)
        if xscale == 'linear':
            bins = np.linspace(xmin,xmax,nbins)
    
        ax1.set_axisbelow(True)
        ax1.grid(which='both', linestyle='--', alpha=0.5, zorder=0)
        ax1.set_title(title, fontsize=fontsize + 1)

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
            fill_between_steps(bin_center, upper_value,lower_value,  color = colors[i],alpha = alpha,lw=lw,ax=ax1)
            ax1.plot([1e14,1e14], label = labels[i],color = colors[i],alpha = alpha,linewidth = 2)
            
            

        ax1.set_yscale(yscale)
        ax1.set_xscale(xscale)
        ax1.legend(fontsize=fontsize - 2, loc=loc, fancybox=True, frameon=True)
        ax1.set_ylabel(r'Rate/bin [s$^{-1}$]', size=fontsize)
        ax1.set_xlim(xmin, xmax)
        ax1.set_ylim(ymin, ymax)

        
        ax1.tick_params(axis='both', which='major', labelsize=fontsize-3)
        ax1.tick_params(axis='both', which='minor', labelsize=fontsize-3) 

        # --- Ratio Plot ---
        reference_hist = hist_data[0]
        for i in range(1, len(hist_data)):
            #ratio = np.divide(hist_data[i], reference_hist, out=np.zeros_like(hist_data[i]), where=reference_hist != 0)
            upper_value = np.divide(std[i][0], reference_hist, out=np.zeros_like(std[i][0]), where=reference_hist != 0)
            lower_value = np.divide(std[i][1], reference_hist, out=np.zeros_like(std[i][1]), where=reference_hist != 0)
            fill_between_steps(bin_centers[0], upper_value, lower_value, ax=ax2, color=colors[i], alpha=0.85, lw=lw)
            #ax2.plot(bin_centers[0], ratio, marker='.', linestyle='-', color=colors[i], alpha=alpha, label=f'{labels[i]} / {labels[0]}')

        ax2.set_yscale('linear')

        ax2.axhline(1.0, color='black', linestyle='--', linewidth=1)  # Reference line at 1
        ax2.set_ylabel("Ratio", size=fontsize)
        ax2.set_xlabel(xlabel, labelpad=10, size=fontsize)
        ax2.set_ylim(0., 1.)  # Adjust as needed
        ax2.grid(which='both', linestyle='--', alpha=0.5)

        ax2.tick_params(axis='both', which='major', labelsize=fontsize - 3)
        ax2.tick_params(axis='both', which='major', labelsize=fontsize - 3)

        plt.tight_layout()
        
        if pdf_name != '':
            print('Saving Figure to: '+os.getcwd() +  '/'+pdf_name)
            plt.savefig(os.getcwd() + '/Figures/'+pdf_name, format='pdf',transparent =True)
        plt.show()

class ratePlot():
    def __init__(self,
                 time,
                 count_rates,
                 count_rates_err,
                 colors,
                 labels,
                 xmin,xmax,ymin,ymax,
                 figsize = [8,8],fontsize = 16, alpha = 0.9,
                 xscale = 'linear',yscale = 'linear',
                 xlabel = '',ylabel = '',
                 loc = 2, pdf_name='',title = ''):
        
        f = plt.figure(figsize=(figsize[0], figsize[1])) 
        ax1 = f.add_subplot(111)

        ax1.set_axisbelow(True)
        ax1.grid(which='both', linestyle='--', alpha=0.5, zorder=0)
        
        for i in range(len(count_rates)):
            plt.errorbar(time[i], 
                           count_rates[i],
                           xerr=0, yerr=count_rates_err[i],
                           fmt= 'ko',label = labels[i], linewidth = 2, ecolor = colors[i], markersize = 2)

        plt.yscale(yscale)
        plt.xscale(xscale)
        plt.ylabel(ylabel,size=fontsize)
        plt.xlabel(xlabel,size=fontsize)
        plt.axis([xmin, xmax, ymin,ymax])
        
        ax1.tick_params(axis='both', which='major', labelsize=fontsize-3)
        ax1.tick_params(axis='both', which='minor', labelsize=fontsize-3) 
        ax1.xaxis.labelpad = 0 

        plt.legend(fontsize=fontsize-3,loc = loc,  fancybox = True,frameon=True)
        
        plt.title(title,fontsize=fontsize+1)
        plt.tight_layout()
        if pdf_name != '':
            print('Saving Figure to: '+os.getcwd() +  '/'+pdf_name)
            plt.savefig(os.getcwd() + '/Figures/'+pdf_name, format='pdf',transparent =True)
        plt.show()
        
def main():
    parser = argparse.ArgumentParser(description="Process CosmicWatch data.")
    parser.add_argument('-i', '--input', required=True, help="Input file name or full path")
    args = parser.parse_args()

    infile_name = args.input.split('/')[-1].split('.')[0]
    print("Plotting infile name: ", infile_name + '.txt')
    # Check if input is a full path or just a file name
    if os.path.isfile(args.input):
        file_path = args.input  # Full path provided by user
    else:
        # Assume the file is in the current working directory
        file_path = os.path.join(os.getcwd(), args.input)
        if not os.path.isfile(file_path):
            print(f"Error: File '{args.input}' not found in the current directory.")
            print(f"    -- Example Usage: >> python plot.py -i ExampleData/AxLab_000.txt")
            sys.exit(1)

    # Load the data file, set the binsize for the rate as a function of time plot.
    f1 = CWClass(file_path, bin_size = 600)

    
    # Plot the ADC values from the coincident and non-coincident events
    c = NPlot(
        data=[ f1.adc,f1.adc[~f1.select_coincident],f1.adc[f1.select_coincident]],
        weights=[f1.weights,f1.weights[~f1.select_coincident],f1.weights[f1.select_coincident]],
        colors=[mycolors[7], mycolors[3],mycolors[1]],
        labels=[r'All Events:  ' + str(f1.count_rate) + '+/-' + str(f1.count_rate_err) +' Hz',
                r'Non-Coincident:  ' + str(f1.count_rate_coincident) + '+/-' + str(f1.count_rate_err_coincident) +' Hz',
                r'Coincident: ' + str(f1.count_rate_coincident) + '+/-' + str(f1.count_rate_err_coincident) +' Hz'],
        xmin=10, xmax=4095, ymin=0.1e-3, ymax=1.9,nbins=4001,
        xlabel='Meausred ADC peak value [0-4095]',
        pdf_name=infile_name+'_ADC.pdf',title = 'ADC Measurement')
    
    # Plot the Calculated SiPM Peak voltages coincident and non-coincident events
    c = NPlot(
        data=[f1.sipm, f1.sipm[~f1.select_coincident],f1.sipm[f1.select_coincident]],
        weights=[f1.weights, f1.weights[~f1.select_coincident],f1.weights[f1.select_coincident]],
        colors=[mycolors[7], mycolors[3],mycolors[1]],
        labels=[r'All Events:  ' + str(f1.count_rate) + '+/-' + str(f1.count_rate_err) +' Hz',
                r'Non-Coincident:  ' + str(f1.count_rate_coincident) + '+/-' + str(f1.count_rate_err_coincident) +' Hz',
                r'Coincident: ' + str(f1.count_rate_coincident) + '+/-' + str(f1.count_rate_err_coincident) +' Hz'],
        xmin=5, xmax=600, ymin=0.1e-6, ymax=0.9,xscale='log',
        xlabel='SiPM Peak Voltage [mV]',fit_gaussian=True,
        pdf_name=infile_name+'_SiPM_peak_voltage.pdf',title = 'SiPM Peak Voltage Measurement',)
    
    c = NPlot(
        data=[f1.sipm, f1.sipm[~f1.select_coincident],f1.sipm[f1.select_coincident]],
        weights=[f1.weights, f1.weights[~f1.select_coincident],f1.weights[f1.select_coincident]],
        colors=[mycolors[7], mycolors[3],mycolors[1]],
        labels=[r'All Events:  ' + str(f1.count_rate) + '+/-' + str(f1.count_rate_err) +' Hz',
                r'Non-Coincident:  ' + str(f1.count_rate_coincident) + '+/-' + str(f1.count_rate_err_coincident) +' Hz',
                r'Coincident: ' + str(f1.count_rate_coincident) + '+/-' + str(f1.count_rate_err_coincident) +' Hz'],
        xmin=5, xmax=600, ymin=0.1e-6, ymax=0.9,xscale='linear',
        xlabel='SiPM Peak Voltage [mV]',
        pdf_name=infile_name+'_SiPM_peak_voltage_linear.pdf',title = 'SiPM Peak Voltage Measurement',)
    
    # Plot rate as a function of time
    c = ratePlot(time = [f1.binned_time_m,f1.binned_time_m,f1.binned_time_m],
        count_rates = [f1.binned_count_rate,f1.binned_count_rate_non_coincident,f1.binned_count_rate_coincident],
        count_rates_err = [f1.binned_count_rate_err,f1.binned_count_rate_err_non_coincident,f1.binned_count_rate_err_coincident], 
        colors=[mycolors[7], mycolors[3], mycolors[1]],
        labels=[r'All Events: ' + str(f1.count_rate) + '+/-' + str(f1.count_rate_err) +' Hz', 
                r'Non-Coincident:  ' + str(f1.count_rate_coincident) + '+/-' + str(f1.count_rate_err_coincident) +' Hz',
                r'Coincident:  ' + str(f1.count_rate_coincident) + '+/-' + str(f1.count_rate_err_coincident) +' Hz'],
        xmin = min(f1.PICO_timestamp_s/60), xmax = max(f1.PICO_timestamp_s/60),ymin = 0,ymax = 1.5*max(f1.binned_count_rate),
        figsize = [7,5],
        fontsize = 16,alpha = 1,
        xscale = 'linear',yscale = 'linear',xlabel = 'Time [min]',ylabel = r'Rate [s$^{-1}$]',
        loc = 1, pdf_name=infile_name+'_rate.pdf',title = 'Detector Count Rate')


    c = ratePlot(time = [f1.binned_time_m,],
        count_rates = [f1.binned_pressure],
        count_rates_err = [np.ones(len(f1.binned_pressure)) * 100], # Uncertainty on pressure is 100 Pa
        colors =[mycolors[6]],
        xmin = min(f1.binned_time_m),xmax = max(f1.binned_time_m),ymin = min(f1.binned_pressure) -100,ymax =max(f1.binned_pressure)+100,
        figsize = [7,5],labels=['Pressure Data'],
        fontsize = 16,alpha = 1,
        xscale = 'linear',yscale = 'linear',xlabel = 'Time [min]',ylabel = r'Pressure [Pa]',
        loc = 4,pdf_name=infile_name+'_pressure.pdf',title = 'Pressure Measurement')
    # Plot rate of coincident and non coincident events

    c = ratePlot(time = [f1.binned_time_m,],
        count_rates = [f1.binned_temperature],
        count_rates_err = [np.ones(len(f1.binned_temperature))*0.1], # Uncertainty on pressure is 0.1C
        colors =[mycolors[5]],
        xmin = min(f1.binned_time_m),xmax = max(f1.binned_time_m),ymin = min(f1.binned_temperature)-0.4,ymax = max(f1.binned_temperature)+0.4,
        figsize = [7,5],
        fontsize = 16,alpha = 1,labels=['Temperature Data'],
        xscale = 'linear',yscale = 'linear',xlabel = 'Time [min]',ylabel = r'Temperature [$^{\circ}$C]',
        loc = 4,pdf_name=infile_name+'_temperature.pdf',title = 'Temperature Measurement')

if __name__ == "__main__":
    main()
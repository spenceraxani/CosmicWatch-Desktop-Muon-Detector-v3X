================================================================================
  CosmicWatch Desktop Muon Detector v3X — GUI User Guide
  Questions? saxani@udel.edu
================================================================================


OVERVIEW
--------
This GUI lets you record, load, and analyze data from the CosmicWatch v3X
muon detector. You can visualize muon rates, energy spectra, environmental
sensor data, and more — either in real time from a connected detector or
from a previously saved data file.


--------------------------------------------------------------------------------
INSTALLATION
--------------------------------------------------------------------------------

macOS / Linux:
  bash install.sh

Windows (double-click or run from Command Prompt):
  install.bat

These scripts install all required Python packages automatically. Python 3.8+
must already be installed. Get it from https://www.python.org/downloads/


--------------------------------------------------------------------------------
GETTING STARTED
--------------------------------------------------------------------------------

1. Launch the app by running:

     python GUI.py

2. The window will open in dark mode showing:
   - A plot area on the left
   - A status panel and 3D detector rendering on the right
   - A log/readout panel at the bottom
   - A toolbar of buttons at the top


--------------------------------------------------------------------------------
TOOLBAR BUTTONS (top-left)
--------------------------------------------------------------------------------

  [Upload icon]  Load File
      Open a previously recorded .txt data file for analysis.
      The dialog defaults to the ExampleData/ folder.

  [Pause icon]   Stop Recording
      Stops an active live recording session and closes the data file.

  [Play icon]    Start Recording (Live Mode)
      Begins recording data from a connected detector.
      A save dialog will open first — choose where to save your data file.
      The dialog defaults to cwd/data/ with a timestamped filename.
      Recording will not start if you cancel the dialog.

  [Refresh icon] Refresh Ports
      Rescans available serial ports. Use this if you plug in the detector
      after the app has launched.

  [Sun/Moon button]  Toggle Light/Dark Theme


--------------------------------------------------------------------------------
CONNECTING A DETECTOR (Live Mode)
--------------------------------------------------------------------------------

1. Plug in your CosmicWatch detector via USB.
2. Click the Refresh button to scan for serial ports.
3. Select the correct port from the dropdown menu (e.g. /dev/cu.usbmodem...
   on Mac, COM3 on Windows). The detector connects automatically on selection.
4. Click the Play button to start recording. Choose a save location in the
   dialog that appears.
5. Data will stream in real time to the log panel and be written to your file.
6. Click the Stop button to end recording.


--------------------------------------------------------------------------------
ANALYZING A DATA FILE
--------------------------------------------------------------------------------

1. Click the Load File button and select a .txt data file.
   Example data files are in the ExampleData/ folder.
2. Choose a binning time (see below) and click a plot type button.
3. Use the matplotlib toolbar (below the plot) to zoom, pan, and save figures.


--------------------------------------------------------------------------------
BINNING TIME
--------------------------------------------------------------------------------

Choose how wide each time bin is when plotting the muon rate over time.
Options: 30s, 60s, 120s, 180s, 240s, 600s, or enter a custom value in the
text box next to the binning buttons.

The app will auto-select a sensible default based on the length of your run
when you load a file. You can change it at any time and click a plot button
to refresh.


--------------------------------------------------------------------------------
ADC MINIMUM FILTER
--------------------------------------------------------------------------------

The ADC Min slider (0-4095) sets a lower threshold on the SiPM pulse height.
Events below this value are excluded from all plots. Drag the slider and click
"Update" to reprocess the loaded file with the new threshold.

This is useful for rejecting low-energy noise.


--------------------------------------------------------------------------------
PLOT TYPES (bottom buttons)
--------------------------------------------------------------------------------

  Rate              Muon detection rate over time (counts/min)
  ADC               Histogram of raw ADC pulse heights (energy spectrum)
  SiPM              SiPM voltage spectrum
  Pressure          Atmospheric pressure over time
  Temperature       Temperature over time (degrees C)
  Linear Accel.     X/Y/Z accelerometer readings over time (g)
  Angular Velocity  X/Y/Z gyroscope readings over time (deg/s)
  Deadtime          Detector deadtime per event over time
  Rate Distribution Histogram of binned count rates (Poisson fit)
  Dt Distribution   Inter-event time histogram (exponential fit)

The active plot button is highlighted in cyan.


--------------------------------------------------------------------------------
CORRELATION PLOTS
--------------------------------------------------------------------------------

You can plot one variable against another to look for correlations:

  macOS:   Cmd + click two plot buttons
  Windows / Linux: Ctrl + click two plot buttons

The first selected button turns cyan. Click a second button to generate a
scatter plot of the two variables with a linear fit and Pearson r value.

Note: Rate vs ADC and Rate vs SiPM are not available as correlation pairs
since they measure different things (time-binned vs per-event).


--------------------------------------------------------------------------------
DATA FILE FORMAT
--------------------------------------------------------------------------------

Recorded files are plain text with one event per line:

  Event  Timestamp[s]  Coincident[bool]  ADC[0-4095]  SiPM[mV]  Deadtime[s]
  Temp[C]  Pressure[Pa]  Accel(X:Y:Z)[g]  Gyro(X:Y:Z)[deg/sec]  Name  Time  Date

Files are saved with a header block of comment lines starting with #.
Files from the detector's onboard SD card are also supported.


--------------------------------------------------------------------------------
FILE LOCATIONS
--------------------------------------------------------------------------------

  Live recordings    cwd/data/          (chosen at start of each recording)
  Example data       ExampleData/
  Images & 3D model  Images/            (PNG icons, logo, bare_assembly.stl)


--------------------------------------------------------------------------------
REQUIREMENTS
--------------------------------------------------------------------------------

  Python 3.8+
  PyQt5
  pyqtgraph
  numpy
  scipy
  matplotlib
  pyserial
  numpy-stl
  Pillow
  PyOpenGL
  PyOpenGL_accelerate  (optional, improves 3D rendering performance)

See install.sh (macOS/Linux) or install.bat (Windows) to install automatically.


================================================================================

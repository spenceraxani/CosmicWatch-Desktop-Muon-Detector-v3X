# CosmicWatch-Desktop-Muon-Detector-v3X

The CosmicWatch Detector v3X is a compact, self-contained particle detector optimized for detecting ionizing radiation—most notably cosmic-ray muons. It is ideally suited for education, outreach, citizen-science projects, and scientific/industry projects.

Key features include:

Penetrating Muon Detection:
v3X records secondary cosmic-ray muons produced by interactions of primary cosmic radiation in the upper atmosphere. At sea level, these muons follow an energy spectrum peaking around 4 GeV.
Scintillation-Based Sensor:
A polystyrene scintillator, doped with primary (POP) and secondary (POPOP) fluorophores, emits photons (λ ≈ 420 nm) when charged particles pass through.
Silicon Photomultiplier (SiPM):
Photons from the scintillator are converted into electrical signals by an SiPM operating in Geiger mode. Compared to traditional PMTs, the SiPM runs at low voltage, is magnetic-field-resistant, rugged, and compact. In v3X it is biased ~6 V over breakdown, yielding a gain of ~6×10⁶ and ~43 % photon-detection efficiency at 420 nm.
Multiple Experimental Modes:
Users can measure muon angular distributions, attenuation through materials (e.g., underground or underwater), characterize the electromagnetic component of showers, and investigate techniques for background reduction. Advanced experiments include determining muon velocity and lifetime or observing solar-induced variations such as Forbush decreases.
Coincidence Operation:
Linking two v3X units in coincidence (∼3 µs window) suppresses uncorrelated backgrounds, greatly enhancing muon signal fidelity.
Rich Data Logging:
Each event record includes timestamp, coincidence flag, ADC value (proportional to deposited energy and SiPM peak voltage), cumulative dead time, ambient temperature, barometric pressure, and three-axis acceleration.
Flexible Readout:
Data may be stored locally on a microSD card for standalone use or streamed via micro-USB to a computer. Provided Python scripts simplify logging and analysis.
Open-Source, Educational Design:
Full build instructions—including parts sourcing and PCB assembly—are published, making v3X an excellent hands-on tool for learning detector physics and electronics.
High Portability & Low Power:
With just 0.5 W power draw and USB power compatibility, v3X is readily deployed in diverse environments.
Overall, the CosmicWatch Detector v3X delivers a versatile, affordable, and user-friendly platform for exploring cosmic-ray physics, from classroom demonstrations to sophisticated field measurements.

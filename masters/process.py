import os.path as op
import logging
logging.getLogger("rootpy.plotting.style").setLevel(logging.WARNING)

from rootpy import ROOT

# data format
# {
#   "workflow": "workflow-abc123",
#   "particle": "q*",
#   "sim-file": "QStar/dataLikeHistograms.QStar{0}.root",
#   "sim-hist": "mjj_Scaled_QStar{0}_30fb",
#   "mass-points": [1000, 2000, ...]
#   "expected-limits": {
#     3000: [100, 90, 110, 105, 98, ...],
#   }
#   "data-limits": {
#     1000: 100,
#     2000: 50,
#   }
# }

DATA_DIR = "data"

class TheoryData:
    def __init__(self):
        self.x = []
        self.y = []

def process_theory_data(data):
    td = TheoryData()
    td.x = sorted(data["mass-points"])
    for m in td.x:
        root_file = ROOT.TFile.Open(op.join(DATA_DIR, data["sim-file"].format(m)))
        nominal = root_file.GetDirectory("Nominal")
        hist = nominal.Get(data["sim-hist"].format(m))
        td.y.append(hist.Integral()/30/1000)
    return td

class Data:
    def __init__(self):
        self.mass_points = []
        self.mean = []
        self.rms = []
        self.low_2sigma = []
        self.low_1sigma = []
        self.high_1sigma = []
        self.high_2sigma = []
    
    def add(self, mass, mean, rms, low2, low1, high1, high2):
        self.mass_points.append(mass)
        self.mean.append(mean)
        self.rms.append(rms)
        self.low_2sigma.append(low2)
        self.low_1sigma.append(low1)
        self.high_1sigma.append(high1)
        self.high_2sigma.append(high2)
    
    def sort(self):
        # sort all lists based on the mass
        self.mass_points, self.mean, self.rms, self.low_2sigma, self.low_1sigma, self.high_1sigma, self.high_2sigma = \
            zip(*sorted(zip(self.mass_points, self.mean, self.rms, self.low_2sigma, self.low_1sigma, self.high_1sigma, self.high_2sigma)))

def process(data):
    mass_points = []

    fb = float(data["fb"])
    pb = fb*1000
    
    processed_data = Data()
    
    for mass in sorted(data["expected-limits"]):
        if len(data["expected-limits"][mass]) == 0:
            continue
        
        binsize = 5
        bin_end = 10000
        if mass >= 4000:
            binsize = 1
            bin_end = 2000
        if mass >= 5000:
            binsize = 0.2
            bin_end = 1000
        if mass >= 6000:
            binsize = 0.1
            bin_end = 500
        if mass >= 6500:
            binsize = 0.05
            bin_end = 250
            
        bins = int(bin_end/binsize)
        hist = ROOT.TH1D("dist{0}".format(mass), "dist", bins, 0, bin_end)
        r = ROOT.TRandom1()

        for limit in data["expected-limits"][mass]:
            # Apply an uncertainty on the luminosity of 3.2%
            lum = r.Gaus(fb, 0.032*fb)
            limit2 = fb*limit/lum
            hist.Fill(limit2)

        mean = hist.GetMean()
        rms = hist.GetRMS()

        one_sigma = 0.3173
        two_sigma = 4.55e-2

        low_2sigma = None
        low_1sigma = None
        high_1sigma = None
        high_2sigma = None

        bin_content = [hist.GetBinContent(b) for b in range(1, hist.GetNbinsX()+1)]
        bin_locations = [hist.GetBinCenter(b) for b in range(1, hist.GetNbinsX()+1)]
        # cumulative = [sum(bin_content[0:b]) for b in range(0, len(bin_content))]
        cumulative = []
        current_sum = 0
        for content in bin_content:
            current_sum += content
            cumulative.append(current_sum)
        cumulative.append(current_sum)
        binsize = 1
        cumulative = [c/max(cumulative) for c in cumulative]
        m = None
        for x, y in zip(bin_locations, cumulative):
            if y >= two_sigma/2 and low_2sigma is None:
                low_2sigma = abs(x-mean)
                if mass == m:
                    print(x, y, two_sigma/2)
            if y >= one_sigma/2 and low_1sigma is None:
                low_1sigma = abs(x-mean)
                if mass == m:
                    print(x, y, one_sigma/2)
            if y >= 1-(one_sigma/2) and high_1sigma is None:
                high_1sigma = abs(x-mean)
                if mass == m:
                    print(x, y, 1-one_sigma/2)
            if y >= 1-(two_sigma/2) and high_2sigma is None:
                high_2sigma = abs(x-mean)
                if mass == m:
                    print(x, y, 1-two_sigma/2)
        
        


        # Convert to cross sections by dividing the number of events by the luminosity
        processed_data.add(mass, mean/pb, rms/pb, low_2sigma/pb, low_1sigma/pb, high_1sigma/pb, high_2sigma/pb)

    return processed_data

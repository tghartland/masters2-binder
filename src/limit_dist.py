from __future__ import print_function, division

import argparse
import math
import os
import os.path as op
import operator
from scipy.stats import poisson
import random
import time
import sys

from array import array
import ROOT
from ROOT import TFile, TH1D, TCanvas, gStyle, gROOT, TF1, TColor, TMinuit, gPad, TPad, TMath
from ROOT import TGaxis, TBox, THStack, TLegend, TGraphErrors, TVectorD, TGraph
from ROOT.Math import GaussIntegrator, WrappedTF1, IParamFunction
from ROOT.Math import IParametricFunctionOneDim

parser = argparse.ArgumentParser(description="Find C.L limits")
parser.add_argument("--workflow", help="workflow name")
parser.add_argument("--pod", help="pod name")
parser.add_argument("--mass", help="mass to test with")
parser.add_argument("--sim-file", help="file with simulated data to use")
parser.add_argument("--sim-hist", help="histogram to use from simulated data file")
parser.add_argument("--data-dir", help="directory where data files are located")
parser.add_argument("--output-dir", help="directory to write output to")
parser.add_argument("--successes", type=int, help="stop after this number of successful fits")
args = parser.parse_args()

print("Loading macros")

st = time.time()

gROOT.SetBatch(True)
gROOT.LoadMacro("FitFunction.cpp+g")

print("Loaded macros, time=", time.time()-st)

CLUSTER_ID = args.workflow
JOB_NAME = args.pod
MASS_FILE = args.mass
SIM_FILE_NAME = args.sim_file
SIM_HIST_NAME = args.sim_hist



print("MASS = {0} GeV".format(MASS_FILE))
print("JOB_NAME = {0}".format(JOB_NAME))
print("SIM_FILE_NAME = {0}".format(SIM_FILE_NAME.format(MASS_FILE)))
print("SIM_HIST_NAME = {0}".format(SIM_HIST_NAME.format(MASS_FILE)))
sys.stdout.flush()


class Fits:
    def __init__(self):
        
        self.p_n = [0,]*100
        self.e_n = [0,]*100
        self.stored_parameters = [0,]*100
        
        self.num_bins = 0
        self.xmins = []
        self.xmaxes = []
        
        self.data = []
        self.errors = []
        self.data_fits = []

        self.model_scale_values = []
        self.final = False
        
        self.exclude_regions = ((0, 0),)
        
        self.col1 = 1
        self.col2 = TColor.GetColor(27, 158, 119)
        self.col3 = TColor.GetColor(217, 95, 2)
        self.col4 = TColor.GetColor(117, 112, 179)
        
    def run_mass_fit(self, peak_scale_initial):
        self.gMinuit = TMinuit(30)
        self.gMinuit.SetPrintLevel(-1)
        self.gMinuit.SetFCN(self.Fitfcn_max_likelihood)

        self.fit_failed = False
        
        arglist = array("d", [0,]*10)
        ierflg = ROOT.Long(0)
        arglist[0] = ROOT.Double(1)
        
        # peak_scale_initial = ROOT.Double(peak_scale_initial)

        tmp = array("d", [0,])
        self.gMinuit.mnexcm("SET NOWarnings", tmp, 0, ierflg); 

        self.gMinuit.mnexcm("SET ERR", arglist, 1, ierflg)
        self.gMinuit.mnparm(0, "p1", 30, 20, 0, 100, ierflg)
        self.gMinuit.mnparm(1, "p2", 10, 1, 0, 0, ierflg)
        self.gMinuit.mnparm(2, "p3", -5.3, 1, 0, 0, ierflg)
        self.gMinuit.mnparm(3, "p4", -4e-2, 1e-2, 0, 0, ierflg)
        self.gMinuit.mnparm(4, "p5", peak_scale_initial, 1, 0, 10000, ierflg)

        
        self.background_fit_only = [0,]*len(self.data)

        arglist[0] = ROOT.Double(0)
        arglist[1] = ROOT.Double(0)

        #self.exclude_regions = ((2.2, 3.3),)
        self.gMinuit.FixParameter(1)
        self.gMinuit.FixParameter(2)
        self.gMinuit.FixParameter(3)
        self.gMinuit.FixParameter(4)
        
        self.gMinuit.mnexcm("simplex", arglist, 2, ierflg)
        self.gMinuit.mnexcm("MIGRAD", arglist, 2, ierflg)
        print("Run fit 0")
        if self.fit_failed:
            print("Fit failed, returning")
            return None, None


        self.gMinuit.Release(1)
        self.gMinuit.mnexcm("simplex", arglist, 2, ierflg)
        self.gMinuit.mnexcm("MIGRAD", arglist, 2, ierflg)
        print("Run fit 1")
        if self.fit_failed:
            print("Fit failed, returning")
            return None, None
        
        self.gMinuit.Release(2)
        self.gMinuit.mnexcm("simplex", arglist, 2, ierflg)
        self.gMinuit.mnexcm("MIGRAD", arglist, 2, ierflg)
        print("Run fit 2")
        if self.fit_failed:
            print("Fit failed, returning")
            return None, None
        
        self.gMinuit.Release(3)
        self.gMinuit.mnexcm("simplex", arglist, 2, ierflg)
        self.gMinuit.mnexcm("MIGRAD", arglist, 2, ierflg)
        print("Run fit 3")
        if self.fit_failed:
            print("Fit failed, returning")
            return None, None

        # Find an actual best fit
        #self.exclude_regions = ((0, 2), (3.3, 100),)
        self.gMinuit.FixParameter(0)
        self.gMinuit.FixParameter(1)
        self.gMinuit.FixParameter(2)
        self.gMinuit.FixParameter(3)
        self.gMinuit.Release(4)
        self.gMinuit.mnexcm("simplex", arglist, 2, ierflg)
        self.gMinuit.mnexcm("MIGRAD", arglist, 2, ierflg)
        print("Run fit 4")
        if self.fit_failed:
            print("Fit failed, returning")
            return None, None

        self.exclude_regions = ()
        self.gMinuit.Release(0)
        self.gMinuit.Release(1)
        self.gMinuit.Release(2)
        self.gMinuit.Release(3)
        self.gMinuit.mnexcm("simplex", arglist, 2, ierflg)
        self.gMinuit.mnexcm("MIGRAD", arglist, 2, ierflg)

        print("Run last fitting stage")
        if self.fit_failed:
            print("Fit failed, returning")
            return None, None

        best_fit_value = ROOT.Double(0)
        self.gMinuit.mnstat(best_fit_value, ROOT.Double(0), ROOT.Double(0),
                            ROOT.Long(0), ROOT.Long(0), ROOT.Long(0))
        #print("Best fit value", best_fit_value)

        # And prepare for iterating over fit values for N injected events
        self.gMinuit.Release(0)
        self.gMinuit.Release(1)
        self.gMinuit.Release(2)
        self.gMinuit.Release(3)
        self.exclude_regions = ()

        #self.data_fits = no_peak_data_fits
        x_values = []
        y_values = []

        for i in range(0, 5):
            p = ROOT.Double(0)
            self.gMinuit.GetParameter(i, p, ROOT.Double(0))
            print("Parameter", i, p)

        fitted_N = ROOT.Double(0)
        self.gMinuit.GetParameter(4, fitted_N, ROOT.Double(0))
        best_fit_likelihood = self.calc_likelihood(fitted_N)

        self.iflag = int(ierflg)
        print("iflag:", self.iflag)

        step = 5
        if int(MASS_FILE) >= 4000:
            step = 1
        if int(MASS_FILE) >= 5000:
            step = 0.2
        if int(MASS_FILE) >= 6000:
            step = 0.1
        if int(MASS_FILE) >= 6500:
            step = 0.1
        N = 0
        print("About to start likelihood testing loop")
        while N < 10000:
            start_sum = sum([math.exp(-a) for a in y_values])
            fit_likelihood = self.calc_likelihood(N)
            if fit_likelihood is None:
                print("Bad fit_likelihood")
                return None, None
            x_values.append(N)
            y_values.append(fit_likelihood-best_fit_likelihood)

            probabilities = [math.exp(-a) for a in y_values]
            end_sum = sum(probabilities)

            max_prob = max(probabilities)
            if max_prob == 0:
                print("max_prob == 0")
                return None, None
            normalised_probabilities = [a/max_prob for a in probabilities]

            if N/step > 50  and all([v > 0.99 for v in normalised_probabilities]):
                print("Probability=1 everywhere, probably something wrong with fit")
                print(normalised_probabilities)
                return None, None

            # if new value changes total by less than 0.1%, end loop
            if N > 0 and (end_sum-start_sum)/start_sum < 0.0001:
                print("Iterated up to {0}".format(N))
                break

            N += step

        self.iflag = int(ierflg)
        return x_values, y_values
    

    def Fitfcn_max_likelihood(self, npar, gin, fcnVal, par, iflag):
        likelihood = 0
        mf = ROOT.MyMassSpectrum()
        mf.SetParameters(par)
        ig = GaussIntegrator()
        ig.SetFunction(mf)
        ig.SetRelTolerance(0.00001)
        
        for i in range(0, self.num_bins):
            for lower, higher in self.exclude_regions:
                if lower < self.xmins[i] < higher:
                    continue

            model_val = ig.Integral(self.xmins[i], self.xmaxes[i]) / (self.xmaxes[i]-self.xmins[i])
            self.background_fit_only[i] = model_val
            model_val += self.model_scale_values[i]*par[4]
            self.data_fits[i] = model_val

            mv = model_val
            di = self.data[i]

            #print("mv", mv, "di", di)
            #sys.stdout.flush()
            if di > 1e10 or math.isinf(mv) or math.isnan(mv):
                self.fit_failed = True
                return

            likelihood += mv - di
            if di > 0 and mv > 0:
                likelihood += di*(TMath.Log(di)-TMath.Log(mv))
            #sys.stderr.flush()
        
        fcnVal[0] = likelihood


    def calc_likelihood(self, peak_scale):
        like = 0

        for i in range(0, self.num_bins):
            if self.data_fits[i] <= 0:
                continue

            p = peak_scale*self.model_scale_values[i]
            tmp = ROOT.TMath.PoissonI(self.data[i], self.background_fit_only[i]+p)
            if tmp == 0:
                return None
                print("tmp == 0 here")
                logtmp = math.log(sys.float_info.min)
            else:
                logtmp = math.log(tmp)
            like += logtmp

        return -like


root_file_location = op.join(args.data_dir, "mjj_mc15_13TeV_361023_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ3W_total_final.root")
root_file = TFile.Open(root_file_location)
hist = root_file.Get("mjj_mc15_13TeV_361023_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ3W_total_final")
#root_file_new = TFile.Open("data/mjj_data_new.root")
#hist_new = root_file_new.Get("mjj_data17_13TeV_00325713_physics_Main_total_final")

hist_bins = hist.GetNbinsX()
hist_contents = [hist.GetBinContent(b) for b in range(1, hist_bins+1)]
hist_center = [hist.GetBinCenter(b) for b in range(1, hist_bins+1)]
hist_left = [hist.GetBinLowEdge(b) for b in range(1, hist_bins+1)]
hist_right = [hist.GetBinLowEdge(b+1) for b in range(1, hist_bins+1)]

root_file_model = TFile.Open(op.join(args.data_dir, SIM_FILE_NAME.format(MASS_FILE)))
nominal = root_file_model.GetDirectory("Nominal")
hist_model = nominal.Get(SIM_HIST_NAME.format(MASS_FILE))
#hist_model.Smooth(1)
hist_model.Scale(1/hist_model.Integral())
hist_model_contents = [hist_model.GetBinContent(b) for b in range(1, hist_model.GetNbinsX()+1)]

root_file.Close()
root_file_model.Close()

def fit_significance(num_injected_events, plot=True, name=""):
    #ROOT.IABstyles.global_style()
    TGaxis.SetMaxDigits(3)

    hist_model_scaled = [num_injected_events*bin_content for bin_content in hist_model_contents]

    hist_random_bg = []
    hist_random = []

    seed = random.randrange(sys.maxsize)
    print("Random seed: ", seed)
    random.seed(seed)


    for bg_content, model_content, center in zip(hist_contents, hist_model_scaled, hist_center):
        if bg_content > 0 or center/1000 > 2:
            x = center/1000
            bg = poisson.ppf(random.random(), bg_content)
            peak = 0
            if model_content > 0:
                peak = poisson.ppf(random.random(), model_content)
            hist_random.append(bg+peak)
        else:
            hist_random.append(0)

    fits = Fits()
    fits.model_scale_values = list(hist_model_contents)

    nbins = hist_bins
    fits.xwidth = [(a-b)/1000/2 for a, b in zip(hist_left[1:], hist_left)]
    fits.xmiddle = [x/1000 for x in hist_center]
    fits.xmins = [x/1000 for x in hist_left]
    fits.xmaxes = [x/1000 for x in hist_right]
    fits.data = hist_random
    fits.data_fits = [0,]*nbins
    fits.errors = [math.sqrt(x) for x in fits.data]
    fits.num_bins = nbins

    
    remove_bins = 0
    for x in fits.data:
        if x == 0:
            remove_bins += 1
        else:
            break
    
    nbins -= remove_bins
    fits.xwidth = fits.xwidth[remove_bins:]
    fits.xmiddle = fits.xmiddle[remove_bins:]
    fits.xmins = fits.xmins[remove_bins:]
    fits.xmaxes = fits.xmaxes[remove_bins:]
    fits.data = fits.data[remove_bins:]
    fits.data_fits = fits.data_fits[remove_bins:]
    fits.errors = fits.errors[remove_bins:]
    fits.num_bins = nbins
    fits.model_scale_values = fits.model_scale_values[remove_bins:]
    
    #for i in range(0, nbins):
    #    print(i, fits.data[i], fits.model_scale_values[i])
    sys.stdout.flush()

    x, y = fits.run_mass_fit(num_injected_events)
    """for i in range(0, 5):
        k = ROOT.Double(0)
        fits.gMinuit.GetParameter(i, k, ROOT.Double(0))
        print("Parameter {0}: {1}".format(i, k))"""
    if x is None or y is None:
        return None

    plot_only = False
    if fits.iflag != 4:
        plot_only = True
        print("Fit did not converge")
        return None

    if not plot_only:
        y = [math.exp(-a) for a in y]
        
        ycumulative = [sum(y[0:i]) for i in range(0, len(y))]
        if max(ycumulative) == 0:
            print("Iteration {0} max(ycumulative)=0".format(name))
            return None
        ycumulative = [yval/max(ycumulative) for yval in ycumulative]
        limit_x = 0
        limit_y = 0
        for xv, yv in zip(x, ycumulative):
            if yv >= 0.95:
                limit_x = xv
                limit_y = yv
                break
        
    return limit_x



def generate_expected_limits():
    start_time = time.time()

    want_successes = args.successes
    successes = 0
    total_iterations = 0
    times_taken = []
    
    if not op.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_file = op.join(args.output_dir, "job-{}-{}.txt".format(MASS_FILE, JOB_NAME))

    with open(output_file, "w") as out_file:
        while successes < want_successes and total_iterations < 2*want_successes:
            total_iterations += 1
            st = time.time()
            limit = fit_significance(0, plot=False, name="{0}.i{1}".format(JOB_NAME, total_iterations))
            sys.stdout.flush()
            sys.stderr.flush()
            print("Iteration {0}, limit = {1}, time = {2:.02f}\n".format(total_iterations, limit, time.time()-st))
            sys.stdout.flush()
            times_taken.append((limit, time.time()-st))
            if limit is None:
                continue
            out_file.write("{0}\n".format(limit))
            successes += 1

    print("successes={0}, total_iterations={1}".format(successes, total_iterations))
    print("Took {0:.0f} seconds".format(time.time()-start_time))

generate_expected_limits()

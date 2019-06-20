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
#gROOT.LoadMacro("IABStyle.cpp+g")

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

#print("Waiting 10s")
#time.sleep(10)
#print("Waiting done")

sys.stdout.flush()

class BackgroundModel:
    def __init__(self, p1, p2, p3, p4):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4

    def model_at(self, x):
        scale = x/14
        a1 = self.p1 * math.pow(1-scale, self.p2)
        a2 = self.p3 + (self.p4 * math.log(scale))
        a3 = math.pow(scale, a2)
        return a1*a3

    def random_at(self, x):
        return poisson.ppf(random.random(), self.model_at(x))


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
            #if peak_scale == 40000:
            #    print(i, "\txmin", self.xmins[i], "\tdata", self.data[i], "\tdata_fit", self.data_fits[i], "\tp", p, "\tdata_fit+p", self.data_fits[i]+p)
            if tmp == 0:
                return None
                print("tmp == 0 here")
                logtmp = math.log(sys.float_info.min)
            else:
                logtmp = math.log(tmp)
            like += logtmp

        return -like



def plot_data_histogram(fits, name):
    #return
    test_canvas = TCanvas("TestCanvas", "Ds Fit", 0, 0, 800, 575)

    gStyle.SetPadBorderMode(0)
    gStyle.SetFrameBorderMode(0)

    test_canvas.Divide(1, 2, 0, 0)
    upper_pad = test_canvas.GetPad(1)
    lower_pad = test_canvas.GetPad(2)
    low, high = 0.05, 0.95
    upper_pad.SetPad(low, 0.4, high, high)
    lower_pad.SetPad(low, low, high, 0.4)
    
    test_canvas.cd(1)
    
    #ROOT.IABstyles.canvas_style(test_canvas, 0.25, 0.05, 0.02, 0.15, 0, 0)
    
    h_Mjj = TH1D("h_Mjj", "Mass Spectrum", 100, 0.2, 12)
    h_Mjj.GetYaxis().SetTitle("num. events")
    h_Mjj.GetXaxis().SetTitle("M [Tev/c^{-2}]")
    
    #ROOT.IABstyles.h1_style(h_Mjj, ROOT.IABstyles.lWidth//2, ROOT.IABstyles.Scolor, 1, 0, 0, -1111.0, -1111.0, 508, 508, 8, ROOT.IABstyles.Scolor, 0.1, 0)
    
    h_Mjj.GetYaxis().SetRangeUser(0.1, 2.5e6)
    h_Mjj.GetXaxis().SetRangeUser(1, 10)
    h_Mjj.GetXaxis().SetTitleOffset(1)
    h_Mjj.GetYaxis().SetTitleOffset(1.1)
    
    upper_pad.SetLogy(1)
    upper_pad.SetLogx(1)
    lower_pad.SetLogx(1)
    
    gr = TGraphErrors(fits.num_bins, array("d", fits.xmiddle), array("d", fits.data), array("d", fits.xwidth), array("d", fits.errors))
    #ROOT.IABstyles.h1_style(gr, ROOT.IABstyles.lWidth//2, ROOT.IABstyles.Scolor, 1, 0, 0, -1111, -1111, 505, 505, 8, ROOT.IABstyles.Scolor, 0.1, 0)
    
    grFit = TGraph(fits.num_bins, array("d", fits.xmiddle), array("d", fits.data_fits))
    #ROOT.IABstyles.h1_style(grFit, ROOT.IABstyles.lWidth//2, 632, 1, 0, 0, -1111, -1111, 505, 505, 8, 632, 0.1, 0)
    
    h_Mjj.Draw("axis")
    gr.Draw("P")
    grFit.Draw("c")

    test_canvas.Update()
    
    gPad.SetBottomMargin(1e-5)
    
    test_canvas.cd(2)
    gPad.SetTopMargin(1e-5)
    
    h2 = TH1D("h2", "", 100, 0.2, 12)
    h2.GetXaxis().SetRangeUser(1, 10)
    h2.GetYaxis().SetRangeUser(-10, 10)
    h2.SetStats(False) # don't show stats box
    h2.Draw("axis")
    sig_values = [(data-theory)/theory if (data!= 0 and theory != 0) else -100 for data, theory in zip(fits.data, fits.data_fits)]
    sig = TGraph(fits.num_bins, array("d", fits.xmiddle), array("d", sig_values))
    #ROOT.IABstyles.h1_style(sig, ROOT.IABstyles.lWidth/2, 632, 1, 0, 0, -1111, -1111, 505, 505, 8, 632, 0.1, 0)
    #ROOT.IABstyles.h1_style(gr, ROOT.IABstyles.lWidth//2, ROOT.IABstyles.Scolor, 1, 0, 0, -1111, -1111, 505, 505, 8, ROOT.IABstyles.Scolor, 0.1, 0)
    sig.SetMarkerStyle(22) # triangle
    sig.SetMarkerColor(2)  # red
    sig.SetMarkerSize(0.8)
    sig.Draw("P")
    #lower_pad.Draw()
    
    label = ROOT.TText()
    label.SetNDC()
    label.SetTextSize(0.03)
    label.DrawText(0.5, 0.7, "{0}GeV".format(MASS_FILE))

    test_canvas.SaveAs("plots/{0}/{2}/plot-{1}-{2}-hist.png".format(CLUSTER_ID, name, MASS_FILE))
    test_canvas.SaveAs("plots/{0}/{2}/plot-{1}-{2}-hist.pdf".format(CLUSTER_ID, name, MASS_FILE))

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
    #seed = 947971000508381793 # 6000, inf
    #seed = 3391513505279461724 # 6500, nan
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
            plot_data_histogram(fits, name)
            return None
        ycumulative = [yval/max(ycumulative) for yval in ycumulative]
        limit_x = 0
        limit_y = 0
        for xv, yv in zip(x, ycumulative):
            if yv >= 0.95:
                limit_x = xv
                limit_y = yv
                break
        
        """if limit_x < 4500:
            return limit_x
        else:
            limit_x = None"""
    return limit_x

    for _x, _y in zip(x, y):
        if _x == limit_x:
            limit_y_2 = _y


    canvas = TCanvas("dist", "dist", 0, 0, 525, 450)
    canvas.SetLeftMargin(0.12)
    canvas.SetBottomMargin(0.12)
    gStyle.SetOptTitle(0)
    canvas.SetTopMargin(0.04)
    canvas.SetRightMargin(0.04)
    graph = TGraph(len(x), array("d", x), array("d", y))
    #ROOT.IABstyles.h1_style(graph, ROOT.IABstyles.lWidth//2, ROOT.IABstyles.Scolor, 1, 0, 0, -1111.0, -1111.0, 508, 508, 8, ROOT.IABstyles.Scolor, 0.1, 0)
    graph.SetMarkerColor(4)
    graph.SetMarkerStyle(3)
    graph.SetMarkerSize(1)
    graph.GetXaxis().SetTitle("Number of signal events")
    graph.GetYaxis().SetTitle("Probability")
    graph.GetXaxis().SetNoExponent(True)
    graph.GetXaxis().SetTitleOffset(1.04)
    graph.GetYaxis().SetTitleOffset(1.04)
    graph.GetXaxis().SetLabelSize(0.045)
    graph.GetYaxis().SetLabelSize(0.045)
    graph.GetXaxis().SetTitleSize(0.05)
    graph.GetYaxis().SetTitleSize(0.05)
    graph.Draw("ap")
    line = ROOT.TLine()
    line.SetLineColor(2)
    line.SetLineWidth(2)
    line.DrawLine(limit_x, 0, limit_x, limit_y_2)
    #label = ROOT.TText()
    #label.SetNDC()
    #label.SetTextSize(0.03)
    #label.DrawText(0.5, 0.7, "{0}GeV".format(MASS_FILE))
    canvas.SaveAs("plots/{0}/{2}/plot-{1}-{2}-sig_dist.png".format(CLUSTER_ID, name, MASS_FILE))
    canvas.SaveAs("plots/{0}/{2}/plot-{1}-{2}-sig_dist.pdf".format(CLUSTER_ID, name, MASS_FILE))

    canvas2 = TCanvas("cumsum", "cumsum", 0, 0, 525, 450)
    gStyle.SetOptTitle(0)
    canvas2.SetTopMargin(0.04)
    canvas2.SetRightMargin(0.04)
    canvas2.SetLeftMargin(0.12)
    canvas2.SetBottomMargin(0.12)
    graph = TGraph(len(x), array("d", x), array("d", ycumulative))
    #ROOT.IABstyles.h1_style(graph, ROOT.IABstyles.lWidth//2, ROOT.IABstyles.Scolor, 1, 0, 0, -1111.0, -1111.0, 508, 508, 8, ROOT.IABstyles.Scolor, 0.1, 0)
    graph.SetMarkerColor(4)
    graph.SetMarkerStyle(3)
    graph.SetMarkerSize(1)
    graph.GetXaxis().SetTitle("Number of signal events")
    graph.GetYaxis().SetTitle("Cumulative probability")
    graph.GetXaxis().SetNoExponent(True)
    graph.GetXaxis().SetTitleOffset(1.04)
    graph.GetYaxis().SetTitleOffset(1.04)
    graph.GetXaxis().SetLabelSize(0.045)
    graph.GetYaxis().SetLabelSize(0.045)
    graph.GetXaxis().SetTitleSize(0.05)
    graph.GetYaxis().SetTitleSize(0.05)
    graph.Draw("ap")
    line = ROOT.TLine()
    line.SetLineColor(2)
    line.SetLineWidth(2)
    line.DrawLine(limit_x, 0, limit_x, limit_y)
    line = ROOT.TLine()
    line.SetLineColor(2)
    line.SetLineWidth(2)
    line.DrawLine(0, 0.95, limit_x, 0.95)
    #label = ROOT.TText()
    #label.SetNDC()
    #label.SetTextSize(0.03)
    #label.DrawText(0.5, 0.7, "{0}GeV particle {1:.02f} confidence limit = {2} events".format(MASS_FILE, limit_y, limit_x))
    canvas2.SaveAs("plots/{0}/{2}/plot-{1}-{2}-sig_cumsum.png".format(CLUSTER_ID, name, MASS_FILE))
    canvas2.SaveAs("plots/{0}/{2}/plot-{1}-{2}-sig_cumsum.pdf".format(CLUSTER_ID, name, MASS_FILE))

    plot_data_histogram(fits, name)

    return limit_x


def plot_95pc_confidence_dist():
    start_time = time.time()

    if not op.exists("results/{0}/{1}".format(CLUSTER_ID, MASS_FILE)):
        os.makedirs("results/{0}/{1}".format(CLUSTER_ID, MASS_FILE))

    if not op.exists("plots/{0}/{1}".format(CLUSTER_ID, MASS_FILE)):
        os.makedirs("plots/{0}/{1}".format(CLUSTER_ID, MASS_FILE))
    
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

    """with open("results/{0}/times-{1}-{2}.txt".format(CLUSTER_ID, JOB_NAME, MASS_FILE), "w") as time_file:
        for limit, t in times_taken:
            if limit is None:
                limit = "None"
            else:
                limit = "{0:.02f}".format(limit)
            time_file.write("{0}:{1:.02f}\n".format(limit, t))"""

    print("successes={0}, total_iterations={1}".format(successes, total_iterations))
    print("Took {0:.0f} seconds".format(time.time()-start_time))

plot_95pc_confidence_dist()

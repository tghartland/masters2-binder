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

import redis

from array import array
import ROOT
from ROOT import TFile, TH1D, TCanvas, gStyle, gROOT, TF1, TColor, TMinuit, gPad, TPad
from ROOT import TGaxis, TBox, THStack, TLegend, TGraphErrors, TVectorD, TGraph
from ROOT.Math import GaussIntegrator, WrappedTF1, IParamFunction
from ROOT.Math import IParametricFunctionOneDim

parser = argparse.ArgumentParser(description="Find C.L limits for data distribution")
parser.add_argument("--workflow", help="workflow name")
#parser.add_argument("--pod", help="pod name")
parser.add_argument("--data-dir", help="directory where data files are located")
parser.add_argument("--output-dir", help="directory to write output to")
parser.add_argument("--redis-host", help="redis hostname", required=True)
parser.add_argument("--redis-port", help="redis port", required=True)
args = parser.parse_args()


gROOT.SetBatch(True)
gROOT.LoadMacro("FitFunction.cpp+g")
gROOT.LoadMacro("IABStyle.cpp+g")



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
        
    def run_mass_fit(self, peak_scale_initial, mass=0):
        self.gMinuit = TMinuit(30)
        self.gMinuit.SetPrintLevel(-1)
        self.gMinuit.SetFCN(self.Fitfcn_max_likelihood)
        
        arglist = array("d", [0,]*10)
        ierflg = ROOT.Long(0)
        arglist[0] = ROOT.Double(1)
        
        # peak_scale_initial = ROOT.Double(peak_scale_initial)

        tmp = array("d", [0,])
        self.gMinuit.mnexcm("SET NOWarnings", tmp, 0, ierflg); 

        self.gMinuit.mnexcm("SET ERR", arglist, 1, ierflg)
        self.gMinuit.mnparm(0, "p1", 5e-6, 1e-7, 0, 0, ierflg)
        self.gMinuit.mnparm(1, "p2", 10, 10, 0, 0, ierflg)
        self.gMinuit.mnparm(2, "p3", -5.3, 1, 0, 0, ierflg)
        self.gMinuit.mnparm(3, "p4", -4e-2, 1e-2, 0, 0, ierflg)
        self.gMinuit.mnparm(4, "p5", peak_scale_initial, peak_scale_initial/50, 0, 0, ierflg)
        
        
        self.background_fit_only = [0,]*len(self.data)

        arglist[0] = ROOT.Double(0)
        arglist[1] = ROOT.Double(0)

        #self.exclude_regions = ((2.2, 3.3),)
        self.gMinuit.FixParameter(2)
        self.gMinuit.FixParameter(3)
        self.gMinuit.FixParameter(4)
        
        self.gMinuit.mnexcm("simplex", arglist, 2, ierflg)
        self.gMinuit.mnexcm("MIGRAD", arglist, 2, ierflg)
        
        self.gMinuit.Release(2)
        self.gMinuit.mnexcm("simplex", arglist, 2, ierflg)
        self.gMinuit.mnexcm("MIGRAD", arglist, 2, ierflg)
        
        self.gMinuit.Release(3)
        self.gMinuit.mnexcm("simplex", arglist, 2, ierflg)
        self.gMinuit.mnexcm("MIGRAD", arglist, 2, ierflg)

        # Find an actual best fit
        #self.exclude_regions = ((0, 2), (3.3, 100),)
        self.gMinuit.FixParameter(0)
        self.gMinuit.FixParameter(1)
        self.gMinuit.FixParameter(2)
        self.gMinuit.FixParameter(3)
        self.gMinuit.Release(4)
        self.gMinuit.mnexcm("simplex", arglist, 2, ierflg)
        self.gMinuit.mnexcm("MIGRAD", arglist, 2, ierflg)

        self.exclude_regions = ()
        self.gMinuit.Release(0)
        self.gMinuit.Release(1)
        self.gMinuit.Release(2)
        self.gMinuit.Release(3)
        self.gMinuit.mnexcm("simplex", arglist, 2, ierflg)
        self.gMinuit.mnexcm("MIGRAD", arglist, 2, ierflg)

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

        fitted_N = ROOT.Double(0)
        self.gMinuit.GetParameter(4, fitted_N, ROOT.Double(0))
        best_fit_likelihood = self.calc_likelihood(fitted_N)
        
        step = 5
        if int(mass) >= 4000:
            step = 1
        if int(mass) >= 5000:
            step = 0.2
        if int(mass) >= 6000:
            step = 0.1
        if int(mass) >= 6500:
            step = 0.05

        N = 0

        while N < 5000:
            start_sum = sum([math.exp(-a) for a in y_values])
            fit_likelihood = self.calc_likelihood(N)
            x_values.append(N)
            y_values.append(fit_likelihood-best_fit_likelihood)

            probabilities = [math.exp(-a) for a in y_values]
            end_sum = sum(probabilities)

            max_prob = max(probabilities)
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

            likelihood += model_val - self.data[i]
            if self.data[i] > 0 and model_val > 0:
                likelihood += self.data[i]*(math.log(self.data[i])-math.log(model_val))
        
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
                print("tmp == 0 here")
                logtmp = math.log(sys.float_info.min)
            else:
                logtmp = math.log(tmp)
            like += logtmp

        return -like



def plot_data_histogram(fits, name):
    return
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
    
    ROOT.IABstyles.canvas_style(test_canvas, 0.25, 0.05, 0.02, 0.15, 0, 0)
    
    h_Mjj = TH1D("h_Mjj", "Mass Spectrum", 100, 0.2, 12)
    h_Mjj.GetYaxis().SetTitle("num. events")
    h_Mjj.GetXaxis().SetTitle("M [Tev/c^{-2}]")
    
    ROOT.IABstyles.h1_style(h_Mjj, ROOT.IABstyles.lWidth//2, ROOT.IABstyles.Scolor, 1, 0, 0, -1111.0, -1111.0, 508, 508, 8, ROOT.IABstyles.Scolor, 0.1, 0)
    
    h_Mjj.GetYaxis().SetRangeUser(0.1, 1e6)
    h_Mjj.GetXaxis().SetRangeUser(1, 10)
    h_Mjj.GetXaxis().SetTitleOffset(1)
    h_Mjj.GetYaxis().SetTitleOffset(1.1)
    
    upper_pad.SetLogy(1)
    upper_pad.SetLogx(1)
    lower_pad.SetLogx(1)
    
    gr = TGraphErrors(fits.num_bins, array("d", fits.xmiddle), array("d", fits.data), array("d", fits.xwidth), array("d", fits.errors))
    ROOT.IABstyles.h1_style(gr, ROOT.IABstyles.lWidth//2, ROOT.IABstyles.Scolor, 1, 0, 0, -1111, -1111, 505, 505, 8, ROOT.IABstyles.Scolor, 0.1, 0)
    
    grFit = TGraph(fits.num_bins, array("d", fits.xmiddle), array("d", fits.data_fits))
    ROOT.IABstyles.h1_style(grFit, ROOT.IABstyles.lWidth//2, 632, 1, 0, 0, -1111, -1111, 505, 505, 8, 632, 0.1, 0)
    
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
    ROOT.IABstyles.h1_style(gr, ROOT.IABstyles.lWidth//2, ROOT.IABstyles.Scolor, 1, 0, 0, -1111, -1111, 505, 505, 8, ROOT.IABstyles.Scolor, 0.1, 0)
    sig.SetMarkerStyle(22) # triangle
    sig.SetMarkerColor(2)  # red
    sig.SetMarkerSize(0.8)
    sig.Draw("P")
    #lower_pad.Draw()
    
    label = ROOT.TText()
    label.SetNDC()
    label.SetTextSize(0.03)
    label.DrawText(0.5, 0.7, "{0}GeV q*".format(MASS_FILE))

    # test_canvas.SaveAs("output_qstar.pdf")
    test_canvas.SaveAs("plots/{0}/{2}/plot-{1}-{2}-hist.png".format(CLUSTER_ID, name, MASS_FILE))


def fit_significance(num_injected_events, plot=True, name=""):
    ROOT.IABstyles.global_style()
    TGaxis.SetMaxDigits(3)

    hist_model_scaled = [num_injected_events*bin_content for bin_content in hist_model_contents]

    hist_random_bg = []
    hist_random = []
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

    """root_file = TFile.Open("data/mjj_mc15_13TeV_361023_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ3W_total_final.root")
    hist = root_file.Get("mjj_mc15_13TeV_361023_Pythia8EvtGen_A14NNPDF23LO_jetjet_JZ3W_total_final")

    root_file_qstar = TFile.Open("data/dataLikeHistograms.QStar{0}.root".format(MASS_FILE))
    nominal = root_file_qstar.GetDirectory("Nominal")
    hist_model = nominal.Get("mjj_Scaled_QStar{0}_30fb".format(MASS_FILE))
    hist_model.Smooth(1)
    hist_model.Scale(1/hist_model.Integral())
    model_scale_values = [hist_model.GetBinContent(b) for b in range(1, hist_model.GetNbinsX()+1)]
    hist_model.Scale(num_injected_events)

    hist_original_bin_contents = [hist.GetBinContent(b) for b in range(0, hist.GetNbinsX()+1)]
    for b in range(0, hist.GetNbinsX()+1):
        hist.SetBinContent(b, 0)

    for b in range(1, hist.GetNbinsX()+1):
        if hist_original_bin_contents[b] > 0 or hist.GetBinLowEdge(b)/1000 > 2:
            x = hist.GetBinCenter(b)/1000
            bg = poisson.ppf(random.random(), hist_original_bin_contents[b])
            peak = 0
            if hist_model.GetBinContent(b) > 0:
                peak = poisson.ppf(random.random(), hist_model.GetBinContent(b))
            hist.SetBinContent(b, bg+peak)
    
    fits = Fits()
    fits.model_scale_values = model_scale_values 

    nbins = hist.GetNbinsX()

    fits.xwidth = [(hist.GetBinLowEdge(b+1)/1000-hist.GetBinLowEdge(b)/1000)/2 for b in range(1, nbins+1)]
    fits.xmiddle = [hist.GetBinCenter(b)/1000 for b in range(1, nbins+1)]
    fits.xmins = [hist.GetBinLowEdge(b)/1000 for b in range(1, nbins+1)]
    fits.xmaxes = [hist.GetBinLowEdge(b+1)/1000 for b in range(1, nbins+1)]
    fits.data = [hist.GetBinContent(b) for b in range(1, nbins+1)]
    fits.data_fits = [0,]*nbins
    

    root_file.Close()
    root_file_qstar.Close()

    fits.errors = [math.sqrt(x) for x in fits.data]
    fits.num_bins = nbins"""
    
    
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

    x, y = fits.run_mass_fit(num_injected_events)

    if x is None and y is None:
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
        
        if limit_x < 4500:
            return limit_x
        else:
            limit_x = None
    return limit_x
    canvas = TCanvas("dist", "dist", 0, 0, 650, 450)
    graph = TGraph(len(x), array("d", x), array("d", y))
    ROOT.IABstyles.h1_style(graph, ROOT.IABstyles.lWidth//2, ROOT.IABstyles.Scolor, 1, 0, 0, -1111.0, -1111.0, 508, 508, 8, ROOT.IABstyles.Scolor, 0.1, 0)
    graph.SetMarkerColor(2)
    graph.SetMarkerStyle(3)
    graph.SetMarkerSize(1.25)
    graph.Draw("ap")
    label = ROOT.TText()
    label.SetNDC()
    label.SetTextSize(0.03)
    label.DrawText(0.5, 0.7, "{0}GeV q*".format(MASS_FILE))
    canvas.SaveAs("plots/{0}/{2}/plot-{1}-{2}-sig_dist.png".format(CLUSTER_ID, name, MASS_FILE))

    canvas2 = TCanvas("cumsum", "cumsum", 0, 0, 650, 450)
    graph = TGraph(len(x), array("d", x), array("d", ycumulative))
    ROOT.IABstyles.h1_style(graph, ROOT.IABstyles.lWidth//2, ROOT.IABstyles.Scolor, 1, 0, 0, -1111.0, -1111.0, 508, 508, 8, ROOT.IABstyles.Scolor, 0.1, 0)
    graph.SetMarkerColor(4)
    graph.SetMarkerStyle(3)
    graph.SetMarkerSize(1.25)
    graph.Draw("ap")
    line = ROOT.TLine(limit_x, 0, limit_x, limit_y)
    line.SetLineColor(2)
    line.Draw("same")
    label = ROOT.TText()
    label.SetNDC()
    label.SetTextSize(0.03)
    label.DrawText(0.5, 0.7, "{0}GeV q* {1:.02f} confidence limit = {2} events".format(MASS_FILE, limit_y, limit_x))
    canvas2.SaveAs("plots/{0}/{2}/plot-{1}-{2}-sig_cumsum.png".format(CLUSTER_ID, name, MASS_FILE))

    plot_data_histogram(fits, name)

    return limit_x


def fit_data_likelihood(simulation_mass):
    data_file_location = op.join(args.data_dir, "mjj_data15_13TeV_00276262_physics_Main_total_final.root")
    root_file_data_old = TFile.Open(data_file_location)
    hist_data_old = root_file_data_old.Get("mjj_data15_13TeV_00276262_physics_Main_total_final")
    
    #root_file_data_new = TFile.Open("data/mjj_data_new.root")
    #hist_data_new = root_file_data_new.Get("mjj_data17_13TeV_00325713_physics_Main_total_final")
    
    hist_data = hist_data_old

    hist_bins = hist_data_old.GetNbinsX()
    #hist_data_contents = [hist_data_old.GetBinContent(b)+hist_data_new.GetBinContent(b) for b in range(1, hist_bins+1)]
    hist_data_contents = [hist_data.GetBinContent(b) for b in range(1, hist_bins+1)]
    hist_center = [hist_data.GetBinCenter(b) for b in range(1, hist_bins+1)]
    hist_left = [hist_data.GetBinLowEdge(b) for b in range(1, hist_bins+1)]
    hist_right = [hist_data.GetBinLowEdge(b+1) for b in range(1, hist_bins+1)]

    root_file_qstar_location = op.join(args.data_dir, "QStar/dataLikeHistograms.QStar{0}.root".format(simulation_mass))
    root_file_qstar = TFile.Open(root_file_qstar_location)
    nominal = root_file_qstar.GetDirectory("Nominal")
    hist_model = nominal.Get("mjj_Scaled_QStar{0}_30fb".format(simulation_mass))
    hist_model.Smooth(1)
    hist_model.Scale(1/hist_model.Integral())
    hist_model_contents = [hist_model.GetBinContent(b) for b in range(1, hist_model.GetNbinsX()+1)]

    hist_model_scaled = [0*bin_content for bin_content in hist_model_contents]

    hist_random_bg = []
    hist_random = []
    for bg_content, model_content, center in zip(hist_data_contents, hist_model_scaled, hist_center):
        if bg_content > 0 or center/1000 > 2:
            x = center/1000
            bg = bg_content# poisson.ppf(random.random(), bg_content)
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

    x, y = fits.run_mass_fit(0, mass=simulation_mass)

    if x is None and y is None:
        return None

    if fits.iflag != 4:
        print("Fit did not converge mass={0}".format(simulation_mass))
        return None

    y = [math.exp(-a) for a in y]
    
    ycumulative = [sum(y[0:i]) for i in range(0, len(y))]
    if max(ycumulative) == 0:
        print("Iteration mass={0} max(ycumulative)=0".format(simulation_mass))
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
    
    if limit_x < 5000:
        return limit_x
    else:
        limit_x = None


def data_confidence_dist():
    r = redis.Redis(host=args.redis_host, port=args.redis_port, db=0)

    start_time = time.time()

    if not op.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(op.join(args.output_dir, "data_cl_dist_qstar.txt"), "w") as out_file:
        for mass in range(2000, 7001, 500):
            print("Testing mass {0}".format(mass))
            limit = fit_data_likelihood(mass)
            r.hset("{}:data-limits".format(args.workflow), mass, limit)
            out_file.write("{0}:{1}\n".format(mass, limit))

    print("Took {0:.02f} seconds".format(time.time()-start_time))

data_confidence_dist()



'''def plot_95pc_confidence_dist():
    """canvas = TCanvas("dist", "dist", 0, 0, 650, 450)
    hist = TH1D("dist", "95% C.L dist", 100, 0, 1500)
    for i in range(0, 300):
        if i %25 == 0:
            print(i)
        hist.Fill(fit_significance(0, plot=False))
    hist.Draw()
    canvas.SaveAs("95pcCL_dist.png")"""
    
    start_time = time.time()

    if not op.exists("results/{0}/{1}".format(CLUSTER_ID, MASS_FILE)):
        os.makedirs("results/{0}/{1}".format(CLUSTER_ID, MASS_FILE))

    if not op.exists("plots/{0}/{1}".format(CLUSTER_ID, MASS_FILE)):
        os.makedirs("plots/{0}/{1}".format(CLUSTER_ID, MASS_FILE))
    
    want_successes = 500
    successes = 0
    total_iterations = 0
    
    with open("results/{0}/{1}/job-{2}-{1}.txt".format(CLUSTER_ID, MASS_FILE, JOB_NAME), "w") as out_file:
        while successes < want_successes and total_iterations < 2*want_successes:
            total_iterations += 1
            st = time.time()
            limit = fit_significance(0, plot=False, name="{0}.i{1}".format(JOB_NAME, total_iterations))
            print("Iteration {0}, limit = {1}, time = {2:.02f}".format(total_iterations, limit, time.time()-st))
            if limit is None:
                continue
            out_file.write("{0}\n".format(limit))
            successes += 1
    print("successes={0}, total_iterations={1}".format(successes, total_iterations))
    print("Took {0:.0f} seconds".format(time.time()-start_time))

plot_95pc_confidence_dist()'''

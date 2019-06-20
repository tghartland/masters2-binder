from __future__ import print_function, division

import argparse
import sys
import math
import os
import os.path as op
from array import array
from operator import itemgetter

import ROOT
from ROOT import TH1D, TCanvas, gROOT, gStyle, gPad, TGraph, TGraphErrors, TGraphAsymmErrors, TMultiGraph, TFile, TRandom, TRandom1, TLatex, TLegend

parser = argparse.ArgumentParser(description="Make brazil plot")
parser.add_argument("--title", help="title for plot")
parser.add_argument("--particle", help="particle name")
parser.add_argument("--workflow", help="workflow name")
parser.add_argument("--sim-file", help="file with simulated data to use")
parser.add_argument("--sim-hist", help="histogram to use from simulated data file")
parser.add_argument("--data-dir", help="directory where data files are located")
parser.add_argument("--output-dir", help="directory to write output to")
parser.add_argument("--expected-limit-dir", help="directory where generated expected limit files are")
parser.add_argument("--data-limit-dir", help="directory where data limit file is")
parser.add_argument("--fb", type=int, help="number of inverse femtobarns of data")
args = parser.parse_args()

gROOT.SetBatch(True)
#gStyle.SetLegendBorderSize(0)
#gStyle.SetLegendFillStyle(0);

job_id = args.workflow

#with open(op.join(base_path, "desc.txt"), "r") as desc_file:
#    title, particle = desc_file.readline().strip().split(",")
#    sim_file = desc_file.readline().strip()
#    sim_hist = desc_file.readline().strip()

title = args.title
sim_file = args.sim_file
sim_hist = args.sim_hist
particle = args.particle

#with open(op.join(base_path, "fb.txt"), "r") as fb_file:
#    fb = int(fb_file.read())

fb = args.fb

print("{0} fb^-1".format(fb))
fb2 = 1000*fb

print("Title: {0}".format(title))
print("Particle: {0}".format(particle))
print("Sim file: {0}".format(sim_file))
print("Sim hist: {0}".format(sim_hist))

y_limits = {
    "q*": (1e-4, 4),
    "qbh": (1e-5, 1e-1),
    "wprime": (5e-5, 4),
}[particle]

data_cl_file = {
    "q*":     op.join(args.data_limit_dir, "data_cl_dist_qstar.txt"),
    "qbh":    op.join(args.data_limit_dir, "data_cl_dist_qbh.txt"),
    "wprime": op.join(args.data_limit_dir, "data_cl_dist_wprime.txt"),
}[particle]

particle_symbol = {
    "q*": "q*",
    "qbh": "QBH",
    "wprime": "W'",
}[particle]

rapidity = {
    "q*": 0.6,
    "qbh": 0.6,
    "wprime": 0.6,
}[particle]


if not op.exists(args.output_dir):
    os.makedirs(args.output_dir)

brazil_data = [] # (mass, mean, rms) tuples
data_mass = []
data_mean = []
data_rms = []
data_2sl = [] # 2 sigma low
data_1sl = [] # 1 sigma low
data_1sh = [] # 1 sigma high
data_2sh = [] # 2 sigma high

mass_points = []

for mass_folder in sorted(os.listdir(args.expected_limit_dir)):
    if not mass_folder.isdigit():
        continue

    mass = int(mass_folder)

    mass_points.append(mass)

    if mass < 2000:
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
    hist = TH1D("dist{0}".format(mass), "dist", bins, 0, bin_end)
    r = TRandom1()

    expected_limits_file = op.join(args.expected_limit_dir, mass_folder, "combined-{}.txt".format(mass))
    with open(expected_limits_file, "r") as result_file:
        for i, line in enumerate(result_file.readlines()):
            if len(line) == 0:
                continue
            limit = float(line)
            lum = r.Gaus(fb, 0.032*fb)
            limit2 = fb*limit/lum
            #if i % 1000 == 0:
            #    print(limit, lum, limit2)
            hist.Fill(limit2)
            #hist.Fill(fb*limit/r.Gaus(fb, 0.032*fb))
            #hist.Fill(limit)
    
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


    brazil_data.append((mass, mean, rms, low_2sigma, low_1sigma, high_1sigma, high_2sigma))

mass_points = sorted(mass_points)

brazil_data = sorted(brazil_data, key=itemgetter(0))


"""print("With luminosity uncertainty")
print("Mass\tMean\tlow2s\tlow1s\thigh1s\thigh2s")
for (mass, mean, rms, low_2, low_1, high_1, high_2) in brazil_data:
    print("{0:.0f}\t{1:.02f}\t{2:.02f}\t{3:.02f}\t{4:.02f}\t{5:.02f}".format(
    mass, mean, mean-low_2, mean-low_1, mean+high_1, mean+high_2))
sys.exit(0)"""

brazil_data = [(x, m/fb2, r/fb2, a/fb2, b/fb2, c/fb2, d/fb2) for x, m, r, a, b, c, d in brazil_data]

with open(op.join(args.output_dir, "b_{0}.txt".format(job_id)), "w") as out_file:
    out_file.write("{0}\n".format(particle))
    for x, m, r, a, b, c,d in brazil_data:
        out_file.write("e:{0}:{1}\n".format(x, m))

canvas = TCanvas(particle, particle, 0, 0, 600, 550)
canvas.SetLogy(True)
#canvas.SetLeftMargin(0.15)
gPad.SetTicky(2)
gPad.SetTickx(1)


x = array("d", list(map(itemgetter(0), brazil_data)))
mean = array("d", list(map(itemgetter(1), brazil_data)))
xerr = array("d", [0,]*len(brazil_data))
rms = array("d", list(map(itemgetter(2), brazil_data)))
rms2 = array("d", [2*r for r in rms])
low_2sigma = array("d", list(map(itemgetter(3), brazil_data)))
low_1sigma = array("d", list(map(itemgetter(4), brazil_data)))
high_1sigma = array("d", list(map(itemgetter(5), brazil_data)))
high_2sigma = array("d", list(map(itemgetter(6), brazil_data)))

mg = TMultiGraph()
#mg.SetTitle(title)
mg.SetTitle(job_id)

brazil_yellow = TGraphAsymmErrors(len(brazil_data), x, mean, xerr, xerr, low_2sigma, high_2sigma)
brazil_yellow.SetFillColor(5)
brazil_yellow.SetFillStyle(1001)
brazil_yellow.SetLineColor(3)
brazil_yellow.SetLineWidth(10)
brazil_yellow.SetMarkerColor(3)


brazil_green = TGraphAsymmErrors(len(brazil_data), x, mean, xerr, xerr, low_1sigma, high_1sigma)
brazil_green.SetFillColor(3)
brazil_green.SetFillStyle(1001)
brazil_green.SetLineColor(3)
brazil_green.SetLineWidth(3)


mg.Add(brazil_yellow)
mg.Add(brazil_green)
mg.Draw("a3")

yaxistitle = "#sigma #times A #times BR [pb]"
if particle == "qbh":
    yaxistitle = "#sigma #times A [pb]"

mg.GetXaxis().SetTitle("m_{%s} [GeV]" % particle_symbol)
mg.GetYaxis().SetTitle(yaxistitle)
mg.GetYaxis().SetRangeUser(*y_limits)
mg.GetYaxis().SetTitleOffset(1.35)
mg.GetXaxis().SetTitleOffset(1.2)
mg.GetXaxis().SetNdivisions(508)

brazil_line = TGraph(len(brazil_data), x, mean)
brazil_line.SetLineStyle(7)
brazil_line.SetLineColor(1)
brazil_line.Draw("same")

data_mass_limit_pairs = []
with open(data_cl_file, "r") as data_file:
    for line in data_file.readlines():
        data_mass_limit_pairs.append(tuple(map(float, line.split(":"))))

with open(op.join(args.output_dir, "b_{0}.txt".format(job_id)), "a") as out_file:
    for a, b in data_mass_limit_pairs:
        out_file.write("o:{0}:{1}\n".format(a, b/fb2))

data_x, data_y = zip(*sorted(data_mass_limit_pairs, key=itemgetter(0)))
data_y = [dy/fb2 for dy in data_y]
data_line = TGraph(len(data_x), array("d", data_x), array("d", data_y))
data_line.SetLineStyle(1)
data_line.SetLineColor(1)
data_line.SetLineWidth(2)
data_line.SetMarkerStyle(8)
data_line.SetMarkerSize(0.75)
data_line.SetMarkerColor(1)
data_line.Draw("samePL")

print(mass_points)
theory_x = list(mass_points)
theory_y = []
for m in theory_x:
    root_file = TFile.Open(op.join(args.data_dir, sim_file.format(m)))
    nominal = root_file.GetDirectory("Nominal")
    hist = nominal.Get(sim_hist.format(m))
    theory_y.append(hist.Integral()/30/1000)
theory_line = TGraph(len(theory_x), array("d", theory_x), array("d", theory_y))
theory_line.SetLineStyle(9)
theory_line.SetLineColor(4)
theory_line.SetLineWidth(2)
theory_line.Draw("same")

x_intersect = 0
y_intersect = 0

for x1, x2, data1, theory1, data2, theory2 in zip(theory_x, theory_x[1:], data_y, theory_y, data_y[1:], theory_y[1:]):
    if data1 < theory1 and data2 > theory2:
        data1, theory1, data2, theory2 = math.log(data1), math.log(theory1), math.log(data2), math.log(theory2)
        print("Setting intersect")
        x = (theory1-data1)/((data2-data1-theory2+theory1)/(x2-x1))
        x_intersect = x1+x
        y_intersect = math.exp(theory1 + x*(theory2-theory1)/(x2-x1))

print("observed x intersect: {0}".format(x_intersect))
print("observed y intersect: {0}".format(y_intersect))

intersect_line = TGraph(1, array("d", [x_intersect,]), array("d", [y_intersect,]))
intersect_line.SetMarkerStyle(8)
intersect_line.SetMarkerSize(0.6)
intersect_line.SetMarkerColor(2)
#intersect_line.Draw("same P")

label = ROOT.TText()
label.SetNDC()
label.SetTextSize(0.04)
label.SetTextFont(42)
#label.DrawText(0.13, 0.85, "Observed data/theory intersect: {0:.0f} GeV".format(x_intersect))




exp_x_intersect = 0
exp_y_intersect = 0

for x1, x2, sim1, theory1, sim2, theory2 in zip(theory_x, theory_x[1:], mean, theory_y, mean[1:], theory_y[1:]):
    if sim1 < theory1 and sim2 > theory2:
        sim1, theory1, sim2, theory2 = math.log(sim1), math.log(theory1), math.log(sim2), math.log(theory2)
        print("Setting intersect")
        x = (theory1-sim1)/((sim2-sim1-theory2+theory1)/(x2-x1))
        exp_x_intersect = x1+x
        exp_y_intersect = math.exp(theory1 + x*(theory2-theory1)/(x2-x1))

print("expected x intersect: {0}".format(exp_x_intersect))
print("expected y intersect: {0}".format(exp_y_intersect))

intersect_line1 = TGraph(1, array("d", [exp_x_intersect,]), array("d", [exp_y_intersect,]))
intersect_line1.SetMarkerStyle(8)
intersect_line1.SetMarkerSize(0.6)
intersect_line1.SetMarkerColor(2)
#intersect_line1.Draw("same P")

label = ROOT.TText()
label.SetNDC()
label.SetTextSize(0.04)
label.SetTextFont(42)
#label.DrawText(0.13, 0.8, "Expected data/theory intersect: {0:.0f} GeV".format(exp_x_intersect))

text_x = 0.6

ATLAS_label = ROOT.TText()
ATLAS_label.SetNDC()
ATLAS_label.SetTextSize(0.04)
ATLAS_label.SetTextFont(72)
#ATLAS_label.DrawText(0.575, 0.7, "ATLAS preliminary")
ATLAS_label.DrawText(text_x, 0.84, "ATLAS internal")


label = ROOT.TLatex()
label.SetNDC()
label.SetTextSize(0.04)
label.SetTextFont(42)
#label.DrawLatex(0.575, 0.645, "#sqrt{s} = 13 TeV, 70fb^{-1}")
label.DrawLatex(text_x, 0.785, "#sqrt{s} = 13 TeV, %sfb^{-1}" % str(fb))


label = ROOT.TLatex()
label.SetNDC()
label.SetTextSize(0.04)
label.SetTextFont(42)
#label.DrawLatex(0.575, 0.645, "#sqrt{s} = 13 TeV, 70fb^{-1}")
label.DrawLatex(text_x, 0.73, "|y*| < %s" % str(rapidity))

legend = TLegend(0.125, 0.125, 0.6, 0.3)
legend.SetMargin(0.15)
legend.SetBorderSize(0)
legend.SetFillStyle(0)
legend.AddEntry(theory_line, particle_symbol, "l")
legend.AddEntry(data_line, "Observed 95% CL upper limit", "lp")
legend.AddEntry(brazil_line, "Expected 95% CL upper limit", "l")
legend.AddEntry(brazil_yellow, "Expected #pm1#sigma and #pm2#sigma")
legend.Draw()

canvas.SaveAs(op.join(args.output_dir, "brazil-{0}.png".format(job_id)))
canvas.SaveAs(op.join(args.output_dir, "brazil-{0}.pdf".format(job_id)))

with open(op.join(args.output_dir, "limits.txt"), "w") as limit_file:
    limit_file.write("Observed mass limit: {0} GeV\n".format(x_intersect))
    limit_file.write("Observed cross section limit: {0} pb\n".format(y_intersect))
    limit_file.write("Expected mass limit: {0} GeV\n".format(exp_x_intersect))
    limit_file.write("Expected cross section limit: {0} pb\n".format(exp_y_intersect))

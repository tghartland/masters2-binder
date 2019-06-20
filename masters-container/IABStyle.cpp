#include "IABStyle.h"

namespace IABstyles {

void enter() {
    // pause before finishing or lose canvas
    char s;
    std::cout << std::endl << " Hit <enter> to continue" << std::endl;
    // cin.get(s);
}

void global_style() {
    gStyle->SetCanvasColor(0);
    gStyle->SetCanvasBorderMode(0);
    gStyle->SetPadColor(0);
    gStyle->SetPadBorderMode(0);
    gStyle->SetFrameBorderMode(0);

    gStyle->SetTitleColor(1); // 0
    gStyle->SetTitleBorderSize(1);
    gStyle->SetTitleX(0.10);
    gStyle->SetTitleY(0.94);
    gStyle->SetTitleW(0.5);
    gStyle->SetTitleH(0.06);

    gStyle->SetLabelOffset(1e-04);
    gStyle->SetLabelSize(0.2);

    gStyle->SetStatColor(0);
    gStyle->SetStatBorderSize(1);
    gStyle->SetStatX(0.975);
    gStyle->SetStatY(0.95);
    gStyle->SetStatW(0.20);
    gStyle->SetStatH(0.10);

    gStyle->SetErrorX(0.0); // Horizontal error bar size
}

void canvas_style(TCanvas *c, float left_margin, float right_margin, float top_margin, float bottom_margin, int canvas_color,
                  int frame_color) {
    c->SetLeftMargin(left_margin);
    c->SetRightMargin(right_margin);
    c->SetTopMargin(top_margin);
    c->SetBottomMargin(bottom_margin);
    c->SetFillColor(canvas_color);
    c->SetFrameFillColor(frame_color);
    c->SetBorderMode(0);
    c->SetBorderSize(1);
    c->SetFrameBorderMode(0);
}

void h1_style(TH1 *h, int line_width, int line_color, int line_style, int fill_style, int fill_color, float y_min, float y_max, int ndivx,
              int ndivy, int marker_style, int marker_color, float marker_size, int optstat) {
    h->SetLineWidth(line_width);
    h->SetLineColor(line_color);
    h->SetLineStyle(line_style);
    h->SetFillColor(fill_color);
    h->SetFillStyle(fill_style);
    h->SetMaximum(y_max);
    h->SetMinimum(y_min);
    h->GetXaxis()->SetNdivisions(ndivx);
    h->GetYaxis()->SetNdivisions(ndivy);
    h->SetMarkerStyle(marker_style);
    h->SetMarkerColor(marker_color);
    h->SetMarkerSize(marker_size);
    h->SetStats(optstat);
    h->SetLabelFont(63, "X");      // 42
    h->SetLabelFont(63, "Y");      // 42
    h->SetLabelOffset(0.005, "X"); // D=0.005
    h->SetLabelOffset(0.005, "Y"); // D=0.005
    h->SetLabelSize(12, "X");
    h->SetLabelSize(12, "Y");
    h->SetTitleOffset(0.8, "X");
    h->SetTitleOffset(1.00, "Y");
    h->SetTitleSize(0.06, "X");
    h->SetTitleSize(0.06, "Y");
    h->SetLabelFont(63, "X");
    h->SetLabelFont(63, "Y");
    h->SetTitle(0);
}

void h1_style(TGraph *h, int line_width, int line_color, int line_style, int fill_style, int fill_color, float y_min, float y_max,
              int ndivx, int ndivy, int marker_style, int marker_color, float marker_size, int optstat) {
    h->SetLineWidth(line_width);
    h->SetLineColor(line_color);
    h->SetLineStyle(line_style);
    h->SetFillColor(fill_color);
    h->SetFillStyle(fill_style);
    h->SetMaximum(y_max);
    h->SetMinimum(y_min);
    h->GetXaxis()->SetNdivisions(ndivx);
    h->GetYaxis()->SetNdivisions(ndivy);
    h->SetMarkerStyle(marker_style);
    h->SetMarkerColor(marker_color);
    h->SetMarkerSize(marker_size);
    h->SetTitle(0);
    h->GetXaxis()->SetLabelFont(63);      // 42
    h->GetYaxis()->SetLabelFont(63);      // 42
    h->GetXaxis()->SetLabelOffset(0.005); // D=0.005
    h->GetYaxis()->SetLabelOffset(0.005); // D=0.005
    h->GetXaxis()->SetLabelSize(12);
    h->GetYaxis()->SetLabelSize(12);
    h->GetXaxis()->SetTitleOffset(0.8);
    h->GetYaxis()->SetTitleOffset(0.8);
    h->GetXaxis()->SetTitleSize(0.06);
    h->GetYaxis()->SetTitleSize(0.06);
    optstat = 0;
}

void h1_style(TGraphErrors *h, int line_width, int line_color, int line_style, int fill_style, int fill_color, float y_min, float y_max,
              int ndivx, int ndivy, int marker_style, int marker_color, float marker_size, int optstat) {
    h->SetLineWidth(line_width);
    h->SetLineColor(line_color);
    h->SetLineStyle(line_style);
    h->SetFillColor(fill_color);
    h->SetFillStyle(fill_style);
    h->SetMaximum(y_max);
    h->SetMinimum(y_min);
    h->GetXaxis()->SetNdivisions(ndivx);
    h->GetYaxis()->SetNdivisions(ndivy);

    h->SetMarkerStyle(marker_style);
    h->SetMarkerColor(marker_color);
    h->SetMarkerSize(marker_size);
    h->SetTitle(0);
    optstat = 0;

    h->GetXaxis()->SetLabelFont(63);      // 42
    h->GetYaxis()->SetLabelFont(63);      // 42
    h->GetXaxis()->SetLabelOffset(0.005); // D=0.005
    h->GetYaxis()->SetLabelOffset(0.005); // D=0.005
    h->GetXaxis()->SetLabelSize(12);
    h->GetYaxis()->SetLabelSize(12);
    h->GetXaxis()->SetTitleOffset(0.8);
    h->GetYaxis()->SetTitleOffset(0.8);
    h->GetXaxis()->SetTitleSize(0.06);
    h->GetYaxis()->SetTitleSize(0.06);

    h->SetTitle(0);
}
}

#ifndef IABstyles_h
#define IABstyles_h
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <sstream>
#include <string>
#include <iostream>
#include <fstream>

#include "TMath.h"
#include "TFile.h"
#include "TTree.h"
#include "TBrowser.h"
#include "TH1.h"
#include "TH2.h"
#include "TChain.h"
#include "TString.h"
#include "TSystem.h"
#include "TROOT.h"
#include "TStopwatch.h"
#include "TH1F.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TF1.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TGraph.h"
#include "TGraphErrors.h"
#include "TLatex.h"
#include "TString.h"

namespace IABstyles{

	int Scolor=kBlack;
	//int S2color=kMagenta;
	//int B1color=kGreen;
	//int B2color=kCyan;
	int lWidth=3;

    
    void enter();
    
    void global_style();
    
    void canvas_style(TCanvas *c, float left_margin=0.15, float right_margin=0.05, float top_margin=0.05, float bottom_margin=0.15, int canvas_color=0, int frame_color=0);
        
    void h1_style(TH1 *h,
    	int line_width=3,
        int line_color=1,
        int line_style=1, 
        int fill_style=1001,
        int fill_color=50,
        float y_min=-1111.,
        float y_max=-1111.,
        int ndivx=510,
        int ndivy=510,
        int marker_style=20,
        int marker_color=1,
        float marker_size=1.2,
        int optstat=0);
            
    void h1_style(TGraph *h,
            int line_width=3,
            int line_color=1,
            int line_style=1, 
            int fill_style=1001,
            int fill_color=50,
            float y_min=-1111.,
            float y_max=-1111.,
            int ndivx=505,
            int ndivy=505,
            int marker_style=20,
            int marker_color=1,
            float marker_size=1.2,
            int optstat=0); 
            
    void h1_style(TGraphErrors *h,
        	int line_width=3,
        	int line_color=1,
        	int line_style=1, 
        	int fill_style=1001,
        	int fill_color=50,
        	float y_min=-1111.,
        	float y_max=-1111.,
        	int ndivx=505,
        	int ndivy=505,
        	int marker_style=20,
        	int marker_color=1,
        	float marker_size=1.2,
        	int optstat=0);             
                
}
#endif

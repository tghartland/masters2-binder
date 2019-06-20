#include <fstream>
#include <iomanip>
#include <iostream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
//#include <string>
#include <vector>

#include "IABStyle.h"
//#include "IABFunctions.h"
#include "TDatime.h"

#include "TMinuit.h"
#include "TRandom3.h"

#include "Math/GaussIntegrator.h"
#include "Math/IFunction.h"
#include "Math/IParamFunction.h"
#include "Math/WrappedParamFunction.h"
#include "Math/WrappedTF1.h"
#include "TBox.h"
#include "TGaxis.h"
#include "THStack.h"
#include "TLegend.h"

#include "TGraphErrors.h"
#include "TVectorD.h"

#include "TColor.h"

using std::left;
using std::right;

using namespace std;

// c++ class used for integrating a function in root (yes it is complicated)
class MyMassSpectrum : public ROOT::Math::IParametricFunctionOneDim {
  private:
    const double *pars;

  public:
    // this method is the actual function. In this case x is the mass
    // scale =x/sqrt{s}
    double DoEval(double x) const {

        Double_t scale = x / 14.;
        Double_t arg1 = pars[0] * TMath::Power((1. - scale), pars[1]);
        Double_t arg2 = pars[2] + pars[3] * TMath::Log(scale);
        Double_t arg3 = TMath::Power(scale, arg2);
        
        return (arg1 * arg3);
    }

    // implementation that allows you to set and change paramaters
    double DoEvalPar(double x, const double *p) const {

        Double_t scale = x / 14.;
        Double_t arg1 = p[0] * TMath::Power((1. - scale), p[1]);
        Double_t arg2 = p[2] + p[3] * TMath::Log(scale);
        Double_t arg3 = TMath::Power(scale, arg2);

        
        return (arg1 * arg3);
    }

    ROOT::Math::IBaseFunctionOneDim *Clone() const { return new MyMassSpectrum(); }

    const double *Parameters() const { return pars; }

    void SetParameters(const double *p) { pars = p; }

    unsigned int NPar() const { return 5; }
};

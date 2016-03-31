/*
// @HEADER
// ***********************************************************************
//
//       xSDKTrilinos: Extreme-scale Software Development Kit Package
//                 Copyright (2016) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Alicia Klinvex    (amklinv@sandia.gov)
//                    James Willenbring (jmwille@sandia.gov)
//                    Michael Heroux    (maherou@sandia.gov)         
//
// ***********************************************************************
// @HEADER
*/

#include "Tpetra_DefaultPlatform.hpp"
#include "Tpetra_Vector.hpp"
#include "BelosMinresSolMgr.hpp"
#include "BelosTpetraAdapter.hpp"

// Specify types used in this example.
typedef double                                       Scalar;
typedef int                                          LO;
typedef int                                          GO;
typedef Teuchos::ScalarTraits<Scalar>::magnitudeType Magnitude;
typedef Teuchos::Comm<int>                           Comm;
typedef Tpetra::MultiVector<Scalar,LO,GO>            MV;
typedef Tpetra::Vector<Scalar,LO,GO>                 Vector;
typedef Tpetra::Operator<Scalar,LO,GO>               OP;
typedef Tpetra::Map<LO,GO>                           Map;
typedef Tpetra::Import<LO,GO>                        Import;
typedef Tpetra::DefaultPlatform::DefaultPlatformType Platform;
typedef Belos::MultiVecTraits<Scalar,MV>             MVT;
typedef Belos::LinearProblem<Scalar,MV,OP>           LinearProblem;
typedef Belos::MinresSolMgr<Scalar,MV,OP>            MinresSolMgr;

using Teuchos::RCP;
using Teuchos::rcp;
using Teuchos::ParameterList;
using Teuchos::Array;
using Teuchos::ArrayRCP;
using std::vector;
using std::cout;
using std::endl;

// Define a class for our user-defined operator, which is the 2D Poisson matrix.
class MyOp : public virtual OP {
public:
  // Constructor
  MyOp (const int nx, const int ny, const RCP<const Comm> comm);

  // Destructor
  virtual ~MyOp() {}

  // Returns the maps
  RCP<const Map> getDomainMap() const { return sourceMap_; }
  RCP<const Map> getRangeMap() const { return sourceMap_; }

  // Computes Y = alpha Op X + beta Y
  // TODO: I do not currently use alpha and beta...but I should
  void apply (const MV& X, MV& Y,
         Teuchos::ETransp mode = Teuchos::NO_TRANS,
         Scalar alpha = Teuchos::ScalarTraits<Scalar>::one (),
         Scalar beta = Teuchos::ScalarTraits<Scalar>::zero ()) const;

  // Whether the operator supports applying the transpose
  bool hasTransposeApply() const { return true; }

private:
  int nx_, ny_;
  RCP<const Map> sourceMap_, targetMap_;
  RCP<const Comm> comm_;
  RCP<const Import> importer_;
};


int main (int argc, char *argv[])
{
  // Initialize MPI.
  Teuchos::oblackholestream blackhole;
  Teuchos::GlobalMPISession mpiSession (&argc, &argv, &blackhole);

  // Get the default communicator and node
  RCP<const Comm> comm = Tpetra::DefaultPlatform::getDefaultPlatform ().getComm ();
  const int myRank = comm->getRank ();

  // Get parameters from command-line processor
  Scalar tol = 1e-5;
  int maxit = 100;
  int nx = 3;
  int ny = 3;
  int nrhs = 1;
  bool verbose = false;
  Teuchos::CommandLineProcessor cmdp (false, true);
  cmdp.setOption ("nx", &nx, "Number of grid points along the x axis.");
  cmdp.setOption ("ny", &ny, "Number of grid points along the y axis.");
  cmdp.setOption ("maxit", &maxit, "Maximum number of iterations used for solver.");
  cmdp.setOption ("tolerance", &tol, "Relative residual used for solver.");
  cmdp.setOption ("nrhs", &nrhs, "Number of right hand sides.");
  cmdp.setOption ("verbose","quiet", &verbose,
                  "Whether to print a lot of info or a little bit.");
  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
    return -1;
  }

  // Construct the operator.
  RCP<MyOp> A = rcp (new MyOp(nx,ny,comm));

  // Construct the initial guess and set of RHS
  RCP<MV> X = rcp(new Vector(A->getDomainMap(),false));
  RCP<MV> B = rcp(new Vector(A->getRangeMap(),false));
  MVT::MvInit(*X);
  MVT::MvRandom(*B);

  // Construct the linear problem
  RCP<LinearProblem> problem = rcp (new LinearProblem(A,X,B));
  problem->setProblem();

  // Set the parameters
  RCP<ParameterList> pList = rcp (new ParameterList());
  pList->set("Convergence Tolerance", tol);
  pList->set("Maximum Iterations", maxit);
  if(verbose) {
    pList->set("Verbosity", Belos::IterationDetails + Belos::TimingDetails + Belos::StatusTestDetails);
    pList->set("Output Frequency", 1);
  }

  // Construct the solver manager
  MinresSolMgr solver(problem, pList);

  // Solve the linear system
  solver.solve();

  // Retrieve the residual
  RCP<MV> R = MVT::Clone(*B, nrhs);
  problem->computeCurrResVec(R.get(), problem->getLHS().get(), problem->getRHS().get());

  // Compute the residual norms
  vector<Magnitude> resNorms(nrhs), rhsNorms(nrhs);
  MVT::MvNorm(*R,resNorms);
  MVT::MvNorm(*B,rhsNorms);
  for(int i=0; i<nrhs && myRank == 0; i++)
    cout << "Residual norm of RHS #" << i << ": " << resNorms[i] / rhsNorms[i] << endl;

  return 0;
}


//
// Constructor
//
MyOp::MyOp (const int nx, const int ny, const RCP<const Comm> comm) :
  nx_(nx),
  ny_(ny),
  comm_(comm)
{
  int numProcs = comm_->getSize();
  if (2*numProcs > ny) { // ny must be >= 2*numProc (to avoid degenerate cases)
    ny_ = 2*numProcs;
    cout << " Increasing ny to " << ny_ << " to avoid degenerate distribution on " << numProcs << " processors.\n";
  }

  // Create the map describing the distribution of rows over MPI processes
  sourceMap_ = rcp(new Map(nx_*ny_,0,comm_));

  if(numProcs > 1)
  {
    int myRank = comm_->getRank();

    // Determine the rows of the vector required by this MPI process for a mat-vec
    Array<int> requiredIndices;
    int myFirstRow = sourceMap_->getMinGlobalIndex();
    int myLastRow = sourceMap_->getMaxGlobalIndex();
    if(myRank > 0)
    {
      requiredIndices.resize(nx_);
      for(int i=0; i<nx_; i++)
        requiredIndices[i] = myFirstRow + i - nx_;
    }
    if(myRank < numProcs-1)
    {
      int currentSize = requiredIndices.size();
      requiredIndices.resize(currentSize+nx_);
      for(int i=0; i<nx_; i++)
        requiredIndices[currentSize+i] = myLastRow + 1 + i;
    }

    // Create a map describing the rows of the vector required by this MPI process
    int numMapEntries = 2*(numProcs-1)*nx_;
    targetMap_ = rcp(new Map(numMapEntries,requiredIndices(),0,comm_));

    // Create the importer
    importer_ = rcp(new Import(sourceMap_,targetMap_));
  }
}


//
// A non-threaded apply
//
void MyOp::apply(const MV& X, MV& Y,
         Teuchos::ETransp mode,
         Scalar alpha,
         Scalar beta) const
{
  int nLocalRows = sourceMap_->getNodeNumElements();
  int n = sourceMap_->getGlobalNumElements();
  int nCols = X.getNumVectors();
  int myFirstRow = sourceMap_->getMinGlobalIndex();
  int myLastRow = sourceMap_->getMaxGlobalIndex();

  Y.scale(beta);

  // Get a view of X's raw data
  ArrayRCP<ArrayRCP<const Scalar> > Xdata = X.get2dView();

  // Compute the local part of the mat-vec
  for(int c=0; c < nCols; c++)
  {
    for(int r=0; r < nLocalRows; r++)
    {
      Y.sumIntoLocalValue(r,c,4*alpha*Xdata[c][r]);

      if((myFirstRow+r)%nx_ > 0 && r > 0)
        Y.sumIntoLocalValue(r,c,-alpha*Xdata[c][r-1]);
      if((myFirstRow+r+1)%nx_ > 0 && r < nLocalRows - 1)
        Y.sumIntoLocalValue(r,c,-alpha*Xdata[c][r+1]);
    }

    for(int i=0; i+nx_ < nLocalRows; i++)
    {
      Y.sumIntoLocalValue(i,c,-alpha*Xdata[c][nx_+i]);
      Y.sumIntoLocalValue(nx_+i,c,-alpha*Xdata[c][i]);
    }
  }

  // Import the rest of the required data, if necessary, and resume computation
  int numProcs = comm_->getSize();
  if(numProcs > 1)
  {
    RCP<MV> redistX = rcp(new MV(targetMap_,nCols,false));
    redistX->doImport(X,*importer_,Tpetra::REPLACE);

    ArrayRCP<ArrayRCP<const Scalar> > redistXdata = redistX->get2dView();

    int startingIndex2=0;
    for(int c=0; c < nCols; c++)
    {
      if(myFirstRow > 0)
      {
        startingIndex2=nx_;

        if(myFirstRow % nx_ > 0)
          Y.sumIntoLocalValue(0,c,-alpha*redistXdata[c][nx_-1]);
        for(int i=0; i < nx_; i++)
          Y.sumIntoLocalValue(i,c,-alpha*redistXdata[c][i]);
      }
      if(myLastRow < n-1)
      {
        if((myLastRow+1) % nx_ > 0)
          Y.sumIntoLocalValue(nLocalRows-1,c,-alpha*redistXdata[c][startingIndex2]);
        for(int i=1; i <= nx_; i++)
          Y.sumIntoLocalValue(nLocalRows-i,c,-alpha*redistXdata[c][startingIndex2+nx_-i]);
      }
    }
  }
}

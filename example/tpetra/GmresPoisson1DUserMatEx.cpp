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

#include "Tpetra_DefaultPlatform.hpp"
#include "Tpetra_Vector.hpp"
#include "Tpetra_RowMatrix.hpp"
#include "BelosPseudoBlockGmresSolMgr.hpp"
#include "BelosTpetraAdapter.hpp"
#include "Ifpack2_Factory.hpp"

// Specify types used in this example.
typedef double                                       Scalar;
typedef int                                          LO;
typedef int                                          GO;
typedef Teuchos::ScalarTraits<Scalar>::magnitudeType Magnitude;
typedef Teuchos::Comm<int>                           Comm;
typedef Tpetra::MultiVector<Scalar,LO,GO>            MV;
typedef Tpetra::Vector<Scalar,LO,GO>                 Vector;
typedef Tpetra::RowMatrix<Scalar,LO,GO>              RowMatrix;
typedef Tpetra::RowGraph<LO,GO>                      RowGraph;
typedef Tpetra::Operator<Scalar,LO,GO>               OP;
typedef Tpetra::Map<LO,GO>                           Map;
typedef Tpetra::Import<LO,GO>                        Import;
typedef Tpetra::Export<LO,GO>                        Export;
typedef Tpetra::DefaultPlatform::DefaultPlatformType Platform;
typedef RowMatrix::node_type                         Node;
typedef Belos::MultiVecTraits<Scalar,MV>             MVT;
typedef Belos::LinearProblem<Scalar,MV,OP>           LinearProblem;
typedef Belos::PseudoBlockGmresSolMgr<Scalar,MV,OP>  GmresSolMgr;
typedef Ifpack2::Preconditioner<Scalar,LO,GO>        Prec;

using Teuchos::RCP;
using Teuchos::rcp;
using Teuchos::ParameterList;
using Teuchos::Array;
using Teuchos::ArrayRCP;
using Teuchos::ArrayView;
using Tpetra::global_size_t;
using std::vector;
using std::cout;
using std::endl;


// Define a class for our user-defined graph, required by the user-defined matrix
class MyGraph : public virtual RowGraph {
public:
  // Constructor
  MyGraph(const int n, const RCP<const Comm> comm);

  // Destructor
	virtual ~MyGraph() {}

	// Other required methods
  RCP<const Comm> getComm() const { return comm_; }
  RCP<Node> getNode() const { return rowMap_->getNode(); }
  RCP<const Map> getRowMap() const { return rowMap_; }
  RCP<const Map> getColMap() const { return colMap_; }
  RCP<const Map> getDomainMap() const { return rowMap_; }
  RCP<const Map> getRangeMap() const { return rowMap_; }
  RCP<const Import> getImporter() const { return importer_; }
  RCP<const Export> getExporter() const { return exporter_; }
  global_size_t getGlobalNumRows() const { return rowMap_->getGlobalNumElements(); }
  global_size_t getGlobalNumCols() const { return getGlobalNumRows(); }
  size_t getNodeNumRows() const { return rowMap_->getNodeNumElements(); }
  size_t getNodeNumCols() const { return colMap_->getNodeNumElements(); }
  GO getIndexBase() const { return rowMap_->getIndexBase(); }
  global_size_t getGlobalNumEntries() const { return 3*getGlobalNumRows()-2; }
  size_t getNodeNumEntries() const;
  size_t getNumEntriesInGlobalRow(GO globalRow) const;
  size_t getNumEntriesInLocalRow(LO localRow) const
      { return getNumEntriesInGlobalRow(rowMap_->getGlobalElement(localRow)); }
  global_size_t getGlobalNumDiags() const { return getGlobalNumRows(); }
  size_t getNodeNumDiags() const { return getNodeNumRows(); }
  size_t getGlobalMaxNumRowEntries() const { return 3; }
  size_t getNodeMaxNumRowEntries() const { return 3; }
  bool hasColMap() const { return true; }
  bool isLocallyIndexed() const { return true; }
  bool isGloballyIndexed() const { return false; }
  void getGlobalRowCopy(GO globalRow, const ArrayView<GO> &indices,
      size_t &numIndices) const;
  void getLocalRowCopy(LO localRow, const ArrayView<LO> &indices,
      size_t &numIndices) const
      { getGlobalRowCopy(rowMap_->getGlobalElement(localRow),indices,numIndices); }
  bool isLowerTriangular() const { return false; }
  bool isUpperTriangular() const { return false; }
  bool isFillComplete() const { return true; }

private:
  RCP<const Map> rowMap_, colMap_;
  RCP<const Comm> comm_;
  RCP<Import> importer_;
  RCP<Export> exporter_;
};

// Define a class for our user-defined matrix, which is the 1D Poisson matrix.
class MyMat : public virtual RowMatrix {
public:

  // Constructor
  MyMat (const int n, const RCP<const Comm> comm);

  // Destructor
  virtual ~MyMat() {}

  // Other required methods implemented by MyGraph
  RCP<const Comm> getComm() const { return graph_->getComm(); }
  RCP<Node> getNode() const { return graph_->getNode(); }
  RCP<const Map> getRowMap() const { return graph_->getRowMap(); }
  RCP<const Map> getColMap() const { return graph_->getColMap(); }
  RCP<const Map> getDomainMap() const { return graph_->getDomainMap(); }
  RCP<const Map> getRangeMap() const { return graph_->getRangeMap(); }
  global_size_t getGlobalNumRows() const { return graph_->getGlobalNumRows(); }
  global_size_t getGlobalNumCols() const { return graph_->getGlobalNumCols(); }
  size_t getNodeNumRows() const { return graph_->getNodeNumRows(); }
  size_t getNodeNumCols() const { return graph_->getNodeNumCols(); }
  GO getIndexBase() const { return graph_->getIndexBase(); }
  global_size_t getGlobalNumEntries() const { return graph_->getGlobalNumEntries(); }
  size_t getNodeNumEntries() const { return graph_->getNodeNumEntries(); }
  size_t getNumEntriesInGlobalRow(GO globalRow) const { return graph_->getNumEntriesInGlobalRow(globalRow); }
  size_t getNumEntriesInLocalRow(LO localRow) const { return graph_->getNumEntriesInLocalRow(localRow); }
  global_size_t getGlobalNumDiags() const { return graph_->getGlobalNumDiags(); }
  size_t getNodeNumDiags() const { return graph_->getNodeNumDiags(); }
  size_t getGlobalMaxNumRowEntries() const { return graph_->getGlobalMaxNumRowEntries(); }
  size_t getNodeMaxNumRowEntries() const { return graph_->getNodeMaxNumRowEntries(); }
  bool hasColMap() const { return graph_->hasColMap(); }
  bool isLowerTriangular() const { return graph_->isLowerTriangular(); }
  bool isUpperTriangular() const { return graph_->isUpperTriangular(); }
  bool isLocallyIndexed() const { return graph_->isLocallyIndexed(); }
  bool isGloballyIndexed() const { return graph_->isGloballyIndexed(); }
  bool isFillComplete() const { return graph_->isFillComplete(); }
  void getGlobalRowCopy(GO globalRow, const ArrayView<GO> &indices,
      const ArrayView<Scalar> &values, size_t &numEntries) const;
  void getLocalRowCopy(LO localRow, const ArrayView<LO> & indices,
      const ArrayView<Scalar> &values, size_t &numEntries) const
      { getGlobalRowCopy(graph_->getRowMap()->getGlobalElement(localRow),indices,values,numEntries); }
  void getLocalDiagCopy(Vector &diag) const { diag.putScalar(2); }
  void leftScale(const Vector &x)
      { throw std::runtime_error("MyMat does not support leftScale."); }
  void rightScale(const Vector &x)
      { throw std::runtime_error("MyMat does not support rightScale."); }
  Magnitude getFrobeniusNorm() const { return 6*getGlobalNumRows()-2; }
  RCP<const RowGraph> getGraph() const { return graph_; }
  bool hasTransposeApply() const { return true; }

  // It doesn't make sense to support row views when we do not explicitly store the matrix
  bool supportsRowViews() const { return false; }
  void getGlobalRowView(GO globalRow, ArrayView<const GO> &indices,
		  ArrayView<const Scalar> &values) const
      { throw std::runtime_error("MyMat does not support row views."); }
  void getLocalRowView(LO localRow, ArrayView<const LO> &indices,
		  ArrayView<const Scalar> &values) const
      { throw std::runtime_error("MyMat does not support row views."); }

  // Computes Y = alpha Op X + beta Y
  void apply (const MV& X, MV& Y,
         Teuchos::ETransp mode = Teuchos::NO_TRANS,
         Scalar alpha = Teuchos::ScalarTraits<Scalar>::one (),
         Scalar beta = Teuchos::ScalarTraits<Scalar>::zero ()) const;

private:
  RCP<MyGraph> graph_;
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
  int n = 10;
  int nrhs = 1;
  bool verbose = false;
  bool withPrec = true;
  Teuchos::CommandLineProcessor cmdp (false, true);
  cmdp.setOption ("n", &n, "Number of rows.");
  cmdp.setOption ("maxit", &maxit, "Maximum number of iterations used for solver.");
  cmdp.setOption ("tolerance", &tol, "Relative residual used for solver.");
  cmdp.setOption ("nrhs", &nrhs, "Number of right hand sides.");
  cmdp.setOption ("verbose","quiet", &verbose,
                  "Whether to print a lot of info or a little bit.");
  cmdp.setOption ("withPrec","noPrec", &withPrec, "Whether to use a preconditioner.");
  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
    return -1;
  }

  // Construct the operator.
  RCP<MyMat> A = rcp (new MyMat(n,comm));

  // Construct the preconditioner
  RCP<Prec> M;
  ParameterList ifpackPL;
  ifpackPL.set("fact: ilut level-of-fill", 5.0);
  M = Ifpack2::Factory::create<RowMatrix> ("ILUT", A);
  M->setParameters(ifpackPL);
  M->initialize ();
  M->compute ();

  // Construct the initial guess and set of RHS
  RCP<MV> X = rcp(new Vector(A->getDomainMap(),false));
  RCP<MV> B = rcp(new Vector(A->getRangeMap(),false));
  MVT::MvInit(*X);
  MVT::MvRandom(*B);

  MVT::MvInit(*B,1);

  // Construct the linear problem
  RCP<LinearProblem> problem = rcp (new LinearProblem(A,X,B));
  if(withPrec)
    problem->setLeftPrec(M);
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
  GmresSolMgr solver(problem, pList);

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
// MyGraph constructor
//
MyGraph::MyGraph(const int n, const RCP<const Comm> comm) :
  comm_(comm)
{
  // Create the map describing the distribution of rows over MPI processes
  rowMap_ = rcp(new Map(n,0,comm_));

  // Determine the rows of the vector required by this MPI process for a mat-vec
  Array<int> requiredIndices;
  int myFirstRow = rowMap_->getMinGlobalIndex();
  int myLastRow = rowMap_->getMaxGlobalIndex();
  for(int i=myFirstRow; i<=myLastRow; i++)
    requiredIndices.push_back(i);
  if(myFirstRow > 0)
    requiredIndices.push_back(myFirstRow-1);
  if(myLastRow < n-1)
    requiredIndices.push_back(myLastRow+1);

  int numProcs = comm_->getSize();
  int numMapEntries = n+2*(numProcs-1);
  colMap_ = rcp(new Map(numMapEntries,requiredIndices(),0,comm_));

  // Create the importer and exporter
  importer_ = rcp(new Import(rowMap_,colMap_));
  exporter_ = rcp(new Export(rowMap_,rowMap_));
}


//
// Returns the number of entries on this node
//
size_t MyGraph::getNodeNumEntries() const
{
  size_t numEntries = 3*getNodeNumRows();
  if(rowMap_->getMinGlobalIndex() == 0)
    numEntries--;
  if(rowMap_->getMaxGlobalIndex() == getGlobalNumRows()-1)
    numEntries--;
  return numEntries;
}


//
// Returns the number of entries in a given row
//
size_t MyGraph::getNumEntriesInGlobalRow(LO globalRow) const
{
  if(globalRow == 0 || globalRow == getGlobalNumRows()-1)
    return 2;
  return 3;
}


//
// Returns a copy of a given row
//
void MyGraph::getGlobalRowCopy(LO globalRow, const ArrayView<LO> &indices,
    size_t &numIndices) const
{
  if(globalRow == 0)
  {
    indices[0] = colMap_->getLocalElement(globalRow);
    indices[1] = colMap_->getLocalElement(globalRow+1);
    numIndices = 2;
  }
  else if(globalRow == getGlobalNumRows()-1)
  {
    indices[0] = colMap_->getLocalElement(globalRow-1);
    indices[1] = colMap_->getLocalElement(globalRow);
    numIndices = 2;
  }
  else
  {
    indices[0] = colMap_->getLocalElement(globalRow-1);
    indices[1] = colMap_->getLocalElement(globalRow);
    indices[2] = colMap_->getLocalElement(globalRow+1);
    numIndices = 3;
  }
}


//
// MyMat constructor
//
MyMat::MyMat (const int n, const RCP<const Comm> comm)
{
  graph_ = rcp(new MyGraph(n,comm));
}


//
// Returns a copy of a given row
//
void MyMat::getGlobalRowCopy(LO globalRow, const ArrayView<LO> & indices,
    const ArrayView<Scalar> &values, size_t &numEntries) const
{
  graph_->getGlobalRowCopy(globalRow,indices,numEntries);
  if(globalRow == 0)
  {
    values[0] = 2;
    values[1] = -1;
  }
  else if(globalRow == getGlobalNumRows()-1)
  {
    values[0] = -1;
    values[1] = 2;
  }
  else
  {
    values[0] = -1;
    values[1] = 2;
    values[2] = -1;
  }
}


//
// A non-threaded apply
//
void MyMat::apply(const MV& X, MV& Y,
         Teuchos::ETransp mode,
         Scalar alpha,
         Scalar beta) const
{
  int nLocalRows = X.getLocalLength();
  int nCols = X.getNumVectors();

  Y.scale(beta);

  int numProcs = getComm()->getSize();
  // Parallel matvec
  if(numProcs > 1)
  {
    // Import the necessary data
    RCP<MV> redistX = rcp(new MV(graph_->getColMap(),nCols,false));
    redistX->doImport(X,*graph_->getImporter(),Tpetra::REPLACE);

    // Get a view of X's raw data
    ArrayRCP<ArrayRCP<const Scalar> > Xdata = redistX->get2dView();

    for(int c=0; c<nCols; c++)
    {
      for(int r=0; r<nLocalRows; r++)
      {
        if(r > 0)
          Y.sumIntoLocalValue(r,c,-alpha*Xdata[c][r-1]);
        Y.sumIntoLocalValue(r,c,2*alpha*Xdata[c][r]);
        if(r < nLocalRows-1)
          Y.sumIntoLocalValue(r,c,-alpha*Xdata[c][r+1]);
      }

      // Contribution from previous process
      // TODO: FIX THE BUG HERE!
      GO firstRow = graph_->getRowMap()->getMinGlobalIndex();
      GO lastRow = graph_->getRowMap()->getMaxGlobalIndex();
      if(firstRow > 0)
        Y.sumIntoLocalValue(0,c,-alpha*Xdata[c][nLocalRows]);
      // Contribution from next process
      if(lastRow < getGlobalNumRows()-1)
      {
        int index = graph_->getColMap()->getLocalElement(lastRow);
        Y.sumIntoLocalValue(nLocalRows-1,c,-alpha*Xdata[c][index]);
      }
    } // end for each column
  } // end if parallel
  // Sequential matvec
  else
  {
    ArrayRCP<ArrayRCP<const Scalar> > Xdata = X.get2dView();
    for(int c=0; c<nCols; c++)
    {
      for(int r=0; r<nLocalRows; r++)
      {
        Y.sumIntoLocalValue(r,c,2*alpha*Xdata[c][r]);
        if(r > 0)
          Y.sumIntoLocalValue(r,c,-alpha*Xdata[c][r-1]);
        if(r < nLocalRows-1)
          Y.sumIntoLocalValue(r,c,-alpha*Xdata[c][r+1]);
      }
    }
  }
} // end apply

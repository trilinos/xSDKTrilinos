#include "Tpetra_DefaultPlatform.hpp"
#include "Tpetra_Vector.hpp"
#include "BelosMinresSolMgr.hpp"
#include "BelosTpetraAdapter.hpp"

// Specify types used in this example.
typedef double                                       Scalar;
typedef int                                          LO;
typedef int                                          GO;
typedef Teuchos::ScalarTraits<Scalar>::magnitudeType Magnitude;
typedef Tpetra::MultiVector<Scalar,LO,GO>            MV;
typedef Tpetra::Vector<Scalar,LO,GO>                 Vector;
typedef Tpetra::Operator<Scalar,LO,GO>               OP;
typedef Tpetra::Map<LO,GO>                           Map;
typedef Tpetra::DefaultPlatform::DefaultPlatformType Platform;
typedef Belos::MultiVecTraits<Scalar,MV>             MVT;
typedef Belos::LinearProblem<Scalar,MV,OP>           LinearProblem;
typedef Belos::MinresSolMgr<Scalar,MV,OP>            MinresSolMgr;

using Teuchos::RCP;
using Teuchos::rcp;
using Teuchos::ParameterList;
using std::vector;
using std::cout;
using std::endl;


// Define a class for our user-defined operator, which is a diagonal matrix.
class MyOp : public virtual OP {
public:
  // Constructor
  MyOp (RCP<Vector> diagonal) : diagonal_(diagonal) {}

  // Destructor
  virtual ~MyOp() {}

  // Returns the maps
  RCP<const Map> getDomainMap() const { return diagonal_->getMap(); }
  RCP<const Map> getRangeMap() const { return diagonal_->getMap(); }

  // Computes Y = alpha Op X + beta Y
  void apply (const MV& X, MV& Y,
         Teuchos::ETransp mode = Teuchos::NO_TRANS,
         Scalar alpha = Teuchos::ScalarTraits<Scalar>::one (),
         Scalar beta = Teuchos::ScalarTraits<Scalar>::zero ()) const
  {
    Y.elementWiseMultiply(alpha,*diagonal_,X,beta);
  }

  // Whether the operator supports applying the transpose
  bool hasTransposeApply() const { return true; }

private:
  RCP<Vector> diagonal_;
};


int main (int argc, char *argv[])
{
  // Initialize MPI.
  Teuchos::oblackholestream blackhole;
  Teuchos::GlobalMPISession mpiSession (&argc, &argv, &blackhole);

  // Get the default communicator and node
  RCP<const Teuchos::Comm<int> > comm = Tpetra::DefaultPlatform::getDefaultPlatform ().getComm ();
  const int myRank = comm->getRank ();

  // Get parameters from command-line processor
  Scalar tol = 1e-5;
  int maxit = 100;
  GO n = 100;
  int nrhs = 1;
  bool verbose = false;
  Teuchos::CommandLineProcessor cmdp (false, true);
  cmdp.setOption ("n", &n, "Number of rows of our operator.");
  cmdp.setOption ("maxit", &maxit, "Maximum number of iterations used for solver.");
  cmdp.setOption ("tolerance", &tol, "Relative residual used for solver.");
  cmdp.setOption ("nrhs", &nrhs, "Number of right hand sides.");
  cmdp.setOption ("verbose","quiet", &verbose,
                  "Whether to print a lot of info or a little bit.");
  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
    return -1;
  }

  // Create a map describing the distribution of our matrix
  RCP<Map> rowMap = rcp(new Map(n,0,comm));

  // Generate a random diagonal
  RCP<Vector> diag = rcp(new Vector(rowMap,false));
  MVT::MvRandom(*diag);

  // Construct the operator.
  RCP<MyOp> A = rcp (new MyOp(diag));

  // Construct the initial guess and set of RHS
  RCP<MV> X = MVT::Clone(*diag, nrhs);
  RCP<MV> B = MVT::Clone(*diag, nrhs);
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
  RCP<MV> R = MVT::Clone(*diag, nrhs);
  problem->computeCurrResVec(R.get(), problem->getLHS().get(), problem->getRHS().get());

  // Compute the residual norms
  vector<Magnitude> resNorms(nrhs), rhsNorms(nrhs);
  MVT::MvNorm(*R,resNorms);
  MVT::MvNorm(*B,rhsNorms);
  for(int i=0; i<nrhs && myRank == 0; i++)
    cout << "Residual norm of RHS #" << i << ": " << resNorms[i] / rhsNorms[i] << endl;

  return 0;
}

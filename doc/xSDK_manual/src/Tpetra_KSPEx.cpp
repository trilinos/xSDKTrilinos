#include "BelosConfigDefs.hpp"
#include "BelosLinearProblem.hpp"
#include "BelosTpetraAdapter.hpp"
#include "BelosPETScSolMgr.hpp"
#include "Ifpack2_Factory.hpp"
#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_StandardCatchMacros.hpp"
#include "Tpetra_CrsMatrix.hpp"
#include "Tpetra_DefaultPlatform.hpp"
#include "Tpetra_MultiVector.hpp"
#include "MatrixMarket_Tpetra.hpp"

int main(int argc, char *argv[]) {
  typedef Tpetra::MultiVector<>                   MV;
  typedef Tpetra::Operator<>                      OP;
  typedef Tpetra::CrsMatrix<>              CrsMatrix;
  typedef Ifpack2::Preconditioner<>             Prec;

  using Teuchos::ParameterList;
  using Teuchos::RCP;
  using Teuchos::rcp;

  // Initialize MPI
  Teuchos::oblackholestream blackhole;
  Teuchos::GlobalMPISession mpiSession (&argc, &argv, &blackhole);

  // Get the default communicator
  RCP<const Teuchos::Comm<int> > comm = Tpetra::DefaultPlatform::getDefaultPlatform().getComm();

  // Read the command line arguments
  int numrhs = 2;
  int maxiters = 100;
  std::string filename("cage4.mtx");
  double tol = 1.0e-5;
  Teuchos::CommandLineProcessor cmdp(false,false);
  cmdp.setOption("filename",&filename,"Filename for test matrix.");
  cmdp.setOption("tol",&tol,"Relative residual tolerance used by GMRES solver.");
  cmdp.setOption("num-rhs",&numrhs,"Number of right-hand sides to be solved for.");
  cmdp.setOption("max-iters",&maxiters,"Maximum number of iterations per linear system.");
  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
    return -1;
  }

  // Get the matrix from a file
  RCP<CrsMatrix> A = Tpetra::MatrixMarket::Reader<CrsMatrix>::readSparseFile(filename,comm);

  // Create a random RHS and set the initial guess to 0
  RCP<MV> B = rcp(new MV(A->getRowMap(),numrhs,false));
  RCP<MV> X = rcp(new MV(A->getRowMap(),numrhs,false));
  RCP<MV> trueX = rcp(new MV(A->getRowMap(),numrhs,false));
  trueX->randomize();
  A->apply(*trueX,*B);
  X->putScalar(0);

  // Construct preconditioner
  Ifpack2::Factory factory;
  RCP<Prec> M = factory.create("RELAXATION", A.getConst());
  ParameterList ifpackParams;
  ifpackParams.set("relaxation: type","Jacobi");
  M->setParameters(ifpackParams);
  M->initialize();
  M->compute();

  // Create parameter list for the Belos solver manager
  ParameterList belosList;
  belosList.set( "Maximum Iterations", maxiters );       // Maximum number of iterations allowed
  belosList.set( "Convergence Tolerance", tol );         // Relative convergence tolerance requested
  belosList.set( "Verbosity", Belos::IterationDetails ); // Print convergence information
  belosList.set( "Solver", "bcgs" );                     // Use BiCGStab as the linear solver

  // Construct a preconditioned linear problem
  RCP<Belos::LinearProblem<double,MV,OP> > problem
    = rcp( new Belos::LinearProblem<double,MV,OP>( A, X, B ) );
  problem->setLeftPrec( M );
  problem->setProblem();

  // Create an iterative solver manager
  RCP< Belos::PETScSolMgr<double,MV,OP> > solver
    = rcp( new Belos::PETScSolMgr<double,MV,OP>(problem, rcp(&belosList,false)) );

  // Perform solve
  solver->solve();
}

/*
   This example demonstrates how to use a PETSc KSP linear solver with
   an Ifpack2  to compute the
   eigenpairs of a PETSc matrix.

   The PETSc matrix is a 2D 5-point Laplace operator stored in AIJ format.
   This matrix is wrapped as an PETScAIJMatrix.  The associated eigenvalue
   problem is solved using Anasazi.

   To invoke this example, use something like:

       mpirun -np 5 ./AnasaziTest.exe -mx 150 -my 150
*/

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
  typedef double                            ST;
  typedef Teuchos::ScalarTraits<ST>        SCT;
  typedef SCT::magnitudeType                MT;
  typedef Tpetra::MultiVector<>             MV;
  typedef Tpetra::Operator<>                OP;
  typedef Belos::MultiVecTraits<ST,MV>     MVT;
  typedef Belos::OperatorTraits<ST,MV,OP>  OPT;
  typedef Tpetra::CrsMatrix<>              CrsMatrix;
  typedef Ifpack2::Preconditioner<>        Prec;

  using Teuchos::ParameterList;
  using Teuchos::RCP;
  using Teuchos::rcp;

  // ************************* Initialize MPI **************************
  Teuchos::oblackholestream blackhole;
  Teuchos::GlobalMPISession mpiSession (&argc, &argv, &blackhole);

  // ************** Get the default communicator and node **************
  RCP<const Teuchos::Comm<int> > comm = Tpetra::DefaultPlatform::getDefaultPlatform().getComm();
  const int myRank = comm->getRank();

  bool verbose = true;
  bool success = true;
  bool proc_verbose = false;
  bool leftprec = true;      // left preconditioning or right.
  int frequency = -1;        // frequency of status test output.
  int numrhs = 1;            // number of right-hand sides to solve for
  int maxiters = -1;         // maximum number of iterations allowed per linear system
  std::string filename("/home/amklinv/matrices/cage4.mtx");
  MT tol = 1.0e-5;           // relative residual tolerance

  // ***************** Read the command line arguments *****************
  Teuchos::CommandLineProcessor cmdp(false,false);
  cmdp.setOption("verbose","quiet",&verbose,"Print messages and results.");
  cmdp.setOption("left-prec","right-prec",&leftprec,"Left preconditioning or right.");
  cmdp.setOption("frequency",&frequency,"Solvers frequency for printing residuals (#iters).");
  cmdp.setOption("filename",&filename,"Filename for test matrix.  Acceptable file extensions: *.hb,*.mtx,*.triU,*.triS");
  cmdp.setOption("tol",&tol,"Relative residual tolerance used by GMRES solver.");
  cmdp.setOption("num-rhs",&numrhs,"Number of right-hand sides to be solved for.");
  cmdp.setOption("max-iters",&maxiters,"Maximum number of iterations per linear system (-1 = adapted to problem/block size).");
  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
    return -1;
  }
  if (!verbose)
    frequency = -1;  // reset frequency if test is not verbose
  proc_verbose = verbose && (myRank==0); /* Only print on the zero processor */

  // ************************* Get the problem *************************
  RCP<CrsMatrix> A = Tpetra::MatrixMarket::Reader<CrsMatrix>::readSparseFile(filename,comm);
  RCP<MV> B = rcp(new MV(A->getRowMap(),numrhs,false));
  RCP<MV> X = rcp(new MV(A->getRowMap(),numrhs,false));
  OPT::Apply(*A, *X, *B);
  MVT::MvInit(*X);
  MVT::MvInit(*B,1);

  // ******************** Construct preconditioner *********************
  Ifpack2::Factory factory;
  RCP<Prec> M = factory.create("RELAXATION", A.getConst());
  ParameterList ifpackParams;
  ifpackParams.set("relaxation: type","Jacobi");
  M->setParameters(ifpackParams);
  M->initialize();
  M->compute();

  // ******* Create parameter list for the Belos solver manager ********
  const int NumGlobalElements = MVT::GetGlobalLength(*B);
  if (maxiters == -1)
    maxiters = NumGlobalElements - 1; // maximum number of iterations to run
  //
  ParameterList belosList;
  belosList.set( "Maximum Iterations", maxiters );       // Maximum number of iterations allowed
  belosList.set( "Convergence Tolerance", tol );         // Relative convergence tolerance requested
  if (numrhs > 1) {
    belosList.set( "Show Maximum Residual Norm Only", true );  // Show only the maximum residual norm
  }
  if (verbose) {
    belosList.set( "Verbosity", Belos::Errors + Belos::Warnings +
		   Belos::TimingDetails + Belos::StatusTestDetails );
    if (frequency > 0)
      belosList.set( "Output Frequency", frequency );
  }
  else
    belosList.set( "Verbosity", Belos::Errors + Belos::Warnings + Belos::FinalSummary );

  // ************ Construct a preconditioned linear problem ************
  RCP<Belos::LinearProblem<double,MV,OP> > problem
    = rcp( new Belos::LinearProblem<double,MV,OP>( A, X, B ) );
  if (leftprec) {
    problem->setLeftPrec( M );
  }
  else {
    problem->setRightPrec( M );
  }
  bool set = problem->setProblem();
  if (set == false) {
    if (proc_verbose)
      std::cout << std::endl << "ERROR:  Belos::LinearProblem failed to set up correctly!" << std::endl;
    return -1;
  }

  // **************** Create an iterative solver manager ***************
  RCP< Belos::PETScSolMgr<double,MV,OP> > solver
    = rcp( new Belos::PETScSolMgr<double,MV,OP>(problem, rcp(&belosList,false)) );
  solver->setCLA(argc,argv);

  // ****************** Start the block CG iteration *******************
  if (proc_verbose) {
    std::cout << std::endl << std::endl;
    std::cout << "Dimension of matrix: " << NumGlobalElements << std::endl;
    std::cout << "Number of right-hand sides: " << numrhs << std::endl;
    std::cout << "Max number of Krylov iterations: " << maxiters << std::endl;
    std::cout << "Relative residual tolerance: " << tol << std::endl;
    std::cout << std::endl;
  }

  // ************************** Perform solve **************************
  Belos::ReturnType ret = solver->solve();

  // ********************* Compute actual residuals ********************
  bool badRes = false;
  std::vector<double> actual_resids( numrhs );
  std::vector<double> rhs_norm( numrhs );
  RCP<MV> resid = MVT::Clone(*X, numrhs);
  OPT::Apply( *A, *X, *resid );
  MVT::MvAddMv( -1.0, *resid, 1.0, *B, *resid );
  MVT::MvNorm( *resid, actual_resids );
  MVT::MvNorm( *B, rhs_norm );
  if (proc_verbose) {
    std::cout<< "---------- Actual Residuals (normalized) ----------"<<std::endl<<std::endl;
    for ( int i=0; i<numrhs; i++) {
      double actRes = actual_resids[i]/rhs_norm[i];
      std::cout<<"Problem "<<i<<" : \t"<< actRes <<std::endl;
      if (actRes > tol) badRes = true;
    }
  }

if (ret!=Belos::Converged || badRes) {
  success = false;
  if (proc_verbose)
    std::cout << std::endl << "ERROR:  Belos did not converge!" << std::endl;
} else {
  success = true;
  if (proc_verbose)
    std::cout << std::endl << "SUCCESS:  Belos converged!" << std::endl;
}

return success ? EXIT_SUCCESS : EXIT_FAILURE;
}

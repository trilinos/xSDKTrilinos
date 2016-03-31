#include "BelosTpetraAdapter.hpp"
#include "BelosPETScSolMgr.hpp"
#include "Ifpack2_Factory.hpp"
#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_ParameterList.hpp"
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

  Teuchos::oblackholestream blackhole;
  Teuchos::GlobalMPISession mpiSession (&argc, &argv, &blackhole);
  RCP<const Teuchos::Comm<int> > comm = Tpetra::DefaultPlatform::getDefaultPlatform().getComm();
  const int myRank = comm->getRank();

  double tol = 1e-6;
  std::string filename("/home/amklinv/matrices/cage4.mtx");
  std::string ksptype("gmres");
  Teuchos::CommandLineProcessor cmdp(false,false);
  cmdp.setOption("filename",&filename,"Filename for test matrix.");
  cmdp.setOption("tol",&tol,"Relative residual tolerance.");
  cmdp.setOption("ksptype",&ksptype,"Type of linear solver to be used.");
  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
    return -1;
  }

  RCP<CrsMatrix> A = Tpetra::MatrixMarket::Reader<CrsMatrix>::readSparseFile(filename,comm);
  RCP<MV> B = rcp(new MV(A->getRowMap(),1,false));
  RCP<MV> X = rcp(new MV(A->getRowMap(),1,false));
  MVT::MvInit(*X);
  MVT::MvInit(*B,1);

  Ifpack2::Factory factory;
  RCP<Prec> M = factory.create("RELAXATION", A.getConst());
  ParameterList ifpackParams;
  ifpackParams.set("relaxation: type","Jacobi");
  M->setParameters(ifpackParams);
  M->initialize();
  M->compute();

  ParameterList belosList;
  belosList.set( "Maximum Iterations", 100 );
  belosList.set( "Convergence Tolerance", tol );
  belosList.set( "Solver", ksptype );

  RCP<Belos::LinearProblem<double,MV,OP> > problem
    = rcp( new Belos::LinearProblem<double,MV,OP>( A, X, B ) );
  problem->setLeftPrec( M );
  problem->setProblem();

  RCP< Belos::PETScSolMgr<double,MV,OP> > solver
    = rcp( new Belos::PETScSolMgr<double,MV,OP>(problem, rcp(&belosList,false)) );
  solver->solve();
}

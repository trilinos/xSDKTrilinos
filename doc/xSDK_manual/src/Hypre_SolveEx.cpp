#include "Tpetra_Map.hpp"
#include "Tpetra_CrsMatrix.hpp"
#include "Tpetra_DefaultPlatform.hpp"
#include "Ifpack2_Preconditioner.hpp"
#include "Ifpack2_Hypre.hpp"
#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_StandardCatchMacros.hpp"

int main(int argc, char *argv[]) {
  using Teuchos::Array;
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::ParameterList;
  using Ifpack2::FunctionParameter;
  using Ifpack2::Hypre::Prec;
  using Ifpack2::Hypre::Solver;

  typedef Tpetra::CrsMatrix<>::scalar_type Scalar;
  typedef Tpetra::CrsMatrix<>::local_ordinal_type LO;
  typedef Tpetra::CrsMatrix<>::global_ordinal_type GO;
  typedef Tpetra::CrsMatrix<>::node_type Node;
  typedef Tpetra::DefaultPlatform::DefaultPlatformType Platform;
  typedef Tpetra::CrsMatrix<Scalar> CrsMatrix;
  typedef Tpetra::MultiVector<Scalar> MV;
  typedef Tpetra::Operator<Scalar> OP;
  typedef Ifpack2::Preconditioner<Scalar> Preconditioner;
  typedef Tpetra::Map<> Map;

  // Initialize the MPI session
  Teuchos::oblackholestream blackhole;
  Teuchos::GlobalMPISession mpiSession(&argc,&argv,&blackhole);

  // Get the default communicator
  Platform &platform = Tpetra::DefaultPlatform::getDefaultPlatform();
  RCP<const Teuchos::Comm<int> > comm = platform.getComm();

  // Get parameters from command-line processor
  int nx = 10;
  Scalar tol = 1e-6;
  bool verbose = false;
  Teuchos::CommandLineProcessor cmdp(false,true);
  cmdp.setOption("nx",&nx, "Number of mesh points in x direction.");
  cmdp.setOption("tolerance",&tol, "Relative residual used for solver.");
  cmdp.setOption("verbose","quiet",&verbose, "Whether to print a lot of info or a little bit.");
  if(cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
    return -1;
  }

  // Create the row map
  int n = nx*nx;
  RCP<Map> map = rcp(new Map(n,0,comm));

  // Create the 2D Laplace operator
  RCP<CrsMatrix> A = rcp(new CrsMatrix(map,5));
  for(LO i = 0; i<nx; i++) {
    for(LO j = 0; j<nx; j++) {
      GO row = i*nx+j;
      if(!map->isNodeGlobalElement(row))
        continue;

      Array<LO> indices;
      Array<Scalar> values;

      if(i > 0) {
        indices.push_back(row - nx);
        values.push_back(-1.0);
      }
      if(i < nx-1) {
        indices.push_back(row + nx);
        values.push_back(-1.0);
      }
      indices.push_back(row);
      values.push_back(4.0);
      if(j > 0) {
        indices.push_back(row-1);
        values.push_back(-1.0);
      }
      if(j < nx-1) {
        indices.push_back(row+1);
        values.push_back(-1.0);
      }
      A->insertGlobalValues(row,indices,values);
    }
  }
  A->fillComplete();

  // Create the initial guess and right hand side
  RCP<MV> trueX = rcp(new MV(A->getRowMap(),1,false));
  RCP<MV> X = rcp(new MV(A->getRowMap(),1));
  RCP<MV> B = rcp(new MV(A->getRowMap(),1,false));
  trueX->randomize();
  A->apply(*trueX,*B);

  // Create the parameters for hypre
  RCP<FunctionParameter> functs[10];
  functs[0] = rcp(new FunctionParameter(Prec, &HYPRE_BoomerAMGSetPrintLevel, 1));  // print AMG solution info
  functs[1] = rcp(new FunctionParameter(Prec, &HYPRE_BoomerAMGSetCoarsenType, 6)); // Falgout coarsening
  functs[2] = rcp(new FunctionParameter(Prec, &HYPRE_BoomerAMGSetRelaxType, 6));   // Sym GS/Jacobi hybrid
  functs[3] = rcp(new FunctionParameter(Prec, &HYPRE_BoomerAMGSetNumSweeps, 1));   // Sweeps on each level
  functs[4] = rcp(new FunctionParameter(Prec, &HYPRE_BoomerAMGSetTol, 0.0));       // Conv tolerance zero
  functs[5] = rcp(new FunctionParameter(Prec, &HYPRE_BoomerAMGSetMaxIter, 1));     // Do only one iteration!
  functs[6] = rcp(new FunctionParameter(Solver, &HYPRE_PCGSetMaxIter, 1000));      // Maximum iterations
  functs[7] = rcp(new FunctionParameter(Solver, &HYPRE_PCGSetTol, tol));           // Convergence tolerance
  functs[8] = rcp(new FunctionParameter(Solver, &HYPRE_PCGSetTwoNorm, 1));         // Use the two-norm as the stopping criteria
  functs[9] = rcp(new FunctionParameter(Solver, &HYPRE_PCGSetPrintLevel, 2));      // Print solve info

  // Create the hypre solver
  RCP<Preconditioner> prec = rcp(new Ifpack2::Ifpack2_Hypre<Scalar,LO,GO,Node>(A));
  ParameterList hypreList;
  hypreList.set("SolveOrPrecondition", Solver);
  hypreList.set("Solver", Ifpack2::Hypre::PCG);
  hypreList.set("Preconditioner", Ifpack2::Hypre::BoomerAMG);
  hypreList.set("SetPreconditioner", true);
  hypreList.set("NumFunctions", 10);
  hypreList.set<RCP<FunctionParameter>*>("Functions", functs);
  prec->setParameters(hypreList);
  prec->compute();

  // Perform solve
  prec->apply(*B,*X);
}

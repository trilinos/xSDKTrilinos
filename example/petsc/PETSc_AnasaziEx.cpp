#include "petscksp.h"

#include "AnasaziBasicEigenproblem.hpp"
#include "AnasaziConfigDefs.hpp"
#include "AnasaziTpetraAdapter.hpp"
#include "AnasaziTraceMinDavidsonSolMgr.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Tpetra_PETScAIJMatrix.hpp"

  using Teuchos::RCP;
  using Teuchos::rcp;
  using std::cout;
  using std::endl;

  typedef Tpetra::PETScAIJMatrix<>           PETScAIJMatrix;
  typedef PETScAIJMatrix::scalar_type               Scalar;
  typedef PETScAIJMatrix::local_ordinal_type        LO;
  typedef PETScAIJMatrix::global_ordinal_type       GO;
  typedef PETScAIJMatrix::node_type                 Node;
  typedef Tpetra::Vector<Scalar,LO,GO,Node>         Vector;
  typedef Tpetra::Map<LO,GO,Node>                   Map;
  typedef Tpetra::Operator<Scalar,LO,GO,Node>       OP;
  typedef Tpetra::MultiVector<Scalar,LO,GO,Node>    MV;
  typedef Tpetra::Vector<Scalar,LO,GO,Node>         Vector;
  typedef Anasazi::Experimental::TraceMinDavidsonSolMgr<Scalar,MV,OP>  SolMgr;
  typedef Anasazi::BasicEigenproblem<Scalar,MV,OP>  Problem;
  typedef Anasazi::OperatorTraits<Scalar,MV,OP>     OPT;
  typedef Anasazi::MultiVecTraits<Scalar,MV>        MVT;

/* 
   This example demonstrates how to use a Trilinos eigensolver to compute the
   eigenpairs of a PETSc matrix.

   For information on configuring and building Trilinos with the PETSc aij
   interface enabled, please see EpetraExt's doxygen documentation at
   http://trilinos.sandia.gov/packages/epetraext, development version
   or release 9.0 or later.

   The PETSc matrix is a 2D 5-point Laplace operator stored in AIJ format.
   This matrix is wrapped as an PETScAIJMatrix.  The associated eigenvalue
   problem is solved using Anasazi.

   To invoke this example, use something like:

       mpirun -np 5 ./AnasaziTest.exe -m 150 -n 150
*/

static char help[] = "Demonstrates how to compute the eigenpairs of a PETSc matrix with \
a Trilinos eigensolver.\n\
Input parameters include:\n\
  -nev              : number of desired eigenvalues \n\
  -m <mesh_x>       : number of mesh points in x-direction\n\
  -n <mesh_n>       : number of mesh points in y-direction\n\n";

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat            A;        /* PETSc matrix */
  PetscInt       m = 50,n = 50; /* #mesh points in x & y directions, resp. */
  PetscInt       nev = 4;
  PetscErrorCode ierr;
  PetscInt rank=0;
  MPI_Comm comm;
  PetscInt Istart, Iend, Ii, i, j, J;
  PetscScalar v;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-m",&m,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-nev",&nev,PETSC_NULL);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n);CHKERRQ(ierr);
  ierr = MatSetType(A, MATAIJ);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(A,5,PETSC_NULL,5,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  PetscObjectGetComm( (PetscObject)A, &comm);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) printf("Matrix has %d (%dx%d) rows\n",m*n,m,n);

  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);

  for (Ii=Istart; Ii<Iend; Ii++) { 
    v = -1.0; i = Ii/n; j = Ii - i*n;  
    if (i>0)   {J = Ii - n; ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
    if (i<m-1) {J = Ii + n; ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
    if (j>0)   {J = Ii - 1; ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
    if (j<n-1) {J = Ii + 1; ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
    v = 4.0; ierr = MatSetValues(A,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Wrap the PETSc matrix as a PETScAIJMatrix. This is lightweight,
     i.e., no deep data copies. */
  RCP<PETScAIJMatrix> epA = rcp(new PETScAIJMatrix(A));

  /* Create an initial guess */
  RCP<MV> initGuess = rcp(new MV(epA->getDomainMap(),4,false));
  initGuess->randomize();

  /* Create an eigenproblem */
  RCP<Problem> problem = rcp(new Problem(epA,initGuess));
  problem->setNEV(nev);
  problem->setHermitian(true);
  problem->setProblem();

  /* Create the parameter list */
  Teuchos::ParameterList pl;
  pl.set("Verbosity", Anasazi::IterationDetails + Anasazi::FinalSummary);
  pl.set("Convergence Tolerance", 1e-6);

  /* Create an Anasazi eigensolver */
  RCP<SolMgr> solver = rcp(new SolMgr(problem, pl));

  /* Solve the problem to the specified tolerances */
  Anasazi::ReturnType returnCode = solver->solve();
  if (returnCode != Anasazi::Converged && rank == 0) {
    cout << "Anasazi::EigensolverMgr::solve() returned unconverged." << endl;
  }
  else if (rank == 0)
    cout << "Anasazi::EigensolverMgr::solve() returned converged." << endl;

  /* Get the eigenvalues and eigenvectors from the eigenproblem */
  Anasazi::Eigensolution<Scalar,MV> sol = problem->getSolution();
  std::vector<Anasazi::Value<Scalar> > evals = sol.Evals;
  RCP<MV> evecs = sol.Evecs;
  int numev = sol.numVecs;

  /* Compute the residual, just as a precaution */
  if (numev > 0) {
    Teuchos::SerialDenseMatrix<int,Scalar> T(numev,numev);
    MV tempvec(epA->getRowMap(), MVT::GetNumberVecs( *evecs ));
    std::vector<Scalar> normR(sol.numVecs);
    MV Kvec( epA->getRowMap(), MVT::GetNumberVecs( *evecs ) );

    OPT::Apply( *epA, *evecs, Kvec );
    MVT::MvTransMv( 1.0, Kvec, *evecs, T );
    MVT::MvTimesMatAddMv( -1.0, *evecs, T, 1.0, Kvec );
    MVT::MvNorm( Kvec, normR );

    if (rank == 0) {
      cout.setf(std::ios_base::right, std::ios_base::adjustfield);
      cout<<"Actual Eigenvalues (obtained by Rayleigh quotient) : "<<endl;
      cout<<"------------------------------------------------------"<<endl;
      cout<<std::setw(16)<<"Real Part"
          <<std::setw(16)<<"Error"<<endl;
      cout<<"------------------------------------------------------"<<endl;
      for (int i=0; i<numev; i++) {
        cout<<std::setw(16)<<T(i,i)
            <<std::setw(16)<<normR[i]/evals[i].realpart
            <<endl;
      }
      cout<<"------------------------------------------------------"<<endl;
    }
  }

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
} /*main*/

/* ***************************************************************** */

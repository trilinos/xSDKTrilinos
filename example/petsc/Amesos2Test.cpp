#include <Teuchos_ScalarTraits.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_oblackholestream.hpp>
#include <Teuchos_Tuple.hpp>
#include <Teuchos_VerboseObject.hpp>

#include <Tpetra_DefaultPlatform.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_CrsMatrix.hpp>

#include "Amesos2.hpp"
#include "Amesos2_Version.hpp"

#include "petscksp.h"
#include "Tpetra_PETScAIJMatrix.hpp"
#include "Tpetra_Vector.hpp"

  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::ArrayView;

  typedef double Scalar;
  typedef int LO;
  typedef int GO;

#ifdef HAVE_AMESOS2_EXPLICIT_INSTANTIATION
// Explicitly instantiate for RowMatrix (not done in Amesos2 by
// default, currently).
template class Amesos2::KLU2<Tpetra::RowMatrix<Scalar, LO, GO>,
			     Tpetra::MultiVector<Scalar, LO, GO> >;

#include <Amesos2_SolverCore_def.hpp>
// template class Amesos2::SolverCore<Amesos2::KLU2, Tpetra::RowMatrix<Scalar, LO, GO>, 
// 				   Tpetra::MultiVector<Scalar, LO, GO> >;
#endif // HAVE_AMESOS2_EXPLICIT_INSTANTIATION

  typedef Tpetra::CrsMatrix<Scalar,LO,GO> MAT;
  typedef Tpetra::MultiVector<Scalar,LO,GO> MV;

//  typedef Tpetra::CrsMatrix<>                       CrsMatrix;
//  typedef Tpetra::Tpetra_PETScAIJMatrix<>           PETScAIJMatrix;
//  typedef PETScAIJMatrix::scalar_type               Scalar;
//  typedef PETScAIJMatrix::local_ordinal_type        LO;
//  typedef PETScAIJMatrix::global_ordinal_type       GO;
//  typedef PETScAIJMatrix::node_type                 Node;
  typedef Tpetra::Vector<Scalar,LO,GO>         Vector;
  typedef Tpetra::Map<LO,GO>                   Map;
  typedef Tpetra::Operator<Scalar,LO,GO>       OP;
  typedef Tpetra::MultiVector<Scalar,LO,GO>    MV;
  typedef Tpetra::Vector<Scalar,LO,GO>         Vector;
  typedef Amesos2::Solver<Tpetra::RowMatrix<Scalar,LO,GO>,MV>        Solver;

/* 
   This example demonstrates how to apply a Trilinos preconditioner to a PETSc
   linear system.

   For information on configuring and building Trilinos with the PETSc aij
   interface enabled, please see EpetraExt's doxygen documentation at
   http://trilinos.sandia.gov/packages/epetraext, development version
   or release 9.0 or later.

   The PETSc matrix is a 2D 5-point Laplace operator stored in AIJ format.
   This matrix is wrapped as an Epetra_PETScAIJMatrix, and an ML AMG
   preconditioner is created for it.  The associated linear system is solved twice,
   the first time using AztecOO's preconditioned CG, the second time using PETSc's.

   To invoke this example, use something like:

       mpirun -np 5 ./TrilinosCouplings_petsc.exe -m 150 -n 150 -petsc_smoother -ksp_monitor_true_residual
*/

static char help[] = "Demonstrates how to solve a PETSc linear system with KSP\
and a Trilinos AMG preconditioner.  In particular, it shows how to wrap a PETSc\
AIJ matrix as a Tpetra matrix, create the AMG preconditioner, and wrap it as a\
shell preconditioner for a PETSc Krylov method.\
Input parameters include:\n\
  -random_exact_sol : use a random exact solution vector\n\
  -view_exact_sol   : write exact solution vector to stdout\n\
  -m <mesh_x>       : number of mesh points in x-direction\n\
  -n <mesh_n>       : number of mesh points in y-direction\n\n";

RCP<MAT> PETScAIJMatrixToTpetraCrsMatrix(Mat A);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Vec            x,b,u;  /* approx solution, RHS, exact solution */
  Mat            A;        /* linear system matrix */
  KSP            ksp;     /* linear solver context */
  PC             pc;
  PetscRandom    rctx;     /* random number generator context */
  PetscReal      norm;     /* norm of solution error */
  PetscInt       i,j,Ii,J,Istart,Iend,its;
  PetscInt       m = 50,n = 50; /* #mesh points in x & y directions, resp. */
  PetscErrorCode ierr;
  PetscBool     flg;
  PetscScalar    v,one = 1.0,neg_one = -1.0;
  PetscInt rank=0;
  MPI_Comm comm;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-m",&m,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);

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

  /* Wrap the PETSc matrix as a Tpetra_PETScAIJMatrix. This is lightweight,
     i.e., no deep data copies. */
//  RCP<PETScAIJMatrix> epA = rcp(new PETScAIJMatrix(A));
  RCP<MAT> epA = PETScAIJMatrixToTpetraCrsMatrix(A);

  ierr = VecCreate(PETSC_COMM_WORLD,&u);CHKERRQ(ierr);
  ierr = VecSetSizes(u,PETSC_DECIDE,m*n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(u);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&b);CHKERRQ(ierr); 

  ierr = VecDuplicate(b,&x);CHKERRQ(ierr);

  ierr = PetscOptionsHasName(PETSC_NULL,"-random_exact_sol",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rctx);CHKERRQ(ierr);
    ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
    ierr = VecSetRandom(u,rctx);CHKERRQ(ierr);
    ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);
  } else {
    ierr = VecSet(u,one);CHKERRQ(ierr);
  }
  ierr = MatMult(A,u,b);CHKERRQ(ierr);
  ierr = VecNorm(b,NORM_2,&norm);CHKERRQ(ierr);
  if (rank==0) printf("||b|| = %f\n",norm);

  /* Copy the PETSc vectors to Tpetra vectors. */
  PetscScalar *vals;
  ierr = VecGetArray(u,&vals);CHKERRQ(ierr);
  PetscInt length;
  ierr = VecGetLocalSize(u,&length);CHKERRQ(ierr);
  PetscScalar* valscopy = (PetscScalar*) malloc(length*sizeof(PetscScalar));
  memcpy(valscopy,vals,length*sizeof(PetscScalar));
  ierr = VecRestoreArray(u,&vals);CHKERRQ(ierr);
  ArrayView<PetscScalar> epuView(valscopy,length);
  RCP<Vector> epu = rcp(new Vector(epA->getRowMap(),epuView));
  RCP<Vector> epb = rcp(new Vector(epA->getRowMap()));
  epA->apply(*epu, *epb);

  /* Check norms of the Tpetra and PETSc vectors. */
  norm = epu->norm2();
  if (rank == 0) printf("||tpetra u||_2 = %f\n",norm);
  ierr = VecNorm(u,NORM_2,&norm);CHKERRQ(ierr);
  if (rank == 0) printf("||petsc u||_2  = %f\n",norm);
  norm = epb->norm2();
  if (rank == 0) printf("||tpetra b||_2 = %f\n",norm);
  ierr = VecNorm(b,NORM_2,&norm);CHKERRQ(ierr);
  if (rank == 0) printf("||petsc b||_2  = %f\n",norm);

  /* Create an Amesos2 linear solver */
  RCP<Solver> solver = Amesos2::create<Tpetra::RowMatrix<Scalar,LO,GO>,MV>("KLU2", epA, epu, epb);

  /* Perform a linear solve with Amesos2 */
  solver->symbolicFactorization().numericFactorization().solve();

  /* Perform a linear solve with PETSc's KSP */
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
  ierr = PCFactorSetMatSolverPackage(pc,MATSOLVERSUPERLU_DIST);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

  ierr = KSPView(ksp,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = VecAXPY(x,neg_one,u);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error %e iterations %D\n",
                     norm,its);CHKERRQ(ierr);

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);  ierr = MatDestroy(&A);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
} /*main*/

/* ***************************************************************** */

RCP<MAT> PETScAIJMatrixToTpetraCrsMatrix(Mat A)
{
  PetscErrorCode ierr;

  // Get the communicator
  RCP< Teuchos::Comm<int> > TrilinosComm;
#ifdef HAVE_MPI
  MPI_Comm PETScComm;
  PetscObjectGetComm( (PetscObject)A, &PETScComm);
  TrilinosComm = rcp(new Teuchos::MpiComm<int>(PETScComm));
#else
  TrilinosComm = rcp(new Teuchos::SerialComm<int>());
#endif 

  // Get information about the distribution from PETSc
  // Note that this is only valid for a block row distribution
  PetscInt numLocalRows, numLocalCols;
  ierr = MatGetLocalSize(A,&numLocalRows,&numLocalCols);
  PetscInt numGlobalRows, numGlobalCols;
  ierr = MatGetSize(A,&numGlobalRows,&numGlobalCols);

  // Create a Tpetra map reflecting this distribution
  RCP<Map> map = rcp(new Map(numGlobalRows,numLocalRows,0,TrilinosComm));

  // Create an array containing the number of entries in each row
  LO minLocalIndex = map->getMinGlobalIndex();
  Teuchos::ArrayRCP<size_t> ncolsPerRow(numLocalRows);
  for(int i=0; i < numLocalRows; i++)
  {
    ierr = MatGetRow(A,minLocalIndex+i,&numLocalCols,NULL,NULL);
    ncolsPerRow[i] = numLocalCols;
    ierr = MatRestoreRow(A,minLocalIndex+i,&numLocalCols,NULL,NULL);
  }

  // Create the matrix and set its values
  RCP<MAT> TrilinosMat = rcp(new MAT(map,ncolsPerRow,Tpetra::StaticProfile));
  const PetscInt * cols;
  const PetscScalar * vals;
  for(int i=0; i < numLocalRows; i++)
  {
    ierr = MatGetRow(A,i+minLocalIndex,&numLocalCols,&cols,&vals);
    Teuchos::ArrayView<const LO> colsToInsert(cols,numLocalCols);
    Teuchos::ArrayView<const Scalar> valsToInsert(vals,numLocalCols);
    TrilinosMat->insertGlobalValues(minLocalIndex+i,colsToInsert,valsToInsert);
    ierr = MatRestoreRow(A,minLocalIndex+i,&numLocalCols,&cols,&vals);
  }

  // Let the matrix know you're done changing it
  TrilinosMat->fillComplete();

  return TrilinosMat;
}

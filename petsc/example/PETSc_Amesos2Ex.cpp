//
// This driver constructs Laplace 2D as a PETSc matrix along with a PETSc Vec right
// hand side.  It then uses Amesos2 to solve the linear system via sparse factorization.
//
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
#include "Amesos2_KLU2.hpp"

#include "petscksp.h"
#include "Tpetra_PETScAIJMatrix.hpp"
#include "Tpetra_Vector.hpp"

int main(int argc,char **args)
{
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::ArrayView;

  typedef Tpetra::PETScAIJMatrix<>             PETScAIJMatrix;
  typedef PETScAIJMatrix::scalar_type          Scalar;
  typedef PETScAIJMatrix::local_ordinal_type   LO;
  typedef PETScAIJMatrix::global_ordinal_type  GO;
  typedef PETScAIJMatrix::node_type            Node;

  typedef Tpetra::CrsMatrix<Scalar,LO,GO>      CrsMatrix;
  typedef Tpetra::Vector<Scalar,LO,GO>         Vector;
  typedef Tpetra::Map<LO,GO>                   Map;
  typedef Tpetra::Operator<Scalar,LO,GO>       OP;
  typedef Tpetra::MultiVector<Scalar,LO,GO>    MV;
  typedef Tpetra::Vector<Scalar,LO,GO>         Vector;
  typedef Amesos2::Solver<CrsMatrix,MV>        Solver;

  Vec            x,b;         /* approx solution, RHS, exact solution */
  Mat            A;             /* linear system matrix */
  PetscRandom    rctx;          /* random number generator context */
  PetscInt       i,j,Ii,J,Istart,Iend;
  PetscInt       m = 50,n = 50; /* #mesh points in x & y directions, resp. */
  PetscErrorCode ierr;
  PetscScalar    v;
  PetscInt rank=0;
  MPI_Comm comm;
  PetscBool set;

  //
  // Start PETSc and get the problem dimensions via command line argument
  //
  PetscInitialize(&argc,&args,NULL,NULL);
//  ierr = PetscOptionsGetInt(PETSC_NULL,"-mx",&m,&set);
//  ierr = PetscOptionsGetInt(PETSC_NULL,"-my",&n,&set);

  //
  // Create the matrix
  //
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n);CHKERRQ(ierr);
  ierr = MatSetType(A, MATAIJ);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(A,5,PETSC_NULL,5,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  PetscObjectGetComm( (PetscObject)A, &comm);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

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

  //
  // Copy the matrix from a PETSc data structure to a Tpetra CrsMatrix
  // Note that this is actually copying the data rather than wrapping it
  //
  RCP<CrsMatrix> epA = xSDKTrilinos::deepCopyPETScAIJMatrixToTpetraCrsMatrix<Scalar,LO,GO,Node>(A);

  //
  // Create the solution vector and random right-hand-side
  //
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,m*n);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&b);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rctx);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
  ierr = VecSetRandom(b,rctx);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);

  //
  // Copy the PETSc vectors to Tpetra vectors
  //
  RCP<Vector> tpetraX = xSDKTrilinos::deepCopyPETScVecToTpetraVector<Scalar,LO,GO,Node>(x);
  RCP<Vector> tpetraB = xSDKTrilinos::deepCopyPETScVecToTpetraVector<Scalar,LO,GO,Node>(b);

  //
  // Create an Amesos2 linear solver
  //
  RCP<Solver> solver = Amesos2::create<CrsMatrix,MV>("KLU2", epA, tpetraX, tpetraB);
  solver->symbolicFactorization();
  solver->numericFactorization();

  //
  // Perform a linear solve with Amesos2
  //
  solver->solve();

  //
  // Terminate PETSc
  //
  ierr = PetscFinalize(); CHKERRQ(ierr);
  return 0;
} /*main*/

/* ***************************************************************** */

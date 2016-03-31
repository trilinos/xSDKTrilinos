#include <Teuchos_ScalarTraits.hpp>
#include <Teuchos_RCP.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Vector.hpp>
#include "Amesos2.hpp"
#include "Amesos2_Version.hpp"
#include "Amesos2_KLU2.hpp"

#include "petscksp.h"
#include "Tpetra_PETScAIJMatrix.hpp"

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
  typedef Amesos2::Solver<CrsMatrix,MV>        Solver;

  Vec            x,b;
  Mat            A;
  PetscRandom    rctx;
  PetscInt       i,j,Ii,J,Istart,Iend;
  PetscInt       m = 50,n = 50;
  PetscErrorCode ierr;
  PetscScalar    v;
  PetscInt rank=0;
  MPI_Comm comm;

  PetscInitialize(&argc,&args,NULL,NULL);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-mx",&m,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-my",&n,PETSC_NULL);CHKERRQ(ierr);

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

  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,m*n);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&b);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rctx);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
  ierr = VecSetRandom(b,rctx);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);

  RCP<CrsMatrix> tpetraA = xSDKTrilinos::deepCopyPETScAIJMatrixToTpetraCrsMatrix<Scalar,LO,GO,Node>(A);
  RCP<Vector> tpetraX = xSDKTrilinos::deepCopyPETScVecToTpetraVector<Scalar,LO,GO,Node>(x);
  RCP<Vector> tpetraB = xSDKTrilinos::deepCopyPETScVecToTpetraVector<Scalar,LO,GO,Node>(b);

  RCP<Solver> solver = Amesos2::create<CrsMatrix,MV>("KLU2", tpetraA, tpetraX, tpetraB);
  solver->symbolicFactorization();
  solver->numericFactorization();
  solver->solve();

  ierr = PetscFinalize(); CHKERRQ(ierr);
  return 0;
}

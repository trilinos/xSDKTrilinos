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

/*
   This example demonstrates how to use Ifpack2 and Belos with a PETSc Mat.

   The PETSc matrix has the same sparsity pattern as a 2D 5-point Laplace 
   operator, but with random entries on the diagonal.
   This matrix is wrapped as an PETScAIJMatrix.  The associated linear system
   is solved with Belos using an Ifpack2 preconditioner.
*/

#include "BelosConfigDefs.hpp"
#include "BelosLinearProblem.hpp"
#include "BelosTpetraAdapter.hpp"
#include "BelosMinresSolMgr.hpp"

#include "Ifpack2_Factory.hpp"

#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_StandardCatchMacros.hpp"

#include "Tpetra_CrsMatrix.hpp"
#include "Tpetra_DefaultPlatform.hpp"
#include "Tpetra_MultiVector.hpp"
#include "Tpetra_PETScAIJMatrix.hpp"

int main(int argc, char *args[]) {
  typedef Tpetra::PETScAIJMatrix<>              PETScAIJMatrix;
  typedef PETScAIJMatrix::scalar_type           Scalar;
  typedef PETScAIJMatrix::local_ordinal_type    LO;
  typedef PETScAIJMatrix::global_ordinal_type   GO;
  typedef PETScAIJMatrix::node_type             Node;
  typedef Tpetra::Vector<Scalar,LO,GO>          Vector;
  typedef Tpetra::Operator<Scalar,LO,GO>        OP;
  typedef Tpetra::MultiVector<Scalar,LO,GO>     MV;
  typedef Ifpack2::Preconditioner<Scalar,LO,GO> Prec;

  using Teuchos::ParameterList;
  using Teuchos::RCP;
  using Teuchos::rcp;

  Vec            x,b;           /* approx solution, RHS, exact solution */
  Mat            A;             /* linear system matrix */
  PetscRandom    rctx;          /* random number generator context */
  PetscInt       i,j,Ii,J,Istart,Iend;
  PetscInt       m = 4,n = 4;   /* #mesh points in x & y directions, resp. */
  PetscErrorCode ierr;
  PetscScalar    v;
  PetscInt       rank=0;
  MPI_Comm       comm;

  //
  // Start PETSc 
  //
  PetscInitialize(&argc,&args,NULL,NULL);

  //
  // Create the matrix
  //
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n);CHKERRQ(ierr);
  ierr = MatSetType(A, MATAIJ);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(A,5,NULL,5,NULL);CHKERRQ(ierr);
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
    v = rand(); ierr = MatSetValues(A,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  //
  // Wrap the PETSc matrix as a PETScAIJMatrix. This is lightweight,
  // i.e., no deep data copies.
  //
  RCP<PETScAIJMatrix> tpetraA = rcp(new PETScAIJMatrix(A));

  //
  // Create a random solution vector and corresponding right-hand-side
  //
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,m*n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rctx);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
  ierr = VecSetRandom(x,rctx);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&b);CHKERRQ(ierr);
  ierr = MatMult(A,x,b);CHKERRQ(ierr);

  //
  // Copy the PETSc vectors to Tpetra vectors
  //
  RCP<Vector> tpetraX = xSDKTrilinos::deepCopyPETScVecToTpetraVector<Scalar,LO,GO,Node>(x);
  RCP<Vector> tpetraB = xSDKTrilinos::deepCopyPETScVecToTpetraVector<Scalar,LO,GO,Node>(b);

  //
  // Set initial guess to 0
  //
  tpetraX->putScalar(0);

  //
  // Construct preconditioner
  //
  Ifpack2::Factory factory;
  RCP<Prec> M = factory.create("RELAXATION", tpetraA.getConst());
  ParameterList ifpackParams;
  ifpackParams.set("relaxation: type","Jacobi");
  M->setParameters(ifpackParams);
  M->initialize();
  M->compute();

  //
  // Create parameter list for the Belos solver manager
  //
  ParameterList belosList;
  belosList.set( "Maximum Iterations", 100 );            // Maximum number of iterations allowed
  belosList.set( "Convergence Tolerance", 1e-6 );        // Relative convergence tolerance requested
  belosList.set( "Verbosity", Belos::IterationDetails + Belos::TimingDetails );

  //
  // Construct a preconditioned linear problem 
  //
  RCP<Belos::LinearProblem<double,MV,OP> > problem
    = rcp( new Belos::LinearProblem<double,MV,OP>( tpetraA, tpetraX, tpetraB ) );
  problem->setLeftPrec( M );
  problem->setProblem();

  //
  // Create an iterative solver manager 
  //
  RCP< Belos::MinresSolMgr<double,MV,OP> > solver
    = rcp( new Belos::MinresSolMgr<double,MV,OP>(problem, rcp(&belosList,false)) );

  //
  // Perform solve 
  //
  solver->solve();

  //
  // Check the residual
  //
  Vector R( tpetraA->getRowMap() );
  tpetraA->apply(*tpetraX,R);
  R.update(1,*tpetraB,-1);
  std::vector<double> normR(1), normB(1);
  R.norm2(normR);
  tpetraB->norm2(normB);
  if(rank == 0) std::cout << "Relative residual: " << normR[0] / normB[0] << std::endl;
  if(normR[0] / normB[0] > 1e-8)
    return EXIT_FAILURE;
  
  //
  // Check the error
  //
  RCP<Vector> trueX = xSDKTrilinos::deepCopyPETScVecToTpetraVector<Scalar,LO,GO,Node>(x);
  Vector errorVec( tpetraA->getRowMap() );
  errorVec.update(1,*tpetraX,-1,*trueX,0);
  std::vector<double> normErrorVec(1);
  errorVec.norm2(normErrorVec);
  if(rank == 0) std::cout << "Error: " << normErrorVec[0] << std::endl;
  
  //
  // Terminate PETSc
  //
  ierr = PetscFinalize(); CHKERRQ(ierr);
  return EXIT_SUCCESS;
}

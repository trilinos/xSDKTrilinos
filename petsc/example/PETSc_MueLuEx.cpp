/*
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
*/

/*
   This example demonstrates how to apply a Trilinos preconditioner to a PETSc
   linear system.

   The PETSc matrix is a 2D 5-point Laplace operator stored in AIJ format.
   This matrix is wrapped as an Epetra_PETScAIJMatrix, and a MueLu AMG
   preconditioner is created for it.  The associated linear system is solved twice,
   the first time using Belos's preconditioned CG, the second time using PETSc's.

   To invoke this example, use something like:

       mpirun -np 5 ./PETSc_MueLu_example.exe -mx 150 -my 150 -petsc_smoother -ksp_monitor_true_residual
*/

// This needs to be here; I can guess why.
#include "MueLu_CreateTpetraPreconditioner.hpp"

// Put these below all the stuff above, because MueLu is weird.
#include "petscksp.h"
#include "Tpetra_ConfigDefs.hpp"
#include "Tpetra_PETScAIJMatrix.hpp"
#include "Tpetra_Vector.hpp"
#include "Tpetra_Map.hpp"
#include "BelosPseudoBlockCGSolMgr.hpp"
#include "BelosTpetraAdapter.hpp"

  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::ArrayView;

  typedef Tpetra::PETScAIJMatrix<>                                         PETScAIJMatrix;
  typedef PETScAIJMatrix::scalar_type                                      Scalar;
  typedef PETScAIJMatrix::local_ordinal_type                               LO;
  typedef PETScAIJMatrix::global_ordinal_type                              GO;
  typedef PETScAIJMatrix::node_type                                        Node;
  typedef Tpetra::CrsMatrix<Scalar,LO,GO,Node>                             CrsMatrix;
  typedef Tpetra::Vector<Scalar,LO,GO,Node>                                Vector;
  typedef MueLu::TpetraOperator<Scalar,LO,GO,Node>                         MueLuOp;
  typedef Tpetra::Map<LO,GO,Node>                                          Map;
  typedef Tpetra::Operator<Scalar,LO,GO,Node>                              OP;
  typedef Tpetra::MultiVector<Scalar,LO,GO,Node>                           MV;
  typedef Tpetra::Vector<Scalar,LO,GO,Node>                                Vector;
  typedef Belos::LinearProblem<Scalar,MV,OP>                               LP;
  typedef Belos::PseudoBlockCGSolMgr<Scalar,MV,OP>                         SolMgr;

PetscErrorCode ShellApplyML(PC pc,Vec x,Vec y);

int main(int argc,char **args)
{
  Vec            x,b;            /* approx solution, RHS  */
  Mat            A;              /* linear system matrix */
  KSP            ksp;            /* linear solver context */
  KSP            kspSmoother=0;  /* solver context for PETSc fine grid smoother */
  PC pc;
  PetscRandom    rctx;           /* random number generator context */
  PetscReal      norm;           /* norm of solution error */
  PetscInt       i,j,Ii,J,Istart,Iend,its;
  PetscInt       m = 50,n = 50;  /* #mesh points in x & y directions, resp. */
  PetscErrorCode ierr;
  PetscScalar    v,neg_one = -1.0;
  MPI_Comm comm;

  //
  // Initialize PETSc and get command line arguments
  //
  PetscInitialize(&argc,&args,PETSC_NULL,PETSC_NULL);
//  ierr = PetscOptionsGetInt(PETSC_NULL,"-mx",&m,PETSC_NULL);CHKERRQ(ierr);
//  ierr = PetscOptionsGetInt(PETSC_NULL,"-my",&n,PETSC_NULL);CHKERRQ(ierr);

  //
  // Create the PETSc matrix
  //
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n);CHKERRQ(ierr);
  ierr = MatSetType(A, MATAIJ);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(A,5,PETSC_NULL,5,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  ierr = PetscObjectGetComm( (PetscObject)A, &comm);CHKERRQ(ierr);

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
  // Create the solution vector and RHS
  //
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,m*n);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&b);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rctx);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rctx);CHKERRQ(ierr);
  ierr = VecSetRandom(b,rctx);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);

  //
  // Copy the PETSc matrix to a Tpetra CrsMatrix
  //
  RCP<CrsMatrix> epA = xSDKTrilinos::deepCopyPETScAIJMatrixToTpetraCrsMatrix<Scalar,LO,GO,Node>(A);

  /* Copy the PETSc vectors to Tpetra vectors. */
  PetscScalar *vals;
  ierr = VecGetArray(x,&vals);CHKERRQ(ierr);
  PetscInt length;
  ierr = VecGetLocalSize(x,&length);CHKERRQ(ierr);
  PetscScalar* valscopy = (PetscScalar*) malloc(length*sizeof(PetscScalar));
  memcpy(valscopy,vals,length*sizeof(PetscScalar));
  ierr = VecRestoreArray(x,&vals);CHKERRQ(ierr);
  ArrayView<PetscScalar> epxView(valscopy,length);
  RCP<Vector> epx = rcp(new Vector(epA->getRowMap(),epxView));
  RCP<Vector> epb = rcp(new Vector(epA->getRowMap()));
  epA->apply(*epx, *epb);

  /* Create the MueLu AMG preconditioner. */

  /* Parameter list that holds options for AMG preconditioner. */
  Teuchos::ParameterList mlList;
  mlList.set("parameterlist: syntax", "ml");
  /* Set recommended defaults for Poisson-like problems. */
//  ML_Epetra::SetDefaults("SA",mlList);
  /* Specify how much information ML prints to screen.
     0 is the minimum (no output), 10 is the maximum. */
  mlList.set("ML output",10);
  mlList.set("smoother: type (level 0)","symmetric Gauss-Seidel");

  /* how many fine grid pre- or post-smoothing sweeps to do */
  mlList.set("smoother: sweeps (level 0)",2);

  mlList.print();

  RCP<MueLuOp> Prec = MueLu::CreateTpetraPreconditioner(Teuchos::rcp_dynamic_cast<OP>(epA), mlList);

  /* Trilinos CG */
  epx->putScalar(0.0);
  RCP<LP> Problem = rcp(new LP(epA, epx, epb));
  Problem->setLeftPrec(Prec);
  Problem->setProblem();
  RCP<Teuchos::ParameterList> cgPL = rcp(new Teuchos::ParameterList());
  cgPL->set("Maximum Iterations", 200);
  cgPL->set("Verbosity", Belos::IterationDetails + Belos::FinalSummary + Belos::TimingDetails);
  cgPL->set("Convergence Tolerance", 1e-12);
  SolMgr solver(Problem,cgPL);
  solver.solve();

  /* PETSc CG */
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = KSPSetTolerances(ksp,1e-12,1.e-50,PETSC_DEFAULT,
                          PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = KSPSetType(ksp,KSPCG);CHKERRQ(ierr);

  /* Wrap the ML preconditioner as a PETSc shell preconditioner. */
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCSHELL);CHKERRQ(ierr);
  ierr = PCShellSetApply(pc,ShellApplyML);CHKERRQ(ierr);
  ierr = PCShellSetContext(pc,(void*)Prec.get());CHKERRQ(ierr);
  ierr = PCShellSetName(pc,"MueLu AMG");CHKERRQ(ierr); 

  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

  ierr = KSPView(ksp,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = VecAXPY(x,neg_one,x);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error %e iterations %D\n",
                     norm,its);CHKERRQ(ierr);

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);  ierr = MatDestroy(&A);CHKERRQ(ierr);

  if (kspSmoother) {ierr = KSPDestroy(&kspSmoother);CHKERRQ(ierr);}

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
} /*main*/

/* ***************************************************************** */

PetscErrorCode ShellApplyML(PC pc,Vec x,Vec y)
{
  PetscErrorCode  ierr;
  MueLuOp *mlp = 0;
  void* ctx;

  ierr = PCShellGetContext(pc,&ctx); CHKERRQ(ierr);  
  mlp = (MueLuOp*)ctx;

  /* Wrap x and y as Tpetra_Vectors. */
  PetscInt length;
  ierr = VecGetLocalSize(x,&length);CHKERRQ(ierr);
  const PetscScalar *xvals;
  PetscScalar *yvals;

  ierr = VecGetArrayRead(x,&xvals);CHKERRQ(ierr);
  ierr = VecGetArray(y,&yvals);CHKERRQ(ierr);

  ArrayView<const PetscScalar> xView(xvals,length);
  ArrayView<PetscScalar> yView(yvals,length);

  Vector epx(mlp->getDomainMap(),xView); // TODO: see if there is a way to avoid copying the data
  Vector epy(mlp->getDomainMap(),yView);

  /* Apply ML. */
  mlp->apply(epx,epy);

  /* Rip the data out of epy */
  Teuchos::ArrayRCP<const Scalar> epxData = epx.getData();
  Teuchos::ArrayRCP<const Scalar> epyData = epy.getData();

  for(int i=0; i< epyData.size(); i++)
  {
    yvals[i] = epyData[i];
  }
  
  /* Clean up and return. */
  ierr = VecRestoreArrayRead(x,&xvals);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&yvals);CHKERRQ(ierr);

  return 0;
} /*ShellApplyML*/


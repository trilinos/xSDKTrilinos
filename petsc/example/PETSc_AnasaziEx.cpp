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
   This example demonstrates how to use a Trilinos eigensolver to compute the
   eigenpairs of a PETSc matrix.

   The PETSc matrix is a 2D 5-point Laplace operator stored in AIJ format.
   This matrix is wrapped as an PETScAIJMatrix.  The associated eigenvalue
   problem is solved using Anasazi.

   To invoke this example, use something like:

       mpirun -np 5 ./AnasaziTest.exe -mx 150 -my 150
*/

#include "petscksp.h"

#include "AnasaziBasicEigenproblem.hpp"
#include "AnasaziConfigDefs.hpp"
#include "AnasaziTpetraAdapter.hpp"
#include "AnasaziRTRSolMgr.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Tpetra_PETScAIJMatrix.hpp"

int main(int argc,char **args)
{
  using Teuchos::RCP;
  using Teuchos::rcp;
  using std::cout;
  using std::endl;

  typedef Tpetra::PETScAIJMatrix<>                  PETScAIJMatrix;
  typedef PETScAIJMatrix::scalar_type               Scalar;
  typedef PETScAIJMatrix::local_ordinal_type        LO;
  typedef PETScAIJMatrix::global_ordinal_type       GO;
  typedef PETScAIJMatrix::node_type                 Node;
  typedef Tpetra::Vector<Scalar,LO,GO,Node>         Vector;
  typedef Tpetra::Map<LO,GO,Node>                   Map;
  typedef Tpetra::Operator<Scalar,LO,GO,Node>       OP;
  typedef Tpetra::MultiVector<Scalar,LO,GO,Node>    MV;
  typedef Anasazi::RTRSolMgr<Scalar,MV,OP>          SolMgr;
  typedef Anasazi::BasicEigenproblem<Scalar,MV,OP>  Problem;
  typedef Anasazi::OperatorTraits<Scalar,MV,OP>     OPT;
  typedef Anasazi::MultiVecTraits<Scalar,MV>        MVT;

  Mat            A;        /* PETSc matrix */
  PetscInt       m = 50,n = 50; /* #mesh points in x & y directions, resp. */
  PetscInt       nev = 4;
  PetscErrorCode ierr;
  MPI_Comm comm;
  PetscInt Istart, Iend, Ii, i, j, J, rank;
  PetscScalar v;


  //
  // Initialize PETSc and get the command line arguments
  //
  PetscInitialize(&argc,&args,NULL,NULL);
//  ierr = PetscOptionsGetInt(PETSC_NULL,"-mx",&m,PETSC_NULL);CHKERRQ(ierr);
//  ierr = PetscOptionsGetInt(PETSC_NULL,"-my",&n,PETSC_NULL);CHKERRQ(ierr);
//  ierr = PetscOptionsGetInt(PETSC_NULL,"-nev",&nev,PETSC_NULL);CHKERRQ(ierr);

  //
  // Create the matrix
  //
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n);CHKERRQ(ierr);
  ierr = MatSetType(A, MATAIJ);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(A,5,PETSC_NULL,5,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  ierr = PetscObjectGetComm( (PetscObject)A, &comm);CHKERRQ(ierr);
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
  // Wrap the PETSc matrix as a PETScAIJMatrix. This is lightweight,
  // i.e., no deep data copies.
  //
  RCP<PETScAIJMatrix> epA = rcp(new PETScAIJMatrix(A));

  //
  // Create an initial guess
  //
  RCP<MV> initGuess = rcp(new MV(epA->getDomainMap(),4,false));
  initGuess->randomize();

  //
  // Create an eigenproblem
  //
  RCP<Problem> problem = rcp(new Problem(epA,initGuess));
  problem->setNEV(nev);
  problem->setHermitian(true);
  problem->setProblem();

  //
  // Create the parameter list
  //
  Teuchos::ParameterList pl;
  pl.set("Verbosity", Anasazi::IterationDetails + Anasazi::FinalSummary);
  pl.set("Convergence Tolerance", 1e-6);

  //
  // Create an Anasazi eigensolver
  //
  RCP<SolMgr> solver = rcp(new SolMgr(problem, pl));

  //
  // Solve the problem to the specified tolerances
  //
  Anasazi::ReturnType returnCode = solver->solve();
  if (returnCode != Anasazi::Converged && rank == 0) {
    cout << "Anasazi::EigensolverMgr::solve() returned unconverged." << endl;
  }
  else if (rank == 0)
    cout << "Anasazi::EigensolverMgr::solve() returned converged." << endl;

  //
  // Get the eigenvalues and eigenvectors from the eigenproblem
  //
  Anasazi::Eigensolution<Scalar,MV> sol = problem->getSolution();
  std::vector<Anasazi::Value<Scalar> > evals = sol.Evals;
  RCP<MV> evecs = sol.Evecs;
  int numev = sol.numVecs;

  //
  // Compute the residual, just as a precaution
  //
  if (numev > 0) {
    RCP<const MV> R = problem->computeCurrResVec();
    std::vector<Scalar> normR(sol.numVecs);
    MVT::MvNorm( *R, normR );

    if (rank == 0) {
      cout.setf(std::ios_base::right, std::ios_base::adjustfield);
      cout<<"Actual Eigenvalues (obtained by Rayleigh quotient) : "<<endl;
      cout<<"------------------------------------------------------"<<endl;
      cout<<std::setw(16)<<"Real Part"
          <<std::setw(16)<<"Error"<<endl;
      cout<<"------------------------------------------------------"<<endl;
      for (int i=0; i<numev; i++) {
        cout<<std::setw(16)<<evals[i].realpart
            <<std::setw(16)<<normR[i]/evals[i].realpart
            <<endl;
      }
      cout<<"------------------------------------------------------"<<endl;
    }
  }

  //
  // Terminate PETSc
  //
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
} /*main*/

/* ***************************************************************** */

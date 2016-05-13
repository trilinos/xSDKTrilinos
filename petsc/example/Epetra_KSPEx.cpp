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

//
// This driver reads a matrix from a Matrix Market file, and solves a linear
// system with that matrix as A.  The initial guesses are all set to zero.  
// An ILU preconditioner is constructed using the Ifpack factory and passed 
// to a PETSc linear solver.
//
#include "BelosConfigDefs.hpp"
#include "BelosLinearProblem.hpp"
#include "BelosEpetraAdapter.hpp"
#include "BelosPETScSolMgr.hpp"

#include "EpetraExt_CrsMatrixIn.h"
#include "Epetra_Map.h"
#ifdef EPETRA_MPI
  #include "Epetra_MpiComm.h"
#else
  #include "Epetra_SerialComm.h"
#endif
#include "Epetra_CrsMatrix.h"

#include "Ifpack.h"

#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_StandardCatchMacros.hpp"

int main(int argc, char *argv[]) {
  typedef double                            ST;
  typedef Teuchos::ScalarTraits<ST>        SCT;
  typedef SCT::magnitudeType                MT;
  typedef Epetra_MultiVector                MV;
  typedef Epetra_Operator                   OP;
  typedef Belos::MultiVecTraits<ST,MV>     MVT;
  typedef Belos::OperatorTraits<ST,MV,OP>  OPT;

  using Teuchos::ParameterList;
  using Teuchos::RCP;
  using Teuchos::rcp;

  //
  // Initialize communicator
  //
#ifdef EPETRA_MPI
  MPI_Init(&argc,&argv);
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
#else
  Epetra_SerialComm Comm;
#endif

  //
  // Process command line arguments
  //
  int numrhs = 2;            // number of right-hand sides to solve for
  std::string filename("cage4.mtx");
  MT tol = 1.0e-4;           // relative residual tolerance

  Teuchos::CommandLineProcessor cmdp(false,false);
  cmdp.setOption("filename",&filename,"Filename for test matrix.");
  cmdp.setOption("tol",&tol,"Relative residual tolerance used by GMRES solver.");
  cmdp.setOption("num-rhs",&numrhs,"Number of right-hand sides to be solved for.");
  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
    return -1;
  }

  //
  // Get the matrix from a file
  //
  Epetra_CrsMatrix* A;
  EpetraExt::MatrixMarketFileToCrsMatrix(filename.c_str(), Comm, A, false);

  //
  // Create a random RHS and set the initial guess to 0
  //
  RCP<Epetra_MultiVector> B, X, trueX;
  X = rcp( new Epetra_MultiVector( A->RowMap(), numrhs ) );
  trueX = rcp( new Epetra_MultiVector( A->RowMap(), numrhs ) );
  B = rcp( new Epetra_MultiVector( A->RowMap(), numrhs ) );
  trueX->Random();
  OPT::Apply( *A, *trueX, *B );
  X->PutScalar( 0.0 );

  //
  // Construct preconditioner
  //
  Ifpack Factory;
  std::string PrecType = "point relaxation stand-alone"; // incomplete Cholesky
  int OverlapLevel = 0; // must be >= 0. If Comm.NumProc() == 1, it is ignored.
  RCP<Ifpack_Preconditioner> Prec = Teuchos::rcp( Factory.Create(PrecType, &*A, OverlapLevel) );
  ParameterList ifpackList;
  ifpackList.set("relaxation: type", "Jacobi");
  Prec->SetParameters(ifpackList);
  Prec->Initialize();
  Prec->Compute();

  // Create the Belos preconditioned operator from the Ifpack preconditioner.
  // NOTE:  This is necessary because Belos expects an operator to apply the
  //        preconditioner with Apply() NOT ApplyInverse().
  RCP<Belos::EpetraPrecOp> belosPrec = rcp( new Belos::EpetraPrecOp( Prec ) );

  //
  // Create parameter list for the GMRES solver manager
  //
  ParameterList belosList;
  int maxiters = 100;
  belosList.set( "Maximum Iterations", maxiters );       // Maximum number of iterations allowed
  belosList.set( "Convergence Tolerance", tol );         // Relative convergence tolerance requested
  belosList.set( "Verbosity", Belos::IterationDetails ); // Print convergence information

  //
  // Construct a preconditioned linear problem
  //
  RCP<Belos::LinearProblem<double,MV,OP> > problem
  = rcp( new Belos::LinearProblem<double,MV,OP>( rcp(A,false), X, B ) );
  problem->setLeftPrec( belosPrec );
  problem->setProblem();

  //
  // Create an iterative solver manager.
  //
  RCP< Belos::PETScSolMgr<double,MV,OP> > solver
  = rcp( new Belos::PETScSolMgr<double,MV,OP>(problem, rcp(&belosList,false)) );

  //
  // Perform solve
  //
  solver->solve();
  
  //
  // Check the residual
  //
  MV R( A->RowMap(), numrhs );
  A->Apply(*X,R);
  R.Update(1,*B,-1);
  std::vector<double> normR(numrhs), normB(numrhs);
  R.Norm2(normR.data());
  B->Norm2(normB.data());
  for(int i=0; i<numrhs; i++)
  {
    if(Comm.MyPID() == 0) std::cout << "Relative residual: " << normR[i] / normB[i] << std::endl;
    if(normR[i] / normB[i] > tol)
      return EXIT_FAILURE;
  }

  //
  // Check the error
  //
  MV errorVec( A->RowMap(), numrhs );
  errorVec.Update(1,*X,-1,*trueX,0);
  std::vector<double> normErrorVec(numrhs);
  errorVec.Norm2(normErrorVec.data());
  for(int i=0; i<numrhs; i++)
  {
    if(Comm.MyPID() == 0) std::cout << "Error: " << normErrorVec[i] << std::endl;
  }

  //
  // Finalize MPI
  //
#ifdef EPETRA_MPI
  MPI_Finalize();
#endif

  return EXIT_SUCCESS;
}

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

#include "BelosConfigDefs.hpp"
#include "BelosLinearProblem.hpp"
#include "BelosTpetraAdapter.hpp"
#include "BelosPETScSolMgr.hpp"

#include "Ifpack2_Factory.hpp"

#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_StandardCatchMacros.hpp"

#include "Tpetra_CrsMatrix.hpp"
#include "Tpetra_DefaultPlatform.hpp"
#include "Tpetra_MultiVector.hpp"
#include "MatrixMarket_Tpetra.hpp"

int main(int argc, char *argv[]) {
  typedef Tpetra::MultiVector<>                   MV;
  typedef Tpetra::Operator<>                      OP;
  typedef Tpetra::CrsMatrix<>              CrsMatrix;
  typedef Ifpack2::Preconditioner<>             Prec;

  using Teuchos::ParameterList;
  using Teuchos::RCP;
  using Teuchos::rcp;

  //
  // Initialize MPI 
  //
  Teuchos::oblackholestream blackhole;
  Teuchos::GlobalMPISession mpiSession (&argc, &argv, &blackhole);

  //
  // Get the default communicator
  //
  RCP<const Teuchos::Comm<int> > comm = Tpetra::DefaultPlatform::getDefaultPlatform().getComm();

  int numrhs = 2;            // number of right-hand sides to solve for
  int maxiters = 100;        // maximum number of iterations allowed per linear system
  std::string filename("cage4.mtx");
  double tol = 1.0e-5;           // relative residual tolerance

  //
  // Read the command line arguments
  //
  Teuchos::CommandLineProcessor cmdp(false,false);
  cmdp.setOption("filename",&filename,"Filename for test matrix.");
  cmdp.setOption("tol",&tol,"Relative residual tolerance used by GMRES solver.");
  cmdp.setOption("num-rhs",&numrhs,"Number of right-hand sides to be solved for.");
  cmdp.setOption("max-iters",&maxiters,"Maximum number of iterations per linear system (-1 = adapted to problem/block size).");
  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
    return -1;
  }

  //
  // Get the matrix from a file
  //
  RCP<CrsMatrix> A = Tpetra::MatrixMarket::Reader<CrsMatrix>::readSparseFile(filename,comm);

  //
  // Create a random RHS and set the initial guess to 0
  //
  RCP<MV> B = rcp(new MV(A->getRowMap(),numrhs,false));
  RCP<MV> X = rcp(new MV(A->getRowMap(),numrhs,false));
  RCP<MV> trueX = rcp(new MV(A->getRowMap(),numrhs,false));
  trueX->randomize();
  A->apply(*trueX,*B);
  X->putScalar(0);

  //
  // Construct preconditioner 
  //
  Ifpack2::Factory factory;
  RCP<Prec> M = factory.create("RELAXATION", A.getConst());
  ParameterList ifpackParams;
  ifpackParams.set("relaxation: type","Jacobi");
  M->setParameters(ifpackParams);
  M->initialize();
  M->compute();

  //
  // Create parameter list for the Belos solver manager 
  //
  ParameterList belosList;
  belosList.set( "Maximum Iterations", maxiters );       // Maximum number of iterations allowed
  belosList.set( "Convergence Tolerance", tol );         // Relative convergence tolerance requested
  belosList.set( "Verbosity", Belos::IterationDetails ); // Print convergence information
  belosList.set( "Solver", "bcgs" );                     // Use BiCGStab as the linear solver

  //
  // Construct a preconditioned linear problem
  //
  RCP<Belos::LinearProblem<double,MV,OP> > problem
    = rcp( new Belos::LinearProblem<double,MV,OP>( A, X, B ) );
  problem->setLeftPrec( M );
  problem->setProblem();

  //
  // Create an iterative solver manager
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
  MV R( A->getRowMap(), numrhs, false );
  A->apply(*X,R);
  R.update(1,*B,-1);
  std::vector<double> normR(numrhs), normB(numrhs);
  R.norm2(normR);
  B->norm2(normB);
  for(int i=0; i<numrhs; i++)
  {
    if(comm->getRank() == 0) std::cout << "Relative residual: " << normR[i] / normB[i] << std::endl;
    if(normR[i] / normB[i] > tol)
      return EXIT_FAILURE;
  }

  //
  // Check the error
  //
  MV errorVec( A->getRowMap(), numrhs, false );
  errorVec.update(1,*X,-1,*trueX,0);
  std::vector<double> normErrorVec(numrhs);
  errorVec.norm2(normErrorVec);
  for(int i=0; i<numrhs; i++)
  {
    if(comm->getRank() == 0) std::cout << "Error: " << normErrorVec[i] << std::endl;
  }

  return EXIT_SUCCESS;
}

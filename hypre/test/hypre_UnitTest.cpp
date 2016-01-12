/*@HEADER
// ***********************************************************************
//
//       Ifpack: Object-Oriented Algebraic Preconditioner Package
//                 Copyright (2002) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
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
// Questions? Contact Michael A. Heroux (maherou@sandia.gov)
//
// ***********************************************************************
//@HEADER
*/

#include "Ifpack2_ETIHelperMacros.h"
#include "Ifpack2_Hypre.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_Comm.hpp"
#include "Teuchos_OrdinalTraits.hpp"
#include "Teuchos_UnitTestHarness.hpp"
#include "Tpetra_ConfigDefs.hpp"
#include "Tpetra_CrsMatrix.hpp"
#include "Tpetra_DefaultPlatform.hpp"

namespace {

typedef double Scalar;
typedef int LO;
typedef int GO;

using Ifpack2::FunctionParameter;
using Ifpack2::Hypre::Solver;
using Ifpack2::Hypre::Prec;
using Teuchos::Array;
using Teuchos::ArrayRCP;
using Teuchos::NO_TRANS;
using Teuchos::RCP;
using Teuchos::rcp;
using Teuchos::reduceAll;
using Teuchos::REDUCE_AND;
using Teuchos::Comm;
using Tpetra::global_size_t;

// Tests whether two vectors are equivalent
template<class Node>
bool EquivalentVectors(const Tpetra::MultiVector<Scalar,LO,GO,Node> &X,
                       const Tpetra::MultiVector<Scalar,LO,GO,Node> &Y,
                       double tol)
{
  bool retVal = true;

  size_t num_vectors = X.getNumVectors();
  if(Y.getNumVectors() != num_vectors){
    printf("Multivectors do not have same number of vectors.\n");
    return false;
  }

  ArrayRCP<ArrayRCP<const Scalar> > xData, yData;
  xData = X.get2dView();
  yData = Y.get2dView();


  for(size_t j = 0; j < num_vectors; j++){
    if(X.getLocalLength() != Y.getLocalLength()){
      printf("Vectors are not same size on local processor.\n");
      return false;
    }
    for(global_size_t i = 0; i < X.getGlobalLength(); i++){
      int X_local = X.getMap()->getLocalElement(i);
      int Y_local = Y.getMap()->getLocalElement(i);
      if(X_local == Teuchos::OrdinalTraits<LO>::invalid() ||
         Y_local == Teuchos::OrdinalTraits<LO>::invalid()){
        continue;
      }
      if(fabs(xData[j][X_local] - yData[j][Y_local]) > tol){
        printf("Vector number[%lu] ", j);
        printf("Val1[%lu] = %f != Val2[%lu] = %f\n", i, xData[j][X_local], i, yData[j][Y_local]);
        retVal = false;
      }
    }
  }

  int localRetVal = (int)retVal;
  int globalRetVal;
  RCP<const Comm<int> > comm = X.getMap()->getComm();
  reduceAll(*comm,REDUCE_AND,1,&localRetVal,&globalRetVal);

  if(globalRetVal == 0){
    return false;
  }
  return true;
}


// Tests hypre interface's ability to initialize correctly
TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( Ifpack_Hypre, Construct, Node ) {
  typedef Tpetra::CrsMatrix<Scalar,LO,GO,Node>      Matrix;
  typedef Tpetra::Map<LO,GO,Node>                   Map;
  typedef Ifpack2::Ifpack2_Hypre<Scalar,LO,GO,Node> Hypre;
  const global_size_t N = 10;

  // get a comm
  RCP<const Comm<int> > comm =
        Tpetra::DefaultPlatform::getDefaultPlatform ().getComm ();
  // create a map
  RCP<Map> map = rcp(new Map(N,0,comm));
  // create the matrix
  RCP<Matrix> matrix = rcp(new Matrix(map,1));
  // put stuff in the matrix
  Array<LO> indices(1);
  Array<Scalar> values(1);
  values[0] = 1.0;
  for(LO i = 0; i < (LO)map->getNodeNumElements(); i++){
    indices[0] = map->getGlobalElement(i);
    matrix->insertGlobalValues(indices[0], indices(), values());
  }
  matrix->fillComplete();
  Hypre preconditioner(matrix);
  preconditioner.initialize();
}


// Tests hypre's ability to work when A has a funky row map, but the vectors have
// a contiguous row map
TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( Ifpack_Hypre, ParameterList, Node ){
  typedef Tpetra::CrsMatrix<Scalar,LO,GO,Node>      Matrix;
  typedef Tpetra::MultiVector<Scalar,LO,GO,Node>    MV;
  typedef Tpetra::Map<LO,GO,Node>                   Map;
  typedef Ifpack2::Ifpack2_Hypre<Scalar,LO,GO,Node> Hypre;
  const double tol = 1e-7;
  GO N = 10;

  // get a comm
  RCP<const Comm<int> > comm =
        Tpetra::DefaultPlatform::getDefaultPlatform ().getComm ();

  // create a funky map
  Array<GO> indices;
  for(GO i = comm->getRank(); i < N; i+= comm->getSize())
    indices.push_back(i);
  RCP<Map> funkyMap = rcp(new Map(N,indices,0,comm));

  // Create a tridiagonal matrix
  RCP<Matrix> matrix = rcp(new Matrix(funkyMap,3));
  for(LO i = 0; i<(LO)funkyMap->getNodeNumElements(); i++)
  {
    GO globalIndex = funkyMap->getGlobalElement(i);
    Array<LO> indices;
    Array<Scalar> values;
    if(globalIndex > 0)
    {
      indices.push_back(globalIndex-1);
      values.push_back(-1.0);
    }
    indices.push_back(globalIndex);
    values.push_back(2.0);
    if(globalIndex < N-1)
    {
      indices.push_back(globalIndex+1);
      values.push_back(-1.0);
    }
    matrix->insertGlobalValues(globalIndex,indices,values);
  }
  RCP<Map> contigMap = rcp(new Map(N,0,comm));
  matrix->fillComplete(contigMap,contigMap);

  // Create the parameter list
  Teuchos::ParameterList list("Preconditioner List");
  RCP<FunctionParameter> functs[11];
  functs[0] = rcp(new FunctionParameter(Solver, &HYPRE_PCGSetMaxIter, 1000));               // max iterations
  functs[1] = rcp(new FunctionParameter(Solver, &HYPRE_PCGSetTol, tol));                   // conv. tolerance
  functs[2] = rcp(new FunctionParameter(Solver, &HYPRE_PCGSetTwoNorm, 1));                  // use the two norm as the stopping criteria
  functs[3] = rcp(new FunctionParameter(Solver, &HYPRE_PCGSetPrintLevel, 0));               // print solve info
  functs[4] = rcp(new FunctionParameter(Solver, &HYPRE_PCGSetLogging, 1));                  // needed to get run info later
  functs[5] = rcp(new FunctionParameter(Prec, &HYPRE_BoomerAMGSetPrintLevel, 1)); // print amg solution info
  functs[6] = rcp(new FunctionParameter(Prec, &HYPRE_BoomerAMGSetCoarsenType, 6));
  functs[7] = rcp(new FunctionParameter(Prec, &HYPRE_BoomerAMGSetRelaxType, 6));  //Sym G.S./Jacobi hybrid
  functs[8] = rcp(new FunctionParameter(Prec, &HYPRE_BoomerAMGSetNumSweeps, 1));
  functs[9] = rcp(new FunctionParameter(Prec, &HYPRE_BoomerAMGSetTol, 0.0));      // conv. tolerance zero
  functs[10] = rcp(new FunctionParameter(Prec, &HYPRE_BoomerAMGSetMaxIter, 1));   //do only one iteration!

  list.set("Solver", Ifpack2::Hypre::PCG);
  list.set("Preconditioner", Ifpack2::Hypre::BoomerAMG);
  list.set("SolveOrPrecondition", Ifpack2::Hypre::Solver);
  list.set("SetPreconditioner", true);
  list.set("NumFunctions", 11);
  list.set<RCP<FunctionParameter>*>("Functions", functs);

  // Create the preconditioner
  Hypre preconditioner(matrix);
  preconditioner.setParameters(list);
  preconditioner.compute();

  // Create the RHS and solution vector
  int numVec = 2;
  MV X(preconditioner.getDomainMap(), numVec);
  MV KnownX(preconditioner.getDomainMap(), numVec);
  KnownX.randomize();
  MV B(preconditioner.getRangeMap(), numVec);
  matrix->apply(KnownX,B,NO_TRANS); // This is a problem.

  preconditioner.apply(B,X);
  TEST_EQUALITY(EquivalentVectors(X, KnownX, tol*10*pow(10.0,comm->getSize())), true);
}


// Tests the hypre interface's ability to work with both a preconditioner and linear
// solver via ApplyInverse
TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( Ifpack_Hypre, Ifpack, Node ){
  typedef Tpetra::CrsMatrix<Scalar,LO,GO,Node>      Matrix;
  typedef Tpetra::MultiVector<Scalar,LO,GO,Node>    MV;
  typedef Tpetra::Map<LO,GO,Node>                   Map;
  typedef Ifpack2::Ifpack2_Hypre<Scalar,LO,GO,Node> Hypre;
  const double tol = 1E-7;
  const int nx = 4;

  // get a comm
  RCP<const Comm<int> > comm =
        Tpetra::DefaultPlatform::getDefaultPlatform ().getComm ();

  //
  // Create Laplace2D with a contiguous row distribution
  //
  RCP<Map> map = rcp(new Map(nx*nx,0,comm));
  RCP<Matrix> matrix = rcp(new Matrix(map,5));
  for(LO i = 0; i<nx; i++)
  {
    for(LO j = 0; j<nx; j++)
    {
      GO row = i*nx+j;
      if(!map->isNodeGlobalElement(row))
        continue;

      Array<LO> indices;
      Array<Scalar> values;

      if(i > 0)
      {
        indices.push_back(row - nx);
        values.push_back(-1.0);
      }
      if(i < nx-1)
      {
        indices.push_back(row + nx);
        values.push_back(-1.0);
      }
      indices.push_back(row);
      values.push_back(4.0);
      if(j > 0)
      {
        indices.push_back(row-1);
        values.push_back(-1.0);
      }
      if(j < nx-1)
      {
        indices.push_back(row+1);
        values.push_back(-1.0);
      }
      matrix->insertGlobalValues(row,indices,values);
    }
  }
  matrix->fillComplete();

  //
  // Create the parameter list
  //
  Teuchos::ParameterList list("New List");
  RCP<FunctionParameter> functs[5];
  functs[0] = rcp(new FunctionParameter(Solver, &HYPRE_PCGSetMaxIter, 1000));
  functs[1] = rcp(new FunctionParameter(Solver, &HYPRE_PCGSetTol, 1e-9));
  functs[2] = rcp(new FunctionParameter(Solver, &HYPRE_PCGSetLogging, 1));
  functs[3] = rcp(new FunctionParameter(Solver, &HYPRE_PCGSetPrintLevel, 0));
  functs[4] = rcp(new FunctionParameter(Prec, &HYPRE_ParaSailsSetLogging, 0));
  list.set("NumFunctions", 5);
  list.set<RCP<FunctionParameter>*>("Functions", functs);
  list.set("SolveOrPrecondition", Solver);
  list.set("Solver", Ifpack2::Hypre::PCG);
  list.set("Preconditioner", Ifpack2::Hypre::ParaSails);
  list.set("SetPreconditioner", true);

  // Create the preconditioner
  Hypre preconditioner(matrix);
  preconditioner.setParameters(list);
  preconditioner.compute();

  // Create the RHS and solution vector
  int numVec = 2;
  MV X(preconditioner.getDomainMap(), numVec);
  MV KnownX(preconditioner.getDomainMap(), numVec);
  KnownX.randomize();
  MV B(preconditioner.getRangeMap(), numVec);
  matrix->apply(KnownX,B,NO_TRANS);

  preconditioner.apply(B,X);
  TEST_EQUALITY(EquivalentVectors(X, KnownX, tol*10*pow(10.0,comm->getSize())), true);
}


// This example uses contiguous maps, so hypre should not have problems
TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( Ifpack_Hypre, DiagonalMatrixInOrder, Node ) {
  typedef Tpetra::CrsMatrix<Scalar,LO,GO,Node>      Matrix;
  typedef Tpetra::MultiVector<Scalar,LO,GO,Node>    MV;
  typedef Tpetra::Map<LO,GO,Node>                   Map;
  typedef Ifpack2::Ifpack2_Hypre<Scalar,LO,GO,Node> Hypre;

  // Get a comm
  RCP<const Comm<int> > comm =
        Tpetra::DefaultPlatform::getDefaultPlatform ().getComm ();
  int myRank = comm->getRank();
  int numProcs = comm->getSize();

  //
  // Construct the contiguous map
  //
  RCP<Map> map = rcp(new Map(2*numProcs,0,comm));

  //
  // Construct the diagonal matrix
  //
  RCP<Matrix> matrix = rcp(new Matrix(map,1));
  Array<LO> indices(1);
  Array<Scalar> values(1);
  indices[0] = 2*myRank;
  values[0] = 2*myRank+1;
  matrix->insertGlobalValues(indices[0], indices(), values());
  indices[0] = 2*myRank+1;
  values[0] = 2*myRank+2;
  matrix->insertGlobalValues(indices[0], indices(), values());
  matrix->fillComplete();

  //
  // Create the parameter list
  //
  const double tol = 1e-7;
  Teuchos::ParameterList list("Preconditioner List");
  RCP<FunctionParameter> functs[5];
  functs[0] = rcp(new FunctionParameter(Solver, &HYPRE_PCGSetMaxIter, 100));               // max iterations
  functs[1] = rcp(new FunctionParameter(Solver, &HYPRE_PCGSetTol, tol));                   // conv. tolerance
  functs[2] = rcp(new FunctionParameter(Solver, &HYPRE_PCGSetTwoNorm, 1));                  // use the two norm as the stopping criteria
  functs[3] = rcp(new FunctionParameter(Solver, &HYPRE_PCGSetPrintLevel, 2));               // print solve info
  functs[4] = rcp(new FunctionParameter(Solver, &HYPRE_PCGSetLogging, 1));
  list.set("Solver", Ifpack2::Hypre::PCG);
  list.set("SolveOrPrecondition", Solver);
  list.set("SetPreconditioner", false);
  list.set("NumFunctions", 5);
  list.set<RCP<FunctionParameter>*>("Functions", functs);

  //
  // Create the preconditioner (which is actually a PCG solver)
  //
  Hypre preconditioner(matrix);
  preconditioner.setParameters(list);
  preconditioner.compute();

  // Create the RHS and solution vector
  int numVec = 2;
  MV X(preconditioner.getDomainMap(), numVec);
  MV KnownX(preconditioner.getDomainMap(), numVec);
  KnownX.randomize();
  MV B(preconditioner.getRangeMap(), numVec);
  matrix->apply(KnownX,B,NO_TRANS);

  //
  // Solve the linear system
  //
  preconditioner.apply(B,X);
  TEST_EQUALITY(EquivalentVectors(X, KnownX, tol*10*pow(10.0,comm->getSize())), true);
}


// hypre does not like the distribution of the vectors in this example.
// Our interface should detect that and return an error code.
TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( Ifpack_Hypre, DiagonalMatrixOutOfOrder, Node ) {
  typedef Tpetra::CrsMatrix<Scalar,LO,GO,Node>      Matrix;
  typedef Tpetra::MultiVector<Scalar,LO,GO,Node>    MV;
  typedef Tpetra::Map<LO,GO,Node>                   Map;
  typedef Ifpack2::Ifpack2_Hypre<Scalar,LO,GO,Node> Hypre;

  // Get a comm
  RCP<const Comm<int> > comm =
        Tpetra::DefaultPlatform::getDefaultPlatform ().getComm ();
  int myRank = comm->getRank();
  int numProcs = comm->getSize();

  //
  // Construct the map [0 0 1 1]
  // Note that the rows will be out of order
  //
  Array<int> elementList(2);
  elementList[0] = 2*myRank+1;
  elementList[1] = 2*myRank;
  RCP<Map> badMap = rcp(new Map(2*numProcs,elementList,0,comm));
  RCP<Map> goodMap = rcp(new Map(2*numProcs,0,comm));

  //
  // Construct the diagonal matrix
  //
  RCP<Matrix> matrix = rcp(new Matrix(badMap,1));
  Array<LO> indices(1);
  Array<Scalar> values(1);
  indices[0] = 2*myRank+1;
  values[0] = 2*myRank+2;
  matrix->insertGlobalValues(indices[0], indices(), values());
  indices[0] = 2*myRank;
  values[0] = 2*myRank+1;
  matrix->insertGlobalValues(indices[0], indices(), values());
  matrix->fillComplete(goodMap,goodMap);

  //
  // Create the parameter list
  //
  const double tol = 1e-7;
  Teuchos::ParameterList list("Preconditioner List");
  RCP<FunctionParameter> functs[5];
  functs[0] = rcp(new FunctionParameter(Solver, &HYPRE_PCGSetMaxIter, 100));               // max iterations
  functs[1] = rcp(new FunctionParameter(Solver, &HYPRE_PCGSetTol, tol));                   // conv. tolerance
  functs[2] = rcp(new FunctionParameter(Solver, &HYPRE_PCGSetTwoNorm, 1));                  // use the two norm as the stopping criteria
  functs[3] = rcp(new FunctionParameter(Solver, &HYPRE_PCGSetPrintLevel, 2));               // print solve info
  functs[4] = rcp(new FunctionParameter(Solver, &HYPRE_PCGSetLogging, 1));
  list.set("Solver", Ifpack2::Hypre::PCG);
  list.set("SolveOrPrecondition", Solver);
  list.set("SetPreconditioner", false);
  list.set("NumFunctions", 5);
  list.set<RCP<FunctionParameter>*>("Functions", functs);

  //
  // Create the preconditioner (which is actually a PCG solver)
  //
  Hypre preconditioner(matrix);
  preconditioner.setParameters(list);
  preconditioner.compute();

  // Create the RHS and solution vector
  int numVec = 2;
  MV X(badMap, numVec);
  MV B(badMap, numVec);
  B.randomize();

  //
  // Solve the linear system
  //
  TEST_THROW(preconditioner.apply(B,X),std::runtime_error);
}



// Creates the identity matrix with a non-contiguous row map
// Even though the Epetra identity matrix has a map that hypre should not be happy with,
// hypre should be able to redistribute it.  It should also be able to accept the
// vectors we give it, since they're using the same distribution as the hypre matrix.
// This tests hypre's ability to perform as a linear solver via ApplyInverse.
TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( Ifpack_Hypre, NonContiguousRowMap, Node ) {
  typedef Tpetra::CrsMatrix<Scalar,LO,GO,Node>      Matrix;
  typedef Tpetra::MultiVector<Scalar,LO,GO,Node>    MV;
  typedef Tpetra::Map<LO,GO,Node>                   Map;
  typedef Ifpack2::Ifpack2_Hypre<Scalar,LO,GO,Node> Hypre;

  // Get a comm
  RCP<const Comm<int> > comm =
        Tpetra::DefaultPlatform::getDefaultPlatform ().getComm ();
  int myRank = comm->getRank();
  int numProcs = comm->getSize();

  //
  // Construct the map [0 1 0 1]
  //
  Array<int> elementList(2);
  elementList[0] = myRank;
  elementList[1] = myRank+numProcs;
  RCP<Map> badMap = rcp(new Map(2*numProcs,elementList,0,comm));
  RCP<Map> goodMap = rcp(new Map(2*numProcs,0,comm));

  //
  // Construct the identity matrix
  //
  RCP<Matrix> matrix = rcp(new Matrix(badMap,1));
  Array<LO> indices(1);
  Array<Scalar> values(1);
  values[0] = 1.0;
  indices[0] = myRank;
  matrix->insertGlobalValues(indices[0], indices(), values());
  indices[0] = myRank + numProcs;
  matrix->insertGlobalValues(indices[0], indices(), values());
  matrix->fillComplete(goodMap,goodMap);

  //
  // Create the parameter list
  //
  const double tol = 1e-7;
  Teuchos::ParameterList list("Preconditioner List");
  RCP<FunctionParameter> functs[5];
  functs[0] = rcp(new FunctionParameter(Solver, &HYPRE_PCGSetMaxIter, 1));               // max iterations
  functs[1] = rcp(new FunctionParameter(Solver, &HYPRE_PCGSetTol, tol));                   // conv. tolerance
  functs[2] = rcp(new FunctionParameter(Solver, &HYPRE_PCGSetTwoNorm, 1));                  // use the two norm as the stopping criteria
  functs[3] = rcp(new FunctionParameter(Solver, &HYPRE_PCGSetPrintLevel, 2));               // print solve info
  functs[4] = rcp(new FunctionParameter(Solver, &HYPRE_PCGSetLogging, 1));
  list.set("Solver", Ifpack2::Hypre::PCG);
  list.set("SolveOrPrecondition", Solver);
  list.set("SetPreconditioner", false);
  list.set("NumFunctions", 5);
  list.set<RCP<FunctionParameter>*>("Functions", functs);

  //
  // Create the preconditioner (which is actually a PCG solver)
  //
  Hypre preconditioner(matrix);
  preconditioner.setParameters(list);
  preconditioner.compute();

  // Create the RHS and solution vector
  int numVec = 2;
  MV X(goodMap, numVec);
  MV B(goodMap, numVec);
  B.randomize();

  //
  // Solve the linear system
  //
  preconditioner.apply(B,X);
  TEST_EQUALITY(EquivalentVectors(X, B, tol*10*pow(10.0,numProcs)), true);
}

// Define typedefs that make the Tpetra macros work.
IFPACK2_ETI_MANGLING_TYPEDEFS()

// Macro that instantiates the unit test
#define LCLINST( NT ) \
TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( Ifpack_Hypre, Construct, NT ) \
TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( Ifpack_Hypre, ParameterList, NT ) \
TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( Ifpack_Hypre, Ifpack, NT ) \
TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( Ifpack_Hypre, DiagonalMatrixInOrder, NT ) \
TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( Ifpack_Hypre, DiagonalMatrixOutOfOrder, NT ) \
TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( Ifpack_Hypre, NonContiguousRowMap, NT )

// Ifpack2's ETI will instantiate the unit test for all enabled type
// combinations.
IFPACK2_INSTANTIATE_N( LCLINST )

} // end anonymous namespace

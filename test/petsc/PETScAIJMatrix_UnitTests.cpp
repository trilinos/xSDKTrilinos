/*
// @HEADER
// ***********************************************************************
//
//          Tpetra: Templated Linear Algebra Services Package
//                 Copyright (2008) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
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
// Questions? Contact Michael A. Heroux (maherou@sandia.gov)
//
// ************************************************************************
// @HEADER
*/

#ifdef HAVE_PETSC
#include <Tpetra_ConfigDefs.hpp>
#include <Tpetra_TestingUtilities.hpp>
#include <Tpetra_PETScAIJMatrix.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Teuchos_CommHelpers.hpp>

// TODO: add test where some nodes have zero rows
// TODO: add test where non-"zero" graph is used to build matrix; if no values are added to matrix, the operator effect should be zero. This tests that matrix values are initialized properly.
// TODO: add test where dynamic profile initially has no allocation, then entries are added. this will test new view functionality.

namespace Teuchos {
  template <>
    ScalarTraits<int>::magnitudeType
    relErr( const int &s1, const int &s2 )
    {
      typedef ScalarTraits<int> ST;
      return ST::magnitude(s1-s2);
    }

  template <>
    ScalarTraits<char>::magnitudeType
    relErr( const char &s1, const char &s2 )
    {
      typedef ScalarTraits<char> ST;
      return ST::magnitude(s1-s2);
    }
}

namespace {

  // no ScalarTraits<>::eps() for integer types

  template <class Scalar, bool hasMachineParameters> struct TestingTolGuts {};

  template <class Scalar>
  struct TestingTolGuts<Scalar, true> {
    static typename Teuchos::ScalarTraits<Scalar>::magnitudeType testingTol()
      { return Teuchos::ScalarTraits<Scalar>::eps(); }
  };

  template <class Scalar>
  struct TestingTolGuts<Scalar, false> {
    static typename Teuchos::ScalarTraits<Scalar>::magnitudeType testingTol()
      { return 0; }
  };

  template <class Scalar>
  static typename Teuchos::ScalarTraits<Scalar>::magnitudeType testingTol()
  {
    return TestingTolGuts<Scalar, Teuchos::ScalarTraits<Scalar>::hasMachineParameters>::
      testingTol();
  }

  using Tpetra::TestingUtilities::getNode;
  using Tpetra::TestingUtilities::getDefaultComm;

  using std::endl;
  using std::swap;

  using std::string;

  using Teuchos::TypeTraits::is_same;
  using Teuchos::as;
  using Teuchos::FancyOStream;
  using Teuchos::RCP;
  using Teuchos::ArrayRCP;
  using Teuchos::rcp;
  using Teuchos::arcp;
  using Teuchos::outArg;
  using Teuchos::arcpClone;
  using Teuchos::arrayView;
  using Teuchos::broadcast;
  using Teuchos::OrdinalTraits;
  using Teuchos::ScalarTraits;
  using Teuchos::Comm;
  using Teuchos::Array;
  using Teuchos::ArrayView;
  using Teuchos::tuple;
  using Teuchos::null;
  using Teuchos::VERB_NONE;
  using Teuchos::VERB_LOW;
  using Teuchos::VERB_MEDIUM;
  using Teuchos::VERB_HIGH;
  using Teuchos::VERB_EXTREME;
  using Teuchos::ETransp;
  using Teuchos::NO_TRANS;
  using Teuchos::TRANS;
  using Teuchos::CONJ_TRANS;
  using Teuchos::EDiag;
  using Teuchos::UNIT_DIAG;
  using Teuchos::NON_UNIT_DIAG;
  using Teuchos::EUplo;
  using Teuchos::UPPER_TRI;
  using Teuchos::LOWER_TRI;
  using Teuchos::ParameterList;
  using Teuchos::parameterList;

  using Tpetra::Map;
  using Tpetra::MultiVector;
  using Tpetra::Vector;
  using Tpetra::Operator;
  using Tpetra::PETScAIJMatrix;
  using Tpetra::RowMatrix;
  using Tpetra::Import;
  using Tpetra::global_size_t;
  using Tpetra::createNonContigMapWithNode;
  using Tpetra::createUniformContigMapWithNode;
  using Tpetra::createContigMapWithNode;
  using Tpetra::createLocalMapWithNode;
  using Tpetra::createVector;
  using Tpetra::DefaultPlatform;
  using Tpetra::ProfileType;
  using Tpetra::StaticProfile;
  using Tpetra::DynamicProfile;
  using Tpetra::OptimizeOption;
  using Tpetra::DoOptimizeStorage;
  using Tpetra::DoNotOptimizeStorage;
  using Tpetra::GloballyDistributed;
  using Tpetra::INSERT;


  double errorTolSlack = 1e+1;
  string filedir;

template <class tuple, class T>
inline void tupleToArray(Array<T> &arr, const tuple &tup)
{
  arr.assign(tup.begin(), tup.end());
}

#define STD_TESTS(matrix) \
  { \
    using Teuchos::outArg; \
    RCP<const Comm<int> > STCOMM = matrix.getComm(); \
    ArrayView<const GO> STMYGIDS = matrix.getRowMap()->getNodeElementList(); \
    ArrayView<const LO> loview; \
    ArrayView<const Scalar> sview; \
    size_t STMAX = 0; \
    for (size_t STR=0; STR < matrix.getNodeNumRows(); ++STR) { \
      const size_t numEntries = matrix.getNumEntriesInLocalRow(STR); \
      TEST_EQUALITY( numEntries, matrix.getNumEntriesInGlobalRow( STMYGIDS[STR] ) ); \
      matrix.getLocalRowView(STR,loview,sview); \
      TEST_EQUALITY( static_cast<size_t>(loview.size()), numEntries ); \
      TEST_EQUALITY( static_cast<size_t>( sview.size()), numEntries ); \
      STMAX = std::max( STMAX, numEntries ); \
    } \
    TEST_EQUALITY( matrix.getNodeMaxNumRowEntries(), STMAX ); \
    global_size_t STGMAX; \
    Teuchos::reduceAll<int,global_size_t>( *STCOMM, Teuchos::REDUCE_MAX, STMAX, outArg(STGMAX) ); \
    TEST_EQUALITY( matrix.getGlobalMaxNumRowEntries(), STGMAX ); \
  }


  TEUCHOS_STATIC_SETUP()
  {
    Teuchos::CommandLineProcessor &clp = Teuchos::UnitTestRepository::getCLP();
    clp.setOption(
        "filedir",&filedir,"Directory of expected matrix files.");
    clp.addOutputSetupOptions(true);
    clp.setOption(
        "test-mpi", "test-serial", &Tpetra::TestingUtilities::testMpi,
        "Test MPI (if available) or force test of serial.  In a serial build,"
        " this option is ignored and a serial comm is always used." );
    clp.setOption(
        "error-tol-slack", &errorTolSlack,
        "Slack off of machine epsilon used to check test results" );
  }


  //
  // UNIT TESTS
  //

  ////
  TEUCHOS_UNIT_TEST_TEMPLATE_2_DECL( PETScAIJMatrix, BadCalls, GO, Node )
  {
    RCP<Node> node = getNode<Node>();
    typedef PetscScalar Scalar;
    typedef int LO;
    typedef ScalarTraits<Scalar> ST;
    typedef MultiVector<Scalar,LO,GO,Node> MV;
    typedef PETScAIJMatrix<Scalar,LO,GO,Node> MAT;
    typedef typename ST::magnitudeType Mag;
    typedef RCP<const Map<LO,GO,Node> > RCPMap;
    typedef ScalarTraits<Mag> MT;
    const global_size_t INVALID = OrdinalTraits<global_size_t>::invalid();
    PetscErrorCode ierr;
    // get a comm
    RCP<const Comm<int> > comm = getDefaultComm();
    // create a Map
    const size_t numLocal = 10;
    RCP<MAT> zero;
    {
      Mat A;
      PetscInt Istart, Iend, Ii;
      PetscScalar v;
      int argc = 0;
      char ** argv;

      ierr = PetscInitialize(&argc,&argv,NULL,NULL);CHKERRV(ierr);

      ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRV(ierr);
      ierr = MatSetSizes(A,numLocal,numLocal,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRV(ierr);
      ierr = MatSetType(A, MATAIJ);CHKERRV(ierr);
      ierr = MatSetFromOptions(A);CHKERRV(ierr);
      ierr = MatMPIAIJSetPreallocation(A,0,PETSC_NULL,0,PETSC_NULL);CHKERRV(ierr);
      ierr = MatSetUp(A);CHKERRV(ierr);

      ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRV(ierr);
      ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRV(ierr);

      zero = rcp(new MAT(A));
    }
//amk TODO    STD_TESTS((*zero));
    TEST_EQUALITY_CONST( zero->getRangeMap() == zero->getDomainMap(), true );
    TEST_EQUALITY_CONST( zero->getFrobeniusNorm(), MT::zero() );
    const RCPMap drmap = zero->getDomainMap();
    {
      MV mv1(drmap,1), mv2(drmap,2), mv3(drmap,3);
      TEST_THROW(zero->apply(mv2,mv1), std::runtime_error); // MVs have different number of vectors
      TEST_THROW(zero->apply(mv2,mv3), std::runtime_error); // MVs have different number of vectors
    }
    // test that our assumptions on the maps are correct:
    // that is, that badmap is not equal to the range, domain, row or colum map of the matrix
    const RCPMap badmap = createContigMapWithNode<LO,GO>(INVALID,1,comm,node);
    TEST_EQUALITY_CONST( badmap != zero->getRowMap(), true );
    TEST_EQUALITY_CONST( badmap != zero->getColMap(), true );
    TEST_EQUALITY_CONST( badmap != zero->getDomainMap(), true );
    TEST_EQUALITY_CONST( badmap != zero->getRangeMap(),  true );
    TEST_EQUALITY_CONST( *badmap != *zero->getRowMap(), true );
    TEST_EQUALITY_CONST( *badmap != *zero->getColMap(), true );
    TEST_EQUALITY_CONST( *badmap != *zero->getDomainMap(), true );
    TEST_EQUALITY_CONST( *badmap != *zero->getRangeMap(),  true );

    ierr = PetscFinalize();CHKERRV(ierr);
  }


  ////
  TEUCHOS_UNIT_TEST_TEMPLATE_2_DECL( PETScAIJMatrix, TheEyeOfTruth, GO, Node )
  {
    RCP<Node> node = getNode<Node>();
    typedef PetscScalar Scalar;
    typedef int LO;
    typedef ScalarTraits<Scalar> ST;
    typedef PETScAIJMatrix<Scalar,LO,GO,Node> MAT;
    typedef MultiVector<Scalar,LO,GO,Node> MV;
    typedef typename ST::magnitudeType Mag;
    typedef ScalarTraits<Mag> MT;
    const global_size_t INVALID = OrdinalTraits<global_size_t>::invalid();
    PetscErrorCode ierr;
    // get a comm
    RCP<const Comm<int> > comm = getDefaultComm();
    const size_t numImages = comm->getSize();
    const size_t myImageID = comm->getRank();
    // create a Map
    const size_t numLocal = 10;
    const size_t numVecs  = 5;
    RCP<const Map<LO,GO,Node> > map = createContigMapWithNode<LO,GO>(INVALID,numLocal,comm,node);
    MV mvrand(map,numVecs,false), mvres(map,numVecs,false);
    mvrand.randomize();
    // create the identity matrix
    RCP<RowMatrix<Scalar,LO,GO,Node> > eye;
    {
      Mat A;
      PetscInt Istart, Iend, Ii;
      PetscScalar v;
      int argc = 0;
      char ** argv;

      ierr = PetscInitialize(&argc,&argv,NULL,NULL);CHKERRV(ierr);

      ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRV(ierr);
      ierr = MatSetSizes(A,numLocal,numLocal,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRV(ierr);
      ierr = MatSetType(A, MATAIJ);CHKERRV(ierr);
      ierr = MatSetFromOptions(A);CHKERRV(ierr);
      ierr = MatMPIAIJSetPreallocation(A,1,PETSC_NULL,0,PETSC_NULL);CHKERRV(ierr);
      ierr = MatSetUp(A);CHKERRV(ierr);

      ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRV(ierr);

      for (Ii=Istart; Ii<Iend; Ii++) { 
        v = 1.0; ierr = MatSetValues(A,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRV(ierr);
      }

      ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRV(ierr);
      ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRV(ierr);

      eye = rcp(new MAT(A));
    }
    // test the properties
    TEST_EQUALITY(eye->getGlobalNumEntries()  , numImages*numLocal);
    TEST_EQUALITY(eye->getNodeNumEntries()      , numLocal);
    TEST_EQUALITY(eye->getGlobalNumRows()      , numImages*numLocal);
    TEST_EQUALITY(eye->getNodeNumRows()          , numLocal);
    TEST_EQUALITY(eye->getNodeNumCols()          , numLocal);
    TEST_EQUALITY(eye->getGlobalNumDiags() , numImages*numLocal);
    TEST_EQUALITY(eye->getNodeNumDiags()     , numLocal);
    TEST_EQUALITY(eye->getGlobalMaxNumRowEntries(), 1);
    TEST_EQUALITY(eye->getNodeMaxNumRowEntries()    , 1);
    TEST_EQUALITY(eye->getIndexBase()          , 0);
    TEST_EQUALITY_CONST(eye->getRowMap()!=Teuchos::null, true);
    TEST_EQUALITY_CONST(eye->getColMap()!=Teuchos::null, true);
    TEST_EQUALITY_CONST(eye->getDomainMap()!=Teuchos::null, true);
    TEST_EQUALITY_CONST(eye->getRangeMap()!=Teuchos::null, true);
    TEST_EQUALITY_CONST(eye->getRowMap()->isSameAs(*eye->getColMap())   , true);
    TEST_EQUALITY_CONST(eye->getRowMap()->isSameAs(*eye->getDomainMap()), true);
    TEST_EQUALITY_CONST(eye->getRowMap()->isSameAs(*eye->getRangeMap()) , true);
    // test the action
    mvres.randomize();
    eye->apply(mvrand,mvres);
    mvres.update(-ST::one(),mvrand,ST::one());
    Array<Mag> norms(numVecs), zeros(numVecs,MT::zero());
    mvres.norm1(norms());
    if (ST::isOrdinal) {
      TEST_COMPARE_ARRAYS(norms,zeros);
    } else {
      TEST_COMPARE_FLOATING_ARRAYS(norms,zeros,MT::zero());
    }

    ierr = PetscFinalize();CHKERRV(ierr);
  }


  ////
  TEUCHOS_UNIT_TEST_TEMPLATE_2_DECL( PETScAIJMatrix, SimpleEigTest, GO, Node )
  {
    RCP<Node> node = getNode<Node>();
    typedef PetscScalar Scalar;
    typedef int LO;
    typedef PETScAIJMatrix<Scalar,LO,GO,Node> MAT;
    typedef ScalarTraits<Scalar> ST;
    typedef MultiVector<Scalar,LO,GO,Node> MV;
    typedef typename ST::magnitudeType Mag;
    typedef ScalarTraits<Mag> MT;
    const size_t ONE = OrdinalTraits<size_t>::one();
    const global_size_t INVALID = OrdinalTraits<global_size_t>::invalid();
    PetscErrorCode ierr;
    // get a comm
    RCP<const Comm<int> > comm = getDefaultComm();
    const size_t numImages = comm->getSize();
    const size_t myImageID = comm->getRank();
    if (numImages < 2) return;
    // create a Map
    RCP<const Map<LO,GO,Node> > map = createContigMapWithNode<LO,GO>(INVALID,ONE,comm,node);
    // create a multivector ones(n,1)
    MV ones(map,ONE,false), threes(map,ONE,false);
    ones.putScalar(ST::one());
    /* create the following matrix:
       [2 1           ]
       [1 1 1         ]
       [  1 1 1       ]
       [   . . .      ]
       [     . . .    ]
       [       . . .  ]
       [         1 1 1]
       [           1 2]
     this matrix has an eigenvalue lambda=3, with eigenvector v = [1 ... 1]
    */
    size_t myNNZ;
    if(myImageID == 0 || myImageID == numImages-1)
      myNNZ = 2;
    else
      myNNZ = 3;
    RCP<MAT> A;
    {
      Mat PA;
      PetscInt Istart, Iend, Ii, tempCol;
      PetscScalar v;
      int argc = 0;
      char ** argv;

      ierr = PetscInitialize(&argc,&argv,NULL,NULL);CHKERRV(ierr);

      ierr = MatCreate(PETSC_COMM_WORLD,&PA);CHKERRV(ierr);
      ierr = MatSetSizes(PA,ONE,ONE,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRV(ierr);
      ierr = MatSetType(PA, MATAIJ);CHKERRV(ierr);
      ierr = MatSetFromOptions(PA);CHKERRV(ierr);
      ierr = MatMPIAIJSetPreallocation(PA,1,PETSC_NULL,2,PETSC_NULL);CHKERRV(ierr);
      ierr = MatSetUp(PA);CHKERRV(ierr);

      ierr = MatGetOwnershipRange(PA,&Istart,&Iend);CHKERRV(ierr);

      for (Ii=Istart; Ii<Iend; Ii++) { 
        if(Ii > 0) {v = 1.0; tempCol=Ii-1; ierr = MatSetValues(PA,1,&Ii,1,&tempCol,&v,INSERT_VALUES);CHKERRV(ierr);}
        if(Ii < numImages-1) {v = 1.0; tempCol=Ii+1; ierr = MatSetValues(PA,1,&Ii,1,&tempCol,&v,INSERT_VALUES);CHKERRV(ierr);}
        if(Ii == 0 || Ii == numImages-1) {v = 2.0; ierr = MatSetValues(PA,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRV(ierr);}
        else {v = 1.0; ierr = MatSetValues(PA,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRV(ierr);}
      }

      ierr = MatAssemblyBegin(PA,MAT_FINAL_ASSEMBLY);CHKERRV(ierr);
      ierr = MatAssemblyEnd(PA,MAT_FINAL_ASSEMBLY);CHKERRV(ierr);

      A = rcp(new MAT(PA));
    }
    // test the properties
    TEST_EQUALITY(A->getGlobalNumEntries()   , static_cast<size_t>(3*numImages-2));
    TEST_EQUALITY(A->getNodeNumEntries()       , myNNZ);
    TEST_EQUALITY(A->getGlobalNumRows()       , static_cast<size_t>(numImages));
    TEST_EQUALITY_CONST(A->getNodeNumRows()     , ONE);
    TEST_EQUALITY(A->getNodeNumCols()           , myNNZ);
    TEST_EQUALITY(A->getGlobalNumDiags()  , static_cast<size_t>(numImages));
    TEST_EQUALITY_CONST(A->getNodeNumDiags(), ONE);
    TEST_EQUALITY(A->getGlobalMaxNumRowEntries() , (numImages > 2 ? 3 : 2));
    TEST_EQUALITY(A->getNodeMaxNumRowEntries()     , myNNZ);
    TEST_EQUALITY_CONST(A->getIndexBase()     , 0);
    TEST_EQUALITY_CONST(A->getRowMap()->isSameAs(*A->getColMap())   , false);
    TEST_EQUALITY_CONST(A->getRowMap()->isSameAs(*A->getDomainMap()), true);
    TEST_EQUALITY_CONST(A->getRowMap()->isSameAs(*A->getRangeMap()) , true);
    // test the action
    threes.randomize();
    A->apply(ones,threes);
    // now, threes should be 3*ones
    threes.update(static_cast<Scalar>(-3)*ST::one(),ones,ST::one());
    Array<Mag> norms(1), zeros(1,MT::zero());
    threes.norm1(norms());
    if (ST::isOrdinal) {
      TEST_COMPARE_ARRAYS(norms,zeros);
    } else {
      TEST_COMPARE_FLOATING_ARRAYS(norms,zeros,MT::zero());
    }

    ierr = PetscFinalize();CHKERRV(ierr);
  }


  ////
  TEUCHOS_UNIT_TEST_TEMPLATE_2_DECL( PETScAIJMatrix, ZeroMatrix, GO, Node )
  {
    RCP<Node> node = getNode<Node>();
    typedef PetscScalar Scalar;
    typedef int LO;
    typedef PETScAIJMatrix<Scalar,LO,GO,Node> MAT;
    typedef ScalarTraits<Scalar> ST;
    typedef MultiVector<Scalar,LO,GO,Node> MV;
    typedef typename ST::magnitudeType Mag;
    typedef ScalarTraits<Mag> MT;
    const global_size_t INVALID = OrdinalTraits<global_size_t>::invalid();
    PetscErrorCode ierr;
    // get a comm
    RCP<const Comm<int> > comm = getDefaultComm();
    // create a Map
    const size_t numLocal = 10;
    const size_t numVecs  = 5;
    RCP<const Map<LO,GO,Node> > map = createContigMapWithNode<LO,GO>(INVALID,numLocal,comm,node);
    // create the zero matrix
    RCP<MAT> zero;
    {
      Mat A;
      PetscInt Istart, Iend, Ii;
      PetscScalar v;
      int argc = 0;
      char ** argv;

      ierr = PetscInitialize(&argc,&argv,NULL,NULL);CHKERRV(ierr);

      ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRV(ierr);
      ierr = MatSetSizes(A,numLocal,numLocal,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRV(ierr);
      ierr = MatSetType(A, MATAIJ);CHKERRV(ierr);
      ierr = MatSetFromOptions(A);CHKERRV(ierr);
      ierr = MatMPIAIJSetPreallocation(A,0,PETSC_NULL,0,PETSC_NULL);CHKERRV(ierr);
      ierr = MatSetUp(A);CHKERRV(ierr);

      ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRV(ierr);
      ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRV(ierr);

      zero = rcp(new MAT(A));
    }
    //
    MV mvrand(map,numVecs,false), mvres(map,numVecs,false);
    mvrand.randomize();
    mvres.putScalar(1);
    zero->apply(mvrand,mvres);
    Array<Mag> norms(numVecs), zeros(numVecs,MT::zero());
    mvres.norm1(norms());
    if (ST::isOrdinal) {
      TEST_COMPARE_ARRAYS(norms,zeros);
    } else {
      TEST_COMPARE_FLOATING_ARRAYS(norms,zeros,MT::zero());
    }

    ierr = PetscFinalize();CHKERRV(ierr);
  }


  ////
  TEUCHOS_UNIT_TEST_TEMPLATE_2_DECL( PETScAIJMatrix, FullMatrixTriDiag, GO, Node )
  {
    RCP<Node> node = getNode<Node>();
    typedef PetscScalar Scalar;
    typedef int LO;
    // do a FEM-type communication, then apply to a MultiVector containing the identity
    // this will check non-trivial communication and test multivector apply
    typedef PETScAIJMatrix<Scalar,LO,GO,Node> MAT;
    typedef ScalarTraits<Scalar> ST;
    typedef MultiVector<Scalar,LO,GO,Node> MV;
    typedef typename ST::magnitudeType Mag;
    typedef ScalarTraits<Mag> MT;
    const size_t ONE = OrdinalTraits<size_t>::one();
    const global_size_t INVALID = OrdinalTraits<global_size_t>::invalid();
    PetscErrorCode ierr;
    // get a comm
    RCP<const Comm<int> > comm = getDefaultComm();
    const size_t numImages = comm->getSize();
    const size_t myImageID = comm->getRank();
    if (numImages < 3) return;
    // create a Map
    RCP<const Map<LO,GO,Node> > map = createContigMapWithNode<LO,GO>(INVALID,ONE,comm,node);

    // RCP<FancyOStream> fos = Teuchos::fancyOStream(rcp(&std::cout,false));

    /* Create the following matrix:
    0  [2 1       ]   [2 1]
    1  [1 4 1     ]   [1 2] + [2 1]
    2  [  1 4 1   ]           [1 2] +
    3  [    1     ] =
       [       4 1]
   n-1 [       1 2]
    */
    RCP<MAT> A;
    {
      Mat PA;
      PetscInt Istart, Iend, Ii, tempCol;
      PetscScalar v;
      int argc = 0;
      char ** argv;

      ierr = PetscInitialize(&argc,&argv,NULL,NULL);CHKERRV(ierr);

      ierr = MatCreate(PETSC_COMM_WORLD,&PA);CHKERRV(ierr);
      ierr = MatSetSizes(PA,ONE,PETSC_DECIDE,numImages,numImages);CHKERRV(ierr);
      ierr = MatSetType(PA, MATAIJ);CHKERRV(ierr);
      ierr = MatSetFromOptions(PA);CHKERRV(ierr);
      ierr = MatMPIAIJSetPreallocation(PA,1,PETSC_NULL,2,PETSC_NULL);CHKERRV(ierr);
      ierr = MatSetUp(PA);CHKERRV(ierr);

      ierr = MatGetOwnershipRange(PA,&Istart,&Iend);CHKERRV(ierr);

      for (Ii=Istart; Ii<Iend; Ii++) { 
        if(Ii > 0) {v = 1.0; tempCol=Ii-1; ierr = MatSetValues(PA,1,&Ii,1,&tempCol,&v,INSERT_VALUES);CHKERRV(ierr);}
        if(Ii < numImages-1) {v = 1.0; tempCol=Ii+1; ierr = MatSetValues(PA,1,&Ii,1,&tempCol,&v,INSERT_VALUES);CHKERRV(ierr);}
        if(Ii == 0 || Ii == numImages-1) {v = 2.0; ierr = MatSetValues(PA,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRV(ierr);}
        else {v = 4.0; ierr = MatSetValues(PA,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRV(ierr);}
      }

      ierr = MatAssemblyBegin(PA,MAT_FINAL_ASSEMBLY);CHKERRV(ierr);
      ierr = MatAssemblyEnd(PA,MAT_FINAL_ASSEMBLY);CHKERRV(ierr);

      A = rcp(new MAT(PA));
    }
    size_t myNNZ;
    A->setObjectLabel("The Matrix");
    MV mveye(map,numImages), mvans(map,numImages), mvres(map,numImages,true);
    mveye.setObjectLabel("mveye");
    mvans.setObjectLabel("mvans");
    mvres.setObjectLabel("mvres");
    // put one on the diagonal, setting mveye to the identity
    mveye.replaceLocalValue(0,myImageID,ST::one());
    // divine myNNZ and build multivector with matrix
    if (myImageID == 0) {
      myNNZ = 2;
      mvans.replaceLocalValue(0,0,static_cast<Scalar>(2));
      mvans.replaceLocalValue(0,1,static_cast<Scalar>(1));
    }
    else if (myImageID == numImages-1) {
      myNNZ = 2;
      mvans.replaceLocalValue(0,numImages-2,static_cast<Scalar>(1));
      mvans.replaceLocalValue(0,numImages-1,static_cast<Scalar>(2));
    }
    else {
      myNNZ = 3;
      mvans.replaceLocalValue(0,myImageID-1,static_cast<Scalar>(1));
      mvans.replaceLocalValue(0,myImageID  ,static_cast<Scalar>(4));
      mvans.replaceLocalValue(0,myImageID+1,static_cast<Scalar>(1));
    }

    // test the properties
    TEST_EQUALITY(A->getGlobalNumEntries()     , static_cast<size_t>(3*numImages-2));
    TEST_EQUALITY(A->getNodeNumEntries()       , myNNZ);
    TEST_EQUALITY(A->getGlobalNumRows()       , static_cast<size_t>(numImages));
    TEST_EQUALITY_CONST(A->getNodeNumRows()     , ONE);
    TEST_EQUALITY(A->getNodeNumCols()           , myNNZ);
    TEST_EQUALITY(A->getGlobalNumDiags()  , static_cast<size_t>(numImages));
    TEST_EQUALITY_CONST(A->getNodeNumDiags(), ONE);
    TEST_EQUALITY(A->getGlobalMaxNumRowEntries() , 3);
    TEST_EQUALITY(A->getNodeMaxNumRowEntries()     , myNNZ);
    TEST_EQUALITY_CONST(A->getIndexBase()     , 0);
    TEST_EQUALITY_CONST(A->getRowMap()->isSameAs(*A->getColMap())   , false);
    TEST_EQUALITY_CONST(A->getRowMap()->isSameAs(*A->getDomainMap()), true);
    TEST_EQUALITY_CONST(A->getRowMap()->isSameAs(*A->getRangeMap()) , true);
    // test the action
    A->apply(mveye,mvres);
    mvres.update(-ST::one(),mvans,ST::one());
    Array<Mag> norms(numImages), zeros(numImages,MT::zero());
    mvres.norm1(norms());
    if (ST::isOrdinal) {
      TEST_COMPARE_ARRAYS(norms,zeros);
    } else {
      const Mag tol = TestingTolGuts<Mag, ! MT::isOrdinal>::testingTol ();
      TEST_COMPARE_FLOATING_ARRAYS( norms, zeros, tol );
    }

    ierr = PetscFinalize();CHKERRV(ierr);
  }


  ////
  TEUCHOS_UNIT_TEST_TEMPLATE_2_DECL( PETScAIJMatrix, CopiesAndViews, GO, Node )
  {
    RCP<Node> node = getNode<Node>();
    typedef PetscScalar Scalar;
    typedef int LO;
    // test that an exception is thrown when we exceed statically allocated memory
    typedef ScalarTraits<Scalar> ST;
    typedef PETScAIJMatrix<Scalar,LO,GO,Node> MAT;
    const global_size_t INVALID = OrdinalTraits<global_size_t>::invalid();
    PetscErrorCode ierr;
    // get a comm
    RCP<const Comm<int> > comm = getDefaultComm();
    const size_t numImages = size(*comm);
    const size_t myImageID = rank(*comm);
    if (numImages < 2) return;
    // create a Map, one row per processor
    const size_t numLocal = 1;
    RCP<const Map<LO,GO,Node> > rmap = createContigMapWithNode<LO,GO>(INVALID,numLocal,comm,node);
    GO myrowind = rmap->getGlobalElement(0);
    // specify the column map to control ordering
    // construct tridiagonal graph
    Array<GO> ginds;
    Array<LO> linds;
    if (myImageID==0) {
      tupleToArray( ginds, tuple<GO>(myrowind,myrowind+1) );
      tupleToArray( linds, tuple<LO>(0,1) );
    }
    else if (myImageID==numImages-1) {
      tupleToArray( ginds , tuple<GO>(myrowind-1,myrowind) );
      tupleToArray( linds , tuple<LO>(numImages-2,numImages-1) );
    }
    else {
      tupleToArray( ginds , tuple<GO>(myrowind-1,myrowind,myrowind+1) );
      tupleToArray( linds , tuple<LO>(myImageID-1,myImageID,myImageID+1) );
    }
    Array<Scalar> vals(ginds.size(),ST::one());
    RCP<Map<LO,GO,Node> > cmap = rcp( new Map<LO,GO,Node>(INVALID,ginds(),0,comm,node) );

    RCP<MAT> matrix;
    {
      Mat PA;
      PetscInt Istart, Iend, Ii, tempCol;
      PetscScalar v;
      int argc = 0;
      char ** argv;

      ierr = PetscInitialize(&argc,&argv,NULL,NULL);CHKERRV(ierr);

      ierr = MatCreate(PETSC_COMM_WORLD,&PA);CHKERRV(ierr);
      ierr = MatSetSizes(PA,1,PETSC_DECIDE,numImages,numImages);CHKERRV(ierr);
      ierr = MatSetType(PA, MATAIJ);CHKERRV(ierr);
      ierr = MatSetFromOptions(PA);CHKERRV(ierr);
      ierr = MatMPIAIJSetPreallocation(PA,1,PETSC_NULL,2,PETSC_NULL);CHKERRV(ierr);
      ierr = MatSetUp(PA);CHKERRV(ierr);

      ierr = MatGetOwnershipRange(PA,&Istart,&Iend);CHKERRV(ierr);

      for (Ii=Istart; Ii<Iend; Ii++) { 
        if(Ii > 0) {v = 1.0; tempCol=Ii-1; ierr = MatSetValues(PA,1,&Ii,1,&tempCol,&v,INSERT_VALUES);CHKERRV(ierr);}
        if(Ii < numImages-1) {v = 1.0; tempCol=Ii+1; ierr = MatSetValues(PA,1,&Ii,1,&tempCol,&v,INSERT_VALUES);CHKERRV(ierr);}
        v = 1.0; ierr = MatSetValues(PA,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRV(ierr);
      }

      ierr = MatAssemblyBegin(PA,MAT_FINAL_ASSEMBLY);CHKERRV(ierr);
      ierr = MatAssemblyEnd(PA,MAT_FINAL_ASSEMBLY);CHKERRV(ierr);

      matrix = rcp(new MAT(PA));
    }

    Array<GO> GCopy(4); Array<LO> LCopy(4); Array<Scalar> SCopy(4);
    size_t numentries;
    // check for throws and no-throws/values
    TEST_THROW( matrix->getLocalRowCopy(    0       ,LCopy(0,1),SCopy(0,1),numentries), std::runtime_error );
    TEST_THROW( matrix->getGlobalRowCopy(myrowind,GCopy(0,1),SCopy(0,1),numentries), std::runtime_error );
    //
    TEST_NOTHROW( matrix->getLocalRowCopy(0,LCopy,SCopy,numentries) );
    TEST_COMPARE_ARRAYS( LCopy(0,numentries), linds );
    TEST_COMPARE_ARRAYS( SCopy(0,numentries), vals  );
    //
    TEST_NOTHROW( matrix->getGlobalRowCopy(myrowind,GCopy,SCopy,numentries) );
    TEST_COMPARE_ARRAYS( GCopy(0,numentries), ginds );
    TEST_COMPARE_ARRAYS( SCopy(0,numentries), vals  );
    //
// amk TODO    STD_TESTS(matrix);

    // Teuchos::reduceAll and Teuchos::REDUCE_SUM are failing to compile for some mysterious reason.
    // Use Tpetra::Vector instead to do the final all-reduce to check success.
    RCP<const Tpetra::Map<int, GO, Node> > finalMap (new Tpetra::Map<int, GO, Node> (Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid (), 1, 0, comm));
    Tpetra::Vector<int,int, GO, Node> finalVec (finalMap);
    finalVec.putScalar (success ? 0 : 1);
    const int globalSuccess = finalVec.normInf ();
    TEST_EQUALITY_CONST( globalSuccess, 0 );
/*
    // All procs fail if any node fails
    int globalSuccess_int = -1;
    Teuchos::reduceAll<int, int> (*comm, ::Teuchos::REDUCE_SUM, success ? 0 : 1, Teuchos::outArg (globalSuccess_int));
    TEST_EQUALITY_CONST( globalSuccess_int, 0 );
*/
    ierr = PetscFinalize();CHKERRV(ierr);
  }


  ////
  TEUCHOS_UNIT_TEST_TEMPLATE_2_DECL( PETScAIJMatrix, AlphaBetaMultiply, GO, Node )
  {
    RCP<Node> node = getNode<Node>();
    typedef PetscScalar Scalar;
    typedef int LO;
    typedef PETScAIJMatrix<Scalar,LO,GO,Node> MAT;
    typedef  Operator<Scalar,LO,GO,Node> OP;
    typedef ScalarTraits<Scalar> ST;
    typedef MultiVector<Scalar,LO,GO,Node> MV;
    typedef typename ST::magnitudeType Mag;
    const size_t THREE = 3;
    const global_size_t INVALID = OrdinalTraits<global_size_t>::invalid();
    PetscErrorCode ierr;
    // get a comm
    RCP<const Comm<int> > comm = getDefaultComm();
    const size_t myImageID = comm->getRank();
    // create a Map
    RCP<const Map<LO,GO,Node> > map = createContigMapWithNode<LO,GO>(INVALID,THREE,comm,node);

    // Create the identity matrix, three rows per proc 
    RCP<RowMatrix<Scalar,LO,GO,Node> > AOp;
    {
      Mat A;
      PetscInt Istart, Iend, Ii;
      PetscScalar v;
      int argc = 0;
      char ** argv;

      ierr = PetscInitialize(&argc,&argv,NULL,NULL);CHKERRV(ierr);

      ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRV(ierr);
      ierr = MatSetSizes(A,THREE,THREE,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRV(ierr);
      ierr = MatSetType(A, MATAIJ);CHKERRV(ierr);
      ierr = MatSetFromOptions(A);CHKERRV(ierr);
      ierr = MatMPIAIJSetPreallocation(A,1,PETSC_NULL,0,PETSC_NULL);CHKERRV(ierr);
      ierr = MatSetUp(A);CHKERRV(ierr);

      ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRV(ierr);

      for (Ii=Istart; Ii<Iend; Ii++) { 
        v = 1.0; ierr = MatSetValues(A,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRV(ierr);
      }

      ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRV(ierr);
      ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRV(ierr);

      AOp = rcp(new MAT(A));
    }
    MV X(map,1), Y(map,1), Z(map,1);
    const Scalar alpha = ST::random(),
                  beta = ST::random();
    X.randomize();
    Y.randomize();
    // Z = alpha*X + beta*Y
    Z.update(alpha,X,beta,Y,ST::zero());
    // test the action: Y = alpha*I*X + beta*Y = alpha*X + beta*Y = Z
    AOp->apply(X,Y,NO_TRANS,alpha,beta);
    //
    Array<Mag> normY(1), normZ(1);
    Z.norm1(normZ());
    Y.norm1(normY());
    if (ST::isOrdinal) {
      TEST_COMPARE_ARRAYS(normY,normZ);
    } else {
      TEST_COMPARE_FLOATING_ARRAYS(normY,normZ,2.0*testingTol<Mag>());
    }

    ierr = PetscFinalize();CHKERRV(ierr);
  }


  ////
  TEUCHOS_UNIT_TEST_TEMPLATE_2_DECL( PETScAIJMatrix, Typedefs, GO, Node )
  {
    typedef PetscScalar Scalar;
    typedef int LO;
    typedef PETScAIJMatrix<Scalar,LO,GO,Node> MAT;
    typedef typename MAT::scalar_type         scalar_type;
    typedef typename MAT::local_ordinal_type  local_ordinal_type;
    typedef typename MAT::global_ordinal_type global_ordinal_type;
    typedef typename MAT::node_type           node_type;
    TEST_EQUALITY_CONST( (is_same< scalar_type         , Scalar >::value) == true, true );
    TEST_EQUALITY_CONST( (is_same< local_ordinal_type  , LO     >::value) == true, true );
    TEST_EQUALITY_CONST( (is_same< global_ordinal_type , GO     >::value) == true, true );
    TEST_EQUALITY_CONST( (is_same< node_type           , Node   >::value) == true, true );
    typedef RowMatrix<Scalar,LO,GO,Node> RMAT;
    typedef typename RMAT::scalar_type         rmat_scalar_type;
    typedef typename RMAT::local_ordinal_type  rmat_local_ordinal_type;
    typedef typename RMAT::global_ordinal_type rmat_global_ordinal_type;
    typedef typename RMAT::node_type           rmat_node_type;
    TEST_EQUALITY_CONST( (is_same< rmat_scalar_type         , Scalar >::value) == true, true );
    TEST_EQUALITY_CONST( (is_same< rmat_local_ordinal_type  , LO     >::value) == true, true );
    TEST_EQUALITY_CONST( (is_same< rmat_global_ordinal_type , GO     >::value) == true, true );
    TEST_EQUALITY_CONST( (is_same< rmat_node_type           , Node   >::value) == true, true );
  }


//
// INSTANTIATIONS
//

#define UNIT_TEST_GROUP( NODE ) \
      TEUCHOS_UNIT_TEST_TEMPLATE_2_INSTANT( PETScAIJMatrix, TheEyeOfTruth,     PetscInt, NODE ) \
      TEUCHOS_UNIT_TEST_TEMPLATE_2_INSTANT( PETScAIJMatrix, ZeroMatrix,        PetscInt, NODE ) \
      TEUCHOS_UNIT_TEST_TEMPLATE_2_INSTANT( PETScAIJMatrix, BadCalls,          PetscInt, NODE ) \
      TEUCHOS_UNIT_TEST_TEMPLATE_2_INSTANT( PETScAIJMatrix, SimpleEigTest,     PetscInt, NODE ) \
      TEUCHOS_UNIT_TEST_TEMPLATE_2_INSTANT( PETScAIJMatrix, FullMatrixTriDiag, PetscInt, NODE ) \
      TEUCHOS_UNIT_TEST_TEMPLATE_2_INSTANT( PETScAIJMatrix, CopiesAndViews,    PetscInt, NODE ) \
      TEUCHOS_UNIT_TEST_TEMPLATE_2_INSTANT( PETScAIJMatrix, AlphaBetaMultiply, PetscInt, NODE ) \
      TEUCHOS_UNIT_TEST_TEMPLATE_2_INSTANT( PETScAIJMatrix, Typedefs,          PetscInt, NODE )


  TPETRA_ETI_MANGLING_TYPEDEFS()

  TPETRA_INSTANTIATE_N( UNIT_TEST_GROUP )

}

#endif

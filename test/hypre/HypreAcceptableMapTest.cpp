// hypre requires a certain parallel distribution
// Each MPI process must own a set of contiguous rows
// This test holds that assumption, even though the rows
// are not in the traditional order

#include "Ifpack2_Hypre.hpp"

#include "Tpetra_CrsMatrix.hpp"
#include "Tpetra_DefaultPlatform.hpp"

int main(int argc, char *argv[]) {
  using Teuchos::RCP;
  using Teuchos::rcp;

  //
  // Specify types used in this example
  //
  typedef Tpetra::CrsMatrix<>::scalar_type             Scalar;
  typedef Tpetra::CrsMatrix<>::local_ordinal_type      LO;
  typedef Tpetra::CrsMatrix<>::global_ordinal_type     GO;
  typedef Tpetra::CrsMatrix<>::node_type               Node;
  typedef Tpetra::Map<>                                Map;
  typedef Tpetra::DefaultPlatform::DefaultPlatformType Platform;
  typedef Tpetra::CrsMatrix<Scalar>                    CrsMatrix;
  typedef Tpetra::Vector<Scalar>                       Vector;
  typedef Tpetra::Operator<Scalar>                     OP;
  typedef Ifpack2::Preconditioner<Scalar>              Preconditioner;

  //
  // Initialize the MPI session
  //
  Teuchos::oblackholestream blackhole;
  Teuchos::GlobalMPISession mpiSession(&argc,&argv,&blackhole);

  //
  // Get the default communicator and node
  //
  Platform &platform = Tpetra::DefaultPlatform::getDefaultPlatform();
  RCP<const Teuchos::Comm<int> > comm = platform.getComm();
  RCP<Node> node = platform.getNode();
  const int myRank = comm->getRank();

  //
  // Construct the map [0 0 1 1]
  // Note that the rows will be out of order
  //
  Teuchos::Array<GO> elementList(2);
  if(myRank == 0)
  {
    elementList[0] = 1;
    elementList[1] = 0;
  }
  else
  {
    elementList[0] = 3;
    elementList[1] = 2;
  }
  RCP<Map> accMap = rcp(new Map(4,elementList(),0,comm,node));

  //
  // Construct the diagonal matrix (5,7,8,2)
  //
  RCP<CrsMatrix> accMat = rcp(new CrsMatrix(accMap,1,Tpetra::StaticProfile));
  Teuchos::Array<GO> cols(1);
  Teuchos::Array<Scalar> vals(1);
  if(myRank == 0)
  {
    cols[0] = 1;
    vals[0] = 7;
    accMat->insertGlobalValues(0,cols(),vals());
    cols[0] = 0;
    vals[0] = 5;
    accMat->insertGlobalValues(2,cols(),vals());
  }
  else
  {
    cols[0] = 3;
    vals[0] = 2;
    accMat->insertGlobalValues(1,cols(),vals());
    cols[0] = 2;
    vals[0] = 8;
    accMat->insertGlobalValues(3,cols(),vals());
  }
  accMat->fillComplete();

  //
  // Construct the RHS [20 14 8 18]
  //
  RCP<Vector> rhs = rcp(new Vector(accMap));
  if(myRank == 0)
  {
    rhs->replaceLocalValue(1,20);
    rhs->replaceLocalValue(0,14);
  }
  else
  {
    rhs->replaceLocalValue(1,8);
    rhs->replaceLocalValue(0,18);
  }

  //
  // Create the preconditioner
  //
  RCP<Preconditioner> prec = rcp(new Ifpack2::Ifpack2_Hypre<Scalar,LO,GO,Node>(accMat));
  prec->initialize();
  prec->compute();

  //
  // Apply the preconditioner to the RHS
  //
  RCP<Vector> sol = rcp(new Vector(accMap));
  prec->apply(*rhs,*sol);

  std::cout << "solution: " << *sol << std::endl;

  //
  // If we reached this point without an exception, the test passed
  //
  if(myRank == 0)
    std::cout << "End Result: TEST PASSED" << std::endl;
  return 0;
}

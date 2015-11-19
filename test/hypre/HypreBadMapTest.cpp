// hypre requires a certain parallel distribution
// Each MPI process must own a set of contiguous rows
// This test violates that assumption

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
  // Construct the map [0 1 0 1]
  //
  Teuchos::Array<GO> elementList(2);
  if(myRank == 0)
  {
    elementList[0] = 0;
    elementList[1] = 2;
  }
  else
  {
    elementList[0] = 1;
    elementList[1] = 3;
  }
  RCP<Map> badMap = rcp(new Map(4,elementList(),0,comm,node));

  //
  // Construct the identity matrix
  //
  RCP<CrsMatrix> badMat = rcp(new CrsMatrix(badMap,1,Tpetra::StaticProfile));
  Teuchos::Array<GO> cols(1);
  Teuchos::Array<Scalar> vals(1);
  vals[0] = 1;
  if(myRank == 0)
  {
    cols[0] = 0;
    badMat->insertGlobalValues(0,cols(),vals());
    cols[0] = 2;
    badMat->insertGlobalValues(2,cols(),vals());
  }
  else
  {
    cols[0] = 1;
    badMat->insertGlobalValues(1,cols(),vals());
    cols[0] = 3;
    badMat->insertGlobalValues(3,cols(),vals());
  }
  badMat->fillComplete();

  //
  // Create the preconditioner
  // This should throw an invalid input exception because we're using a bad map
  //
  try
  {
    RCP<Preconditioner> prec = rcp(new Ifpack2::Ifpack2_Hypre<Scalar,LO,GO,Node>(badMat));
  }
  catch(std::invalid_argument& e)
  {
    if(myRank == 0)
      std::cout << "End Result: TEST PASSED" << std::endl;
    return 0;
  }

  if(myRank == 0)
    std::cout << "End Result: TEST FAILED" << std::endl;
  return -1;
}

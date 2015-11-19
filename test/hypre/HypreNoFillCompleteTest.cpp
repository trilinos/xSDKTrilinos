// This test makes sure an exception is thrown if
// the matrix is not fillComplete

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
    elementList[0] = 0;
    elementList[1] = 1;
  }
  else
  {
    elementList[0] = 2;
    elementList[1] = 3;
  }
  RCP<Map> accMap = rcp(new Map(4,elementList(),0,comm,node));

  //
  // Construct the identity matrix
  //
  RCP<CrsMatrix> nfcMat = rcp(new CrsMatrix(accMap,1,Tpetra::StaticProfile));
  Teuchos::Array<GO> cols(1);
  Teuchos::Array<Scalar> vals(1);
  vals[0] = 1;
  if(myRank == 0)
  {
    cols[0] = 0;
    nfcMat->insertGlobalValues(0,cols(),vals());
    cols[0] = 1;
    nfcMat->insertGlobalValues(1,cols(),vals());
  }
  else
  {
    cols[0] = 2;
    nfcMat->insertGlobalValues(2,cols(),vals());
    cols[0] = 3;
    nfcMat->insertGlobalValues(3,cols(),vals());
  }

  //
  // Create the preconditioner
  // This should fail because we have not called fillComplete
  //
  try
  {
    RCP<Preconditioner> prec = rcp(new Ifpack2::Ifpack2_Hypre<Scalar,LO,GO,Node>(nfcMat));
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

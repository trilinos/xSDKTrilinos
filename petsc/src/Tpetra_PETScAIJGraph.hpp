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

/*#############################################################################
# CVS File Information
#    Current revision: $Revision$
#    Last modified:    $Date$
#    Modified by:      $Author$
#############################################################################*/

#ifndef _TPETRA_PETSCAIJGRAPH_H_
#define _TPETRA_PETSCAIJGRAPH_H_

#include "Tpetra_ConfigDefs.hpp"
#include "Tpetra_Import.hpp"
#include "Tpetra_RowGraph.hpp"
#ifdef HAVE_MPI
#include "Teuchos_DefaultMpiComm.hpp"
#else
#include "Teuchos_DefaultSerialComm.hpp"
#endif
//Petsc headers.
#include "petscmat.h"
#include <type_traits>


namespace Tpetra {

//! Tpetra_PETScAIJMatrix: A class for constructing and using real-valued sparse compressed row matrices.

/*! The Tpetra_PETScAIJMatrix is a wrapper class for PETSc sequential or parallel AIJ matrices.  It is
    derived from the Tpetra_RowMatrix class, and so provides PETSc users access to Trilinos preconditioners.
    This class is lightweight, i.e., there are no deep copies of matrix data.  Whenever possible, class
    methods utilize callbacks to native PETSc functions.  Currently, only sequential and parallel point AIJ
    PETSc matrix types are supported.
*/    

template<class LO = Details::DefaultTypes::local_ordinal_type, 
         class GO = Details::DefaultTypes::global_ordinal_type, 
         class Node = Details::DefaultTypes::node_type>
class PETScAIJGraph : 
  virtual public RowGraph <LO,GO,Node> 
{
private:
  typedef Teuchos::Comm<int> Comm;

public:
  typedef LO local_ordinal_type;
  typedef GO global_ordinal_type; 
  typedef Node node_type;

  //! @name Constructor/Destructor Methods

  //! Constructor
  PETScAIJGraph(Mat PETScMat);

  //! Destructor
  virtual ~PETScAIJGraph() {};

  //@}

  //! @name Graph query methods
  //@{ 

  //! The communicator over which this matrix is distributed. 
  RCP<const Comm> getComm() const {return comm_;};

  //! The Kokkos Node instance.
  RCP<Node> getNode() const { return rowMap_->getNode(); };

  //! The Map that describes the distribution of rows over processes.
  RCP<const Map<LO,GO,Node> > getRowMap() const { return rowMap_; };

  //! The Map that describes the distribution of columns over processes. 
  RCP<const Map<LO,GO,Node> > getColMap() const { return colMap_; };

  //! The Map associated with the domain of this operator, which must be compatible with X.getMap(). 
  RCP<const Map<LO,GO,Node> > getDomainMap() const { return rowMap_; }; // TODO: domain and range map should not be the same

  //! The Map associated with the range of this operator, which must be compatible with Y.getMap(). 
  RCP<const Map<LO,GO,Node> > getRangeMap() const { return rowMap_; };

  //! This graph's Import object.
  RCP<const Import<LO,GO,Node> > getImporter() const { return importer_; };

  //! This graph's Export object.
  RCP<const Export<LO,GO,Node> > getExporter() const { return exporter_; };

  //! The global number of rows of this matrix. 
  global_size_t getGlobalNumRows() const { return numGlobalRows_; };

  //! The global number of columns of this matrix. 
  global_size_t getGlobalNumCols() const { return numGlobalCols_; };

  //! The number of rows owned by the calling process. 
  size_t getNodeNumRows() const { return numLocalRows_; };

  //! The number of columns needed to apply the forward operator on this node. 
  size_t getNodeNumCols() const { return numLocalCols_; };

  //! The index base for global indices in this matrix.
  GO getIndexBase() const { return rowMap_->getIndexBase(); };

  //! The global number of stored (structurally nonzero) entries. 
  global_size_t getGlobalNumEntries() const { return nnzGlobal_; };

  //! The local number of stored (structurally nonzero) entries. 
  size_t getNodeNumEntries() const { return nnzLocal_; };

  //! The current number of entries on the calling process in the specified global row.
  size_t getNumEntriesInGlobalRow(GO globalRow) const;

  //! The current number of entries on the calling process in the specified local row.
  size_t getNumEntriesInLocalRow(LO localRow) const;

  //! The number of global diagonal entries, based on global row/column index comparisons.
  global_size_t getGlobalNumDiags() const;

  //! The number of local diagonal entries, based on global row/column index comparisons. 
  size_t getNodeNumDiags() const;

  //! The maximum number of entries across all rows/columns on all nodes.
  size_t getGlobalMaxNumRowEntries() const;

  //! The maximum number of entries across all rows/columns on this node. 
  size_t getNodeMaxNumRowEntries() const;

  //! Whether this matrix has a well-defined column map.
  bool hasColMap() const { return true; };

  //! Whether this matrix is lower triangular. 
  bool isLowerTriangular() const { return false; }; // TODO

  //! Whether this matrix is upper triangular. 
  bool isUpperTriangular() const { return false; }; // TODO

  //! Whether matrix indices are locally indexed.
  bool isLocallyIndexed() const { return false; };

  //! Whether matrix indices are globally indexed.
  bool isGloballyIndexed() const { return true; };

  //! Whether fillComplete() has been called. 
  bool isFillComplete() const;

  //@}

  //! @name Extraction methods
  //@{ 

  //! Get a copy of the given global row's entries. 
  void getGlobalRowCopy(GO globalRow, const ArrayView<GO> &indices, size_t &numIndices) const;

  //! Get a copy of the given local row's entries. 
  void getLocalRowCopy(LO localRow, const ArrayView<LO> &indices, size_t &numIndices) const;

  //@}*/

private:
  Mat PETScMat_;   // PETSc matrix
  RCP<Comm> comm_; // Teuchos communicator
  RCP<Import<LO,GO,Node> > importer_;
  RCP<Export<LO,GO,Node> > exporter_;

  LO numLocalRows_;
  size_t numLocalCols_;
  global_size_t numGlobalCols_;
  GO numGlobalRows_;
  RCP<const Map<LO,GO,Node> > rowMap_, colMap_;
  global_size_t nnzGlobal_;
  size_t nnzLocal_;
};



//! Constructor
//==============================================================================
template<class LO, class GO, class Node>
PETScAIJGraph<LO,GO,Node>::PETScAIJGraph(Mat PETScMat)
  : PETScMat_(PETScMat)
{
  PetscErrorCode ierr;
  MatType type;
  MatInfo info;
  PetscInt PETScCols, PETScLocalCols, rowStart;
  Mat OffDiagonal;

  // Wrap the communicator in a Teuchos Comm
#ifdef HAVE_MPI
  MPI_Comm comm;
  ierr = PetscObjectGetComm( (PetscObject)PETScMat, &comm); CHKERRV(ierr);
  comm_ = rcp(new Teuchos::MpiComm<int>(comm));
#else
  comm_ = rcp(new Teuchos::SerialComm<int>());
#endif

  // Figure out what kind of matrix it is
  // We currently support both sequential and parallel AIJ format
  ierr = MatGetType(PETScMat, &type); CHKERRV(ierr);

  // Test whether the matrix type is valid
#ifdef HAVE_TPETRA_DEBUG
  TEUCHOS_TEST_FOR_EXCEPT(strcmp(type,MATSEQAIJ) != 0 && strcmp(type,MATMPIAIJ) != 0)
#endif

  // Get the row and column ownership
  // NOTE: This only works for certain parallel layouts
  ierr = MatGetLocalSize(PETScMat, &numLocalRows_, &PETScLocalCols); CHKERRV(ierr);
  ierr = MatGetSize(PETScMat, &numGlobalRows_, &PETScCols); CHKERRV(ierr);
  numGlobalCols_ = PETScCols;
  ierr = MatGetOwnershipRange(PETScMat,&rowStart,NULL); CHKERRV(ierr);

  // Create the row map
  // TODO: Will the index base always be 0?
  rowMap_ = rcp(new Tpetra::Map<LO,GO,Node>(numGlobalRows_, numLocalRows_, 0, comm_));

  // Compute the local number of nonzeros
  ierr = MatGetInfo(PETScMat, MAT_LOCAL, &info); CHKERRV(ierr);
  nnzLocal_ = info.nz_used;

  // Compute the global number of nonzeros
  ierr = MatGetInfo(PETScMat, MAT_GLOBAL_SUM, &info); CHKERRV(ierr);
  nnzGlobal_ = info.nz_used;

  // Get the GIDs of the non-local columns
  const PetscInt * garray;
  ierr = MatMPIAIJGetSeqAIJ(PETScMat,NULL,&OffDiagonal,&garray); CHKERRV(ierr);
  ierr = MatGetSize(OffDiagonal,NULL,&PETScCols); CHKERRV(ierr);
  numLocalCols_ = PETScLocalCols+PETScCols;

//  for(size_t i=0; i<10; i++) std::cerr << "garray[" << i << "] = " << garray[i] << std::endl;

  Array<int> ColGIDs(numLocalCols_);
  for (PetscInt i=0; i<PETScLocalCols; i++) ColGIDs[i] = rowStart + i;
  for (size_t i=PETScLocalCols; i<numLocalCols_; i++) ColGIDs[i] = garray[i-PETScLocalCols];

//  for(size_t i=0; i<numLocalCols_; i++) std::cerr << "ColGIDs[" << i << "] = " <<  ColGIDs[i] << std::endl;

  // Create the column map
  // TODO: Will the index base always be 0?
  global_size_t numOverlappingCols;
  Teuchos::reduceAll(*comm_,Teuchos::SumValueReductionOp<int,global_size_t>(),1,&numLocalCols_,&numOverlappingCols);
  colMap_ = rcp(new Tpetra::Map<LO,GO,Node>(numOverlappingCols, ColGIDs, 0, comm_));

  // Create the importer
  importer_ = rcp(new Import<LO,GO,Node>(rowMap_,colMap_));

  // Create the exporter
  exporter_ = rcp(new Export<LO,GO,Node>(colMap_, rowMap_));
}



//! The current number of entries on the calling process in the specified global row.
//==============================================================================
template<class LO, class GO, class Node>
size_t PETScAIJGraph<LO,GO,Node>::getNumEntriesInGlobalRow(GO globalRow) const
{
  PetscErrorCode ierr;
  PetscInt ncols;
  ierr = MatGetRow(PETScMat_,globalRow,&ncols,NULL,NULL); CHKERRQ(ierr);
  ierr = MatRestoreRow(PETScMat_,globalRow,NULL,NULL,NULL); CHKERRQ(ierr);
  return ncols;
}



//! The current number of entries on the calling process in the specified local row.
//==============================================================================
template<class LO, class GO, class Node>
size_t PETScAIJGraph<LO,GO,Node>::getNumEntriesInLocalRow(LO localRow) const
{
  GO globalRow = localRow + rowMap_->getMinGlobalIndex();
  return getNumEntriesInGlobalRow(globalRow);
}



//! The number of global diagonal entries, based on global row/column index comparisons.
//==============================================================================
template<class LO, class GO, class Node>
global_size_t PETScAIJGraph<LO,GO,Node>::getGlobalNumDiags() const
{
  PetscErrorCode ierr;
  PetscInt globalnz;
  IS is;
  ierr = MatFindZeroDiagonals(PETScMat_,&is); CHKERRQ(ierr);
  ierr = ISGetSize(is,&globalnz); CHKERRQ(ierr);
  return (numGlobalRows_-globalnz);
}



//! The number of local diagonal entries, based on global row/column index comparisons. 
//==============================================================================
template<class LO, class GO, class Node>
size_t PETScAIJGraph<LO,GO,Node>::getNodeNumDiags() const
{
  PetscErrorCode ierr;
  PetscInt localnz;
  IS is;
  ierr = MatFindZeroDiagonals(PETScMat_,&is); CHKERRQ(ierr);
  ierr = ISGetLocalSize(is,&localnz); CHKERRQ(ierr);
  return (numLocalRows_-localnz);
}



//! The maximum number of entries across all rows/columns on all nodes. 
//==============================================================================
template<class LO, class GO, class Node>
size_t PETScAIJGraph<LO,GO,Node>::getGlobalMaxNumRowEntries() const
{
  size_t globalMax;
  size_t localMax = getNodeMaxNumRowEntries();
  Teuchos::reduceAll(*comm_,Teuchos::MaxValueReductionOp<int,global_size_t>(),1,&localMax,&globalMax);
  return globalMax;
}



//! The maximum number of entries across all rows/columns on this node. 
//==============================================================================
template<class LO, class GO, class Node>
size_t PETScAIJGraph<LO,GO,Node>::getNodeMaxNumRowEntries() const
{
  size_t maxEntries = 0;
  size_t tempEntries;

  LO nLocalRows = getNodeNumRows();
  for(LO i=0; i< nLocalRows; i++)
  {
    tempEntries = getNumEntriesInLocalRow(i);
    if(tempEntries > maxEntries)
      maxEntries = tempEntries;
  }

  return maxEntries;
}



//! Get a copy of the given global row's entries. 
//==============================================================================
template<class LO, class GO, class Node>
void PETScAIJGraph<LO,GO,Node>::getGlobalRowCopy(GO globalRow, const ArrayView<GO> &indices, size_t &numIndices) const
{
  PetscErrorCode ierr;
  PetscInt ncols;
  const PetscInt * cols;

  // Get PETSc's row
  ierr = MatGetRow(PETScMat_,globalRow,&ncols,&cols,NULL); CHKERRV(ierr);
  numIndices = ncols;

  // Copy it to a Trilinos Array
  Teuchos::Array<LO> tpCols(ncols);
  for(LO i=0; i<ncols; i++)
  {
    indices[i] = cols[i];
  }
  ierr = MatRestoreRow(PETScMat_,globalRow,&ncols,&cols,NULL); CHKERRV(ierr);
}



//! Get a copy of the given local row's entries. 
//==============================================================================
template<class LO, class GO, class Node>
void PETScAIJGraph<LO,GO,Node>::getLocalRowCopy(LO localRow, const ArrayView<LO> &indices, size_t &numIndices) const
{
  GO globalRow = localRow + rowMap_->getMinGlobalIndex();
  
  getGlobalRowCopy(globalRow, indices, numIndices);
} //ExtractMyRowCopy()



//! Whether fillComplete() has been called. 
//==============================================================================
template<class LO, class GO, class Node>
bool PETScAIJGraph<LO,GO,Node>::isFillComplete() const 
{
  PetscBool assembled;
  PetscErrorCode ierr;

  ierr = MatAssembled(PETScMat_,&assembled); CHKERRQ(ierr);

  return assembled;
}



} // namespace Tpetra
#endif /* _TPETRA_PETSCAIJMATRIX_H_ */

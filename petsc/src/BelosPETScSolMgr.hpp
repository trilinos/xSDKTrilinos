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

#ifndef BELOS_PETSC_SOLMGR_HPP
#define BELOS_PETSC_SOLMGR_HPP

/*! \file BelosPETScSolMgr.hpp
    \brief Pure virtual base class which describes the basic interface for a solver manager.
*/

#include "BelosConfigDefs.hpp"
#include "BelosTypes.hpp"

#include "BelosLinearProblem.hpp"
#include "BelosOutputManager.hpp"
#include "BelosSolverManager.hpp"

#ifdef HAVE_XSDKTRILINOS_EPETRA
  #include "Epetra_DataAccess.h"
  #include "Epetra_MultiVector.h"
#endif
#include "Tpetra_MultiVector.hpp"

#ifdef HAVE_MPI
#include "Teuchos_DefaultMpiComm.hpp"
#ifdef HAVE_XSDKTRILINOS_EPETRA
  #include "Epetra_MpiComm.h"
#endif
#else
#include "Teuchos_DefaultSerialComm.hpp"
#endif

//Petsc headers.
#include "petscksp.h"
#include <type_traits>

// TODO: Because PETSc is using its own vector class, Kokkos is not being used for vector operations.

/*! \class Belos::PETScSolMgr
  \brief The Belos::PETScSolMgr is a wrapper for the PETSc linear solvers.
*/

namespace Belos {

template<class ScalarType, class MV, class OP>
class PETScSolMgrHelper {
public:
  static void getData(const MV& x, const int i, const ScalarType* &rawData)
  { TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "This method is not implemented.");}

  static void getDataNonConst(MV& x, const int i, ScalarType* &rawData)
  { TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "This method is not implemented.");}

  static MPI_Comm getComm(const MV& x)
  { TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "This method is not implemented.");}

  static PetscInt getLocalLength(const MV& x)
  { TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "This method is not implemented.");}

  static PetscInt getGlobalLength(const MV& x)
  { TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "This method is not implemented.");}

  static void wrapVector(ScalarType* x, const MV& helper, Teuchos::RCP<MV>& trilinosX)
  { TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "This method is not implemented.");}

  static void unwrapVector(ScalarType* x, Teuchos::RCP<MV> trilinosX)
  { TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "This method is not implemented.");}
};


// Partial specialization for Tpetra
template<class ScalarType, class OP>
class PETScSolMgrHelper<ScalarType,Tpetra::MultiVector<ScalarType,typename OP::local_ordinal_type,typename OP::global_ordinal_type>,OP> {
private:
  typedef Tpetra::MultiVector<ScalarType,typename OP::local_ordinal_type,typename OP::global_ordinal_type> MV;
  typedef Tpetra::Vector<ScalarType,typename OP::local_ordinal_type,typename OP::global_ordinal_type, typename OP::node_type, OP::node_type::classic> Vector;

public:
  static void getData(const MV& x, const int i, const ScalarType* &rawData)
  { rawData = x.getData(i).get(); }

  static void getDataNonConst(MV& x, const int i, ScalarType* &rawData)
  { rawData = x.getDataNonConst(i).get(); }

  static MPI_Comm getComm(const MV& x)
  { return *(Teuchos::rcp_dynamic_cast<const Teuchos::MpiComm<int> >(x.getMap()->getComm())->getRawMpiComm()); }

  static PetscInt getLocalLength(const MV& x)
  { return x.getLocalLength(); }

  static PetscInt getGlobalLength(const MV& x)
  { return x.getGlobalLength(); }

  // TODO: We are not actually wrapping a vector; we are copying it.
  static void wrapVector(ScalarType* x, const MV& helper, Teuchos::RCP<MV>& trilinosX)
  { 
    Teuchos::ArrayView<ScalarType> data(x,helper.getLocalLength());
    trilinosX = Teuchos::rcp(new Vector(helper.getMap(),data)); 
  }

  // TODO: This is not parallel
  static void unwrapVector(ScalarType* x, Teuchos::RCP<MV> trilinosX)
  {  
     Teuchos::ArrayRCP<const ScalarType> rawData = trilinosX->getData(0);
     for(size_t i=0; i<trilinosX->getLocalLength(); i++) x[i] = rawData[i];
  }
};


// Partial specialization for Epetra
#ifdef HAVE_XSDKTRILINOS_EPETRA
template<class ScalarType, class OP>
class PETScSolMgrHelper<ScalarType,Epetra_MultiVector,OP> {
private:
  typedef Epetra_MultiVector MV;

public:
  static void getData(const MV& x, const int i, const ScalarType* &rawData)
  { rawData = x[i]; }

  static void getDataNonConst(MV& x, const int i, ScalarType* &rawData)
  { rawData = x[i]; }

  static MPI_Comm getComm(const MV& x)
  { return dynamic_cast<const Epetra_MpiComm&>(x.Comm()).GetMpiComm(); }

  static PetscInt getLocalLength(const MV& x)
  { return x.MyLength(); }

  static PetscInt getGlobalLength(const MV& x)
  { return x.GlobalLength(); }

  static void wrapVector(ScalarType* x, const MV& helper, Teuchos::RCP<MV>& trilinosX)
  { trilinosX = Teuchos::rcp(new Epetra_Vector(View,helper.Map(),x)); }

  static void unwrapVector(ScalarType* x, Teuchos::RCP<MV> trilinosX)
  { } // Since we didn't copy any data, this doesn't have to do anything
};
#endif


template<class ScalarType, class MV, class OP>
class PETScSolMgr : public SolverManager<ScalarType,MV,OP> {
  private:
    typedef typename Teuchos::ScalarTraits<ScalarType>::magnitudeType MagnitudeType;
    typedef MultiVecTraits<ScalarType,MV> MVT;

  public:

  //!@name Constructors/Destructor
  //@{

  //! Empty constructor.
  PETScSolMgr();

  PETScSolMgr( const Teuchos::RCP<LinearProblem<ScalarType,MV,OP> > &problem,
               const Teuchos::RCP<Teuchos::ParameterList> &pl );

  //! Destructor.
  virtual ~PETScSolMgr() {};
  //@}

  //! @name Accessor methods
  //@{

  //! Return a reference to the linear problem being solved by this solver manager.
  const LinearProblem<ScalarType,MV,OP>& getProblem() const {
    return *problem_;
  }

  //! Return the valid parameters for this solver manager.
  Teuchos::RCP<const Teuchos::ParameterList> getValidParameters() const;

  //! Return the current parameters being used for this solver manager.
  Teuchos::RCP<const Teuchos::ParameterList> getCurrentParameters() const { return params_; }

  Teuchos::Array<Teuchos::RCP<Teuchos::Time> > getTimers() const {
    return Teuchos::tuple(timerSolve_);
  }

  /// \brief Tolerance achieved by the last \c solve() invocation.
  ///
  /// This is the maximum over all right-hand sides' achieved
  /// convergence tolerances, and is set whether or not the solve
  /// actually managed to achieve the desired convergence tolerance.
  ///
  /// The default implementation throws std::runtime_error.  This is
  /// in case the idea of a single convergence tolerance doesn't make
  /// sense for some solvers.  It also serves as a gradual upgrade
  /// path (since this method is a later addition to the \c
  /// SolverManager interface).
  MagnitudeType achievedTol() const {
    return achievedTol_;
  }

  //! Get the iteration count for the most recent call to \c solve().
  int getNumIters() const {
    return numIters_;
  }

  /*! \brief Returns whether a loss of accuracy was detected in the solver.
   *  \note This method is normally applicable to GMRES-type solvers.
  */
  bool isLOADetected() const { return false; }

  //@}

  //! @name Set methods
  //@{

  //! Set the linear problem that needs to be solved.
  void setProblem( const Teuchos::RCP<LinearProblem<ScalarType,MV,OP> > &problem ) { problem_ = problem; }

  /// \brief Set the parameters to use when solving the linear problem.
  ///
  /// \param params [in/out] List of parameters to use when solving
  ///   the linear problem.  This list will be modified as necessary
  ///   to include default parameters that need not be provided.  If
  ///   params is null, then this method uses default parameters.
  ///
  /// \note The ParameterList returned by \c getValidParameters() has
  ///   all the parameters that the solver understands, possibly
  ///   including human-readable documentation and validators.
  void setParameters( const Teuchos::RCP<Teuchos::ParameterList> &params );

  void setCLA(int argc, char* argv[]) {argc_ = argc; argv_ = argv;}

  //@}

  //! @name Reset methods
  //@{

  /// \brief Reset the solver manager.
  ///
  /// Reset the solver manager in a way specified by the \c
  /// ResetType parameter.  This informs the solver manager that the
  /// solver should prepare for the next call to solve by resetting
  /// certain elements of the iterative solver strategy.
  void reset( const ResetType type ) { if ((type & Belos::Problem) && !Teuchos::is_null(problem_)) problem_->setProblem(); }
  //@}

  //! @name Solver application methods
  //@{

  /// \brief Iterate until the status test tells us to stop.
  //
  /// This method performs possibly repeated calls to the underlying
  /// linear solver's iterate() routine, until the problem has been
  /// solved (as decided by the solver manager via the status
  /// test(s)), or the solver manager decides to quit.
  ///
  /// \return A \c Belos::ReturnType enum specifying:
  ///   - Belos::Converged: the linear problem was solved to the
  ///     specification required by the solver manager.
  ///   - Belos::Unconverged: the linear problem was not solved to the
  ///     specification desired by the solver manager.
  ReturnType solve();
  //@}

private:
  static PetscErrorCode applyMat(Mat A, Vec x, Vec Ax);
  static PetscErrorCode applyPrec(PC M, Vec x, Vec Mx);

    // Linear problem.
    Teuchos::RCP<LinearProblem<ScalarType,MV,OP> > problem_;

    // Output manager.
    Teuchos::RCP<OutputManager<ScalarType> > printer_;
    Teuchos::RCP<std::ostream> outputStream_;

    // Current parameter list.
    Teuchos::RCP<Teuchos::ParameterList> params_;

    /// \brief List of valid parameters and their default values.
    ///
    /// This is declared "mutable" because the SolverManager interface
    /// requires that getValidParameters() be declared const, yet we
    /// want to create the valid parameter list only on demand.
    mutable Teuchos::RCP<const Teuchos::ParameterList> validParams_;

    // Default solver values.
    static const MagnitudeType convtol_default_;
    static const int maxIters_default_;
    static const bool assertPositiveDefiniteness_default_;
    static const int verbosity_default_;
    static const std::string label_default_;
    static const Teuchos::RCP<std::ostream> outputStream_default_;
    static const KSPType solver_default_;

    // Current solver values.
    MagnitudeType convtol_,achievedTol_;
    int maxIters_, numIters_;
    int verbosity_;
    bool assertPositiveDefiniteness_;
    std::string solver_;

    // Timers.
    std::string label_;
    Teuchos::RCP<Teuchos::Time> timerSolve_;

    // Internal state variables.
    bool isSet_;

    // Command line arguments
    int argc_;
    char** argv_;
};


//=============================================================================
// Default solver values.
template<class ScalarType, class MV, class OP>
const typename PETScSolMgr<ScalarType,MV,OP>::MagnitudeType PETScSolMgr<ScalarType,MV,OP>::convtol_default_ = 1e-8;

template<class ScalarType, class MV, class OP>
const int PETScSolMgr<ScalarType,MV,OP>::maxIters_default_ = 1000;

template<class ScalarType, class MV, class OP>
const bool PETScSolMgr<ScalarType,MV,OP>::assertPositiveDefiniteness_default_ = true;

template<class ScalarType, class MV, class OP>
const int PETScSolMgr<ScalarType,MV,OP>::verbosity_default_ = Belos::Errors;

template<class ScalarType, class MV, class OP>
const std::string PETScSolMgr<ScalarType,MV,OP>::label_default_ = "Belos";

template<class ScalarType, class MV, class OP>
const Teuchos::RCP<std::ostream> PETScSolMgr<ScalarType,MV,OP>::outputStream_default_ = Teuchos::rcp(&std::cout,false);

template<class ScalarType, class MV, class OP>
const KSPType PETScSolMgr<ScalarType,MV,OP>::solver_default_ = KSPGMRES;

//=============================================================================
// Empty constructor
template<class ScalarType, class MV, class OP>
PETScSolMgr<ScalarType,MV,OP>::PETScSolMgr() :
  outputStream_(outputStream_default_),
  convtol_(convtol_default_),
  maxIters_(maxIters_default_),
  numIters_(0),
  verbosity_(verbosity_default_),
  assertPositiveDefiniteness_(assertPositiveDefiniteness_default_),
  label_(label_default_),
  solver_(solver_default_),
  isSet_(false),
  argc_(0)
{}


//=============================================================================
// Basic constructor
template<class ScalarType, class MV, class OP>
PETScSolMgr<ScalarType,MV,OP>::
PETScSolMgr( const Teuchos::RCP<LinearProblem<ScalarType,MV,OP> > &problem,
             const Teuchos::RCP<Teuchos::ParameterList> &pl ) :
  problem_(problem),
  outputStream_(outputStream_default_),
  convtol_(convtol_default_),
  maxIters_(maxIters_default_),
  numIters_(0),
  verbosity_(verbosity_default_),
  assertPositiveDefiniteness_(assertPositiveDefiniteness_default_),
  label_(label_default_),
  solver_(solver_default_),
  isSet_(false),
  argc_(0)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
    problem_.is_null (), std::invalid_argument,
    "Belos::PETScSolMgr two-argument constructor: "
    "'problem' is null.  You must supply a non-null Belos::LinearProblem "
    "instance when calling this constructor.");

  if (! pl.is_null ()) {
    // Set the parameters using the list that was passed in.
    setParameters (pl);
  }    
}


//=============================================================================
// Basic constructor
template<class ScalarType, class MV, class OP>
void PETScSolMgr<ScalarType,MV,OP>::setParameters( const Teuchos::RCP<Teuchos::ParameterList> &params )
{
  using Teuchos::ParameterList;
  using Teuchos::parameterList;
  using Teuchos::RCP;

  RCP<const ParameterList> defaultParams = getValidParameters();

  // Create the internal parameter list if one doesn't already exist.
  if (params_.is_null()) {
    params_ = parameterList (*defaultParams);
  } else {
    params->validateParameters (*defaultParams);
  }

  // Check for maximum number of iterations
  if (params->isParameter("Maximum Iterations")) {
    maxIters_ = params->get("Maximum Iterations",maxIters_default_);

    // Update parameter in our list and in status test.
    params_->set("Maximum Iterations", maxIters_);
  }

  // Check if positive definiteness assertions are to be performed
  if (params->isParameter("Assert Positive Definiteness")) {
    assertPositiveDefiniteness_ = params->get("Assert Positive Definiteness",assertPositiveDefiniteness_default_);

    // Update parameter in our list.
    params_->set("Assert Positive Definiteness", assertPositiveDefiniteness_);
  }

  // Check if the user has defined a particular Krylov solver to be used
  if (params->isParameter("Solver")) {
    solver_ = params->get("Solver", solver_default_);

    // Update parameter in our list
    params_->set("Solver", solver_);
  }

  // Check to see if the timer label changed.
  if (params->isParameter("Timer Label")) {
    std::string tempLabel = params->get("Timer Label", label_default_);

    // Update parameter in our list and solver timer
    if (tempLabel != label_) {
      label_ = tempLabel;
      params_->set("Timer Label", label_);
      std::string solveLabel = label_ + ": PETScSolMgr total solve time";
#ifdef BELOS_TEUCHOS_TIME_MONITOR
      timerSolve_ = Teuchos::TimeMonitor::getNewCounter(solveLabel);
#endif
    }
  }

  // Check for a change in verbosity level
  if (params->isParameter("Verbosity")) {
    if (Teuchos::isParameterType<int>(*params,"Verbosity")) {
      verbosity_ = params->get("Verbosity", verbosity_default_);
    } else {
      verbosity_ = (int)Teuchos::getParameter<Belos::MsgType>(*params,"Verbosity");
    }

    // Update parameter in our list.
    params_->set("Verbosity", verbosity_);
    if (printer_ != Teuchos::null)
      printer_->setVerbosity(verbosity_);
  }

  // output stream
  if (params->isParameter("Output Stream")) {
    outputStream_ = Teuchos::getParameter<Teuchos::RCP<std::ostream> >(*params,"Output Stream");

    // Update parameter in our list.
    params_->set("Output Stream", outputStream_);
    if (printer_ != Teuchos::null)
      printer_->setOStream( outputStream_ );
  }

  // Create output manager if we need to.
  if (printer_ == Teuchos::null) {
    printer_ = Teuchos::rcp( new OutputManager<ScalarType>(verbosity_, outputStream_) );
  }

  // Check for convergence tolerance
  if (params->isParameter("Convergence Tolerance")) {
    convtol_ = params->get("Convergence Tolerance",convtol_default_);

    // Update parameter in our list and residual tests.
    params_->set("Convergence Tolerance", convtol_);
  }

  // Create the timer if we need to.
  if (timerSolve_ == Teuchos::null) {
    std::string solveLabel = label_ + ": PETScSolMgr total solve time";
#ifdef BELOS_TEUCHOS_TIME_MONITOR
    timerSolve_ = Teuchos::TimeMonitor::getNewCounter(solveLabel);
#endif
  }

  // Inform the solver manager that the current parameters were set.
  isSet_ = true;
}


//=============================================================================
template<class ScalarType, class MV, class OP>
Teuchos::RCP<const Teuchos::ParameterList>
PETScSolMgr<ScalarType,MV,OP>::getValidParameters() const
{
  using Teuchos::ParameterList;
  using Teuchos::parameterList;
  using Teuchos::RCP;

  if (validParams_.is_null()) {
    // Set all the valid parameters and their default values.
    RCP<ParameterList> pl = parameterList ();
    pl->set("Convergence Tolerance", convtol_default_,
      "The relative residual tolerance that needs to be achieved by the\n"
      "iterative solver in order for the linera system to be declared converged.");
    pl->set("Maximum Iterations", maxIters_default_,
      "The maximum number of block iterations allowed for each\n"
      "set of RHS solved.");
    pl->set("Assert Positive Definiteness", assertPositiveDefiniteness_default_,
      "Whether or not to assert that the linear operator\n"
      "and the preconditioner are indeed positive definite.");
    pl->set("Verbosity", verbosity_default_,
      "What type(s) of solver information should be outputted\n"
      "to the output stream.");
    pl->set("Output Stream", outputStream_default_,
      "A reference-counted pointer to the output stream where all\n"
      "solver output is sent.");
    pl->set("Timer Label", label_default_,
      "The string to use as a prefix for the timer labels.");
    pl->set("Solver", solver_default_,
      "The string to use as the KSP solver name.");
    //  defaultParams_->set("Restart Timers", restartTimers_);
    validParams_ = pl;
  }
  return validParams_;
}


//=============================================================================
template<class ScalarType, class MV, class OP>
ReturnType PETScSolMgr<ScalarType,MV,OP>::solve()
{
  using Teuchos::RCP;
  typedef PETScSolMgrHelper<ScalarType,MV,OP> Helper;

  PetscErrorCode ierr;
  PetscBool isInitialized;
  KSP solver;
  Vec petscX, petscB, petscR;
  Mat petscA;
  PC petscPrec;
  PetscInt localLength, globalLength, tmpInt;
  PetscReal norm;

  bool isConverged = true;

  { // begin timing
#ifdef BELOS_TEUCHOS_TIME_MONITOR
  Teuchos::TimeMonitor slvtimer(*timerSolve_);
#endif

  // Get the LHS and RHS
  RCP<MV> X = problem_->getLHS();
  RCP<const MV> B = problem_->getRHS();

  // Get the distribution
  localLength = Helper::getLocalLength(*X);
  globalLength = Helper::getGlobalLength(*B);

  // Check whether PETSc is initialized.  If not, initialize it
  ierr = PetscInitialized(&isInitialized); CHKERRCONTINUE(ierr);
  if(!isInitialized) {
    // Set the PETSc communicator
    PETSC_COMM_WORLD = Helper::getComm(*X);

    // Pass in the command line arguments
    ierr = PetscInitialize(&argc_,&argv_,NULL,NULL); CHKERRCONTINUE(ierr);
  }

  // Create the PETSc vectors
  ierr = VecCreate(PETSC_COMM_WORLD,&petscR); CHKERRCONTINUE(ierr);

  // Create the solver
  ierr = KSPCreate(PETSC_COMM_WORLD,&solver); CHKERRCONTINUE(ierr);

  // Set the KSP options
  ierr = KSPSetFromOptions(solver); CHKERRCONTINUE(ierr);

  // Select which solver we use
  std::cout << "solver: " << solver << std::endl;
  ierr = KSPSetType(solver, solver_.c_str()); CHKERRCONTINUE(ierr);

  // Tell the solver not to zero out the initial vector
  ierr = KSPSetInitialGuessNonzero(solver, PETSC_TRUE); CHKERRCONTINUE(ierr);

  // Set the tolerance and maximum number of iterations
  ierr = KSPSetTolerances(solver, convtol_, PETSC_DEFAULT, PETSC_DEFAULT, maxIters_); CHKERRCONTINUE(ierr);

  // Tell the solver whether to output convergence information
  if(verbosity_ & IterationDetails || verbosity_ & StatusTestDetails) {
    PetscViewerAndFormat *vf;
    ierr = PetscViewerAndFormatCreate(PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)solve)),PETSC_VIEWER_DEFAULT);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject)vf->viewer);CHKERRQ(ierr);
    ierr = KSPMonitorSet(solver, (PetscErrorCode (*)(KSP,PetscInt,PetscReal,void*))KSPMonitorDefault, vf, (PetscErrorCode (*)(void**))PetscViewerAndFormatDestroy); CHKERRCONTINUE(ierr);
  }

  // Wrap the Trilinos Operator in a PETSc Mat
  ierr = MatCreateShell(PETSC_COMM_WORLD,localLength,localLength,globalLength,globalLength,(void*)problem_.get(),&petscA); CHKERRCONTINUE(ierr);
  ierr = MatShellSetOperation(petscA,MATOP_MULT,(void(*)(void))applyMat); CHKERRCONTINUE(ierr);

  // Wrap the Trilinos Preconditioner in a PETSc Mat
  if(problem_->isRightPrec() || problem_->isLeftPrec()) {
    TEUCHOS_TEST_FOR_EXCEPTION(problem_->isRightPrec() && problem_->isLeftPrec(), std::invalid_argument,
    "Belos::PETScSolMgr solve(): We do not currently support both left and right preconditioning at the same time.");

    if(problem_->isLeftPrec()) {
      ierr = KSPSetPCSide(solver,PC_LEFT); CHKERRCONTINUE(ierr);
    }
    else {
      ierr = KSPSetPCSide(solver,PC_RIGHT); CHKERRCONTINUE(ierr);
    }

    ierr = PCCreate(PETSC_COMM_WORLD, &petscPrec); CHKERRCONTINUE(ierr);
    ierr = PCSetType(petscPrec, PCSHELL); CHKERRCONTINUE(ierr);
    ierr = PCShellSetApply(petscPrec, applyPrec); CHKERRCONTINUE(ierr);
    ierr = PCShellSetContext(petscPrec, (void*)problem_.get()); CHKERRCONTINUE(ierr);
    ierr = KSPSetPC(solver,petscPrec); CHKERRCONTINUE(ierr);
  }

  // Give the Trilinos matrix to the PETSc solver
  ierr = KSPSetOperators(solver,petscA,petscA); CHKERRCONTINUE(ierr);

  // PETSc is currently only capable of handling single right hand sides,
  // so we will loop over all of them one at a time
  int nrhs = MVT::GetNumberVecs(*B);
  ScalarType *xValues;
  const ScalarType *bValues;
  numIters_ = 0;
  achievedTol_ = 0;
  for(int i=0; i<nrhs; i++)
  {
    // Get pointers to the raw vector data
    Helper::getDataNonConst(*X,i,xValues);
    Helper::getData(*B,i,bValues);

    // Wrap the pointers in PETSc vectors
    // TODO: I ignore the block size for now.  What does it do?
    ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD,1,localLength,globalLength,xValues,&petscX); CHKERRCONTINUE(ierr);
    ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD,1,localLength,globalLength,bValues,&petscB); CHKERRCONTINUE(ierr);

    // Call the linear solver
    ierr = KSPSolve(solver,petscB,petscX); CHKERRCONTINUE(ierr);
    if(ierr >= PETSC_ERR_MIN_VALUE && ierr <= PETSC_ERR_MAX_VALUE) {
      isConverged = false;
    }

    // Get the number of iterations
    ierr = KSPGetIterationNumber(solver,&tmpInt); CHKERRCONTINUE(ierr);
    numIters_ = std::max(tmpInt,numIters_);

    // Get the residual
    ierr = KSPBuildResidual(solver,NULL,NULL,&petscR); CHKERRCONTINUE(ierr);
    ierr = VecNorm(petscR,NORM_2,&norm); CHKERRCONTINUE(ierr);
    achievedTol_ = std::max(norm,achievedTol_);

    ierr = VecDestroy(&petscX); CHKERRCONTINUE(ierr);
    ierr = VecDestroy(&petscB); CHKERRCONTINUE(ierr);
  }

  // Free appropriate memory
  ierr = VecDestroy(&petscR); CHKERRCONTINUE(ierr);
  ierr = KSPDestroy(&solver); CHKERRCONTINUE(ierr);
  ierr = MatDestroy(&petscA); CHKERRCONTINUE(ierr);
  if(problem_->isRightPrec() || problem_->isLeftPrec()) {
    ierr = PCDestroy(&petscPrec); CHKERRCONTINUE(ierr);
  }
  
  PetscFinalize();

  } // end timing

  // print timing information
#ifdef BELOS_TEUCHOS_TIME_MONITOR
  // Calling summarize() can be expensive, so don't call unless the
  // user wants to print out timing details.  summarize() will do all
  // the work even if it's passed a "black hole" output stream.
  if (verbosity_ & TimingDetails)
    Teuchos::TimeMonitor::summarize( printer_->stream(TimingDetails) );
#endif

  if (!isConverged ) {
    return Unconverged; // return from BiCGStabSolMgr::solve()
  }
  return Converged; // return from BiCGStabSolMgr::solve()
}


//=============================================================================
template<class ScalarType, class MV, class OP>
PetscErrorCode PETScSolMgr<ScalarType,MV,OP>::applyMat(Mat A, Vec x, Vec Ax)
{
  using Teuchos::RCP;
  typedef PETScSolMgrHelper<ScalarType,MV,OP> Helper;

  PetscErrorCode ierr;
  const PetscScalar * xData;
  PetscScalar * AxData;
  void * ptr;

  // Get the problem out of the context
  ierr = MatShellGetContext(A,&ptr); CHKERRQ(ierr);
  LinearProblem<ScalarType,MV,OP> * problem = (LinearProblem<ScalarType,MV,OP>*)ptr;

  // Rip the raw data out of the PETSc vectors
  ierr = VecGetArrayRead(x, &xData); CHKERRQ(ierr);
  ierr = VecGetArray(Ax, &AxData); CHKERRQ(ierr);

  // Wrap the PETSc data in a Trilinos Vector
  RCP<MV> trilinosX, trilinosAX;
  Helper::wrapVector(const_cast<PetscScalar*>(xData), *problem->getLHS(), trilinosX); // The const_cast is not ideal, but we do promise not to modify the entries of xData.
  Helper::wrapVector(AxData, *problem->getLHS(), trilinosAX);

  // Perform the multiplication
  problem->applyOp(*trilinosX,*trilinosAX); 

  // Unwrap the vectors; this is necessary if we copied data in the wrap step
  Helper::unwrapVector(AxData, trilinosAX);

  // Restore the PETSc vectors
  ierr = VecRestoreArrayRead(x,&xData); CHKERRQ(ierr);
  ierr = VecRestoreArray(Ax,&AxData); CHKERRQ(ierr);

  return 0;
}


//=============================================================================
template<class ScalarType, class MV, class OP>
PetscErrorCode PETScSolMgr<ScalarType,MV,OP>::applyPrec(PC M, Vec x, Vec Mx)
{
  using Teuchos::RCP;
  typedef PETScSolMgrHelper<ScalarType,MV,OP> Helper;

  PetscErrorCode ierr;
  const PetscScalar * xData;
  PetscScalar * MxData;
  void * ptr;

  // Get the problem out of the context
  ierr = PCShellGetContext(M,&ptr); CHKERRQ(ierr);
  LinearProblem<ScalarType,MV,OP> * problem = (LinearProblem<ScalarType,MV,OP>*)ptr;

  // Rip the raw data out of the PETSc vectors
  ierr = VecGetArrayRead(x, &xData); CHKERRQ(ierr);
  ierr = VecGetArray(Mx, &MxData); CHKERRQ(ierr);

  // Wrap the PETSc data in a Trilinos Vector
  RCP<MV> trilinosX, trilinosMX;
  Helper::wrapVector(const_cast<PetscScalar*>(xData), *problem->getLHS(), trilinosX); // The const_cast is not ideal, but we do promise not to modify the entries of xData.
  Helper::wrapVector(MxData, *problem->getLHS(), trilinosMX);

  // Perform the multiplication
  if(problem->isLeftPrec()) {
    problem->applyLeftPrec(*trilinosX, *trilinosMX);
  }
  else {
    problem->applyRightPrec(*trilinosX, *trilinosMX);
  }

  // Unwrap the vectors; this is necessary if we copied data in the wrap step
  Helper::unwrapVector(MxData, trilinosMX);

  // Restore the PETSc vectors
  ierr = VecRestoreArrayRead(x,&xData); CHKERRQ(ierr);
  ierr = VecRestoreArray(Mx,&MxData); CHKERRQ(ierr);
  
  return 0;
}


} // End Belos namespace

#endif /* BELOS_PETSC_SOLMGR_HPP */

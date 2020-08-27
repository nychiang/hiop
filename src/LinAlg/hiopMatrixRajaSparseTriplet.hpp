#pragma once

#include "hiopVector.hpp"
#include "hiopMatrixDense.hpp"
#include "hiopMatrixSparse.hpp"

#include <cassert>
#include <string>

namespace hiop
{

/** 
 * @brief Sparse matrix of doubles in triplet format - it is not distributed
 * @note for now (i,j) are expected ordered: first on rows 'i' and then on cols 'j'
 */
class hiopMatrixRajaSparseTriplet : public hiopMatrixSparse
{
public:
  hiopMatrixRajaSparseTriplet(int rows, int cols, int nnz, std::string memspace);
  virtual ~hiopMatrixRajaSparseTriplet(); 

  virtual void setToZero();
  virtual void setToConstant(double c);
  virtual void copyFrom(const hiopMatrixSparse& dm);

  virtual void copyRowsFrom(const hiopMatrix& src, const long long* rows_idxs, long long n_rows);
  
  virtual void timesVec(double beta,  hiopVector& y,
			double alpha, const hiopVector& x) const;
  virtual void timesVec(double beta,  double* y,
			double alpha, const double* x) const;

  virtual void transTimesVec(double beta,   hiopVector& y,
			     double alpha, const hiopVector& x) const;
  virtual void transTimesVec(double beta,   double* y,
			     double alpha, const double* x) const;

  virtual void timesMat(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const;

  virtual void transTimesMat(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const;

  virtual void timesMatTrans(double beta, hiopMatrix& W, double alpha, const hiopMatrix& X) const;

  virtual void addDiagonal(const double& alpha, const hiopVector& d_);
  virtual void addDiagonal(const double& value);
  virtual void addSubDiagonal(const double& alpha, long long start, const hiopVector& d_);
  /* add to the diagonal of 'this' (destination) starting at 'start_on_dest_diag' elements of
   * 'd_' (source) starting at index 'start_on_src_vec'. The number of elements added is 'num_elems' 
   * when num_elems>=0, or the remaining elems on 'd_' starting at 'start_on_src_vec'. */
  virtual void addSubDiagonal(int start_on_dest_diag, const double& alpha, 
			      const hiopVector& d_, int start_on_src_vec, int num_elems=-1)
  {
    assert(false && "not needed / implemented");
  }
  virtual void addSubDiagonal(int start_on_dest_diag, int num_elems, const double& c) 
  {
    assert(false && "not needed / implemented");
  }

  virtual void addMatrix(double alpha, const hiopMatrix& X);

  /* block of W += alpha*this */
  virtual void addToSymDenseMatrixUpperTriangle(int row_dest_start, int col_dest_start, 
						double alpha, hiopMatrixDense& W) const;
  /* block of W += alpha*transpose(this) */
  virtual void transAddToSymDenseMatrixUpperTriangle(int row_dest_start, int col_dest_start, 
						     double alpha, hiopMatrixDense& W) const;
  virtual void addUpperTriangleToSymDenseMatrixUpperTriangle(int diag_start, 
							     double alpha, hiopMatrixDense& W) const
  {
    assert(false && "counterpart method of hiopMatrixSymSparseTriplet should be used");
  }

  /* diag block of W += alpha * M * D^{-1} * transpose(M), where M=this 
   *
   * Only the upper triangular entries of W are updated.
   */
  virtual void addMDinvMtransToDiagBlockOfSymDeMatUTri(int rowCol_dest_start, const double& alpha, 
						       const hiopVector& D, hiopMatrixDense& W) const;

  /* block of W += alpha * M * D^{-1} * transpose(N), where M=this 
   *
   * Warning: The product matrix M * D^{-1} * transpose(N) with start offsets 'row_dest_start' and 
   * 'col_dest_start' needs to fit completely in the upper triangle of W. If this is NOT the 
   * case, the method will assert(false) in debug; in release, the method will issue a 
   * warning with HIOP_DEEPCHECKS (otherwise NO warning will be issue) and will silently update 
   * the (strictly) lower triangular  elements (these are ignored later on since only the upper 
   * triangular part of W will be accessed)
   */
  virtual void addMDinvNtransToSymDeMatUTri(int row_dest_start, int col_dest_start, const double& alpha,
					    const hiopVector& D, const hiopMatrixSparse& N,
					    hiopMatrixDense& W) const;

  virtual double max_abs_value();

  virtual bool isfinite() const;
  
  //virtual void print(int maxRows=-1, int maxCols=-1, int rank=-1) const;
  virtual void print(FILE* f=NULL, const char* msg=NULL, int maxRows=-1, int maxCols=-1, int rank=-1) const;

  virtual void startingAtAddSubDiagonalToStartingAt(int diag_src_start, const double& alpha, 
					    hiopVector& vec_dest, int vec_start, int num_elems=-1) const 
  {
    assert(0 && "This method should be used only for symmetric matrices.\n");
  }

  virtual hiopMatrix* alloc_clone() const;
  virtual hiopMatrix* new_copy() const;

  inline int* i_row() { return iRow_; }
  inline int* j_col() { return jCol_; }
  inline double* M() { return values_; }

  inline const int* i_row() const { return iRow_; }
  inline const int* j_col() const { return jCol_; }
  inline const double* M() const { return values_; }
  
  inline int* i_row_host() { return iRow_host_; }
  inline int* j_col_host() { return jCol_host_; }
  inline double* M_host() { return values_host_; }

  inline const int* i_row_host() const { return iRow_host_; }
  inline const int* j_col_host() const { return jCol_host_; }
  inline const double* M_host() const { return values_host_; }

  void copyToDev();
  void copyFromDev() const;

#ifdef HIOP_DEEPCHECKS
  virtual bool assertSymmetry(double tol=1e-16) const { return false; }
  virtual bool checkIndexesAreOrdered() const;
#endif
protected:
  int* iRow_; ///< row indices of the nonzero entries
  int* jCol_; ///< column indices of the nonzero entries
  double* values_; ///< values of the nonzero entries

  mutable int* iRow_host_;
  mutable int* jCol_host_;
  mutable double* values_host_;

  std::string mem_space_;// = "DEVICE";

protected:
  struct RowStartsInfo
  {
    int *idx_start_; //size num_rows+1
    int num_rows_;
    std::string mem_space_;
    RowStartsInfo()
      : idx_start_(NULL), num_rows_(0)
    {}
    RowStartsInfo(int n_rows, std::string memspace);
      //: idx_start(new int[n_rows+1]), num_rows(n_rows)
    virtual ~RowStartsInfo();
  };

  mutable RowStartsInfo* row_starts_host;
  //mutable RowStartsInfo* row_starts;

private:
  RowStartsInfo* allocAndBuildRowStarts() const; 
private:
  hiopMatrixRajaSparseTriplet() 
    : hiopMatrixSparse(0, 0, 0), iRow_(NULL), jCol_(NULL), values_(NULL)
  {
  }
  hiopMatrixRajaSparseTriplet(const hiopMatrixRajaSparseTriplet&) 
    : hiopMatrixSparse(0, 0, 0), iRow_(NULL), jCol_(NULL), values_(NULL)
  {
    assert(false);
  }
};

/** Sparse symmetric matrix in triplet format. Only the upper triangle is stored */
class hiopMatrixRajaSymSparseTriplet : public hiopMatrixRajaSparseTriplet 
{
public: 
  hiopMatrixRajaSymSparseTriplet(int n, int nnz, std::string memspace)
    : hiopMatrixRajaSparseTriplet(n, n, nnz, memspace)
  {
  }
  virtual ~hiopMatrixRajaSymSparseTriplet() {}  

  /** y = beta * y + alpha * this * x */
  virtual void timesVec(double beta,  hiopVector& y,
			double alpha, const hiopVector& x) const;
  virtual void timesVec(double beta,  double* y,
			double alpha, const double* x) const;

  virtual void transTimesVec(double beta,   hiopVector& y,
			     double alpha, const hiopVector& x) const
  {
    return timesVec(beta, y, alpha, x);
  }
  virtual void transTimesVec(double beta,   double* y,
			     double alpha, const double* x) const
  {
    return timesVec(beta, y, alpha, x);
  }

  virtual void addToSymDenseMatrixUpperTriangle(int row_dest_start, int col_dest_start,                                                                           
				double alpha, hiopMatrixDense& W) const;
  
  virtual void transAddToSymDenseMatrixUpperTriangle(int row_dest_start, int col_dest_start,                                                                           
				     double alpha, hiopMatrixDense& W) const;

  virtual void addUpperTriangleToSymDenseMatrixUpperTriangle(int diag_start, 
							    double alpha, hiopMatrixDense& W) const
  {
    assert(this->n()+diag_start < W.n());
    addToSymDenseMatrixUpperTriangle(diag_start, diag_start, alpha, W);
  }

  /* extract subdiagonal from 'this' (source) and adds the entries to 'vec_dest' starting at
   * index 'vec_start'. If num_elems>=0, 'num_elems' are copied; otherwise copies as many as
   * are available in 'vec_dest' starting at 'vec_start'
   */
  virtual void startingAtAddSubDiagonalToStartingAt(int diag_src_start, const double& alpha, 
					    hiopVector& vec_dest, int vec_start, int num_elems=-1) const;
					    

  virtual hiopMatrix* alloc_clone() const;
  virtual hiopMatrix* new_copy() const;

#ifdef HIOP_DEEPCHECKS
  virtual bool assertSymmetry(double tol=1e-16) const { return true; }
#endif
  virtual bool isDiagonal() const 
  {
    for(int itnnz = 0; itnnz < nnz; itnnz++)
    {
      if(iRow_[itnnz] != jCol_[itnnz])
      {
        return false;
      }
    }
    return true;
  }
};
} //end of namespace

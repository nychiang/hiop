#include "nlpSparse_ex7.hpp"

#include <cmath>
#include <cstring> //for memcpy
#include <cstdio>

/** Nonlinear *highly nonconvex* and *rank deficient* problem test for the Filter IPM
 * Newton of HiOp. It uses a Sparse NLP formulation. The problem is based on Ex6.
 *
 *  min   -(2*convex_obj-1)*sum 1/4* { (x_{i}-1)^4 : i=1,...,n} + 0.5x^Tx
 *  s.t.
 *            4*x_1 + 2*x_2                     == 10
 *        5<= 2*x_1         + x_3
 *        1<= 2*x_1                 + 0.5*x_i   <= 2*n, for i=4,...,n
 *        x_1 free
 *        0.0 <= x_2
 *        1.5 <= x_3 <= 10
 *        x_i >=0.5, i=4,...,n
 *
 * Optionally, one can add the following constraints to obtain a rank-deficient Jacobian
 *
 *  s.t.  [-inf] <= 4*x_1 + 2*x_3 <= [ 19 ]                  (rnkdef-con1)
 *        4*x_1 + 2*x_2 == 10                                (rnkdef-con2)
 *
 *
 */
Ex7::Ex7(int n, bool convex_obj, bool rankdefic_Jac_eq, bool rankdefic_Jac_ineq)
  : convex_obj_{convex_obj},
    rankdefic_eq_(rankdefic_Jac_eq),
    rankdefic_ineq_(rankdefic_Jac_ineq),
    n_vars{n},
    n_cons{2}
{
  assert(n>=3);
  if(n>3)
    n_cons += n-3;
  n_cons += rankdefic_eq_ + rankdefic_ineq_;
}

Ex7::~Ex7()
{}

bool Ex7::get_prob_sizes(long long& n, long long& m)
  { n=n_vars; m=n_cons; return true; }

bool Ex7::get_vars_info(const long long& n, double *xlow, double* xupp, NonlinearityType* type)
{
  assert(n==n_vars);
  for(long long i=0; i<n; i++) {
    if(i==0) { xlow[i]=-1e20; xupp[i]=1e20; type[i]=hiopNonlinear; continue; }
    if(i==1) { xlow[i]= 0.0;  xupp[i]=1e20; type[i]=hiopNonlinear; continue; }
    if(i==2) { xlow[i]= 1.5;  xupp[i]=10.0 ;type[i]=hiopNonlinear; continue; }
    //this is for x_4, x_5, ... , x_n (i>=3), which are bounded only from below
    xlow[i]= 0.5; xupp[i]=1e20;type[i]=hiopNonlinear;
  }
  return true;
}

bool Ex7::get_cons_info(const long long& m, double* clow, double* cupp, NonlinearityType* type)
{
  assert(m==n_cons);
  long long conidx{0};
  clow[conidx]= 10.0;    cupp[conidx]= 10.0;      type[conidx++]=hiopInterfaceBase::hiopLinear;
  clow[conidx]= 5.0;     cupp[conidx]= 1e20;      type[conidx++]=hiopInterfaceBase::hiopLinear;
  for(long long i=3; i<n_vars; i++) {
    clow[conidx] = 1.0;   cupp[conidx]= 2*n_vars; type[conidx++]=hiopInterfaceBase::hiopLinear;
  }

  if(rankdefic_ineq_) {
    // [-inf] <= 4*x_1 + 2*x_3 <= [ 19 ]
    clow[conidx] = -1e+20;   cupp[conidx] = 19.;  type[conidx++]=hiopInterfaceBase::hiopNonlinear;
  }

  if(rankdefic_eq_) {
    //  4*x_1 + 2*x_2 == 10
    clow[conidx] = 10;    cupp[conidx] = 10;   type[conidx++]=hiopInterfaceBase::hiopNonlinear;
  }
  assert(conidx==m);
  return true;
}

bool Ex7::get_sparse_blocks_info(int& nx,
					    int& nnz_sparse_Jaceq, int& nnz_sparse_Jacineq,
					    int& nnz_sparse_Hess_Lagr)
{
    nx = n_vars;;
    nnz_sparse_Jaceq = 2 + 2*rankdefic_eq_;
    nnz_sparse_Jacineq = 2 + 2*(n_vars-3) + 2*rankdefic_ineq_;
    nnz_sparse_Hess_Lagr = n_vars;
    return true;
}

bool Ex7::eval_f(const long long& n, const double* x, bool new_x, double& obj_value)
{
  assert(n==n_vars);
  obj_value=0.;
  for(auto i=0;i<n;i++) obj_value += (2*convex_obj_-1)*0.25*pow(x[i]-1., 4) + 0.5*pow(x[i], 2);

  return true;
}

bool Ex7::eval_grad_f(const long long& n, const double* x, bool new_x, double* gradf)
{
  assert(n==n_vars);
  for(auto i=0;i<n;i++) gradf[i] = (2*convex_obj_-1)*pow(x[i]-1.,3) + x[i];
  return true;
}

bool Ex7::eval_cons(const long long& n, const long long& m,
			 const long long& num_cons, const long long* idx_cons,
			 const double* x, bool new_x, double* cons)
{
  return false;
}

/* Four constraints no matter how large n is */
bool Ex7::eval_cons(const long long& n, const long long& m,
		    const double* x, bool new_x, double* cons)
{
  assert(n==n_vars); assert(m==n_cons);
  assert(n_cons==2+n-3+rankdefic_eq_+rankdefic_ineq_);

  //local contributions to the constraints in cons are reset
  for(auto j=0;j<m; j++) cons[j]=0.;

  long long conidx{0};
  //compute the constraint one by one.
  // --- constraint 1 body --->  4*x_1 + 2*x_2 == 10
  cons[conidx++] += 4*x[0] + 2*x[1];

  // --- constraint 2 body ---> 2*x_1 + x_3
  cons[conidx++] += 2*x[0] + 1*x[2];

  // --- constraint 3 body --->   2*x_1 + 0.5*x_i, for i>=4
  for(auto i=3; i<n; i++) {
    cons[conidx++] += 2*x[0] + 0.5*x[i];
  }

  if(rankdefic_ineq_) {
    // [-inf] <= 4*x_1 + 2*x_3 <= [ 19 ]
    cons[conidx++] = 4*x[0] + 2*x[2];
  }

  if(rankdefic_eq_) {
    //  4*x_1 + 2*x_2 == 10
    cons[conidx++] += 4*x[0] + 2*x[1];
  }
  assert(conidx==m);

  return true;
}

bool Ex7::eval_Jac_cons(const long long& n, const long long& m,
			     const long long& num_cons, const long long* idx_cons,
			     const double* x, bool new_x,
			     const int& nnzJacS, int* iJacS, int* jJacS, double* MJacS)
{
  return false;
}

bool Ex7::eval_Jac_cons(const long long& n, const long long& m,
			     const double* x, bool new_x,
			     const int& nnzJacS, int* iJacS, int* jJacS, double* MJacS)
{
    assert(n==n_vars); assert(m==n_cons);
    assert(n>=3);

    assert(nnzJacS == 4 + 2*(n-3) + 2*rankdefic_eq_ + 2*rankdefic_ineq_);


    int nnzit{0};
    long long conidx{0};

    if(iJacS!=NULL && jJacS!=NULL){
        // --- constraint 1 body --->  4*x_1 + 2*x_2 == 10
        iJacS[nnzit] = conidx;   jJacS[nnzit++] = 0;
        iJacS[nnzit] = conidx;   jJacS[nnzit++] = 1;
        conidx++;

        // --- constraint 2 body ---> 2*x_1 + x_3
        iJacS[nnzit] = conidx;   jJacS[nnzit++] = 0;
        iJacS[nnzit] = conidx;   jJacS[nnzit++] = 2;
        conidx++;

        // --- constraint 3 body --->   2*x_1 + 0.5*x_i, for i>=4
        for(auto i=3; i<n; i++){
            iJacS[nnzit] = conidx;   jJacS[nnzit++] = 0;
            iJacS[nnzit] = conidx;   jJacS[nnzit++] = i;
            conidx++;
        }

        if(rankdefic_ineq_) {
          // [-inf] <= 4*x_1 + 2*x_3 <= [ 19 ]
          iJacS[nnzit] = conidx;   jJacS[nnzit++] = 0;
          iJacS[nnzit] = conidx;   jJacS[nnzit++] = 2;
          conidx++;
        }

        if(rankdefic_eq_) {
          //  4*x_1 + 2*x_2 == 10
          iJacS[nnzit] = conidx;   jJacS[nnzit++] = 0;
          iJacS[nnzit] = conidx;   jJacS[nnzit++] = 1;
          conidx++;
        }
        assert(conidx==m);
        assert(nnzit == nnzJacS);
    }

    //values for sparse Jacobian if requested by the solver
    nnzit = 0;
    if(MJacS!=NULL) {
        // --- constraint 1 body --->  4*x_1 + 2*x_2 == 10
        MJacS[nnzit++] = 4;
        MJacS[nnzit++] = 2;

        // --- constraint 2 body ---> 2*x_1 + x_3
        MJacS[nnzit++] = 2;
        MJacS[nnzit++] = 1;

        // --- constraint 3 body --->   2*x_1 + 0.5*x_4
        for(auto i=3; i<n; i++){
            MJacS[nnzit++] = 2;
            MJacS[nnzit++] = 0.5;
        }

        if(rankdefic_ineq_) {
          // [-inf] <= 4*x_1 + 2*x_3 <= [ 19 ]
          MJacS[nnzit++] = 4;
          MJacS[nnzit++] = 2;
        }

        if(rankdefic_eq_) {
          //  4*x_1 + 2*x_2 == 10
          MJacS[nnzit++] = 4;
          MJacS[nnzit++] = 2;
        }
        assert(nnzit == nnzJacS);
    }
    return true;
}

bool Ex7::eval_Hess_Lagr(const long long& n, const long long& m,
			      const double* x, bool new_x, const double& obj_factor,
			      const double* lambda, bool new_lambda,
			      const int& nnzHSS, int* iHSS, int* jHSS, double* MHSS)
{
    //Note: lambda is not used since all the constraints are linear and, therefore, do
    //not contribute to the Hessian of the Lagrangian
    assert(nnzHSS == n);

    if(iHSS!=NULL && jHSS!=NULL) {
      for(int i=0; i<n; i++) iHSS[i] = jHSS[i] = i;
    }

    if(MHSS!=NULL) {
      for(int i=0; i<n; i++) MHSS[i] = obj_factor * ( (2*convex_obj_-1) * 3*pow(x[i]-1., 2) + 1);
    }
    return true;
}

bool Ex7::get_starting_point(const long long& n, double* x0)
{
  assert(n==n_vars);
  for(auto i=0; i<n; i++)
    x0[i]=0.0;
  return true;
}

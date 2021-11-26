/*
 * Define the new operator for f^i = Func(f^i_j, f^i_j), 
 * where f^i_j = F^i * GR^i_j, the j and i is the neighbor
 * the code is used to build a conservative force field
 * 
 * The F is a Nf x (NaxNix3) variable
 * The L is a Nf x (NaxNi) variable
 * The Y is a Nf x (Nax3) variable
 * 
 * 		-- Pinghui Mo, 2020/9/22
 * e-mail: pinghui_mo@outlook.com
 */


//- import the library of tensorflow
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <cmath>
#include <stdio.h>

using namespace tensorflow;

// #defin HIGH_PRECISION


//- register the operator
#ifdef HIGH_PRECISION
REGISTER_OP("Mzscatter")
  .Input("f: double")
  .Input("l: int32")
  .Attr("ni: int")
  .Output("y: double");
#else
REGISTER_OP("Mzscatter")
  .Input("f: float")
  .Input("l: int32")
  .Attr("ni: int")
  .Output("y: float");
#endif


//- create the operator class
//* the class must inherit the OpKernel Class
class MzscatterOp : public OpKernel {
public:

  /// Constructor.
  explicit MzscatterOp(OpKernelConstruction* context) : OpKernel(context) {
	  //- define the attribute of context
	  //* the context is the input from your tensorflow code
	  OP_REQUIRES_OK(context, context->GetAttr("ni", &Ni));

	  //- init the private member vairbles
	  
  }
  
  
  /// Compute the descriptor
  /// param: context
  void Compute(OpKernelContext* context) override {
    
    /* 
     * Get input
     * 1.check
     * 2.get tensor
     * 3.get shape and check
     */
	
	//- 1.check
    DCHECK_EQ(2, context->num_inputs());
    
    //- 2.get tensor
    const Tensor& F = context->input(0);
    const Tensor& L = context->input(1);
    
    //- 3. get shape and check
    const TensorShape& shF = F.shape();
    const TensorShape& shL = L.shape();
    
    int Nf   = shF.dim_size(0);
    int Dim  = shF.dim_size(1);
    int Dim2 = shL.dim_size(1);
    int D    = 3;
    int Na   = Dim / Ni / D;
	
    DCHECK_EQ(shF.dims(), 2);
    DCHECK_EQ(shL.dims(), 2);
    DCHECK_EQ(shL.dim_size(0), Nf);
    DCHECK_EQ(shL.dim_size(1), Na*Ni);
    
    /*
     * Calculate the output
     * 1.create tensor
     * 2.allocate the memory
     * 3.calculate
     */
    
    //- 1.create tensor
    TensorShape shY;
    shY.AddDim(Nf);
    shY.AddDim(Na*D);
    
    Tensor* Y = NULL;
    
    //- 2.allocate the memory
    //* allocate memory for the Y tensor which is called output 0
    OP_REQUIRES_OK(context, context->allocate_output(0, shY, &Y));
    #ifdef HIGH_PRECISION
    auto f = F.matrix<double>();
    auto l = L.matrix<int>();
    auto y = Y->matrix<double>();
    #else
    auto f = F.matrix<float>();
    auto l = L.matrix<int>();
    auto y = Y->matrix<float>();
    #endif
    
    // calculate
    int ff, ii, jj;
    int ij, lij, ii3, ij3, lij3;
    // std::cout<<"MZSCATTER: INIT\n";
    for(ff=0; ff<Nf; ff++){
    	for(ii=0; ii<Na; ii++){
    		y(ff, 3*ii+0) = 0.0;
    		y(ff, 3*ii+1) = 0.0;
    		y(ff, 3*ii+2) = 0.0;
      }
    }

    // std::cout<<"MZSCATTER: START\n";
    for(ff=0; ff<Nf; ff++){
    	for(ii=0; ii<Na; ii++){
        ii3 = 3*ii;
    		for(jj=0; jj<Ni; jj++){
          ij = ii*Ni+jj;
          lij = l(ff, ij);
          ij3 = 3*ij;
          lij3 = 3*lij;
          // std::cout<<"F:"<<f(ff, ij3 +0)<<", "<<f(ff, ij3 +1)<<", "<<f(ff, ij3 +2)<<", "<<"| ";
          if(lij >= 0){
            // f^i += f^i_j
            y(ff,ii3 +0) += f(ff, ij3+0);
            y(ff,ii3 +1) += f(ff, ij3+1);
            y(ff,ii3 +2) += f(ff, ij3+2);
            // f^j -= f^i_j
            y(ff,lij3+0) -= f(ff, ij3+0);
            y(ff,lij3+1) -= f(ff, ij3+1);
            y(ff,lij3+2) -= f(ff, ij3+2);
            // std::cout<<"Y:"<<y(ff,ii3 +0)<<", "<<y(ff,ii3 +0)<<", "<<y(ff,ii3 +0)<<", "<<"|| ";
            // std::cout<<"Y2:"<<y(ff,lij3 +0)<<", "<<y(ff,lij3 +0)<<", "<<y(ff,lij3 +0)<<", "<<"|| \n";
          }
    			
    		}
    	}
    }

    // DEBUG
    // std::cout<<"MZSCATTER: Y\n";
    // for(ff=0; ff<Nf; ff++){
    // 	for(ii=0; ii<Na; ii++){
    //     ii3 = 3*ii;
    //     std::cout<<y(ff,ii3 +0)<<", "<<y(ff,ii3 +1)<<", "<<y(ff,ii3 +2)<<", "<<"| ";
    //   }
    //   std::cout<<"\n";
    // }
    
  }
  
//- define the private variable for calculation
private:
float tmp=0;
int Ni;
};

REGISTER_KERNEL_BUILDER(Name("Mzscatter").Device(DEVICE_CPU), MzscatterOp);





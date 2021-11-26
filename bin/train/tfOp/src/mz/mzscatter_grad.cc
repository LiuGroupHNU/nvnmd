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

using namespace tensorflow;

// #defin HIGH_PRECISION


//- register the operator
#ifdef HIGH_PRECISION
REGISTER_OP("MzscatterGrad")
  .Input("f: double")
  .Input("l: int32")
  .Input("dy: double")
  .Attr("ni: int")
  .Output("dydf: double");
#else
REGISTER_OP("MzscatterGrad")
  .Input("f: float")
  .Input("l: int32")
  .Input("dy: float")
  .Attr("ni: int")
  .Output("dydf: float");
#endif


//- create the operator class
//* the class must inherit the OpKernel Class
class MzscatterGradOp : public OpKernel {
public:

	
  /// Constructor.
  explicit MzscatterGradOp(OpKernelConstruction* context) : OpKernel(context) {
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
    DCHECK_EQ(3, context->num_inputs());
    
    //- 2.get tensor
    const Tensor& F  = context->input(0);
    const Tensor& L  = context->input(1);
    const Tensor& DY = context->input(2);
    
    //- 3. get shape and check
    const TensorShape& shF  = F.shape();
    const TensorShape& shL  = L.shape();
    const TensorShape& shDY = DY.shape();
    
    int Nf   = shF.dim_size(0);
    int Dim  = shF.dim_size(1);
    int Dim2 = shL.dim_size(1);
    int D    = 3;
    int Na   = Dim / Ni / D;
	

    DCHECK_EQ(shF.dims(), 2);
    DCHECK_EQ(shL.dims(), 2);
    DCHECK_EQ(shDY.dims(), 2);
    DCHECK_EQ(shL.dim_size(0), Nf);
    DCHECK_EQ(shL.dim_size(1), Na*Ni);
    DCHECK_EQ(shDY.dim_size(0), Nf);
    DCHECK_EQ(shDY.dim_size(1), Na*D);
    
    
    /*
     * Calculate the output
     * 1.create tensor
     * 2.allocate the memory
     * 3.calculate
     */
    
    //- 1.create tensor
    TensorShape shDYDF;
    shDYDF.AddDim(Nf);
    shDYDF.AddDim(Na*Ni*3);
    
    Tensor* DYDF = NULL;
    
    //- 2.allocate the memory
    //* allocate memory 
    OP_REQUIRES_OK(context, context->allocate_output(0, shDYDF, &DYDF));
    
    #ifdef HIGH_PRECISION
    auto f = F.matrix<double>();
    auto l = L.matrix<int>();
    auto dy = DY.matrix<double>();
    auto dydf = DYDF->matrix<double>();
    #else
    auto f = F.matrix<float>();
    auto l = L.matrix<int>();
    auto dy = DY.matrix<float>();
    auto dydf = DYDF->matrix<float>();
    #endif
    
    // calculate
    int ff, ii, jj;
    int ij, lij, ii3, ij3, lij3;
    // std::cout<<"MZSCATTER_GRAD: INIT\n";
    for(ff=0; ff<Nf; ff++){
    	for(ii=0; ii<Na; ii++){
        for(jj=0; jj<Ni; jj++){
          ij3 = (ii*Ni+jj)*3;
          dydf(ff, ij3+0) = 0.0;
          dydf(ff, ij3+1) = 0.0;
          dydf(ff, ij3+2) = 0.0;
        }
      }
    }

    // std::cout<<"MZSCATTER_GRAD: STARt\n";
    for(ff=0; ff<Nf; ff++){
    	for(ii=0; ii<Na; ii++){
        ii3 = 3*ii;
    		for(jj=0; jj<Ni; jj++){
          ij = ii*Ni+jj;
          lij = l(ff, ij);
          ij3 = 3*ij;
          lij3 = 3*lij;
          if(lij >= 0){
            // f^i += f^i_j
            dydf(ff, ij3+0) += dy(ff,ii3 +0);
            dydf(ff, ij3+1) += dy(ff,ii3 +1);
            dydf(ff, ij3+2) += dy(ff,ii3 +2);
            // f^j -= f^i_j
            dydf(ff, ij3+0) -= dy(ff,lij3+0);
            dydf(ff, ij3+1) -= dy(ff,lij3+1);
            dydf(ff, ij3+2) -= dy(ff,lij3+2);
          }
    			
    		}
    	}
    }

    // // DEBUG
    // std::cout<<"MZSCATTER_GRAD: DY\n";
    // for(ff=0; ff<Nf; ff++){
    // 	for(ii=0; ii<Na; ii++){
    //     ii3 = 3*ii;
    //     std::cout<<dy(ff,ii3 +0)<<", "<<dy(ff,ii3 +1)<<", "<<dy(ff,ii3 +2)<<", "<<"| ";
    //   }
    //   std::cout<<"\n";
    // }
    
  }
  
  //- define the private variable for calculation
  private:
  float tmp = 0;
  int Ni;
};

REGISTER_KERNEL_BUILDER(Name("MzscatterGrad").Device(DEVICE_CPU), MzscatterGradOp);

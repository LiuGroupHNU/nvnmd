

//
// --------------------------------------------------------------------
/*

# 功能
将输入的 x 量化为 nbit 位小数位的定点数，然后输出

# 参数
nbit x 量化的小数位
nbit2 dy_dx量化的小数位
nbit3 dy2_dx2量化的小数位

# 注意
1. nbit < 0 时， 表示不需要量化，直接输出浮点数
2. 为了实现简便，且范用，x的维度为2个维度

*/
// --------------------------------------------------------------------
//

//- import the library of tensorflow
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <cmath>
#include <stdio.h>

using namespace tensorflow;


//- register the operator
#ifdef HIGH_PRECISION
REGISTER_OP("Mzmatmul")
  .Input("x: double")
  .Input("w: double")
  .Attr("isround: int")
  .Attr("nbit1: int")
  .Attr("nbit2: int")
  .Attr("nbit3: int")
  .Output("y: double")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
    shape_inference::ShapeHandle shX;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &shX));
    shape_inference::DimensionHandle D1 = c->Dim(shX, 0);
    shape_inference::DimensionHandle D2 = c->Dim(shX, 1);
    c->set_output(0, c->Matrix(D1, D2));
    return Status::OK();
  });
#else
REGISTER_OP("Mzmatmul")
  .Input("x: float")
  .Input("w: float")
  .Attr("isround: int")
  .Attr("nbit1: int")
  .Attr("nbit2: int")
  .Attr("nbit3: int")
  .Output("y: float")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
    shape_inference::ShapeHandle shX, shW;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &shX));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &shW));
    shape_inference::DimensionHandle D1 = c->Dim(shX, 0);
    shape_inference::DimensionHandle D2 = c->Dim(shX, 1);
    shape_inference::DimensionHandle D3 = c->Dim(shW, 0);
    shape_inference::DimensionHandle D4 = c->Dim(shW, 1);
    c->set_output(0, c->Matrix(D1, D4));
    return Status::OK();
  });
#endif



//- create the operator class
//* the class must inherit the OpKernel Class
class MzmatmulOp : public OpKernel {
public:

  /// Constructor.
  explicit MzmatmulOp(OpKernelConstruction* context) : OpKernel(context) {
	  //- define the attribute of context
	  //* the context is the input from your tensorflow code
    OP_REQUIRES_OK(context, context->GetAttr("nbit1", &nbit1));
    OP_REQUIRES_OK(context, context->GetAttr("nbit2", &nbit2));
    OP_REQUIRES_OK(context, context->GetAttr("nbit3", &nbit3));
    OP_REQUIRES_OK(context, context->GetAttr("isround", &isround));
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
    const Tensor& X = context->input(0);
    const Tensor& W = context->input(1);
    
    //- 3. get shape and check
    const TensorShape& shX = X.shape();
    const TensorShape& shW = W.shape();
    
    int D1 = shX.dim_size(0);
    int D2 = shX.dim_size(1);
    int D3 = shW.dim_size(0);
    int D4 = shW.dim_size(1);
    
    /*
     * Calculate the output
     * 1.create tensor
     * 2.allocate the memory
     * 3.calculate
     */
    
    //- 1.create tensor
    TensorShape shY;
    shY.AddDim(D1);
    shY.AddDim(D4);
    
    Tensor* Y = NULL;
    
    //- 2.allocate the memory
    //* allocate memory for the Y tensor which is called output 0
    OP_REQUIRES_OK(context, context->allocate_output(0, shY, &Y));
    #ifdef HIGH_PRECISION
    auto x = X.matrix<double>();
    auto w = W.matrix<double>();
    auto y = Y->matrix<double>();
    double prec;
    #else
    auto x = X.matrix<float>();
    auto w = W.matrix<float>();
    auto y = Y->matrix<float>();
    float prec;
    #endif

    
    // calculate
    int ii, jj, kk;
    
    for(ii=0; ii<D1; ii++){
        for(kk=0; kk<D4; kk++){
          y(ii, kk) = 0.0;
        }
      }

    if (this->nbit1 < 0){
      for(ii=0; ii<D1; ii++){
        for(kk=0; kk<D4; kk++){
          for(jj=0; jj<D2; jj++){
            y(ii, kk) += x(ii, jj) * w(jj, kk);
          }
        }
      }
    }
    //
    else {
      prec = 1 << this->nbit1;

      if (this->isround)
        for(ii=0; ii<D1; ii++){
          for(kk=0; kk<D4; kk++){
            for(jj=0; jj<D2; jj++){
              y(ii, kk) += round(x(ii, jj) * w(jj, kk) * prec)/ prec;
            }
          }
        }
      //
      else
        for(ii=0; ii<D1; ii++){
          for(kk=0; kk<D4; kk++){
            for(jj=0; jj<D2; jj++){
              y(ii, kk) += floor(x(ii, jj) * w(jj, kk) * prec)/ prec;
            }
          }
        }
    }
  }
  
//- define the private variable for calculation
private:
int nbit1, nbit2, nbit3;
int isround;
};

REGISTER_KERNEL_BUILDER(Name("Mzmatmul").Device(DEVICE_CPU), MzmatmulOp);





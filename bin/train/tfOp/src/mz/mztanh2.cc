

//
// --------------------------------------------------------------------
/*

# 功能
输入x, 输出y=tanh2(x)

x1 = clip(x, -2, 2)
x2 = clip(x, -4, 4)
y1 = x1    - x1*|x1|/4
y2 = x2/32 - x2*|x2|/256 
y = y1 + y2

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
REGISTER_OP("Mztanh2")
  .Input("x: double")
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
REGISTER_OP("Mztanh2")
  .Input("x: float")
  .Attr("isround: int")
  .Attr("nbit1: int")
  .Attr("nbit2: int")
  .Attr("nbit3: int")
  .Output("y: float")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
    shape_inference::ShapeHandle shX;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &shX));
    shape_inference::DimensionHandle D1 = c->Dim(shX, 0);
    shape_inference::DimensionHandle D2 = c->Dim(shX, 1);
    c->set_output(0, c->Matrix(D1, D2));
    return Status::OK();
  });
#endif



//- create the operator class
//* the class must inherit the OpKernel Class
class Mztanh2Op : public OpKernel {
public:

  /// Constructor.
  explicit Mztanh2Op(OpKernelConstruction* context) : OpKernel(context) {
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
    DCHECK_EQ(1, context->num_inputs());
    
    //- 2.get tensor
    const Tensor& X = context->input(0);
    
    //- 3. get shape and check
    const TensorShape& shX = X.shape();
    
    int D1 = shX.dim_size(0);
    int D2 = shX.dim_size(1);
    
    /*
     * Calculate the output
     * 1.create tensor
     * 2.allocate the memory
     * 3.calculate
     */
    
    //- 1.create tensor
    TensorShape shY;
    shY.AddDim(D1);
    shY.AddDim(D2);
    
    Tensor* Y = NULL;
    
    //- 2.allocate the memory
    //* allocate memory for the Y tensor which is called output 0
    OP_REQUIRES_OK(context, context->allocate_output(0, shY, &Y));
    #ifdef HIGH_PRECISION
    auto xs = X.matrix<double>();
    auto ys = Y->matrix<double>();
    double prec;
    double x, x1, x2;
    double y, y1, y2;
    #else
    auto xs = X.matrix<float>();
    auto ys = Y->matrix<float>();
    float prec;
    float x, x1, x2;
    float y, y1, y2;
    #endif

    
    // calculate
    int ii, jj;
    bool  sign;
    

    if (this->nbit1 < 0){
      for(ii=0; ii<D1; ii++){
        for(jj=0; jj<D2; jj++){
          sign = xs(ii, jj) < 0;
          x = (sign) ? -xs(ii, jj) : xs(ii, jj);
          x1 = (x >  2) ?  2 : x;
          x2 = (x >  4) ?  4 : x;
          y1 = x1 - x1 * x1 * 0.25;
          y2 = x2 * 0.03125 - x2 * x2 * 0.00390625;
          ys(ii, jj) = (sign) ? -(y1 + y2) : (y1 + y2);
        }
      }
    }
    //
    else {
      prec = 1 << this->nbit1;

      if (this->isround)
      for(ii=0; ii<D1; ii++){
        for(jj=0; jj<D2; jj++){
          sign = xs(ii, jj) < 0;
          x = (sign) ? -xs(ii, jj) : xs(ii, jj);
          x =  round(x * prec) / prec;
          x1 = (x >  2) ?  2 : x;
          x2 = (x >  4) ?  4 : x;
          y1 = round((x1 - x1 * x1 * 0.25) * prec) / prec;
          y2 = round((x2 * 0.03125 - x2 * x2 * 0.00390625) * prec) / prec;
          ys(ii, jj) = (sign) ? -(y1 + y2) : (y1 + y2);
        }
      }
      //
      else
      for(ii=0; ii<D1; ii++){
        for(jj=0; jj<D2; jj++){
          sign = xs(ii, jj) < 0;
          x = (sign) ? -xs(ii, jj) : xs(ii, jj);
          x =  floor(x * prec) / prec;
          x1 = (x >  2) ?  2 : x;
          x2 = (x >  4) ?  4 : x;
          y1 = floor((x1 - x1 * x1 * 0.25) * prec) / prec;
          y2 = floor((x2 * 0.03125 - x2 * x2 * 0.00390625) * prec) / prec;
          ys(ii, jj) = (sign) ? -(y1 + y2) : (y1 + y2);
        }
      }
    }
  }
  
//- define the private variable for calculation
private:
int nbit1, nbit2, nbit3;
int isround;
};

REGISTER_KERNEL_BUILDER(Name("Mztanh2").Device(DEVICE_CPU), Mztanh2Op);





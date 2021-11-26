
//
// --------------------------------------------------------------------
/*

# 功能
使用 X 来检索 W 中的元素
X (D1, D2)
W (D3, D4)
Y (D1, D2*D4)

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
REGISTER_OP("Mzmap")
  .Input("x: double")
  .Input("w: double")
  .Input("w2: double")
  .Output("y: double")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    shape_inference::ShapeHandle shX, shW, shW2;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &shX));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &shW));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &shW2));
    
    shape_inference::DimensionHandle D1 = c->Dim(shX, 0);
    shape_inference::DimensionHandle D2 = c->Dim(shX, 1);
    shape_inference::DimensionHandle D3 = c->Dim(shW, 0);
    shape_inference::DimensionHandle D4 = c->Dim(shW, 1);

    c->set_output(0, c->Matrix(D1, D4));
    return Status::OK();
  });
#else
REGISTER_OP("Mzmap")
  .Input("x: float")
  .Input("w: float")
  .Input("w2: float")
  .Output("y: float")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    shape_inference::ShapeHandle shX, shW, shW2;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &shX));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &shW));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &shW2));
    
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
class MzmapOp : public OpKernel {
public:

  /// Constructor.
  explicit MzmapOp(OpKernelConstruction* context) : OpKernel(context) {	  
    // OP_REQUIRES_OK(context, context->GetAttr("st", &st));
    // OP_REQUIRES_OK(context, context->GetAttr("dt", &dt));
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
    const Tensor& X = context->input(0);
    const Tensor& W = context->input(1);
    
    //- 3. get shape and check
    const TensorShape& shX = X.shape();
    const TensorShape& shW = W.shape();
    
    int D1 = shX.dim_size(0);
    int D2 = shX.dim_size(1);
    int D3 = shW.dim_size(0);
    int D4 = shW.dim_size(1);
	
    DCHECK_EQ(shX.dims(), 2);
    DCHECK_EQ(shW.dims(), 2);
    
    /*
     * Calculate the output
     * 1.create tensor
     * 2.allocate the memory
     * 3.calculate
     */
    
    //- 1.create tensor
    TensorShape shY;
    shY.AddDim(D1);
    shY.AddDim(D2*D4);
    Tensor* Y = NULL;
    
    //- 2.allocate the memory
    //* allocate memory for the Y tensor which is called output 0
    OP_REQUIRES_OK(context, context->allocate_output(0, shY, &Y));
    #ifdef HIGH_PRECISION
    auto x = X.matrix<double>();
    auto w = W.matrix<double>();
    auto y = Y->matrix<double>();
    #else
    auto x = X.matrix<float>();
    auto w = W.matrix<float>();
    auto y = Y->matrix<float>();
    #endif

    float dt = 1 / D3;
    int ii, jj, kk, jk, n;
    for(ii=0; ii<D1; ii++){
      jk = 0;
      for(jj=0; jj<D2; jj++){
        n = floor(x(ii, jj) * D3); // 向下取整
        //check
        if (n < 0)  std::cout<<"ERROR: index ( " << n <<" ) is smaller than 0 \n";
        if (n > D3) std::cout<<"ERROR: index ( " << n <<" ) is bigger  than " << D3 <<" \n";
        n = (n == D3) ? (D3 - 1) : n;
        //map
        for(kk=0; kk<D4; kk++){
          y(ii, jk) = w(n, kk);
          jk ++;
        }
      }
    }

  }
//- define the private variable for calculation
private:
#ifdef HIGH_PRECISION
double st, dt;
#else
float st, dt;
#endif
};

REGISTER_KERNEL_BUILDER(Name("Mzmap").Device(DEVICE_CPU), MzmapOp);





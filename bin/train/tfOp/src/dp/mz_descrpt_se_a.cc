#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <iostream>

#include "ComputeDescriptor.h"
#include "NeighborList.h"

typedef double boxtensor_t ;
typedef double compute_t;

using namespace tensorflow;
using namespace std;

#ifdef HIGH_PREC
typedef double VALUETYPE ;
#else 
typedef float  VALUETYPE ;
#endif

#ifdef HIGH_PREC
REGISTER_OP("MzDescrptSeA")
.Input("coord: double")
.Input("type: int32")
.Input("natoms: int32")
.Input("box: double")
.Input("mesh: int32")
.Input("davg: double")
.Input("dstd: double")
.Attr("rcut_a: float")
.Attr("rcut_r: float")
.Attr("rcut_r_smth: float")
.Attr("sel_a: list(int)")
.Attr("sel_r: list(int)")
.Output("descrpt: double")
.Output("descrpt_deriv: double")
.Output("rij: double")
.Output("nlist: int32");
#else
REGISTER_OP("MzDescrptSeA")
.Input("coord: float")
.Input("type: int32")
.Input("natoms: int32")
.Input("box: float")
.Input("mesh: int32")
.Input("davg: float")
.Input("dstd: float")
.Attr("rcut_a: float")
.Attr("rcut_r: float")
.Attr("rcut_r_smth: float")
.Attr("sel_a: list(int)")
.Attr("sel_r: list(int)")
.Output("descrpt: float")
.Output("descrpt_deriv: float")
.Output("rij: float")
.Output("nlist: int32");
#endif

class MzDescrptSeAOp : public OpKernel {
public:
  explicit MzDescrptSeAOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("rcut_a", &rcut_a));
    OP_REQUIRES_OK(context, context->GetAttr("rcut_r", &rcut_r));
    OP_REQUIRES_OK(context, context->GetAttr("rcut_r_smth", &rcut_r_smth));
    OP_REQUIRES_OK(context, context->GetAttr("sel_a", &sel_a));
    OP_REQUIRES_OK(context, context->GetAttr("sel_r", &sel_r));
    cum_sum (sec_a, sel_a);
    cum_sum (sec_r, sel_r);
    ndescrpt_a = sec_a.back() * 4;
    ndescrpt_r = sec_r.back() * 1;
    ndescrpt = ndescrpt_a + ndescrpt_r;
    nnei_a = sec_a.back();
    nnei_r = sec_r.back();
    nnei = nnei_a + nnei_r;
    fill_nei_a = (rcut_a < 0);
    count_nei_idx_overflow = 0;
    //MZ
    init_precs();
    cal_map_param(rcut_r);
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    int context_input_index = 0;
    const Tensor& coord_tensor	= context->input(context_input_index++);
    const Tensor& type_tensor	= context->input(context_input_index++);
    const Tensor& natoms_tensor	= context->input(context_input_index++);
    const Tensor& box_tensor	= context->input(context_input_index++);
    const Tensor& mesh_tensor	= context->input(context_input_index++);
    const Tensor& avg_tensor	= context->input(context_input_index++);
    const Tensor& std_tensor	= context->input(context_input_index++);

    // set size of the sample
    OP_REQUIRES (context, (coord_tensor.shape().dims() == 2),	errors::InvalidArgument ("Dim of coord should be 2"));
    OP_REQUIRES (context, (type_tensor.shape().dims() == 2),	errors::InvalidArgument ("Dim of type should be 2"));
    OP_REQUIRES (context, (natoms_tensor.shape().dims() == 1),	errors::InvalidArgument ("Dim of natoms should be 1"));
    OP_REQUIRES (context, (box_tensor.shape().dims() == 2),	errors::InvalidArgument ("Dim of box should be 2"));
    OP_REQUIRES (context, (mesh_tensor.shape().dims() == 1),	errors::InvalidArgument ("Dim of mesh should be 1"));
    OP_REQUIRES (context, (avg_tensor.shape().dims() == 2),	errors::InvalidArgument ("Dim of avg should be 2"));
    OP_REQUIRES (context, (std_tensor.shape().dims() == 2),	errors::InvalidArgument ("Dim of std should be 2"));
    OP_REQUIRES (context, (fill_nei_a),				errors::InvalidArgument ("Rotational free descriptor only support the case rcut_a < 0"));
    OP_REQUIRES (context, (sec_r.back() == 0),			errors::InvalidArgument ("Rotational free descriptor only support all-angular information: sel_r should be all zero."));

    OP_REQUIRES (context, (natoms_tensor.shape().dim_size(0) >= 3),		errors::InvalidArgument ("number of atoms should be larger than (or equal to) 3"));
    auto natoms	= natoms_tensor	.flat<int>();
    int nloc = natoms(0);
    int nall = natoms(1);
    int ntypes = natoms_tensor.shape().dim_size(0) - 2;
    int nsamples = coord_tensor.shape().dim_size(0);

    // check the sizes
    OP_REQUIRES (context, (nsamples == type_tensor.shape().dim_size(0)),	errors::InvalidArgument ("number of samples should match"));
    OP_REQUIRES (context, (nsamples == box_tensor.shape().dim_size(0)),		errors::InvalidArgument ("number of samples should match"));
    OP_REQUIRES (context, (ntypes == avg_tensor.shape().dim_size(0)),		errors::InvalidArgument ("number of avg should be ntype"));
    OP_REQUIRES (context, (ntypes == std_tensor.shape().dim_size(0)),		errors::InvalidArgument ("number of std should be ntype"));

    OP_REQUIRES (context, (nall * 3 == coord_tensor.shape().dim_size(1)),	errors::InvalidArgument ("number of atoms should match"));
    OP_REQUIRES (context, (nall == type_tensor.shape().dim_size(1)),		errors::InvalidArgument ("number of atoms should match"));
    OP_REQUIRES (context, (9 == box_tensor.shape().dim_size(1)),		errors::InvalidArgument ("number of box should be 9"));
    OP_REQUIRES (context, (ndescrpt == avg_tensor.shape().dim_size(1)),		errors::InvalidArgument ("number of avg should be ndescrpt"));
    OP_REQUIRES (context, (ndescrpt == std_tensor.shape().dim_size(1)),		errors::InvalidArgument ("number of std should be ndescrpt"));

    int nei_mode = 0;
    if (mesh_tensor.shape().dim_size(0) == 16) {
      // lammps neighbor list
      nei_mode = 3;
    }
    else if (mesh_tensor.shape().dim_size(0) == 12) {
      // user provided extended mesh
      nei_mode = 2;
    }
    else if (mesh_tensor.shape().dim_size(0) == 6) {
      // manual copied pbc
      assert (nloc == nall);
      nei_mode = 1;
    }
    else if (mesh_tensor.shape().dim_size(0) == 0) {
      // no pbc
      nei_mode = -1;
    }
    else {
      throw runtime_error("invalid mesh tensor");
    }
    bool b_pbc = true;
    // if region is given extended, do not use pbc
    if (nei_mode >= 1 || nei_mode == -1) {
      b_pbc = false;
    }
    bool b_norm_atom = false;
    if (nei_mode == 1){
      b_norm_atom = true;
    }

    // Create an output tensor
    TensorShape descrpt_shape ;
    descrpt_shape.AddDim (nsamples);
    descrpt_shape.AddDim (nloc * ndescrpt);
    TensorShape descrpt_deriv_shape ;
    descrpt_deriv_shape.AddDim (nsamples);
    descrpt_deriv_shape.AddDim (nloc * ndescrpt * 3);
    TensorShape rij_shape ;
    rij_shape.AddDim (nsamples);
    rij_shape.AddDim (nloc * nnei * 3);
    TensorShape nlist_shape ;
    nlist_shape.AddDim (nsamples);
    nlist_shape.AddDim (nloc * nnei);

    int context_output_index = 0;
    Tensor* descrpt_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(context_output_index++, 
						     descrpt_shape, 
						     &descrpt_tensor));
    Tensor* descrpt_deriv_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(context_output_index++, 
						     descrpt_deriv_shape, 
						     &descrpt_deriv_tensor));
    Tensor* rij_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(context_output_index++, 
						     rij_shape,
						     &rij_tensor));
    Tensor* nlist_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(context_output_index++, 
						     nlist_shape,
						     &nlist_tensor));
    
    auto coord	= coord_tensor	.matrix<VALUETYPE>();
    auto type	= type_tensor	.matrix<int>();
    auto box	= box_tensor	.matrix<VALUETYPE>();
    auto mesh	= mesh_tensor	.flat<int>();
    auto avg	= avg_tensor	.matrix<VALUETYPE>();
    auto std	= std_tensor	.matrix<VALUETYPE>();
    auto descrpt	= descrpt_tensor	->matrix<VALUETYPE>();
    auto descrpt_deriv	= descrpt_deriv_tensor	->matrix<VALUETYPE>();
    auto rij		= rij_tensor		->matrix<VALUETYPE>();
    auto nlist		= nlist_tensor		->matrix<int>();

    // // check the types
    // int max_type_v = 0;
    // for (int ii = 0; ii < natoms; ++ii){
    //   if (type(0, ii) > max_type_v) max_type_v = type(0, ii);
    // }
    // int ntypes = max_type_v + 1;
    OP_REQUIRES (context, (ntypes == int(sel_a.size())),	errors::InvalidArgument ("number of types should match the length of sel array"));
    OP_REQUIRES (context, (ntypes == int(sel_r.size())),	errors::InvalidArgument ("number of types should match the length of sel array"));

    for (int kk = 0; kk < nsamples; ++kk){
      // set region
      boxtensor_t boxt [9] = {0};
      for (int dd = 0; dd < 9; ++dd) {
	boxt[dd] = box(kk, dd);
      }
      SimulationRegion<compute_t > region;
      region.reinitBox (boxt);

      // set & normalize coord
      vector<compute_t > d_coord3 (nall*3);
      for (int ii = 0; ii < nall; ++ii){
	for (int dd = 0; dd < 3; ++dd){
	  d_coord3[ii*3+dd] = coord(kk, ii*3+dd);
	}
	if (b_norm_atom){
	  compute_t inter[3];
	  region.phys2Inter (inter, &d_coord3[3*ii]);
	  for (int dd = 0; dd < 3; ++dd){
	    if      (inter[dd] < 0 ) inter[dd] += 1.;
	    else if (inter[dd] >= 1) inter[dd] -= 1.;
	  }
	  region.inter2Phys (&d_coord3[3*ii], inter);
	}
      }

      // set type
      vector<int > d_type (nall);
      for (int ii = 0; ii < nall; ++ii) d_type[ii] = type(kk, ii);
      
      // build nlist
      vector<vector<int > > d_nlist_a;
      vector<vector<int > > d_nlist_r;
      vector<int> nlist_map;
      bool b_nlist_map = false;
      if (nei_mode == 3) {	
	int * pilist, *pjrange, *pjlist;
	memcpy (&pilist, &mesh(4), sizeof(int *));
	memcpy (&pjrange, &mesh(8), sizeof(int *));
	memcpy (&pjlist, &mesh(12), sizeof(int *));
	int inum = mesh(1);
	assert (inum == nloc);
	d_nlist_a.resize (inum);
	d_nlist_r.resize (inum);
	for (unsigned ii = 0; ii < inum; ++ii){
	  d_nlist_r.reserve (pjrange[inum] / inum + 10);
	}
	for (unsigned ii = 0; ii < inum; ++ii){
	  int i_idx = pilist[ii];
	  for (unsigned jj = pjrange[ii]; jj < pjrange[ii+1]; ++jj){
	    int j_idx = pjlist[jj];
	    d_nlist_r[i_idx].push_back (j_idx);
	  }
	}
      }
      else if (nei_mode == 2) {
	vector<int > nat_stt = {mesh(1-1), mesh(2-1), mesh(3-1)};
	vector<int > nat_end = {mesh(4-1), mesh(5-1), mesh(6-1)};
	vector<int > ext_stt = {mesh(7-1), mesh(8-1), mesh(9-1)};
	vector<int > ext_end = {mesh(10-1), mesh(11-1), mesh(12-1)};
	vector<int > global_grid (3);
	for (int dd = 0; dd < 3; ++dd) global_grid[dd] = nat_end[dd] - nat_stt[dd];
	::build_nlist (d_nlist_a, d_nlist_r, d_coord3, nloc, rcut_a, rcut_r, nat_stt, nat_end, ext_stt, ext_end, region, global_grid);
      }
      else if (nei_mode == 1) {
	vector<double > bk_d_coord3 = d_coord3;
	vector<int > bk_d_type = d_type;
	vector<int > ncell, ngcell;
	copy_coord(d_coord3, d_type, nlist_map, ncell, ngcell, bk_d_coord3, bk_d_type, rcut_r, region);	
	b_nlist_map = true;
	vector<int> nat_stt(3, 0);
	vector<int> ext_stt(3), ext_end(3);
	for (int dd = 0; dd < 3; ++dd){
	  ext_stt[dd] = -ngcell[dd];
	  ext_end[dd] = ncell[dd] + ngcell[dd];
	}
	::build_nlist (d_nlist_a, d_nlist_r, d_coord3, nloc, rcut_a, rcut_r, nat_stt, ncell, ext_stt, ext_end, region, ncell);
      }
      else if (nei_mode == -1){
	::build_nlist (d_nlist_a, d_nlist_r, d_coord3, rcut_a, rcut_r, NULL);
      }
      else {
	throw runtime_error("unknow neighbor mode");
      }

      // loop over atoms, compute descriptors for each atom
#pragma omp parallel for 
      for (int ii = 0; ii < nloc; ++ii){
	vector<int> fmt_nlist_a;
	vector<int> fmt_nlist_r;
	int ret = -1;
	if (fill_nei_a){
	  if ((ret = format_nlist_fill_a (fmt_nlist_a, fmt_nlist_r, d_coord3, ntypes, d_type, region, b_pbc, ii, d_nlist_a[ii], d_nlist_r[ii], rcut_r, sec_a, sec_r)) != -1){
	    if (count_nei_idx_overflow == 0) {
	      cout << "WARNING: Radial neighbor list length of type " << ret << " is not enough" << endl;
	      flush(cout);
	      count_nei_idx_overflow ++;
	    }
	  }
	}

	vector<compute_t > d_descrpt_a;
	vector<compute_t > d_descrpt_a_deriv;
	vector<compute_t > d_descrpt_r;
	vector<compute_t > d_descrpt_r_deriv;
	vector<compute_t > d_rij_a;
	vector<compute_t > d_rij_r;      
	compute_descriptor_se_a (d_descrpt_a,
				 d_descrpt_a_deriv,
				 d_rij_a,
				 d_coord3,
				 ntypes, 
				 d_type,
				 region, 
				 b_pbc,
				 ii, 
				 fmt_nlist_a,
				 sec_a, 
				 rcut_r_smth, 
				 rcut_r,
         //map
         map_x1,
         map_k1,
         map_b1,
         map_x2,
         map_k2,
         map_b2,
         precs);

	// check sizes
	assert (d_descrpt_a.size() == ndescrpt_a);
	assert (d_descrpt_a_deriv.size() == ndescrpt_a * 3);
	assert (d_rij_a.size() == nnei_a * 3);
	assert (int(fmt_nlist_a.size()) == nnei_a);
	// record outputs
	for (int jj = 0; jj < ndescrpt_a; ++jj) {
	  descrpt(kk, ii * ndescrpt + jj) = (d_descrpt_a[jj] - avg(d_type[ii], jj)) / std(d_type[ii], jj);
	}
	for (int jj = 0; jj < ndescrpt_a * 3; ++jj) {
	  descrpt_deriv(kk, ii * ndescrpt * 3 + jj) = d_descrpt_a_deriv[jj] / std(d_type[ii], jj/3);
	}
	for (int jj = 0; jj < nnei_a * 3; ++jj){
	  rij (kk, ii * nnei * 3 + jj) = d_rij_a[jj];
	}
	for (int jj = 0; jj < nnei_a; ++jj){
	  int record = fmt_nlist_a[jj];
	  if (b_nlist_map && record >= 0) {
	    record = nlist_map[record];
	  }
	  nlist (kk, ii * nnei + jj) = record;
	}
      }
    }
  }
private:
  float rcut_a;
  float rcut_r;
  float rcut_r_smth;
  vector<int32> sel_r;
  vector<int32> sel_a;
  vector<int> sec_a;
  vector<int> sec_r;
  int ndescrpt, ndescrpt_a, ndescrpt_r;
  int nnei, nnei_a, nnei_r;
  bool fill_nei_a;
  int count_nei_idx_overflow;
  void 
  cum_sum (vector<int> & sec,
	   const vector<int32> & n_sel) const {
    sec.resize (n_sel.size() + 1);
    sec[0] = 0;
    for (int ii = 1; ii < sec.size(); ++ii){
      sec[ii] = sec[ii-1] + n_sel[ii-1];
    }
  }
  

  // MZ_DEFINE
  // ==============================
  /*

  由于rij->r2，r2是随距离呈平方关系增长的
  并不好直接使用r2来进行映射
  并且随着r2逐渐增大，应该作用力更弱，曲线更加平滑
  因此，构建操作r2->u
  此操作将r2从[0, rc2]的范围变换到[0, 1]

  */
  // ==============================
  //

  double map_x1[2], map_k1[3], map_b1[3];
  double map_x2[2], map_k2[3], map_b2[3];
  double rc2;
  double precs[3];

  void init_precs(){
    precs[0] = 8192; // 2^14 NBIT_DATA_FL
    // precs[0] = 524288; // 2^19
    // precs[0] = 281474976710656; // 2^48 NBIT_LONG_DATA_FL
    precs[1] = 1024; // NBIT_FEA_X
    precs[2] = 1024; // NBIT_FEA_FL
  }

  void xp(double &x, double &y, double k1, double b1, double k2, double b2){
    /*
    # 功能
    给出线段的斜率和偏移，计算出两条直线的交点
    # Input
    k1, b1, k2, b2
    # Output
    x, y
    */
    x = (b2 - b1) / (k1 - k2);
    y = k1 * x + b1;
  }

  void
  cal_map_param(double rc){
    rc2 = rc * rc;
    double th = log2(1.5);
    // k
    double ln2_k1 = ceil(log2(rc2/4) - th);
    double ln2_k2 = ceil(log2(rc2*1) - th);
    double ln2_k3 = ceil(log2(rc2*4) - th);

    double k1 = 1 / double(1 << int(ln2_k1));
    double k2 = 1 / double(1 << int(ln2_k2));
    double k3 = 1 / double(1 << int(ln2_k3));
    // b
    double b1 = 0;
    double b2 = 0;
    double b3 = 1.0 - rc2 * k3;
    b2 = b3 * 0.5;
    // xp
    double x13, y13;
    double x12, y12;
    double x23, y23;
    double dx;
    // xp(x13, y13, k1, b1, k2, b2);
    // b2 = y13 - x13 * k3;
    // 迭代求解相交的点
    while(1){
      xp(x12, y12, k1, b1, k2, b2);
      xp(x23, y23, k2, b2, k3, b3);
      dx = (x23 - x12) - rc2 / 2;
      if (abs(dx) < 1e-6) break;
      else b2 += 0.01 * dx;
    }

    double prec = precs[0];
    x12 = round(x12 * prec) / prec;
    x23 = round(x23 * prec) / prec;

    k1 = round(k1 * prec) / prec;
    k2 = round(k2 * prec) / prec;
    k3 = round(k3 * prec) / prec;

    b1 = round(b1 * prec) / prec;
    b2 = round(b2 * prec) / prec;
    b3 = round(b3 * prec) / prec;

    map_x1[0] = x12;
    map_x1[1] = x23;

    map_k1[0] = k1;
    map_k1[1] = k2;
    map_k1[2] = k3;

    map_b1[0] = b1;
    map_b1[1] = b2;
    map_b1[2] = b3;
    //print
    // printf("INFO: cal_map_param\n");
    // printf("xp: %16.10f %16.10f \n", map_x1[0], map_x1[1]);
    // printf("k: %16.10f %16.10f %16.10f \n", map_k1[0], map_k1[1], map_k1[2]);
    // printf("b: %16.10f %16.10f %16.10f \n", map_b1[0], map_b1[1], map_b1[2]);
    // u2r
    map_x2[0] = y12;
    map_x2[1] = y23;

    map_k2[0] = 1 / (map_k1[0] * rc2);
    map_k2[1] = 1 / (map_k1[1] * rc2);
    map_k2[2] = 1 / (map_k1[2] * rc2);

    map_b2[0] = 0;
    map_b2[1] = (x12/rc2) - y12 * map_k2[1];
    map_b2[2] = 1.0       - 1.0 * map_k2[2];
  }
};

REGISTER_KERNEL_BUILDER(Name("MzDescrptSeA").Device(DEVICE_CPU), MzDescrptSeAOp);


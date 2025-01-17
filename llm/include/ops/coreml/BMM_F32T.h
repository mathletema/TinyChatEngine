#include "common.h"

class BMM_F32T_coreml {
   public:
    BMM_F32T_coreml(float _alpha);
    BMM_F32T_coreml(){};
    void forward(const Matrix3D<float> &x, const Matrix3D<float> &weight, Matrix3D<float> &output);
    void forward_weight_untransposed(const Matrix3D<float> &x, const Matrix3D<float> &weight, Matrix3D<float> &output);
    float alpha;

   private:
    std::string profile_name = "BMM_F32T_coreml";
};

void load_BMM_F32T_coreml(BMM_F32T_coreml &op, std::string prefix);
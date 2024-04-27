/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// TODO: Check that this part is correct
// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}


// MARK: - Preprocess
// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template<int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const float3* means,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* projmatrix,
	const glm::vec3* campos,
	float3* dL_dmean2D,
	glm::vec3* dL_dmeans,
	float* dL_dcolor,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	// Taking care of gradients from the screenspace points
	//TODO: mean3d->mean2d is not correct
	float3 mean3d = means[idx];
	// const float4 p_hom = transformPoint4x4(mean3d, projmatrix);
	// const float p_w = 1.0f / (p_hom.w + 0.0000001f);
	// const glm::vec3 mean2d = { p_hom.x * p_w, p_hom.y * p_w, 0. };

	// float3 mean3d_det = {dL_dmeans[idx].x + mean3d.x, dL_dmeans[idx].y + mean3d.y, dL_dmeans[idx].z + mean3d.z};
	// // dL_mean3d -> dL_mean2d due to 3d->2d projection
	// const float4 p_hom_det = transformPoint4x4(mean3d_det, projmatrix);
	// const float p_w_det = 1.0f / (p_hom_det.w + 0.0000001f);
	// const glm::vec3 mean2d_det = { p_hom_det.x * p_w_det, p_hom_det.y * p_w_det, 0. };
	// dL_dmean2D[idx].x = mean2d_det.x - mean2d.x;
	// dL_dmean2D[idx].y = mean2d_det.y - mean2d.y;

	// Compute gradient updates due to computing colors from SHs
	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3*)means, *campos, shs, clamped, (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dmeans, (glm::vec3*)dL_dsh);
	dL_dmean2D[idx].x = dL_dmeans[idx].x * mean3d.z;
	dL_dmean2D[idx].y = dL_dmeans[idx].y * mean3d.z;
	
}

//TODO: add Weight and Height
// MARK: - Rendering
// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const float3* means3D,
	const uint32_t* __restrict__ point_list,
	int Weight, int Height,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image, 
	const float* __restrict__ opacity,
	const glm::vec4* rotations,
	const glm::vec3* scales,
	const float scale_modifier,
	const float* projmatrix,
	const float* viewmatrix,
	const float* __restrict__ colors,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float3* __restrict__ ddas,
	const float* __restrict__ dL_dpixels,
	const float* __restrict__ dL_dmedian_depths,
	const float* __restrict__ dL_dloss_dd,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors,
	float3* __restrict__ dL_dmean2D,
	float3* __restrict__ dL_dmean3D,
	float3* __restrict__ dL_dscales,
	float4* __restrict__ dL_drot
	)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (Weight + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, Weight), min(pix_min.y + BLOCK_Y ,Height) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = Weight * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };
	const glm::vec4 h_x = {-1.0f, 0.f, 0.0f, Pix2ndc(pixf.x,Weight)};
	const glm::vec4 h_y = {0.f, -1.0f, 0.0f, Pix2ndc(pixf.y,Height)};

	const bool inside = pix.x < Weight&& pix.y < Height;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;
	bool nc = false;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float collected_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];
	__shared__ float3 collected_mean3d[BLOCK_SIZE];
	__shared__ glm::vec4 collected_rot[BLOCK_SIZE];
	__shared__ glm::vec3 collected_scale[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = { 0 };
	float dL_dpixel[C];
	float3 dda;
	float dL_dmedian_depth;
	if (inside){
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * Height * Weight + pix_id];
		dda = ddas[pix_id];
		dL_dmedian_depth = dL_dmedian_depths[pix_id];
	}

	float last_alpha = 0;
	float last_color[C] = { 0 };
	float A = 0;
	float D_1 = 0;
	float D_2 = 0;
	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_opacity[block.thread_rank()] = opacity[coll_id];
			collected_rot[block.thread_rank()] = rotations[coll_id];
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_scale[block.thread_rank()] = scales[coll_id];
			collected_mean3d[block.thread_rank()] = means3D[coll_id];

			for (int i = 0; i < C; i++){
				collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];
			}
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			bool filter = false;
			contributor--;
			if (contributor >= last_contributor)
				continue;

			float2 xy = collected_xy[j];

			const glm::vec3 scale = collected_scale[j];
			const glm::vec4 rot = collected_rot[j];
			float3 p = collected_mean3d[j];
			glm::mat3 S = glm::mat3(1.0f);
			S[0][0] = scale_modifier * scale.x;
			S[1][1] = scale_modifier * scale.y;
			S[2][2] = 0.;
			const glm::vec4 q = rot;
			const float r = q.x;
			const float x = q.y;
			const float y = q.z;
			const float z = q.w;
			const glm::vec3 tu = glm::vec3(1.f - 2.f * (y * y + z * z), 2.f * (x * y + r * z), 2.f * (x * z - r * y));
			const glm::vec3 tv = glm::vec3(2.f * (x * y - r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z + r * x));
			const glm::vec3 tw = glm::cross(tu, tv); 
			const glm::mat3 R = glm::mat3(tu, tv, tw);
			const glm::mat3 RS = R * S;
			glm::mat4 H;
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					H[i][j] = RS[i][j];
				}
			}
			H[0][3] = 0.;
			H[1][3] = 0.;
			H[2][3] = 0.;
			H[3] = glm::vec4(p.x, p.y, p.z, 1.0f);
			glm::mat4 W = glm::mat4(projmatrix[0], projmatrix[1], projmatrix[2], projmatrix[3],
				projmatrix[4], projmatrix[5], projmatrix[6], projmatrix[7],
				projmatrix[8], projmatrix[9], projmatrix[10], projmatrix[11],
				projmatrix[12], projmatrix[13], projmatrix[14], projmatrix[15]);
				
			const glm::vec4 hu = glm::transpose(W * H) * h_x;
			const glm::vec4 hv = glm::transpose(W * H) * h_y;
			// Compute blending values, as before.
			const float o = collected_opacity[j];
			const float u_num = hu.w * hv.y - hu.y * hv.w;
			const float v_num = - hu.w * hv.x + hu.x * hv.w ;
			const float denom = hu.y * hv.x - hu.x * hv.y;
			const float u = u_num / denom;
			const float v = v_num / denom;
			float power = -0.5f * (u * u + v * v);
			if (power > 0.0f)
				continue;

			const float2 d = {pixf.x - xy.x, pixf.y - xy.y};
			const float power_filter = - (d.x * d.x  + d.y * d.y);
			// if (power_filter > power){
			// 	filter = true;
			// 	power = power_filter;
			// }
			
			const float G = exp(power);
			const float alpha = min(0.99f, o * G);
			if (alpha < 1.0f / 255.0f)
				continue;
			// NC loss
			float dL_du = 0.f;
			float dL_dv = 0.f;

			float test_T = T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;

			float z_origin = (W * H * glm::vec4(u, v, 1, 1)).w;
			float zndc = 1000 / (1000 - 0.2) - 0.2 * 1000 / (1000 - 0.2) / z_origin;
			if (j == 0){
				A = dda.z - alpha * test_T;
				D_1 = dda.x - alpha * test_T * zndc;
				D_2 = dda.y - alpha * test_T * zndc * zndc;
			}
			const float dL_dzndc = 2.f * alpha * test_T * (A - D_1) * dL_dloss_dd[pix_id];
			const float dL_dw = (D_2 + A * zndc * zndc - 2.f * zndc * D_1) * dL_dloss_dd[pix_id];
			glm::vec3 dL_dp = {0,0,0};
			if (j == 0 && T_final > 0.5f){
				dL_dp = {viewmatrix[2] * dL_dmedian_depth, viewmatrix[6] * dL_dmedian_depth, viewmatrix[10] * dL_dmedian_depth};
			}
			//TODO: check if correct
			if (test_T > 0.5f && T < 0.5){
				dL_dp = {viewmatrix[2] * dL_dmedian_depth, viewmatrix[6] * dL_dmedian_depth, viewmatrix[10] * dL_dmedian_depth};
				}

			// dL_dp = f(dL_dzndc);

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).


			float dL_dalpha = 0.0f;
			const int global_id = collected_id[j];
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;

				const float dL_dchannel = dL_dpixel[ch];
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				// Update the gradients w.r.t. color of the Gaussian. 
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
			}
			dL_dalpha *= test_T;
			T = test_T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;
			if (!filter){
				dL_du += o * (dL_dalpha+dL_dw) * G * (-u);
				dL_dv += o * (dL_dalpha+dL_dw) * G * (-v);	
			}

			const float du_dhux = u * hv.y / denom;
			const float du_dhuy = (- hv.w ) / denom - hv.x * u / denom;
			const float du_dhuw = hv.y / denom;

			const float du_dhvx = - u * hu.y / denom;
			const float du_dhvy = (hu.w) / denom + hu.x * u / denom;
			const float du_dhvw = - hu.y / denom;

			const float dv_dhux = (hv.w) / denom + hv.y * v / denom;
			const float dv_dhuy = - hv.x * v / denom;
			const float dv_dhuw = - hv.x / denom;

			const float dv_dhvx = (-hu.w) / denom - hu.y * v / denom;
			const float dv_dhvy = hu.x * v / denom;
			const float dv_dhvw = hu.x / denom;

			// dL_dhu -> dL_dcov3D, dL_mean3d
			//TODO: modify to matrix form
			const glm::vec4 W_T_hx = glm::transpose(W) * h_x;
			const glm::vec4 W_T_hy = glm::transpose(W) * h_y;
			const float3 dhuw_dmean3d = { W_T_hx.x, W_T_hx.y, W_T_hx.z };
			const float3 dhvw_dmean3d = { W_T_hy.x, W_T_hy.y, W_T_hy.z };

			const float3 du_dtu = { W_T_hx.x * S[0][0] * du_dhux + W_T_hy.x * S[1][1] * du_dhvx, 
									W_T_hx.y * S[0][0] * du_dhux + W_T_hy.y * S[1][1] * du_dhvx,
									W_T_hx.z * S[0][0] * du_dhux + W_T_hy.z * S[1][1] * du_dhvx};
			const float3 du_dtv = { W_T_hx.x * S[0][0] * du_dhuy + W_T_hy.x * S[1][1] * du_dhvy, 
									W_T_hx.y * S[0][0] * du_dhuy + W_T_hy.y * S[1][1] * du_dhvy,
									W_T_hx.z * S[0][0] * du_dhuy + W_T_hy.z * S[1][1] * du_dhvy};
			const float3 dv_dtu = { W_T_hx.x * S[0][0] * dv_dhux + W_T_hy.x * S[1][1] * dv_dhvx, 
									W_T_hx.y * S[0][0] * dv_dhux + W_T_hy.y * S[1][1] * dv_dhvx,
									W_T_hx.z * S[0][0] * dv_dhux + W_T_hy.z * S[1][1] * dv_dhvx};
			const float3 dv_dtv = { W_T_hx.x * S[0][0] * dv_dhuy + W_T_hy.x * S[1][1] * dv_dhvy, 
									W_T_hx.y * S[0][0] * dv_dhuy + W_T_hy.y * S[1][1] * dv_dhvy,
									W_T_hx.z * S[0][0] * dv_dhuy + W_T_hy.z * S[1][1] * dv_dhvy};

			const float2 du_dscale = { du_dhux * (W_T_hx.x * tu.x + W_T_hx.y * tu.y + W_T_hx.z * tu.z) + du_dhvx * (W_T_hy.x * tu.x + W_T_hy.y * tu.y + W_T_hy.z * tu.z), 
									   du_dhuy * (W_T_hx.x * tu.x + W_T_hx.y * tu.y + W_T_hx.z * tu.z) + du_dhvy * (W_T_hy.x * tu.x + W_T_hy.y * tu.y + W_T_hy.z * tu.z)};
			const float2 dv_dscale = { dv_dhux * (W_T_hx.x * tv.x + W_T_hx.y * tv.y + W_T_hx.z * tv.z) + dv_dhvx * (W_T_hy.x * tv.x + W_T_hy.y * tv.y + W_T_hy.z * tv.z), 
									   dv_dhuy * (W_T_hx.x * tv.x + W_T_hx.y * tv.y + W_T_hx.z * tv.z) + dv_dhvy * (W_T_hy.x * tv.x + W_T_hy.y * tv.y + W_T_hy.z * tv.z)};

			const float4 du_dr = { 		   			       du_dtu.y *   2.  * z - du_dtu.z * 2. * y
								  +	du_dtv.x * (-2.) * z +                        du_dtv.z * 2. * x,
														   du_dtu.y *   2.  * y + du_dtu.z * 2. * z
								  + du_dtv.x *   2.  * y + du_dtv.y * (-4.) * x + du_dtv.z * 2. * r,
									du_dtu.x * (-4.) * y + du_dtu.y *   2.  * x - du_dtu.z * 2. * r
								  + du_dtv.x *   2.  * x + 						  du_dtv.z * 2. * z,
									du_dtu.x * (-4.) * z + du_dtu.y *   2.  * r + du_dtu.z * 2. * x
								  + du_dtv.x * (-2.) * r + du_dtv.y * (-4.) * z	+ du_dtv.z * 2. * y };
			const float4 dv_dr = { 		   			       dv_dtu.y *   2.  * z - dv_dtu.z * 2. * y
								  + dv_dtv.x * (-2.) * z +                        dv_dtv.z * 2. * x,
								  						   dv_dtu.y *   2.  * y + dv_dtu.z * 2. * z
								  + dv_dtv.x *   2.  * y + dv_dtv.y * (-4.) * x + dv_dtv.z * 2. * r,
								  	dv_dtu.x * (-4.) * y + dv_dtu.y *   2.  * x - dv_dtu.z * 2. * r
								  + dv_dtv.x *   2.  * x + 						  dv_dtv.z * 2. * z,
								  	dv_dtu.x * (-4.) * z + dv_dtu.y *   2.  * r + dv_dtu.z * 2. * x
								  + dv_dtv.x * (-2.) * r + dv_dtv.y * (-4.) * z	+ dv_dtv.z * 2. * y };
			
			float3 du_dmean3d = {du_dhuw * dhuw_dmean3d.x, du_dhuw * dhuw_dmean3d.y, du_dhuw * dhuw_dmean3d.z};
			float3 dv_dmean3d = {dv_dhuw * dhvw_dmean3d.x, dv_dhuw * dhvw_dmean3d.y, dv_dhuw * dhvw_dmean3d.z};
			// p_view (x,y,z) -> mean3d (x,y,z)
			glm::mat3 dp_view_dmean3d = {RS[0][0] * du_dmean3d.x + RS[1][0] * dv_dmean3d.x + 1,
										RS[0][0] * du_dmean3d.y + RS[1][0] * dv_dmean3d.y,
										RS[0][0] * du_dmean3d.z + RS[1][0] * dv_dmean3d.z,
										RS[0][1] * du_dmean3d.x + RS[1][1] * dv_dmean3d.x,
										RS[0][1] * du_dmean3d.y + RS[1][1] * dv_dmean3d.y + 1,
										RS[0][1] * du_dmean3d.z + RS[1][1] * dv_dmean3d.z,
										RS[0][2] * du_dmean3d.x + RS[1][2] * dv_dmean3d.x,
										RS[0][2] * du_dmean3d.y + RS[1][2] * dv_dmean3d.y,
										RS[0][2] * du_dmean3d.z + RS[1][2] * dv_dmean3d.z + 1};
			glm::vec3 dL_dmean3d = dp_view_dmean3d * dL_dp;
								
			glm::mat3 dp_view_dscale = {RS[0][0] * du_dscale.x + RS[1][0] * dv_dscale.x + R[0][0] * u,
										RS[0][0] * du_dscale.y + RS[1][0] * dv_dscale.y + R[1][0] * v,
										0,
										RS[0][1] * du_dscale.x + RS[1][1] * dv_dscale.x + R[0][1] * u,
										RS[0][1] * du_dscale.y + RS[1][1] * dv_dscale.y + R[1][1] * v,
										0,
										RS[0][2] * du_dscale.x + RS[1][2] * dv_dscale.x + R[0][2] * u,
										RS[0][2] * du_dscale.y + RS[1][2] * dv_dscale.y + R[1][2] * v,
										0};
			glm::vec3 dL_dscale = dp_view_dscale * dL_dp;
			// p_view (x,y,z) -> tu
			glm::mat3 dp_view_dtu = {S[0][0] * u + RS[0][0] * du_dtu.x + RS[1][0] * dv_dtu.x,
									RS[0][0] * du_dtu.y + RS[1][0] * dv_dtu.y,
									RS[0][0] * du_dtu.z + RS[1][0] * dv_dtu.z,
									S[1][1] * u + RS[0][1] * du_dtu.x + RS[1][1] * dv_dtu.x,
									RS[0][1] * du_dtu.y + RS[1][1] * dv_dtu.y,
									RS[0][1] * du_dtu.z + RS[1][1] * dv_dtu.z,
									S[2][2] * u + RS[0][2] * du_dtu.x + RS[1][2] * dv_dtu.x,
									RS[0][2] * du_dtu.y + RS[1][2] * dv_dtu.y,
									RS[0][2] * du_dtu.z + RS[1][2] * dv_dtu.z};
			glm::vec3 dL_dtu = dp_view_dtu * dL_dp;
			// p_view (x,y,z) -> tv1, tv2, tv3						 						
			glm::mat3 dp_view_dtv = {RS[0][0] * du_dtv.x + S[1][0] * v + RS[1][0] * dv_dtv.x,
									RS[0][0] * du_dtv.y + RS[1][0] * dv_dtv.y,
									RS[0][0] * du_dtv.z + RS[1][0] * dv_dtv.z,
									RS[0][1] * du_dtv.x + RS[1][1] * dv_dtv.x,
									RS[0][1] * du_dtv.y + S[1][1] * v + RS[1][1] * dv_dtv.y,
									RS[0][1] * du_dtv.z + RS[1][1] * dv_dtv.z,
									RS[0][2] * du_dtv.x + RS[1][2] * dv_dtv.x,
									RS[0][2] * du_dtv.y + RS[1][2] * dv_dtv.y,
									RS[0][2] * du_dtv.z + S[1][2] * v + RS[1][2] * dv_dtv.z};
			glm::vec3 dL_dtv = dp_view_dtv * dL_dp;
			const float4 dL_dr = { 		   			       dL_dtu.y *   2.  * z - dL_dtu.z * 2. * y
				+	dL_dtv.x * (-2.) * z +                        dL_dtv.z * 2. * x,
										 dL_dtu.y *   2.  * y + dL_dtu.z * 2. * z
				+ dL_dtv.x *   2.  * y + dL_dtv.y * (-4.) * x + dL_dtv.z * 2. * r,
				  dL_dtu.x * (-4.) * y + dL_dtu.y *   2.  * x - dL_dtu.z * 2. * r
				+ dL_dtv.x *   2.  * x + 						  dL_dtv.z * 2. * z,
				  dL_dtu.x * (-4.) * z + dL_dtu.y *   2.  * r + dL_dtu.z * 2. * x
				+ dL_dtv.x * (-2.) * r + dL_dtv.y * (-4.) * z	+ dL_dtv.z * 2. * y };


		
			// Update gradients w.r.t. opacity of the Gaussian
			atomicAdd(&(dL_dopacity[global_id]), G * (dL_dalpha + dL_dw));
			atomicAdd(&(dL_dmean3D[global_id].x), dL_du * du_dmean3d.x + dL_dv * dv_dmean3d.x + dL_dmean3d.x);
			atomicAdd(&(dL_dmean3D[global_id].y), dL_du * du_dmean3d.y + dL_dv * dv_dmean3d.y + dL_dmean3d.y);
			atomicAdd(&(dL_dmean3D[global_id].z), dL_du * du_dmean3d.z + dL_dv * dv_dmean3d.z + dL_dmean3d.z);
			atomicAdd(&(dL_dscales[global_id].x), du_dscale.x * dL_du + dv_dscale.x * dL_dv + dL_dscale.x);
			atomicAdd(&(dL_dscales[global_id].y), du_dscale.y * dL_du + dv_dscale.y * dL_dv + dL_dscale.y);
			atomicAdd(&(dL_drot[global_id].x), du_dr.x * dL_du + dv_dr.x * dL_dv + dL_dr.x);
			atomicAdd(&(dL_drot[global_id].y), du_dr.y * dL_du + dv_dr.y * dL_dv + dL_dr.y);
			atomicAdd(&(dL_drot[global_id].z), du_dr.z * dL_du + dv_dr.z * dL_dv + dL_dr.z);
			atomicAdd(&(dL_drot[global_id].w), du_dr.w * dL_du + dv_dr.w * dL_dv + dL_dr.w);
		}
	}
}

void BACKWARD::preprocess(
	int P, int D, int M,
	const float3* means3D,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* cov3Ds,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const glm::vec3* campos,
	float3* dL_dmean2D,
	const float* dL_dconic,
	glm::vec3* dL_dmean3D,
	float* dL_dcolor,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{

	// Propagate gradients for remaining steps: finish 3D mean gradients,
	// propagate color gradients to SH (if desireD), propagate 3D covariance
	// matrix gradients to scale and rotation.
	preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P, D, M,
		(float3*)means3D,
		radii,
		shs,
		clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		projmatrix,
		campos,
		(float3*)dL_dmean2D,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		dL_dscale,
		dL_drot);
}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
	const uint2* ranges,
	const float3* means3D,
	const uint32_t* point_list,
	int W, int H,
	const float* bg_color,
	const float2* means2D,
	const float* opacity,
	const glm::vec4* rotations,
	const glm::vec3* scales,
	const float scale_modifier,
	const float* projmatrix,
	const float* viewmatrix,
	const float* colors,
	const float* final_Ts,
	const uint32_t* n_contrib,
	const float3* ddas,
	const float* dL_dpixels,
	const float* dL_dmedian_depth,
	const float* dL_dloss_dd,
	float* dL_dopacity,
	float* dL_dcolors,
	float3* dL_dmean2D,
	float3* dL_dmean3D,
	float3* dL_dscales,
	float4* dL_drot)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		means3D,
		point_list,
		W, H,
		bg_color,
		means2D,
		opacity,
		rotations,
		scales,
		scale_modifier,
		projmatrix,
		viewmatrix,
		colors,
		final_Ts,
		n_contrib,
		ddas,
		dL_dpixels,
		dL_dmedian_depth,
		dL_dloss_dd,
		dL_dopacity,
		dL_dcolors,
		dL_dmean2D,
		dL_dmean3D,
		dL_dscales,
		dL_drot);
}